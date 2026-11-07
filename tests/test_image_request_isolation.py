"""Real request-routing/cache methods against inert CPU dependencies."""
import ast
import copy
import json
import re
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image
from modules.model_identity import file_signature
from modules.image_guidance import normalize_reference_images
from modules import llama_vision, chat_tools

ROOT = Path(__file__).resolve().parents[1]


def method(name, scope):
    tree = ast.parse((ROOT/'modules/image_pipeline.py').read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node],type_ignores=[]),name,'exec'), scope)
    return scope[name]


class ImageSourceTests(unittest.TestCase):
    def route(self, control, image, native=False, references=None):
        tree = ast.parse((ROOT/'modules/image_pipeline.py').read_text(encoding='utf-8'))
        process = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'process')
        start = next(i for i,n in enumerate(process.body) if isinstance(n,ast.Assign)
                     and ast.unparse(n.targets[0]) == 'controlnet')
        stop = next(i for i,n in enumerate(process.body) if isinstance(n,ast.Assign)
                    and ast.unparse(n.targets[0]) == 'device')
        data = {'input_image':image,'reference_images':references}
        controls = SimpleNamespace(get_settings=lambda data:dict(control),
            controlnet_models={k:None for k in ('img2img','edit','canny','depth','sketch','recolour')})
        scope = dict(gen_data=data, modules=SimpleNamespace(controlnet=controls), architecture='MOSSImage' if native else 'SDXL',
            image_edit=native, hidream_edit=False, guidance={'moss_image_edit_format':'Independent'},
            normalize_reference_images=normalize_reference_images)
        exec(compile(ast.Module(body=process.body[start:stop],type_ignores=[]),'source_route','exec'),scope)
        return scope

    def test_retained_upload_and_extra_references_are_not_used_for_fresh_requests(self):
        image = Image.new('RGBA',(137,93),(1,2,3,42))
        for native in (False,True):
            first = self.route({'type':'img2img','denoise':0.3},image,native)
            self.assertIs(first['input_image'],image)
            fresh = self.route({},image,native,references=[image])
            self.assertIsNone(fresh['input_image'])
            self.assertEqual(fresh['additional_references'],[])
            self.assertEqual(fresh['gen_data']['reference_image_count'],0)
            self.assertIs(fresh['gen_data']['input_image'],image)
            self.assertEqual(image.getpixel((0,0)),(1,2,3,42))

    def test_selected_tool_requires_current_source_and_valid_denoise(self):
        for control in ({'type':'img2img'},{'type':'edit'}):
            with self.subTest(control=control), self.assertRaisesRegex(ValueError,'Add an image'):
                self.route(control,None,native=True)
        with self.assertRaisesRegex(ValueError,'image-edit checkpoint'):
            self.route({'type':'edit'},Image.new('RGB',(64,64)))
        with self.assertRaisesRegex(ValueError,'denoise'):
            self.route({'type':'img2img','denoise':0},Image.new('RGB',(64,64)))

    def test_chat_reference_is_explicit_owned_and_keeps_upload_pixels(self):
        upload=Image.new('RGBA',(137,93),(6,7,8,31));upload.info['note']='original'
        image,prompt=llama_vision.selected_image_reference('New image',upload,[])
        self.assertIsNone(image);self.assertEqual(prompt,'')
        image,prompt=llama_vision.selected_image_reference('Uploaded image',upload,[])
        self.assertIsNot(image,upload);self.assertEqual(image.tobytes(),upload.tobytes())
        self.assertEqual(image.info,upload.info)
        with self.assertRaises(ValueError):
            llama_vision.selected_image_reference('Previous generated image',None,
                [{'role':'assistant','content':'![Image](file:///private.png)'}])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'owned.png';upload.save(path)
            url='gradio_api/file=owned-isolation-test'
            llama_vision.remember_image(url,path,'original visual prompt')
            try:
                history=[{'role':'assistant','content':f'![Image]({url})'},
                         {'role':'user','content':'change the sky'}]
                image,prompt=llama_vision.selected_image_reference('Previous generated image',None,history)
                self.assertEqual(image.tobytes(),upload.tobytes())
                self.assertEqual(prompt,'original visual prompt')
                self.assertEqual(llama_vision.selected_image_reference('New image',None,history),(None,''))
            finally:
                llama_vision._generated_images.pop(url,None);llama_vision._generated_prompts.pop(url,None)


class CacheTests(unittest.TestCase):
    def test_prompt_a_b_a_and_clip_skip_changes_encode_exact_current_text(self):
        encode=Mock(side_effect=lambda **kw:([kw['text'],kw['clip']],))
        owner=SimpleNamespace(conditions={'+':{'text':None,'cache':None}},
            xl_base_patched=SimpleNamespace(clip='native'))
        run=method('textencode',{'CLIPTextEncode':lambda:SimpleNamespace(encode=encode),
            'CLIPSetLastLayer':lambda:SimpleNamespace(set_last_layer=lambda clip,layer:(f'{clip}:{layer}',))})
        for text,skip in [('A "BLUE 42"',1),('B cat',1),('A "BLUE 42"',1),('A "BLUE 42"',2)]:
            run(owner,'+',text,skip)
            self.assertEqual(owner.conditions['+']['cache'][0],text)
        run(owner,'+','A "BLUE 42"',2)
        self.assertEqual(encode.call_count,4)
        self.assertEqual(owner.conditions['+']['cache'][1],'native:-2')
        self.assertEqual(owner.xl_base_patched.clip,'native')

    def test_lora_a_b_a_off_rebuilds_from_base_and_changed_file_invalidates(self):
        def model(**kw): return SimpleNamespace(**kw)
        base=model(unet=SimpleNamespace(patches={}),clip=SimpleNamespace(patcher=SimpleNamespace(patches={})),
                   vae=object(),clip_vision=None)
        calls=[]
        def apply(unet,clip,lora,weight,clip_weight):
            calls.append((unet,lora,weight))
            patched=SimpleNamespace(patches=copy.deepcopy(unet.patches))
            patched.patches.setdefault('weight',[]).append((lora,weight))
            return patched,clip
        with tempfile.TemporaryDirectory() as folder:
            paths={name:Path(folder)/name for name in ('A','B')}
            for path in paths.values(): path.write_bytes(b'model')
            scope=dict(file_signature=file_signature,shared=SimpleNamespace(models=SimpleNamespace(
                get_model_path=lambda kind,name,hash=None:paths[name])),
                comfy=SimpleNamespace(utils=SimpleNamespace(load_torch_file=lambda path,safe_load:Path(path).name),
                    sd=SimpleNamespace(load_lora_for_models=apply)))
            run=method('load_loras',scope)
            owner=SimpleNamespace(xl_base=base,xl_base_patched=base,xl_base_patched_hash=None,
                                  conditions={'old':'tensor'},StableDiffusionModel=model)
            for name in ('A','B','A'):
                run(owner,[{'name':name,'weight':0.7}])
                self.assertEqual(owner.xl_base_patched.unet.patches['weight'],[(name,0.7)])
                self.assertIsNone(owner.conditions)
            run(owner,[{'name':'A','weight':0.7}]);self.assertEqual(len(calls),3)
            paths['A'].write_bytes(b'replaced model')
            run(owner,[{'name':'A','weight':0.7}]);self.assertEqual(len(calls),4)
            run(owner,[])
            self.assertIs(owner.xl_base_patched,base)
            self.assertTrue(all(row[0] is base.unet for row in calls))
            scope['comfy'].sd.load_lora_for_models=lambda unet,clip,*args:(unet,clip)
            with self.assertRaisesRegex(RuntimeError,'No compatible weights'):
                run(owner,[{'name':'B','weight':1}])
            self.assertIs(owner.xl_base_patched,base)


class ChatSourceTests(unittest.TestCase):
    def test_review_enabled_does_not_implicitly_attach_previous_images(self):
        tree=ast.parse((ROOT/'modules/llama_pipeline.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='pipeline')
        process=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='process')
        attach,feedback=Mock(),Mock()
        scope=dict(re=re,json=json,Path=Path,DEFAULT_MODEL='vision.gguf',
            settings=SimpleNamespace(default_settings={'enable_llm_tools':True}),
            worker=SimpleNamespace(add_result=Mock()),TimeIt=lambda label:nullcontext(),
            model_has_vision=lambda model:True,selected_image_reference=llama_vision.selected_image_reference,
            transient_vision_messages=llama_vision.transient_vision_messages,
            image_tool_history=llama_vision.image_tool_history,attach_image=attach,attach_image_feedback=feedback,
            IMAGE_TOOL=chat_tools.IMAGE_TOOL,IMAGE_TOOL_GUIDANCE=chat_tools.IMAGE_TOOL_GUIDANCE)
        exec(compile(ast.Module(body=[process],type_ignores=[]),'chat_process','exec'),scope)
        requests=[]
        def respond(parameters,callback):
            requests.append(parameters)
            callback({'choices':[{'delta':{'content':'Reply'},'finish_reason':'stop'}]})
        owner=SimpleNamespace(llm=SimpleNamespace(handle_chat_completions=respond),model_settings=(),
            current_model_settings=lambda model:(),embeddings=None,embeddings_hash='[]')
        data=dict(task_id=1,model='vision.gguf',system='',vision_enabled=True,
            history=[{'role':'assistant','content':'old image context'}, {'role':'user','content':'A new blue sailboat.'}])
        self.assertEqual(scope['process'](owner,data),'Reply')
        attach.assert_not_called();feedback.assert_not_called()
        self.assertIn('Image source: New image',requests[-1]['messages'][0]['content'])
        upload=Image.new('RGBA',(23,17),(8,9,10,29))
        scope['process'](owner,dict(data,image_source='Uploaded image',image_reference=upload))
        attach.assert_called_once()
        self.assertEqual(attach.call_args.args[1].tobytes(),upload.tobytes())
        feedback.assert_not_called()


if __name__=='__main__': unittest.main()
