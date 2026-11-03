"""CPU-only checks for observational wrappers around Comfy V3 nodes."""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from tools.comfy_video_app import _observe, _recovered_selection, generate


class VideoObserverTests(unittest.TestCase):
    def test_recovered_encoder_is_explicit_boolean_and_minimax_only(self):
        self.assertTrue(_recovered_selection('minimax', True))
        self.assertFalse(_recovered_selection('minimax', False))
        self.assertFalse(_recovered_selection('ltx_av', False))
        for invalid in ('true', 1, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _recovered_selection('minimax', invalid)
        with self.assertRaises(ValueError):
            _recovered_selection('ltx_av', True)

    def test_generate_defaults_to_official_and_forwards_explicit_recovered_choice(self):
        with patch('tools.comfy_video_app._run', return_value={}) as run:
            generate('minimax', 'model', 'clip', 'vae', 'audio')
            self.assertFalse(run.call_args.kwargs['recovered_encoder'])
            generate('minimax', 'model', 'clip', 'vae', 'audio', recovered_encoder=True)
            self.assertTrue(run.call_args.kwargs['recovered_encoder'])

    def test_audio_wrapper_preserves_v3_classmethod_validation_and_results(self):
        audio_vae = object()
        audio_samples = {'samples': np.array([2.])}
        video_samples = {'samples': np.array([3.])}
        frames = np.zeros((1, 2, 2, 3))
        sound = {'waveform': np.array([4.]), 'sample_rate': 24000}
        calls = []

        class AudioNode:
            def __init__(self):
                # Native V3 VALIDATE_CLASS/first_real_override reads __func__.
                self.implementation = getattr(type(self), 'execute').__func__

            @classmethod
            def execute(cls, vae, samples):
                calls.append((cls, vae, samples))
                return (sound,)

        class Decode:
            def decode(self, vae, samples):
                return (frames,)

        class Tiled(Decode):
            pass

        class Guider:
            def inner_set_conds(self, values):
                pass

            def sample(self, *args, **kwargs):
                pass

        owner = types.SimpleNamespace(model_base_patched=types.SimpleNamespace(audio_vae=audio_vae))
        original = AudioNode.execute.__func__
        def process(gen_data, callback):
            instance = AudioNode()
            self.assertTrue(callable(instance.implementation))
            result = instance.execute(vae=audio_vae, samples=audio_samples)
            self.assertIs(result[0], sound)
            self.assertIs(Tiled().decode(None, video_samples)[0], frames)
            return []
        owner.process = process
        comfy = types.ModuleType('comfy')
        samplers = types.ModuleType('comfy.samplers')
        samplers.CFGGuider = Guider
        comfy.samplers = samplers
        nodes = types.ModuleType('nodes')
        nodes.VAEDecode, nodes.VAEDecodeTiled = Decode, Tiled
        extras = types.ModuleType('comfy_extras')
        audio = types.ModuleType('comfy_extras.nodes_audio')
        audio.VAEDecodeAudio = AudioNode
        with patch.dict(sys.modules, {'comfy': comfy, 'comfy.samplers': samplers,
            'nodes': nodes, 'comfy_extras': extras, 'comfy_extras.nodes_audio': audio}):
            observed = _observe(owner, {}, av=True, output_video=None)
        self.assertIs(AudioNode.execute.__func__, original)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], AudioNode)
        self.assertIs(calls[0][1], audio_vae)
        self.assertIs(calls[0][2], audio_samples)
        self.assertIs(observed['audio'], sound)
        self.assertIs(observed['output'], frames)
        self.assertIs(observed['latent']['audio'], audio_samples['samples'])


if __name__ == '__main__':
    unittest.main()
