"""Photo-to-depth options for the existing SDXL depth ControlNet."""
import gradio as gr
import ruinedfooocus.app.state as shared
import ruinedfooocus.features.image_editing.controlnet as controlnet
from ruinedfooocus.features.image_editing.depth_control import DEPTH_DEFAULTS, DEPTH_CHOICES, depth_settings


def create_ui(selection, control_type, checkpoint, checkpoint_features):
    with gr.Group(visible=False) as panel:
        mode = gr.Dropdown(label='Depth input', value='auto',
            choices=[('Estimate from photo', 'auto'), ('Use supplied depth map', 'none')],
            info='Estimate downloads once. Depth maps: white = near.')
        with gr.Accordion('Depth options', open=False) as advanced:
            model = gr.Dropdown(label='Depth model', value='auto',
                choices=[('Auto', 'auto'), ('DA3 Small (137 MB)', 'small'), ('DA3 Base (542 MB)', 'base')],
                info='Auto: Base at 8+ GB VRAM; Small on CPU or smaller GPUs.')
            resolution = gr.Slider(label='Depth resolution', minimum=140, maximum=2520, step=14, value=504,
                info='More detail and VRAM; keeps input size.')
            resize = gr.Dropdown(label='Depth resize method', value='upper_bound_resize',
                choices=[('Fit longest side', 'upper_bound_resize'), ('Fit shortest side (more VRAM)', 'lower_bound_resize')])
            norm = gr.Dropdown(label='Depth contrast', value='v2_style',
                choices=[('Balanced', 'v2_style'), ('Full range', 'min_max')])
            dtype = gr.Dropdown(label='Depth precision', value='default',
                choices=list(DEPTH_CHOICES['depth_dtype']), info='Default follows the native backend and device.')
    controls = dict(depth_preprocess=mode, depth_model=model, depth_resolution=resolution, depth_resize_method=resize,
                    depth_normalization=norm, depth_dtype=dtype)
    for key, component in controls.items():
        shared.add_ctrl('cn_' + key, component)

    def context_changed(preset, kind, name):
        values = (DEPTH_DEFAULTS if preset == controlnet.NEWCN else controlnet.cn_options.get(preset, {}))
        is_depth = (kind.lower() if preset == controlnet.NEWCN else values.get('type')) == 'depth'
        options = depth_settings(values)
        visible = checkpoint_features(name)['sdxl'] and is_depth
        return [gr.update(visible=visible)] + [gr.update(value=options[key]) for key in controls] + [
            gr.update(visible=options['depth_preprocess'] == 'auto')]
    gr.on(triggers=[selection.change, control_type.change, checkpoint.change], fn=context_changed,
        inputs=[selection, control_type, checkpoint], outputs=[panel, *controls.values(), advanced],
        concurrency_id='depth_context', concurrency_limit=1, api_visibility='undocumented')
    mode.change(lambda value:gr.update(visible=value == 'auto'), inputs=mode, outputs=advanced,
                api_visibility='undocumented')
    return controls
