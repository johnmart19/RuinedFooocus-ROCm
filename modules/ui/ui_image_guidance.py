"""Current-image guidance controls; saved startup defaults live in Settings."""
import gradio as gr
from modules.image_guidance import (FIELD_SCHEMA, capabilities, read_preset_metadata,
                                    effective_preset_checkpoint, restore_guidance_metadata)


def family_for_checkpoint(name):
    import shared
    info = shared.models.get_models_by_path('checkpoints', name, fetch=False)
    family = shared.models.get_model_base(info)
    path = shared.models.get_file('checkpoints', name)
    if path is not None:
        detected = shared.models.detect_checkpoint_base(path)
        if detected:
            family = detected
    return family


def control_visibility(family):
    available = capabilities(family)
    return (available['moss_edit'], available['moss_slg'],
            available['sampling_shift'], available['cfg_norm'])


def references_visible(family, edit_format):
    available = capabilities(family)
    return available.get('multiple_refs', False) and (
        not available['moss_edit'] or edit_format != 'Disabled')


def preset_control_state(current, defaults, metadata, mode, family, checkpoint, main_checkpoint, previous):
    """Pure preset transition: never store reference images in backup state."""
    if metadata is None:
        return None
    current = {key: current.get(key, defaults.get(key, spec['default'])) for key, spec in FIELD_SCHEMA.items()}
    previous = previous or {'checkpoint': main_checkpoint, 'backup': None}
    locked = bool(metadata and mode & 2)
    backup = previous.get('backup')
    values = dict(current)
    changed = previous.get('checkpoint') != checkpoint
    if locked:
        if backup is None:
            backup = {'checkpoint': main_checkpoint, 'values': dict(current)}
        values = dict(backup['values'])
        if backup['checkpoint'] != checkpoint:
            values.update(sampling_shift=None, cfg_norm_strength=None)
        values.update(restore_guidance_metadata(metadata, family=family))
    elif backup is not None:
        values = (dict(backup['values']) if backup['checkpoint'] == main_checkpoint else
                  {key: defaults.get(key, spec['default']) for key, spec in FIELD_SCHEMA.items()})
        if backup['checkpoint'] != main_checkpoint:
            values.update(sampling_shift=None, cfg_norm_strength=None)
        backup = None
    elif changed:
        values.update(sampling_shift=None, cfg_norm_strength=None)
    return {'values': values, 'locked': locked, 'clear_references': locked or changed or previous.get('backup') is not None,
            'state': {'checkpoint': checkpoint, 'backup': backup}}


def create_ui(base_model, preset_selection=None, main_model=None):
    import shared
    defaults = shared.settings.default_settings
    family = family_for_checkpoint(base_model.value)
    initial = control_visibility(family)
    preset_notice = gr.Markdown('', visible=False)
    with gr.Accordion('Image guidance', open=False) as guidance_panel:
        gr.Markdown('These controls apply to the current image without restarting. '
                    'Select a checkpoint under Models to see its supported options.')
        with gr.Group(visible=initial[0]) as moss_edit:
            edit_format = gr.Dropdown(label='MOSS edit format',
                choices=list(FIELD_SCHEMA['moss_image_edit_format']['choices']),
                value=defaults.get('moss_image_edit_format', 'Disabled'),
                info='Requires an input image and a matching edit LoRA. Disabled uses ordinary image generation.')
            strength = gr.Slider(label='Reference attention strength', minimum=0, maximum=100,
                step=0.05, value=defaults.get('moss_image_edit_strength', 1.0),
                info='1 is normal; 0 omits vision conditioning. Applies to ai-toolkit formats.')
        slg = gr.Checkbox(label='MOSS skip-layer guidance', visible=initial[1],
            value=defaults.get('moss_image_slg', False),
            info='Optional extra guidance. Off follows the standard MOSS workflow.')
        shift = gr.Number(label='Sampling shift', minimum=0.01, maximum=100,
            value=None, visible=initial[2],
            info='Leave blank for the model default. An override applies only to this generation.')
        norm = gr.Number(label='CFG normalization strength', minimum=0, maximum=100,
            value=None, visible=initial[3],
            info='Optional native CFG normalization. Leave blank to disable; 1 is a normal starting value.')
        references = gr.Gallery(label='Additional edit references (up to 15)', value=[],
            interactive=True, type='pil', file_types=['image'], sources=['upload'],
            columns=3, height=180, buttons=[],
            visible=references_visible(family, edit_format.value))
        gr.Markdown('For Mage, Fibo2 or enabled MOSS editing, put the primary image in Input image. '
                    'Additional references are images 2 onward in upload order.')
        controls = dict(moss_image_edit_format=edit_format, moss_image_edit_strength=strength,
                        moss_image_slg=slg, sampling_shift=shift, cfg_norm_strength=norm,
                        reference_images=references)
        for key, component in controls.items():
            shared.add_ctrl(key, component)

    def model_changed(name, current_format):
        family = family_for_checkpoint(name)
        edit, use_slg, use_shift, use_norm = control_visibility(family)
        # Do not carry model-specific numerical overrides into a different model.
        return (gr.update(visible=edit), gr.update(visible=use_slg),
                gr.update(visible=use_shift, value=None), gr.update(visible=use_norm, value=None),
                gr.update(visible=references_visible(family, current_format), value=[]))

    if preset_selection is None:
        base_model.change(model_changed, inputs=[base_model, edit_format], outputs=[moss_edit, slg, shift, norm, references],
                          api_visibility='undocumented')
    else:
        main_model = main_model or base_model
        state = gr.State({'checkpoint': main_model.value, 'backup': None})
        fields = [controls[key] for key in FIELD_SCHEMA]
        outputs = [guidance_panel, preset_notice, moss_edit] + fields + [references, state]
        def context_changed(main_checkpoint, preset, previous, *values):
            metadata = read_preset_metadata(preset)
            mode = shared.settings.default_settings.get('preset_mode', 7)
            checkpoint = effective_preset_checkpoint(main_checkpoint, preset, mode)
            if checkpoint is None:
                return [gr.update() for _ in outputs]
            family = family_for_checkpoint(checkpoint)
            transition = preset_control_state(dict(zip(FIELD_SCHEMA, values)), defaults, metadata,
                                              mode, family, checkpoint, main_checkpoint, previous)
            if transition is None:
                return [gr.update() for _ in outputs]
            selected = transition['values']
            available = capabilities(family)
            updates = [gr.update(visible=not transition['locked']),
                       gr.update(visible=transition['locked'], value='Preset controls image guidance. Clear the preset to edit these controls.'),
                       gr.update(visible=available['moss_edit'])]
            for key in FIELD_SCHEMA:
                visible = (available['moss_slg'] if key == 'moss_image_slg' else
                           available['sampling_shift'] if key == 'sampling_shift' else
                           available['cfg_norm'] if key == 'cfg_norm_strength' else True)
                updates.append(gr.update(value=selected[key], visible=visible))
            updates.append(gr.update(visible=not transition['locked'] and references_visible(family, selected['moss_image_edit_format']),
                                     **({'value': []} if transition['clear_references'] else {})))
            updates.append(transition['state'])
            return updates
        gr.on(triggers=[main_model.change, preset_selection.change], fn=context_changed,
              inputs=[main_model, preset_selection, state] + fields, outputs=outputs,
              concurrency_id='image_guidance_context', concurrency_limit=1, api_visibility='undocumented')
    def format_changed(name, value):
        visible = references_visible(family_for_checkpoint(name), value)
        return gr.update(visible=visible, **({} if visible else {'value': []}))
    edit_format.change(format_changed, inputs=[base_model, edit_format], outputs=references,
                       api_visibility='undocumented')
    return controls
