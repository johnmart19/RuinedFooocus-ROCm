import modules.controlnet as controlnet
from modules.controlnet import (
    load_cnsettings,
    save_cnsettings,
    NEWCN,
)
import gradio as gr
import shared
from modules.video_settings import VIDEO_FPS
from shared import add_ctrl, path_manager, translate
import modules.ui.ui_evolve as ui_evolve
from modules.inpaint_utils import editor_payload
from modules.ui.ui_image_guidance import family_for_checkpoint

t = translate

SDXL_CONTROLS = frozenset(("canny", "depth", "sketch", "recolour"))


def checkpoint_features(name, moss_format=None):
    """Use header-detected architecture so SDXL metadata aliases stay supported."""
    from modules.image_components import ALIASES
    family = family_for_checkpoint(name)
    canonical = ALIASES.get(family, family)
    sdxl = canonical in ("SDXL", "SDXL 1.0")
    if moss_format is None:
        moss_format = shared.settings.default_settings.get("moss_image_edit_format", "Disabled")
    native_edit = (canonical in ("MageFlow", "Fibo2", "QwenImage")
                   or canonical == "MOSSImage" and moss_format != "Disabled")
    return {"video": family in VIDEO_FPS, "sdxl": sdxl, "native_edit": native_edit}


def control_types(features):
    return [name.capitalize() for name in controlnet.controlnet_models
            if (not features["video"] or name in ("upscale", "rembg"))
            and (features["sdxl"] or name not in SDXL_CONTROLS)]


def input_label(features):
    return "Image to video" if features["video"] else "Image edit reference" if features["native_edit"] else "Image to image"


def powerup_choices(video, sdxl=True, native_edit=False):
    choices = [("None", "None")]
    order = {"img2img": 0, "canny": 1, "depth": 2, "sketch": 3,
             "recolour": 4, "upscale": 5, "rembg": 6, "faceswap": 7}
    if video:
        choices.append(("Image to video", "Image to video"))
    for name, options in sorted(controlnet.cn_options.items(),
                                key=lambda item: (order.get(item[1]["type"], 8), item[0].casefold())):
        if name == "Image to image (preserve details)":
            continue  # Use the image-to-image denoise slider instead of a second preset.
        if options["type"] == "upscale" and name.startswith("Upscale"):
            continue  # Image upscaling is selected with the resolution controls.
        if video and options["type"] not in ("upscale", "rembg"):
            continue
        if not sdxl and options["type"] in SDXL_CONTROLS:
            continue
        label = ("Image edit reference" if native_edit else "Image to image") if name == "Img2Img" else name
        choices.append((label, name))
    return choices + [(NEWCN, NEWCN)]


def add_controlnet_tab(main_view, inpaint_view, prompt, image_number, run_event, base_model,
                       preset_selection=None, main_model=None):
    initial_features = checkpoint_features(base_model.value)
    with gr.Tab(label=t("Image tools")):
        gr.Markdown("Choose a tool, add an input image, then use Generate. "
                    "Upscaling is under Generation > Size and upscale.")
        with gr.Row():
            cn_selection = gr.Dropdown(
                label=t("Image tool"),
                choices=powerup_choices(**initial_features),
                value="None",
            )
            add_ctrl("cn_selection", cn_selection)

        cn_name = gr.Textbox(
            label=t("Tool preset name"),
            show_label=True,
            placeholder=t("Name"),
            interactive=True,
            visible='hidden',
        )
        cn_save_btn = gr.Button(
            value=t("Save tool preset"),
            visible='hidden',
        )

        type_choices = control_types(initial_features)
        cn_type = gr.Dropdown(
            label=t("Type"),
            choices=type_choices,
            value=type_choices[0],
            visible='hidden',
        )
        add_ctrl("cn_type", cn_type)

        cn_edge_low = gr.Slider(
            label=t("Edge (low)"),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.2,
            visible='hidden',
        )
        add_ctrl("cn_edge_low", cn_edge_low)

        cn_edge_high = gr.Slider(
            label=t("Edge (high)"),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.8,
            visible='hidden',
        )
        add_ctrl("cn_edge_high", cn_edge_high)

        cn_start = gr.Slider(
            label=t("Control starts at"),
            info="Fraction of sampling: 0 starts immediately.",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.0,
            visible='hidden',
        )
        add_ctrl("cn_start", cn_start)

        cn_stop = gr.Slider(
            label=t("Control ends at"),
            info="Fraction of sampling: 1 keeps control until the end.",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=1.0,
            visible='hidden',
        )
        add_ctrl("cn_stop", cn_stop)

        cn_strength = gr.Slider(
            label=t("Strength"),
            minimum=0.0,
            maximum=2.0,
            step=0.01,
            value=1.0,
            visible='hidden',
        )
        add_ctrl("cn_strength", cn_strength)

        cn_upscaler = gr.Dropdown(
            label=t("Upscaler"),
            show_label=False,
            choices=["None"] + sorted(set(path_manager.upscaler_filenames) | {
                key for key, entry in path_manager.DOWNLOADABLE_FILES.items()
                if entry["path"] == "path_upscalers"
            } | {
                value["upscaler"] for value in controlnet.cn_options.values()
                if value["type"] == "upscale"
            }),
            value="None",
            visible='hidden',
        )
        add_ctrl("cn_upscale", cn_upscaler)

        cn_outputs = [
            cn_name,
            cn_save_btn,
            cn_type,
        ]
        cn_sliders = [
            cn_start,
            cn_stop,
            cn_strength,
            cn_edge_low,
            cn_edge_high,
            cn_upscaler,
        ]

        from modules.ui.ui_image_guidance import create_ui as create_image_guidance
        guidance_controls = create_image_guidance(base_model, preset_selection=preset_selection, main_model=main_model)
        moss_format_control = guidance_controls["moss_image_edit_format"]

        @cn_selection.change(
            api_visibility='undocumented',
            inputs=[cn_selection, cn_type, base_model, moss_format_control],
            outputs=cn_outputs + cn_sliders
        )
        def cn_changed(selection, control_type, checkpoint, moss_format=None):
            features = checkpoint_features(checkpoint, moss_format)
            if selection != NEWCN:
                options = controlnet.cn_options.get(selection, {})
                kind = options.get("type")
                updates = [gr.update(visible='hidden')] * len(cn_sliders)
                if kind == "img2img" and not features["native_edit"]:
                    updates[2] = gr.update(visible=True, label=t("Denoise"),
                        info="Lower values preserve the input; higher values allow larger changes.",
                        minimum=0.01, maximum=1.0,
                        value=options.get("denoise", options.get("strength", 0.64)))
                elif kind in SDXL_CONTROLS and features["sdxl"]:
                    updates[0] = gr.update(visible=True, value=options.get("start", 0.0))
                    updates[1] = gr.update(visible=True, value=options.get("stop", 1.0))
                    updates[2] = gr.update(visible=True, label=t("Control strength"),
                        info="How strongly the input guides the composition.",
                        minimum=0.0, maximum=2.0, value=options.get("strength", 1.0))
                    if kind == "canny":
                        updates[3] = gr.update(visible=True, value=options.get("edge_low", 0.2))
                        updates[4] = gr.update(visible=True, value=options.get("edge_high", 0.8))
                elif kind == "upscale":
                    updates[5] = gr.update(visible=True, value=options.get("upscaler", "None"))
                return [gr.update(visible='hidden')] * len(cn_outputs) + updates
            else:
                return ([gr.update(value="", visible=True)]
                        + [gr.update(visible=True)] * (len(cn_outputs) - 1)
                        + cn_type_changed(control_type, selection, checkpoint, moss_format))

        @cn_type.change(
            api_visibility='undocumented',
            inputs=[cn_type, cn_selection, base_model, moss_format_control],
            outputs=cn_sliders,
        )
        def cn_type_changed(selection, preset, checkpoint, moss_format=None):
            if preset != NEWCN:
                return cn_changed(preset, selection, checkpoint, moss_format)[len(cn_outputs):]
            features = checkpoint_features(checkpoint, moss_format)
            if selection not in control_types(features) or (selection.lower() == "img2img" and features["native_edit"]):
                return [gr.update(visible='hidden')] * len(cn_sliders)
            # cn_start,cn_stop,cn_strength,cn_edge_low,cn_edge_high, cn_upscaler
            slider_states = {
                "canny": [True, True, True, True, True, False],
                "img2img": [False, False, True, False, False, False],
                "default": [True, True, True, False, False, False],
                "upscale": [False, False, False, False, False, True],
                "faceswap": [False, False, False, False, False, False],
                "rembg": [False, False, False, False, False, False],
            }
            if selection.lower() in slider_states:
                show = slider_states[selection.lower()]
            else:
                show = slider_states["default"]

            result = []
            for vis in show:
                result += [gr.update(visible=True if vis else 'hidden')]

            if selection.lower() == "img2img":
                result[2] = gr.update(visible=True, label=t("Denoise"), minimum=0.01,
                                      maximum=1.0, value=0.64)
            else:
                result[2] = gr.update(visible=show[2], label=t("Strength"), minimum=0.0,
                                      maximum=2.0, value=1.0)

            return result

        @cn_save_btn.click(
            api_visibility='undocumented',
            inputs=cn_outputs + cn_sliders + [base_model, moss_format_control],
            outputs=[cn_selection],
        )
        def cn_save(
            cn_name,
            cn_save_btn,
            cn_type,
            cn_start,
            cn_stop,
            cn_strength,
            cn_edge_low,
            cn_edge_high,
            upscale_model,
            checkpoint,
            moss_format=None,
        ):
            if cn_type not in control_types(checkpoint_features(checkpoint, moss_format)):
                raise gr.Error("This control type is not supported by the selected checkpoint.")
            if cn_name != "":
                cn_options = load_cnsettings()
                opts = {
                    "type": cn_type.lower(),
                    "start": cn_start,
                    "stop": cn_stop,
                    "strength": cn_strength,
                    "upscaler": upscale_model,
                }
                if cn_type.lower() == "canny":
                    opts.update(
                        {
                            "edge_low": cn_edge_low,
                            "edge_high": cn_edge_high,
                        }
                    )
                cn_options[cn_name] = opts
                save_cnsettings(cn_options)
                choices = powerup_choices(**checkpoint_features(checkpoint, moss_format))
                return gr.update(choices=choices, value=cn_name)
            else:
                return gr.update()

        input_image = gr.Image(
            buttons=['download', 'fullscreen'],
            label=t(input_label(initial_features)),
            type="pil",
            image_mode="RGBA",
            visible=True,
        )
        add_ctrl("input_image", input_image)
        random_image_prompt = gr.Checkbox(
            label=t("Generate random prompt"), value=False,
            info="Opt in to replacing the prompt with One Button when an input image is supplied.",
        )
        add_ctrl("obp_image_prompt", random_image_prompt)
        with gr.Group(visible=False) as image_prompt_options:
            gr.Markdown("Uses the Prompt tools settings; it does not describe the input image. "
                        "Advanced prompt options are under Prompt tools.")
            for name, label in (("OBP_preset", "One Button Preset"),
                                ("OBP_promptenhance", "Prompt enhancement"),
                                ("OBP_modeltype", "Prompt style")):
                source = shared.get_ctrl(name)
                mirror = gr.Dropdown(choices=source.choices, value=source.value, label=label)
                # Input fires only for user edits; change also follows preset updates.
                mirror.input(lambda value: value, inputs=mirror, outputs=source,
                             api_visibility="undocumented")
                source.change(lambda value: value, inputs=source, outputs=mirror,
                              api_visibility="undocumented")
            subject_source = shared.get_ctrl("obp_givensubject")
            subject = gr.Textbox(label=t("Subject"), value=subject_source.value,
                                 placeholder="Describe the subject and desired action")
            subject.input(lambda value: value, inputs=subject, outputs=subject_source,
                          api_visibility="undocumented")
            subject_source.change(lambda value: value, inputs=subject_source, outputs=subject,
                                  api_visibility="undocumented")
        random_image_prompt.change(lambda enabled: gr.update(visible=enabled),
                                   inputs=random_image_prompt, outputs=image_prompt_options,
                                   api_visibility="undocumented")
        inpaint_toggle = gr.Checkbox(label=t("Inpainting"), value=False, visible=initial_features["sdxl"])

        add_ctrl("inpaint_toggle", inpaint_toggle)
        with gr.Group(visible=False) as inpaint_options:
            inpaint_denoise = gr.Slider(label=t("Inpainting denoise"), minimum=0.01,
                maximum=1.0, step=0.01, value=1.0,
                info="Amount of regeneration inside the mask. 1 fully redraws the masked area.")
            add_ctrl("inpaint_denoise", inpaint_denoise)
            inpaint_mask_grow = gr.Slider(label=t("Mask expansion (pixels)"),
                minimum=0, maximum=128, step=1, value=20,
                info="Expand the painted mask to blend edits into surrounding pixels.")
            add_ctrl("inpaint_mask_grow", inpaint_mask_grow)
        inpaint_toggle.change(lambda enabled: gr.update(visible=enabled),
            inputs=inpaint_toggle, outputs=inpaint_options, api_visibility="undocumented")

        def model_changed(checkpoint, selection, moss_format=None):
            features = checkpoint_features(checkpoint, moss_format)
            choices = powerup_choices(**features)
            if selection not in [value for _, value in choices]:
                selection = "None"
            types = control_types(features)
            return [gr.update(choices=choices, value=selection),
                    gr.update(label=t(input_label(features))),
                    gr.update(value=False, visible=features["sdxl"]),
                    gr.update(choices=types, value=types[0])]

        base_model.change(model_changed, inputs=[base_model, cn_selection, moss_format_control],
                          outputs=[cn_selection, input_image, inpaint_toggle, cn_type],
                          api_visibility="undocumented")

        def edit_format_changed(moss_format, checkpoint, selection, control_type):
            features = checkpoint_features(checkpoint, moss_format)
            kind = (control_type.lower() if selection == NEWCN else
                    controlnet.cn_options.get(selection, {}).get("type"))
            denoise = gr.update()
            if kind == "img2img":
                denoise = gr.update(visible=not features["native_edit"], label=t("Denoise"),
                    minimum=0.01, maximum=1.0,
                    info="Lower values preserve the input; higher values allow larger changes.")
            # Update presentation only: preserve image pixels, selected preset,
            # and the user's current denoise value when switching edit formats.
            return [gr.update(choices=powerup_choices(**features)),
                    gr.update(label=t(input_label(features))), denoise]

        moss_format_control.change(edit_format_changed,
            inputs=[moss_format_control, base_model, cn_selection, cn_type],
            outputs=[cn_selection, input_image, cn_strength], api_visibility="undocumented")

        @inpaint_toggle.change(
            api_visibility='undocumented',
            inputs=[inpaint_toggle, main_view],
            outputs=[main_view, inpaint_view, inpaint_toggle]
        )
        def inpaint_checked(r, image):
            if r:
                try:
                    payload = editor_payload(image)
                except ValueError as error:
                    gr.Warning(str(error))
                    return {main_view: gr.update(visible=True),
                            inpaint_view: gr.update(visible='hidden', interactive=True),
                            inpaint_toggle: gr.update(value=False)}

                return {
                    main_view: gr.update(visible='hidden'),
                    inpaint_view: gr.update(
                        visible=True,
                        interactive=True,
                        value=payload,
                    )
                }
            else:
                return {
                    main_view: gr.update(visible=True),
                    inpaint_view: gr.update(
                        visible='hidden',
                        # Gradio's editor get_data needs its editable canvas even
                        # while hidden; read-only mode removes that canvas.
                        interactive=True,
                    ),
                }

        ui_evolve.add_evolve_tab(prompt, image_number, run_event)

    return inpaint_toggle

