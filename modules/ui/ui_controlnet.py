import modules.controlnet as controlnet
from modules.controlnet import (
    cn_options,
    load_cnsettings,
    save_cnsettings,
    NEWCN,
)
import gradio as gr
import shared
from modules.video_settings import VIDEO_FPS
from shared import add_ctrl, path_manager, translate
import modules.ui.ui_evolve as ui_evolve
import modules.ui.ui_llama as ui_llama
from modules.ui.ui_checkpoint_tools import create_ui as create_checkpoint_tools
from PIL import Image

t = translate

def is_video_checkpoint(name):
    model = shared.models.get_models_by_path("checkpoints", name)
    return shared.models.get_model_base(model) in VIDEO_FPS


def powerup_choices(video):
    choices = [("None", "None")]
    order = {"img2img": 0, "canny": 1, "depth": 2, "sketch": 3,
             "recolour": 4, "upscale": 5, "rembg": 6, "faceswap": 7}
    if video:
        choices.append(("Image to video", "Image to video"))
    for name, options in sorted(controlnet.cn_options.items(),
                                key=lambda item: (order.get(item[1]["type"], 8), item[0].casefold())):
        if video and options["type"] not in ("upscale", "rembg"):
            continue
        choices.append(("Image to image" if name == "Img2Img" else name, name))
    return choices + [("Checkpoint tools", "Checkpoint tools"), (NEWCN, NEWCN)]


def add_controlnet_tab(main_view, inpaint_view, prompt, image_number, run_event, base_model):
    initial_video = is_video_checkpoint(base_model.value)
    with gr.Tab(label=t("PowerUp")):
        with gr.Row():
            cn_selection = gr.Dropdown(
                label=t("Cheat Code"),
                choices=powerup_choices(initial_video),
                value="None",
            )
            add_ctrl("cn_selection", cn_selection)

        checkpoint_tools = create_checkpoint_tools(base_model)

        cn_name = gr.Textbox(
            show_label=False,
            placeholder=t("Name"),
            interactive=True,
            visible='hidden',
        )
        cn_save_btn = gr.Button(
            value=t("Save"),
            visible='hidden',
        )

        type_choices=[name.capitalize() for name in controlnet.controlnet_models
                      if not initial_video or name in ("upscale", "rembg")]
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
            label=t("Start"),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.0,
            visible='hidden',
        )
        add_ctrl("cn_start", cn_start)

        cn_stop = gr.Slider(
            label=t("Stop"),
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
                value["upscaler"] for value in cn_options.values()
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

        @cn_selection.change(
            api_visibility='undocumented',
            inputs=[cn_selection],
            outputs=[cn_name] + cn_outputs + cn_sliders
        )
        def cn_changed(selection):
            if selection != NEWCN:
                return [gr.update(visible='hidden')] + [gr.update(visible='hidden')] * len(
                    cn_outputs + cn_sliders
                )
            else:
                return [gr.update(value="")] + [gr.update(visible=True)] * len(
                    cn_outputs + cn_sliders
                )

        @cn_type.change(
            api_visibility='undocumented',
            inputs=[cn_type],
            outputs=cn_sliders,
        )
        def cn_type_changed(selection):
            # cn_start,cn_stop,cn_strength,cn_edge_low,cn_edge_high, cn_upscaler
            slider_states = {
                "canny": [True, True, True, True, True, False],
                "img2img": [False, False, True, False, False, False],
                "default": [True, True, True, False, False, False],
                "upscale": [False, False, False, False, False, True],
                "faceswap": [False, False, False, False, False, False],
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
            inputs=cn_outputs + cn_sliders + [base_model],
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
        ):
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
                choices = powerup_choices(is_video_checkpoint(checkpoint))
                return gr.update(choices=choices, value=cn_name)
            else:
                return gr.update()

        input_image = gr.Image(
            label=t("Image to video" if initial_video else "Image to image"),
            type="pil",
            visible=True,
        )
        add_ctrl("input_image", input_image)
        random_image_prompt = gr.Checkbox(
            label=t("Generate random prompt"), value=False,
            info="Opt in to replacing the prompt with One Button when an input image is supplied.",
        )
        add_ctrl("obp_image_prompt", random_image_prompt)
        with gr.Group(visible=False) as image_prompt_options:
            gr.Markdown("Uses the One Button settings below; it does not describe the input image. "
                        "For video, include the desired motion in Subject. Advanced options are in One Button.")
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
        cn_selection.change(
            lambda selection, enabled: [gr.update(visible=selection != "Checkpoint tools"),
                                        gr.update(visible=enabled and selection != "Checkpoint tools")],
            inputs=[cn_selection, random_image_prompt], outputs=[random_image_prompt, image_prompt_options],
            api_visibility="undocumented")
        inpaint_toggle = gr.Checkbox(label=t("Inpainting"), value=False, visible=not initial_video)

        add_ctrl("inpaint_toggle", inpaint_toggle)

        cn_selection.change(
            lambda selection, checkpoint: [gr.update(visible=selection == "Checkpoint tools"),
                               gr.update(visible=selection != "Checkpoint tools"),
                               gr.update(visible=selection != "Checkpoint tools" and not is_video_checkpoint(checkpoint))],
            inputs=[cn_selection, base_model], outputs=[checkpoint_tools, input_image, inpaint_toggle],
            api_visibility="undocumented",
        )

        def model_changed(checkpoint, selection):
            video = is_video_checkpoint(checkpoint)
            choices = powerup_choices(video)
            if selection not in [value for _, value in choices]:
                selection = "None"
            types = [name.capitalize() for name in controlnet.controlnet_models
                     if not video or name in ("upscale", "rembg")]
            return [gr.update(choices=choices, value=selection),
                    gr.update(label=t("Image to video" if video else "Image to image")),
                    gr.update(value=False, visible=not video and selection != "Checkpoint tools"),
                    gr.update(choices=types, value=types[0])]

        base_model.change(model_changed, inputs=[base_model, cn_selection],
                          outputs=[cn_selection, input_image, inpaint_toggle, cn_type],
                          api_visibility="undocumented")

        @inpaint_toggle.change(
            api_visibility='undocumented',
            inputs=[inpaint_toggle, main_view],
            outputs=[main_view, inpaint_view]
        )
        def inpaint_checked(r, image):
            if r:
                base_height = 600
                img = Image.open(image)
                scale = (base_height / float(img.size[1]))
                width = int((float(img.size[0]) * float(scale)))
                img = img.resize((width, base_height), Image.Resampling.LANCZOS)

                return {
                    main_view: gr.update(visible='hidden'),
                    inpaint_view: gr.update(
                        visible=True,
                        interactive=True,
                        value={
                            'background': img,
                            'layers': [Image.new("RGBA", (width, base_height))],
                            'composite': None,
                        },
                    )
                }
            else:
                return {
                    main_view: gr.update(visible=True),
                    inpaint_view: gr.update(
                        visible='hidden',
                        interactive=False,
                    ),
                }

        ui_evolve.add_evolve_tab(prompt, image_number, run_event)

        ui_llama.add_llama_tab(prompt)

    return inpaint_toggle

