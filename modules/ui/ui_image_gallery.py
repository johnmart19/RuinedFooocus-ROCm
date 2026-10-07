import gradio as gr
import json
from shared import path_manager, translate, get_ctrl
from modules.imagebrowser import ImageBrowser

browser = ImageBrowser()
t = translate

METADATA_CONTROLS = {
    "prompt": "prompt", "negative": "negative", "steps": "custom_steps",
    "cfg": "cfg", "sampler_name": "sampler_name", "scheduler": "scheduler",
    "clip_skip": "clip_skip", "width": "custom_width", "height": "custom_height",
    "seed": "seed", "video_duration": "video_duration", "video_fps": "video_fps",
}


def apply_metadata(metadata):
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict):
        raise gr.Error("Select an image with generation metadata first.")
    data = {key.lower(): value for key, value in metadata.items()}
    if "prompt" not in data:
        raise gr.Error("This file has no saved prompt.")
    # Saved width/height describe the final file; sampling must use the native size.
    data["width"] = data.get("generation_width", data.get("width"))
    data["height"] = data.get("generation_height", data.get("height"))
    updates = {get_ctrl(target): gr.update(value=data[key])
               for key, target in METADATA_CONTROLS.items() if data.get(key) is not None}
    updates[get_ctrl("performance_selection")] = gr.update(value="Own settings")
    updates[get_ctrl("style_selection")] = gr.update(value=[])
    updates[get_ctrl("obp_assume_direct_control")] = gr.update(value=False)
    updates[get_ctrl("obp_image_prompt")] = gr.update(value=False)
    if data.get("width") and data.get("height"):
        updates[get_ctrl("aspect_ratios_selection")] = gr.update(value="Custom...")
    if data.get("seed") is not None:
        updates[get_ctrl("seed_random")] = gr.update(value=False)
    from modules.automatic_upscale import choices, PRESETS
    upscale = data.get("automatic_upscale", "Off")
    if upscale not in choices():
        upscale = "Off"
    updates[get_ctrl("automatic_upscale")] = gr.update(choices=choices(), value=upscale)
    updates[get_ctrl("advanced_upscalers")] = gr.update(value=upscale not in PRESETS)
    updates[get_ctrl("automatic_upscale_target")] = gr.update(value="Generated image")
    gr.Info("Prompt and sampling settings applied in Main. Checkpoint, LoRAs and input image stay selected.")
    return updates

def create_image_gallery():
    with gr.Blocks() as app_image_browser:
        with gr.Row(elem_id="browser-layout"):
            # Left side for gallery
            with gr.Column(scale=2, min_width=0, elem_id="browser-images"):
                gallery = gr.Gallery(
                    buttons=['download', 'download_all', 'fullscreen'],
                    label=t("Images"),
                    show_label=False,
                    columns=[2, 2, 3, 4, 6, 8],
                    height="var(--rf-gallery-height)",
                    elem_id="output-browser",
                    object_fit="contain",
                    value=browser.load_images(1)[0],
                )
                pages=browser.num_images_pages()[1]
                ib_page = gr.Slider(
                    label=t("Page"),
                    value=1,
                    step=1,
                    minimum=1,
                    maximum=max(2, pages), # Stupid workaround to make sure max is larger than min... :|
                )
                ib_range = gr.Markdown()

            # Right side for metadata and search
            with gr.Column(scale=1, min_width=0, elem_id="browser-controls"):
                with gr.Row():
                    update_btn = gr.Button(t("Update DB"), scale=3)
                    copymeta_btn = gr.Button(t("Copy to prompt"), scale=3)
                    gr.HTML(value="""<a href="gradio_api/file/html/slideshow.html" style="color: gray; text-decoration: none" target="_blank">🛝</a>""")
                metadata_output = gr.Textbox(
                    label=t("Image Metadata"), interactive=False, lines=15
                )
                search_input = gr.Textbox(
                    label=t("Search Metadata"), placeholder=t("Search term")
                )
                search_btn = gr.Button(t("Search"))
                status_output = gr.Markdown()
                metadata_json = gr.JSON(
                    value={},
                    visible='hidden'
                )

        # Event handlers
        update_btn.click(
            fn=browser.update_images,
            concurrency_id="image_browser_refresh",
            api_visibility='undocumented',
            outputs=[gallery, ib_page, status_output],
        )
        ib_page.change(
            fn=browser.load_images,
            api_visibility='undocumented',
            inputs=[ib_page],
            outputs=[gallery, ib_range],
        )
        copymeta_btn.click(
            fn=apply_metadata,
            api_visibility='undocumented',
            inputs=[metadata_json],
            outputs=[get_ctrl(name) for name in list(METADATA_CONTROLS.values()) + [
                "performance_selection", "style_selection", "obp_assume_direct_control",
                "obp_image_prompt", "aspect_ratios_selection", "seed_random",
                "automatic_upscale", "advanced_upscalers", "automatic_upscale_target"]],
        )
        gallery.select(
            fn=browser.get_image_metadata,
            api_visibility='undocumented',
            outputs=[metadata_json, metadata_output],
        )
        search_btn.click(
            fn=browser.search_metadata,
            api_visibility='undocumented',
            inputs=[search_input],
            outputs=[gallery, ib_page, status_output],
        )
        search_input.submit(
            fn=browser.search_metadata,
            api_visibility='undocumented',
            inputs=[search_input],
            outputs=[gallery, ib_page, status_output],
        )


    return app_image_browser, [gallery, ib_page, status_output]

# Optional: If you want to launch just this gallery
def launch_image_gallery():
    app_image_browser, _ = create_image_gallery()
    app_image_browser.launch()
