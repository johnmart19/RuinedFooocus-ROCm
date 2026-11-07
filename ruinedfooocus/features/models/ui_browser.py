"""Checkpoint browsing in Create, sharing the quick picker's file identities."""

import gradio as gr

from ruinedfooocus.core.util import get_checkpoint_thumbnail
from ruinedfooocus.features.models.selection import (
    checkpoint_records,
    model_label,
    selected_checkpoint,
)
from ruinedfooocus.features.models.ui_download import create_ui as create_download_ui
from ruinedfooocus.features.models.ui_model_components import create_ui as create_components_ui
from ruinedfooocus.features.models.ui_checkpoint_tools import create_ui as create_checkpoint_ui


def create_ui(base_model, models):
    def describe(name):
        info = models.get_models_by_path("checkpoints", name, fetch=False)
        publisher = info.get("model", {})
        title = publisher.get("name", "") if isinstance(publisher, dict) else ""
        return "\n".join(
            filter(
                None,
                (
                    model_label(name, info),
                    f"File: {name}",
                    title,
                    f"Family: {info.get('baseModel', 'Unknown')}",
                ),
            )
        )

    query = gr.Textbox(
        label="Search checkpoints",
        placeholder="Search model name or filename",
        value="",
        elem_id="model-search",
    )

    def refresh(search):
        records = checkpoint_records(
            models.get_names("checkpoints"),
            lambda name: models.get_models_by_path("checkpoints", name, fetch=False),
            search,
        )
        return gr.update(
            value=[(get_checkpoint_thumbnail(r["name"]), r["caption"]) for r in records]
        ), records

    initial, records = refresh("")
    selection = gr.State(records)
    gallery = gr.Gallery(
        label="Image checkpoints",
        show_label=False,
        height="clamp(16rem, 55dvh, 40rem)",
        allow_preview=False,
        preview=False,
        columns=2,
        rows=3,
        object_fit="contain",
        buttons=[],
        min_width=60,
        value=initial["value"],
        elem_id="model-gallery",
    )
    gr.on(
        triggers=[query.input, query.submit],
        fn=refresh,
        inputs=query,
        outputs=[gallery, selection],
        api_visibility="undocumented",
    )

    def select(records, evt: gr.SelectData):
        try:
            name = selected_checkpoint(records, evt.index, evt.value.get("caption"))
        except (ValueError, AttributeError) as error:
            raise gr.Error(str(error)) from None
        if not models.get_file("checkpoints", name):
            raise gr.Error(
                "This checkpoint is no longer installed. Refresh files and select again."
            )
        return gr.update(value=name)

    gallery.select(select, inputs=selection, outputs=base_model, api_visibility="undocumented")
    with gr.Accordion("Selected model details", open=False):
        current = gr.Textbox(
            label="Selected model", value=describe(base_model.value), lines=3, interactive=False
        )
    base_model.change(describe, inputs=base_model, outputs=current, api_visibility="undocumented")
    with gr.Accordion("Add image models", open=False):
        create_download_ui(base_model)
    with gr.Accordion("Image model components", open=False):
        create_components_ui(base_model)
    with gr.Accordion("Checkpoint tools", open=False):
        create_checkpoint_ui(base_model)
    return query, gallery, selection, refresh
