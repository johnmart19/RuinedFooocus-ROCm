import gradio as gr
from ruinedfooocus.features.prompting.local_model import run_prompt
from ruinedfooocus.features.prompting.presets import rewrite_presets


def add_rewrite_controls(prompt):
    def suggest(system_file, current):
        if not current or not current.strip():
            raise gr.Error("Enter a prompt before rewriting it.")
        if not system_file:
            raise gr.Error("Choose a rewrite preset first.")
        result = run_prompt(current, system_file)
        return gr.update(value=result, visible=True), current, gr.update(interactive=True)

    def apply_rewrite(current, original, proposed):
        if current != original:
            raise gr.Error(
                "The prompt changed after this preview. Preview a new rewrite before applying it."
            )
        if not proposed or not proposed.strip():
            raise gr.Error("The rewrite is empty.")
        return (
            proposed,
            (current, proposed),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    def undo_rewrite(current, previous):
        if previous is None:
            return gr.skip(), None, gr.update(interactive=False)
        if current != previous[1]:
            raise gr.Error(
                "The prompt changed after applying the rewrite. Undo would overwrite your edits."
            )
        return gr.update(value=previous[0]), None, gr.update(interactive=False)

    def refresh_presets(selected):
        choices = rewrite_presets()
        # Keep the selection unless its preset file was removed.
        return gr.update(
            choices=choices, value=selected if selected in [path for _, path in choices] else None
        )

    with gr.Group(), gr.Row():
        llama_btn = gr.Button(value="Preview rewrite")
        llama_select = gr.Dropdown(
            choices=rewrite_presets(),
            label="Rewrite preset",
            show_label=True,
            buttons=[llama_btn],
        )
    previous_prompt = gr.State(None)
    original_prompt = gr.State(None)
    preview = gr.Textbox(label="Suggested prompt", lines=4, visible=False)
    with gr.Row():
        apply_btn = gr.Button(value="Apply rewrite", interactive=False)
        undo_btn = gr.Button(value="Undo rewrite", size="sm", interactive=False)
        refresh_btn = gr.Button(value="Refresh presets", size="sm")
    gr.Markdown(
        "Add your own presets as text files in the `prompt_presets` folder (legacy `llamas` is also supported) (first line: name, then the "
        "instructions), or start the prompt with `system: ...` and a blank line for a one-off instruction."
    )
    gr.Markdown(
        "Uses the local model selected in **Chat**. Preview the suggestion, then apply it explicitly."
    )
    llama_btn.click(
        suggest,
        api_visibility="undocumented",
        inputs=[llama_select, prompt],
        outputs=[preview, original_prompt, apply_btn],
    )
    apply_btn.click(
        apply_rewrite,
        inputs=[prompt, original_prompt, preview],
        outputs=[prompt, previous_prompt, undo_btn, apply_btn],
        api_visibility="undocumented",
        queue=False,
    )
    undo_btn.click(
        undo_rewrite,
        inputs=[prompt, previous_prompt],
        outputs=[prompt, previous_prompt, undo_btn],
        api_visibility="undocumented",
        queue=False,
    )
    refresh_btn.click(
        refresh_presets,
        inputs=llama_select,
        outputs=llama_select,
        api_visibility="undocumented",
        queue=False,
    )
