import gradio as gr
from ruinedfooocus.features.chat.llama_pipeline import run_llama
from ruinedfooocus.features.prompting.presets import rewrite_presets

def add_llama_tab(prompt):
    def run_llama_run(system_file, prompt):
        if not prompt or not prompt.strip():
            raise gr.Error("Enter a prompt before rewriting it.")
        if not system_file:
            raise gr.Error("Choose a rewrite preset first.")
        res = run_llama(system_file, prompt)

        # Keep the original so the rewrite can be undone.
        return gr.update(value=res), prompt, gr.update(interactive=True)

    def undo_rewrite(previous):
        if previous is None:
            return gr.skip(), None, gr.update(interactive=False)
        return gr.update(value=previous), None, gr.update(interactive=False)

    def refresh_presets(selected):
        choices = rewrite_presets()
        # Keep the selection unless its preset file was removed.
        return gr.update(choices=choices, value=selected if selected in [path for _, path in choices] else None)

    with gr.Group(), gr.Row():
        llama_btn = gr.Button(value="Rewrite prompt")
        llama_select = gr.Dropdown(
            choices=rewrite_presets(),
            label="Rewrite preset",
            show_label=True,
            buttons=[llama_btn],
        )
    previous_prompt = gr.State(None)
    with gr.Row():
        undo_btn = gr.Button(value="Undo rewrite", size="sm", interactive=False)
        refresh_btn = gr.Button(value="Refresh presets", size="sm")
    gr.Markdown("Add your own presets as text files in the `prompt_presets` folder (legacy `llamas` is also supported) (first line: name, then the "
                "instructions), or start the prompt with `system: ...` and a blank line for a one-off instruction.")
    llama_btn.click(
        run_llama_run,
        api_visibility='undocumented',
        inputs=[llama_select, prompt],
        outputs=[prompt, previous_prompt, undo_btn]
    )
    undo_btn.click(undo_rewrite, inputs=previous_prompt, outputs=[prompt, previous_prompt, undo_btn],
                   api_visibility='undocumented', queue=False)
    refresh_btn.click(refresh_presets, inputs=llama_select, outputs=llama_select,
                      api_visibility='undocumented', queue=False)
