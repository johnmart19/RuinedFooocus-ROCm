import gradio as gr
from modules.llama_pipeline import run_llama, llama_names

def add_llama_tab(prompt):
    def run_llama_run(system_file, prompt):
        if not prompt or not prompt.strip():
            raise gr.Error("Enter a prompt before rewriting it.")
        if not system_file:
            raise gr.Error("Choose a rewrite preset first.")
        res = run_llama(system_file, prompt)

        return gr.update(value=res)

    with gr.Group(), gr.Row():
        llama_btn = gr.Button(value="Rewrite prompt")
        llama_select = gr.Dropdown(
            choices=llama_names(),
            label="Rewrite preset",
            show_label=True,
            buttons=[llama_btn],
        )
    llama_btn.click(
        run_llama_run,
        api_visibility='undocumented',
        inputs=[llama_select, prompt],
        outputs=[prompt]
    )
