css = """
#checkpoint-tools { gap: 10px; padding: 8px; }
#checkpoint-tools .gradio-row { gap: 8px; }
#style-selection button { border-radius: var(--button-small-radius); }
#style-selection .token-remove { background: transparent; color: var(--body-text-color);
  box-shadow: none; border: 0; border-radius: var(--radius-sm); width: 20px; height: 20px; }
#style-selection .token-remove:hover { background: var(--button-secondary-background-fill-hover); }
.gradio-container .main:has([role="tabpanel"]:not([style*="display: none"]) #chat-layout) {
  max-width: none;
}
#chat-layout { --chat-width: 72%; gap: 8px; }
#chat-conversation, #chat-controls { min-width: 0 !important; }
#chat-local-models { background: var(--block-background-fill); }
#chat-model-source .wrap { display: flex; flex-wrap: nowrap; gap: 4px; }
#chat-model-source label { flex: 1; min-width: 0; justify-content: center; }
#chat-model-source label span { margin: 0; }
#chat-model-source label input { position: absolute; opacity: 0; width: 1px; height: 1px; }
#chat-model-source label.selected, #chat-model-source label.selected:hover {
  background: var(--color-accent); border-color: var(--color-accent); color: white; }
#chat-model-source label:not(.selected):hover { background: var(--button-secondary-background-fill-hover); }
#chat-model-source label:focus-within { outline: 2px solid var(--color-accent); outline-offset: 2px; }
#chat-model-actions { align-items: stretch; }
#chat-model-actions button { height: auto; min-height: 34px; padding: 6px 8px; }
#chat-messages .message { max-width: 100%; }
#chat-messages .prose { max-width: none; line-height: 1.6; overflow-wrap: anywhere; }
#chat-messages pre { max-width: 100%; overflow-x: auto; white-space: pre; }
#chat-messages pre code { white-space: pre; overflow-wrap: normal; word-break: normal; }
#chat-divider { align-self: stretch; padding: 0; width: 12px !important; min-width: 12px !important;
  height: 100%; min-height: 120px; position: relative; }
#chat-divider .html-container, #chat-divider .prose { width: 100%; height: 100%; }
#chat-divider [role="separator"] { position: absolute; inset: 0; cursor: col-resize;
  border-radius: 6px; background: var(--border-color-primary); touch-action: none; }
#chat-divider [role="separator"]:hover, #chat-divider [role="separator"]:focus-visible {
  background: var(--color-accent); outline: 2px solid var(--color-accent); }
@media (min-width: 901px) {
  #chat-layout { display: grid; grid-template-columns: minmax(0, var(--chat-width)) 12px minmax(0, 1fr); }
}
@media (max-width: 900px) {
  #chat-layout { flex-direction: column; }
  #chat-divider { display: none; }
}
textarea {
  resize: vertical;
}
.loader-container {
  display: flex; /* Use flex to align items horizontally */
  align-items: center; /* Center items vertically within the container */
  flex-wrap: wrap;
  gap: 4px 12px;
}
/* Style the progress bar */
progress {
  appearance: none; /* Remove default styling */
  height: 20px; /* Set the height of the progress bar */
  border-radius: 5px; /* Round the corners of the progress bar */
  background-color: #f3f3f3; /* Light grey background */
  width: 100%;
}
/* Style the progress bar container */
.progress-container {
  margin-left: 20px;
  margin-right: 20px;
  flex-grow: 1; /* Allow the progress container to take up remaining space */
  min-width: 100px;
}
/* Set the color of the progress bar fill */
progress::-webkit-progress-value {
  background-color: #3498db; /* Blue color for the fill */
}
progress::-moz-progress-bar {
  background-color: #3498db; /* Blue color for the fill in Firefox */
}
/* Style the text on the progress bar */
progress::after {
  content: attr(value '%'); /* Display the progress value followed by '%' */
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: white; /* Set text color */
  font-size: 14px; /* Set font size */
}
/* Style other texts */
.loader-container > span {
  margin-left: 5px; /* Add spacing between the progress bar and the text */
  white-space: normal;
  overflow-wrap: anywhere;
}
#progress-bar > .generating {
  display: none !important;
}
#progress-bar{
  min-height: 30px;
}
.hint-container > .generating {
  display: none !important;
}
.hint-container{
  height: 150px !important;
}
.json-container{
  height: 600px;
  overflow: auto !important;
}
.type_small_row{
  height: 100px !important;
}
.scroll-hide{
  resize: auto !important;
}
.refresh_button{
  border: none !important;
  background: none !important;
  font-size: none !important;
  box-shadow: none !important;
}
.element1 {
  opacity: 0.01;
}
#inpaint_sketch { overflow: overlay !important; resize: auto; background: var(--panel-background-fill); z-index: 5; }
/* Custom CSS for mobile-friendliness */
@media (max-width: 768px) {
  body, .gr-textbox, .gr-button {
    font-size: 14px;
  }
  #component-321
  {
    padding: unset;
  }
  #main_view {
    height: 100% !important;
  }
  #component-19{
    padding: 0px 6% 0 7%;
  }
  #component-22{
    margin-top: 55px;
  }
}
"""
progress_html = """
<div class="loader-container">
  <div class="progress-container">
    <progress value="*number*" max="100"></progress>
  </div>
  <span>*text*</span>
</div>
"""
scripts = """
function generate_shortcut(){
  document.addEventListener('keydown', (e) => {
    let handle = 'none';
    if (e.key !== undefined) {
      if ((e.key === 'Enter' && e.ctrlKey)) handle = 'run';
      if ((e.key === 'q' && e.ctrlKey)) handle = 'edit_mode';
      if ((e.key === 'Escape')) handle = 'edit_mode';
    } else if (e.keyCode !== undefined) {
      if ((e.keyCode === 13 && e.ctrlKey)) handle = 'run';
      if ((e.keyCode === 81 && e.ctrlKey)) handle = 'edit_mode';
      if ((e.keyCode === 27)) handle = 'edit_mode';
    }
    if (handle == 'run') {
      const button = document.getElementById('generate');
      if (button) button.click();
      e.preventDefault();
    } else if (handle == 'edit_mode') {
      const button = document.getElementById('edit_mode');
      if (button) button.click();
      e.preventDefault();
    }
  });
}
generate_shortcut();
"""

from shared import state


def make_progress_html(number, text):
    if number == -1:
        number = state["last_progress"]
    else:
        state["last_progress"] = number
    return progress_html.replace("*number*", str(number)).replace("*text*", text)
