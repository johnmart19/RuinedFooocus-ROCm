from ruinedfooocus.core.paths import asset_path
from pathlib import Path

css = asset_path("ui.css").read_text(encoding="utf-8")

# Gradio scopes component CSS below main; outer page spacing belongs in head.
head = """<style>
/* Text and rem spacing follow the browser default up to 1920px wide, then grow
   smoothly with the window: ~117% at 2560px (1440p), 150% at 3840px (4K). */
html { font-size: var(--rf-user-text-scale, clamp(100%, 50% + .4167vw, 150%)); }
@media (max-width: 600px) {
.gradio-container .main.fillable.app { padding: 8px !important; }
}</style>"""

# The label names the progress bar for assistive technology with its status text.
progress_html = """
<label class="loader-container">
  <span class="progress-container">
    <progress value="*number*" max="100"></progress>
  </span>
  <span>*text*</span>
</label>
"""
scripts = """
function generate_shortcut(){
  const savedTextScale = Number(localStorage.getItem('ruinedfooocus-text-scale'));
  if (savedTextScale >= 75 && savedTextScale <= 200) {
    document.documentElement.style.setProperty('--rf-user-text-scale', `${savedTextScale}%`);
  }
  // Gradio treats non-navigation keys as search edits, even modifiers and
  // PrintScreen, filtering the list to its selected label. Keep these keys
  // out of dropdown handlers without cancelling the OS screenshot chord.
  const preserveDropdown = (e) => {
    if (['Alt', 'Control', 'Shift', 'Meta', 'PrintScreen'].includes(e.key) &&
        e.target?.closest?.('[role="combobox"][aria-expanded="true"]')) {
      e.stopImmediatePropagation();
      if (e.key === 'Alt') e.preventDefault();
    }
  };
  document.addEventListener('keydown', preserveDropdown, {capture: true});
  document.addEventListener('keyup', preserveDropdown, {capture: true});
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

import html as html_text

from ruinedfooocus.app.state import state


def make_progress_html(number, text):
    if number == -1:
        number = state["last_progress"]
    else:
        state["last_progress"] = number
    # Status text can name files or prompts; it is text, never markup.
    return progress_html.replace("*number*", str(number)).replace(
        "*text*", html_text.escape(str(text))
    )
