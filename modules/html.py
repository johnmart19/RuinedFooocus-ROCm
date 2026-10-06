from pathlib import Path

css = (Path(__file__).resolve().parents[1] / "html" / "ui.css").read_text(encoding="utf-8")

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
  // Bare Alt activates the browser menu on Windows and blurs open dropdowns.
  // Prevent that default action only while a dropdown is open; screenshot
  // chords and other Alt shortcuts keep propagating normally.
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Alt' && !e.ctrlKey && !e.metaKey &&
        document.querySelector('[role="combobox"][aria-expanded="true"]')) {
      e.preventDefault();
    }
  }, {capture: true});
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
