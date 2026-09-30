# Responsive UI

RuinedFooocus uses Gradio 6.29.0. The workspace fills the browser width; previews,
the image browser, and chat adapt to its height. Desktop model controls have a
bounded width so wide screens give more space to the output. Below 1001 CSS
pixels, controls stack underneath the workspace. Phone layouts use full-width
generation actions and larger touch targets. Browser zoom remains available.
The four main navigation labels use compact spacing on phones. Nested menus
wrap onto additional rows when needed, so Model/LoRAs and the Main tools remain
visible without an overflow menu. Grouped controls inherit theme border widths.

Display size does not change generation resolution, model presets, or saved
settings. Select output dimensions with Aspect Ratios or Custom as before.

Layout rules live in `html/ui.css`, loaded by `modules/html.py`. Use application
`elem_id` values rather than Gradio's generated component IDs. Gradio scopes
custom CSS: responsive component variables belong on `#app-tabs`, not a nested
`.gradio-container` selector. Keep colors tied to theme variables for light and
dark themes. The chat divider behavior lives in `html/chat_splitter.js`.

## Chat layout

Drag the divider beside chat to resize it. Arrow keys also resize; double-click
or Home resets it. The default keeps model controls between 300 and 480 pixels
wide and gives the conversation the remaining space. Manual resizing keeps the
controls usable when the window shrinks. Narrow screens stack the controls below
the conversation and hide the divider.


## Verification (2026-09-30)

The 6.29.0 dependency update rechecked all four pages at 390×844 and 1920×1080,
including a completed Gradio chat submission and Image browser refresh.
No horizontal page overflow was found. The broader layout pass below used 6.28.0.

Windows/Python 3.12 startup and dependency checks passed with Gradio 6.28.0 and
gradio-client 2.7.1. Main, Image browser, Chat bots, and Settings were checked in
the browser at 360×800, 390×844, 768×1024, 1280×720, 1920×1080, 2560×1440, and
3840×2160 CSS pixels. No horizontal page overflow was found. One Button actions,
Checkpoint tools, dropdowns, and the chat divider were also checked.

The submenu pass also checked Model/LoRAs, custom One Button controls, expanded
Workflow settings and Evolve, vision options, image metadata/search, and expanded
runtime, text-encoder and VAE settings at phone widths. Opening these menus does
not require changing saved settings or running a generation.

Phone sizes are browser viewport simulations, not physical iOS/Android tests.
This UI pass does not repeat model inference or GPU compatibility testing.
