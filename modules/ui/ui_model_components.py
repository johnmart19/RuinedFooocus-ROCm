"""Image checkpoint component inventory for the Main model tab."""
import json
import os
from pathlib import Path

import gradio as gr

from modules.image_components import inventory, download_missing, disk_free


def selected(name):
    import shared
    if name not in shared.models.get_names('checkpoints'):
        raise ValueError('Select an installed image checkpoint')
    path = shared.models.get_file_from_name('checkpoints', name)
    if path is None:
        raise ValueError('The selected checkpoint is no longer installed')
    # Cached local metadata only: selection must not cause network access or model loading.
    cache = (shared.models.cache_paths['checkpoints'] / Path(path).name).with_suffix('.json')
    try:
        metadata = json.loads(cache.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    signature = metadata.get('_rf_file_signature')
    stat = Path(path).stat()
    if signature is not None and signature != [str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]:
        metadata = {}
    return Path(path), metadata.get('baseModel', 'Unknown')


def offline():
    from argparser import args
    return args.offline or any(os.environ.get(name, '').upper() in ('1', 'TRUE', 'YES', 'ON')
                               for name in ('RF_OFFLINE', 'HF_HUB_OFFLINE'))


def refresh(name, message=''):
    import shared
    try:
        path, family = selected(name)
        state = inventory(path, family, shared.settings.default_settings, shared.path_manager)
        rows = [[row['key'], row['filename'], row['status'], 'Custom' if row['override'] else 'Automatic',
                 row['location'], row['source'] or 'Local / bundled',
                 f"{row['bytes'] / 1024**3:.2f} GiB" if row['bytes'] else 'Unknown',
                 'Available' if row['download_available'] else '—'] for row in state['rows']]
        ready = not any(row['status'] == 'Missing' for row in state['rows'])
        status = f"**{state['architecture']}** · {'Components present' if ready else 'Missing components'} · {disk_free(shared.path_manager) / 1024**3:.1f} GiB free on encoder drive\n\n"
        status += state['note'] + '\n\n' + message
        if offline():
            status += '\n\nOffline mode: downloads disabled.'
        elif shared.settings.default_settings.get('local_model_metadata', False):
            status += '\n\nLocal metadata only. Files download only when you click Download missing.'
        return rows, status, gr.update(interactive=not offline() and any(r['download_available'] for r in state['rows']))
    except (OSError, ValueError, TypeError) as error:
        return [], str(error), gr.update(interactive=False)


def download_selected(name, progress=gr.Progress()):
    import shared
    try:
        path, family = selected(name)
        files = download_missing(path, family, shared.settings.default_settings, shared.path_manager,
                                 offline=offline(), progress=progress)
        return refresh(name, 'Downloaded: ' + ', '.join(files) if files else 'No downloadable missing components.')
    except Exception as error:
        return refresh(name, 'Download failed: ' + str(error))


def create_ui(base_model):
    """Create inside the desired model tab; selection changes refresh local status only."""
    with gr.Column(min_width=0, elem_id="model-components-panel") as panel:
        gr.Markdown('Inspect the selected checkpoint’s encoder and VAE files. Refresh never downloads. Bundled components take priority over overrides.')
        table = gr.Dataframe(headers=['Component', 'Resolved file', 'Status', 'Selection', 'Location', 'Source', 'Installed size', 'Download'],
                             datatype=['str'] * 8, interactive=False, wrap=True, elem_id='model-component-inventory')
        status = gr.Markdown('Select an image checkpoint and click Refresh components.')
        with gr.Row():
            refresh_button = gr.Button('Refresh components')
            download_button = gr.Button('Download missing', interactive=False)
        outputs = [table, status, download_button]
        base_model.change(refresh, inputs=base_model, outputs=outputs, api_visibility='undocumented')
        refresh_button.click(refresh, inputs=base_model, outputs=outputs, api_visibility='undocumented')
        download_button.click(download_selected, inputs=base_model, outputs=outputs,
                              concurrency_id='image_component_downloads', concurrency_limit=1, api_visibility='undocumented')
    return panel
