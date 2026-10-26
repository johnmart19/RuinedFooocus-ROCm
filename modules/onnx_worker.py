"""Isolated DirectML mask worker; never imported into the Torch process."""
import argparse
import base64
import json
from pathlib import Path
import sys

CANARY = 'CAk6RgoOCgF4CgF4EgF5IgNBZGQSCmdwdS1jYW5hcnlaEwoBeBIOCgwIARIICgIIAgoCCAJiEwoBeRIOCgwIARIICgIIAgoCCAJCBAoAEBE='


def run(runtime, probe=False, device_id=None, site_dir=None):
    if site_dir:
        import site
        site.addsitedir(site_dir)
        sys.path.remove(site_dir)
        sys.path.insert(0, site_dir)
    sys.path.insert(0, str(Path(runtime).resolve()))
    import onnxruntime as ort
    import numpy as np
    if 'DmlExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('The isolated ONNX runtime has no DirectML provider.')
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    provider = {'device_id': str(device_id)} if device_id is not None else {
        'performance_preference': 'high_performance', 'device_filter': 'gpu'}
    providers = [('DmlExecutionProvider', provider), 'CPUExecutionProvider']
    if probe:
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        session = ort.InferenceSession(base64.b64decode(CANARY), sess_options=options,
                                       providers=providers[:1])
        x = np.ones((2, 2), dtype=np.float32)
        np.testing.assert_array_equal(session.run(None, {'x': x})[0], x * 2)
        print(json.dumps({'ok': True, 'version': ort.__version__}), flush=True)
        return
    import rembg
    from PIL import Image
    # The parent uses its own rembg version's normal postprocessing. Only mask
    # inference occurs here, with the same inherited package/model default.
    session = rembg.new_session(sess_opts=options, providers=providers)
    if 'DmlExecutionProvider' not in session.inner_session.get_providers():
        raise RuntimeError('DirectML did not initialize; refusing silent CPU inference.')
    print(json.dumps({'ready': True, 'model': session.model_name}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('close'):
            return
        try:
            with Image.open(request['input']) as image:
                masks = session.predict(image.convert('RGB'))
            paths = []
            for i, mask in enumerate(masks):
                path = Path(request['output']) / f'mask-{i}.png'
                mask.save(path)
                paths.append(str(path))
            print(json.dumps({'masks': paths}), flush=True)
        except Exception as error:
            print(json.dumps({'error': f'{type(error).__name__}: {error}'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--device-id', type=int)
    parser.add_argument('--site-dir')
    args = parser.parse_args()
    run(args.runtime, args.probe, args.device_id, args.site_dir)
