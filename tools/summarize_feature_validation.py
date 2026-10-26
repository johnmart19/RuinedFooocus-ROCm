"""CPU-only interim report and media contact sheet; parity is not a quality claim."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        return {'read_error': str(error)}


def media(path):
    stats, tiles = {}, []
    with av.open(str(path)) as container:
        stream = next(iter(container.streams.video), None)
        if stream:
            fps = float(stream.average_rate) if stream.average_rate else None
            count = total = squares = variation = 0
            previous = last = None
            selected = {0, max(0, stream.frames // 2)}
            finite = True
            for frame in container.decode(stream):
                pixels = frame.to_ndarray(format='rgb24')
                floating = pixels.astype(np.float32) / 255
                finite = finite and bool(np.isfinite(floating).all())
                total += float(floating.sum(dtype=np.float64))
                squares += float(np.square(floating).sum(dtype=np.float64))
                if previous is not None:
                    variation += float(np.abs(floating - previous).mean())
                previous, last = floating, (count, Image.fromarray(pixels).copy())
                if count in selected:
                    tiles.append((f'{path.name} frame {count}', last[1].copy()))
                count += 1
            if last and last[0] not in selected:
                tiles.append((f'{path.name} frame {last[0]}', last[1]))
            elements = count * stream.width * stream.height * 3
            mean = total / elements if elements else None
            stats['video'] = dict(width=stream.width, height=stream.height, frames=count,
                fps=fps, duration=count / fps if fps else None, finite=bool(count) and finite,
                pixel_mean=mean, pixel_std=float(np.sqrt(max(0, squares / elements - mean**2))) if elements else None,
                mean_adjacent_frame_absolute_change=variation / (count - 1) if count > 1 else None)
    # Reopen because decoding video may consume interleaved audio packets.
    with av.open(str(path)) as container:
        stream = next(iter(container.streams.audio), None)
        if stream:
            samples = elements = 0
            squares = 0.0
            finite = True
            for frame in container.decode(stream):
                values = frame.to_ndarray()
                floating = values.astype(np.float64)
                if values.dtype.kind in 'iu':
                    if values.dtype.kind == 'u':
                        floating -= (np.iinfo(values.dtype).max + 1) / 2
                    floating /= ((np.iinfo(values.dtype).max + 1) / 2 if values.dtype.kind == 'u'
                                 else abs(np.iinfo(values.dtype).min))
                finite = finite and bool(np.isfinite(floating).all())
                squares += float(np.square(floating).sum())
                elements += floating.size
                samples += frame.samples
            rate = stream.codec_context.sample_rate
            stats['audio'] = dict(sample_rate=rate, channels=len(stream.codec_context.layout.channels),
                samples_per_channel=samples, duration=samples / rate if rate else None,
                finite=finite, rms=float(np.sqrt(squares / elements)) if elements and finite else None)
    return stats, tiles


def summarize(root):
    summary = {'interim': True, 'quality_claim': False, 'reports': {}, 'assets': {}, 'media': {}, 'warnings': []}
    tiles = []
    for path in sorted(root.glob('*-comparison.json')):
        name = path.name.removesuffix('-comparison.json')
        process_path = root / (name + '-process.json')
        report, process = read(path), read(process_path)
        comparison = report.get('comparison', {}).get('passed') is True
        exit_code = process.get('exit_code')
        digest = process.get('comparison_sha256')
        correlated = (digest == hashlib.sha256(path.read_bytes()).hexdigest() if digest is not None
                      else process_path.exists() and process_path.stat().st_mtime_ns >= path.stat().st_mtime_ns)
        status = ('failed' if not comparison else 'exit unrecorded' if exit_code is None
                  else 'process record stale' if not correlated else 'passed' if exit_code == 0 else 'failed')
        summary['reports'][name] = dict(status=status, comparison_passed=comparison, exit_code=exit_code,
                                       process_record_correlated=correlated)
    summary['report_counts'] = dict(Counter(r['status'] for r in summary['reports'].values()))
    for path in sorted(root.glob('*assets-results.json')):
        data = read(path)
        rows = data if isinstance(data, list) else data.get('results', [])
        verified = sum(bool(r.get('sha256')) and r.get('actual_sha256') == r['sha256'] and r.get('status') == 'verified' for r in rows)
        summary['assets'][path.name] = dict(entries=len(rows), recorded_sha256_matches=verified,
            statuses=dict(Counter(r.get('status', 'unknown') for r in rows)), rehashed_by_summary=False)
    for path in sorted(root.glob('*.mp4')):
        try:
            stats, selected = media(path)
            summary['media'][path.name] = stats
            tiles.extend(selected)
        except Exception as error:
            summary['media'][path.name] = {'decode_error': str(error)}
    for name, stats in summary['media'].items():
        if not name.endswith('-app.mp4'):
            continue
        native = summary['media'].get(name.removesuffix('-app.mp4') + '-native.mp4')
        if not native:
            summary['warnings'].append(f'{name}: native MP4 missing')
            continue
        mismatches = [f'{kind}.{key}' for kind, keys in [('video', ['width', 'height', 'frames', 'fps']),
                      ('audio', ['sample_rate', 'channels', 'samples_per_channel'])] for key in keys
                      if stats.get(kind, {}).get(key) != native.get(kind, {}).get(key)]
        decoded = all(not value.get('decode_error') and value.get('video', {}).get('frames', 0) > 0
                      for value in (stats, native))
        summary.setdefault('media_pairs', {})[name] = {'structural_counts_match': decoded and not mismatches,
                                                       'both_videos_decoded': decoded, 'mismatches': mismatches}
    for name, stats in summary['media'].items():
        if name.endswith('-native.mp4') and name.removesuffix('-native.mp4') + '-app.mp4' not in summary['media']:
            summary['warnings'].append(f'{name}: app MP4 missing')
        video, audio = stats.get('video', {}), stats.get('audio', {})
        if video.get('duration') and audio.get('duration') and abs(video['duration'] - audio['duration']) > 1 / video['fps']:
            summary['warnings'].append(f'{name}: audio/video durations differ by more than one video frame')
    for path in sorted(root.glob('*.png')):
        try:
            with Image.open(path) as image:
                tiles.append((path.name, image.convert('RGB').copy()))
        except OSError as error:
            summary['warnings'].append(f'{path.name}: {error}')
    if tiles:
        sheet = Image.new('RGB', (960, 210 * ((len(tiles) + 2) // 3)), '#202020')
        draw = ImageDraw.Draw(sheet)
        for index, (label, image) in enumerate(tiles):
            x, y = index % 3 * 320, index // 3 * 210
            sheet.paste(ImageOps.contain(image, (312, 180)), (x + 4, y + 4))
            draw.text((x + 4, y + 186), label[:48], fill='white')
        sheet.save(root / 'contact-sheet.jpg', quality=88)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path('outputs/feature-validation-2026-10-25'))
    args = parser.parse_args()
    print(json.dumps(summarize(args.root.resolve())['report_counts']))
