"""Cherry-pick features and their declared prerequisites onto original upstream.

Uses a separate temporary Git repository with read-only access to source objects.
Does not move the application branch, install packages, launch the app, or run video
tests. Use the application's Python environment for the optional CPU tests.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def dependency_order(features, selected):
    by_id = {item['id']: item for item in features}
    result, active, visited = [], set(), set()
    def visit(key):
        if key not in by_id:
            raise ValueError(f'Unknown feature: {key}')
        if key in active:
            raise ValueError(f'Cyclic feature dependency: {key}')
        if key in visited:
            return
        active.add(key)
        for prerequisite in by_id[key].get('requires', []):
            visit(prerequisite)
        active.remove(key)
        visited.add(key)
        result.append(key)
    for key in selected:
        visit(key)
    return result


def module_exports(tree):
    """Collect module bindings without mistaking function locals for exports."""
    exports = set()
    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports.add(node.name)
            return
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            exports.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            exports.update(a.asname or a.name.split('.')[0] for a in node.names)
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(tree)
    return exports


def cpu_test_code(patterns):
    return f'''import unittest
patterns = {patterns!r}
suites = []
for pattern in patterns:
    suite = unittest.defaultTestLoader.discover('tests', pattern=pattern)
    if suite.countTestCases() == 0:
        raise ValueError('No CPU tests discovered for ' + pattern)
    suites.append(suite)
result = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(suites))
raise SystemExit(not result.wasSuccessful())
'''


def python_contracts(folder):
    """Check syntax, local modules and named imports without executing app code."""
    trees = {}
    for path in folder.rglob('*.py'):
        rel = path.relative_to(folder).as_posix()
        if rel.startswith(('repositories/', '.git/')):
            continue
        trees[rel] = ast.parse(path.read_text(encoding='utf-8-sig'), filename=rel)
    failures = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('modules.'):
                modules = [(node.module, node.names)]
            elif isinstance(node, ast.Import):
                modules = [(alias.name, []) for alias in node.names if alias.name.startswith('modules.')]
            else:
                continue
            for name, imports in modules:
                target = name.replace('.', '/') + '.py'
                package = name.replace('.', '/') + '/'
                if target not in trees:
                    if not any(p.startswith(package) for p in trees):
                        failures.append(f'{path}:{node.lineno}: missing {name}')
                    continue
                exports = module_exports(trees[target])
                for alias in imports:
                    if alias.name != '*' and alias.name not in exports:
                        failures.append(f'{path}:{node.lineno}: {name} has no {alias.name}')
    if failures:
        raise ValueError('\n'.join(failures))
    return len(trees)


def run(root, manifest, selected, ref='HEAD', tests=False):
    root = root.resolve()
    temp_root = root / 'tmp'
    temp_root.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='feature-picks-', dir=temp_root)).resolve()
    if scratch.parent != temp_root.resolve():
        raise ValueError('Validation checkout must remain under repository tmp/')
    objects = Path(subprocess.check_output(['git', 'rev-parse', '--git-path', 'objects'], cwd=root, text=True).strip())
    if not objects.is_absolute():
        objects = root / objects
    subprocess.run(['git', 'init', '--quiet', str(scratch)], check=True)
    (scratch / '.git/objects/info/alternates').write_bytes((objects.resolve().as_posix() + '\n').encode('utf-8'))
    def git(*args):
        return subprocess.run(['git', *args], cwd=scratch, capture_output=True, text=True)
    for key, value in [('user.name', 'RuinedFooocus validation'), ('user.email', 'validation@localhost'),
                       ('commit.gpgSign', 'false'), ('core.autocrlf', 'false')]:
        assert git('config', key, value).returncode == 0
    hooks = scratch / '.git/disabled-hooks'
    hooks.mkdir()
    assert git('config', 'core.hooksPath', str(hooks)).returncode == 0
    checkout = git('checkout', '--detach', manifest['base'])
    if checkout.returncode:
        raise RuntimeError(checkout.stderr)
    history = subprocess.check_output(['git', 'log', '--format=%H%x09%s', manifest['base'] + '..' + ref], cwd=root, text=True)
    subjects = {}
    for row in history.splitlines():
        sha, subject = row.split('\t', 1)
        if subject in subjects:
            raise ValueError(f'Ambiguous commit subject: {subject}')
        subjects[subject] = sha
    by_id = {item['id']: item for item in manifest['features']}
    # Only native source contracts are copied. No backend or package is installed.
    native_files = ['node_helpers.py', 'comfy/cli_args.py', 'comfy/text_encoders/qwen35.py',
                    'comfy_extras/nodes_fibo2.py', 'comfy_extras/nodes_moss_image.py']
    report = dict(base=manifest['base'], ref=ref, scratch=str(scratch), video_tests=False, features=[])
    for key in selected:
        item = by_id[key]
        closure = dependency_order(manifest['features'], [key])
        if git('reset', '--hard', manifest['base']).returncode:
            raise RuntimeError('Could not reset isolated validation repository')
        # Test files are not left behind between features by reset; dependencies
        # are replayed using real cherry-pick, not only text patch application.
        record = dict(id=key, closure=closure, commits=[], status='passed')
        try:
            for dependency in closure:
                sha = subjects[by_id[dependency]['subject']]
                result = git('cherry-pick', '--allow-empty', sha)
                if result.returncode:
                    raise RuntimeError(f'Cherry-pick {dependency} failed:\n{result.stdout}\n{result.stderr}')
                record['commits'].append(sha)
            record['python_files'] = python_contracts(scratch)
            test_patterns = item.get('tests', [])
            if tests and test_patterns and not item.get('skip_tests'):
                (scratch / 'tmp').mkdir(exist_ok=True)
                for rel in native_files:
                    original = root / 'repositories/ComfyUI' / rel
                    if original.is_file():
                        destination = scratch / 'repositories/ComfyUI' / rel
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original, destination)
                env = os.environ.copy()
                env['RF_OFFLINE'] = '1'
                env['PYTHONDONTWRITEBYTECODE'] = '1'
                env['TEMP'] = env['TMP'] = str(temp_root)
                code = cpu_test_code(test_patterns)
                import sys
                result = subprocess.run([sys.executable, '-c', code], cwd=scratch, env=env, capture_output=True, text=True)
                record['test_output'] = result.stdout + result.stderr
                if result.returncode:
                    raise RuntimeError('CPU tests failed:\n' + record['test_output'])
            record['tests'] = test_patterns if tests and not item.get('skip_tests') else []
        except (KeyError, RuntimeError, ValueError, SyntaxError) as error:
            record.update(status='failed', error=str(error))
            git('cherry-pick', '--abort')
        report['features'].append(record)
        print(key + ': ' + record['status'] + ' [' + ', '.join(closure) + ']', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('feature_commits.json'))
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--feature', action='append', help='Feature ID; default: all features')
    parser.add_argument('--tests', action='store_true', help='Run declared CPU tests; video tests are always excluded')
    parser.add_argument('--output', type=Path, default=Path('tmp/feature-cherry-picks.json'))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    selected = args.feature or [item['id'] for item in manifest['features']]
    report = run(Path(__file__).resolve().parents[1], manifest, selected, args.ref, args.tests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return any(item['status'] != 'passed' for item in report['features'])


if __name__ == '__main__':
    raise SystemExit(main())
