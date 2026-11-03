"""Browser runtime asset boundaries, with tiny fake downloads and no code execution."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules import browser_python as runtime


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_bytes(files, extra=()):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:bz2') as archive:
        for name, data in [*files.items(), *extra]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


class BrowserPythonAssetTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        (root / 'tmp').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root / 'tmp')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name) / 'assets'
        self.wheel = b'fake browser-only wheel'
        self.wheel_name = 'numpy-1.0-py3-none-any.whl'
        lock = {'packages': {'numpy': {'file_name': self.wheel_name, 'sha256': digest(self.wheel)}}}
        self.core = {'pyodide.js': b'fake loader', 'pyodide.asm.mjs': b'fake module',
                     'pyodide.asm.wasm': b'fake wasm', 'python_stdlib.zip': b'fake stdlib',
                     'pyodide-lock.json': json.dumps(lock).encode()}
        self.archive = archive_bytes({'pyodide/' + name: data for name, data in self.core.items()})
        self.calls = []
        self.responses = {runtime.ARCHIVE_URL: self.archive,
                          runtime.PACKAGE_URL + self.wheel_name: self.wheel}
        self.patch_core = patch.object(runtime, 'CORE_HASHES', {n: digest(d) for n, d in self.core.items()})
        self.patch_archive = patch.object(runtime, 'ARCHIVE_SHA256', digest(self.archive))
        self.patch_core.start()
        self.patch_archive.start()
        self.addCleanup(self.patch_core.stop)
        self.addCleanup(self.patch_archive.stop)
        self.assets = runtime.RuntimeAssets(self.folder, fetch=self.fetch)

    def fetch(self, url, limit):
        self.calls.append((url, limit))
        if url not in self.responses:
            raise AssertionError('Unexpected URL: ' + url)
        return self.responses[url]

    def client(self):
        app = FastAPI()
        runtime.install_routes(app, self.assets)
        return TestClient(app)

    def test_verified_core_and_lock_package_are_cached_without_repeat_fetch(self):
        self.assertEqual(self.assets.asset('pyodide.js').read_bytes(), self.core['pyodide.js'])
        self.assertEqual(self.assets.asset(self.wheel_name).read_bytes(), self.wheel)
        self.assets.asset(self.wheel_name)
        self.assertEqual([url for url, _ in self.calls],
                         [runtime.ARCHIVE_URL, runtime.PACKAGE_URL + self.wheel_name])

    def test_paths_and_arbitrary_urls_fail_before_fetch(self):
        for name in ('../secret', '/etc/passwd', 'C:\\private', 'a/b', 'a%2fb',
                     'https://example.com/code.py', '..', 'file..whl'):
            with self.subTest(name=name), self.assertRaises(KeyError):
                self.assets.asset(name)
        self.assertEqual(self.calls, [])

    def test_unknown_basename_cannot_proxy_or_read_private_files(self):
        self.folder.mkdir()
        (self.folder / 'private.json').write_text('private app data')
        with self.assertRaises(KeyError):
            self.assets.asset('private.json')
        self.assertEqual([url for url, _ in self.calls], [runtime.ARCHIVE_URL])
        self.assertEqual((self.folder / 'private.json').read_text(), 'private app data')

    def test_archive_checksum_failure_writes_nothing(self):
        self.responses[runtime.ARCHIVE_URL] = b'corrupt archive'
        with self.assertRaisesRegex(ValueError, 'archive checksum mismatch'):
            self.assets.asset('pyodide.js')
        self.assertFalse(self.folder.exists())

    def test_package_checksum_failure_never_publishes_unverified_data(self):
        self.responses[runtime.PACKAGE_URL + self.wheel_name] = b'corrupt wheel'
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.assets.asset(self.wheel_name)
        self.assertFalse((self.folder / self.wheel_name).exists())
        self.assertFalse(list(self.folder.glob('.asset-*')))

    def test_cache_tampering_is_repaired_before_asset_is_returned(self):
        path = self.assets.asset(self.wheel_name)
        path.write_bytes(b'tampered')
        self.assertFalse(self.assets.valid(path, digest(self.wheel)))
        self.assertEqual(self.assets.asset(self.wheel_name).read_bytes(), self.wheel)
        self.assertEqual(sum(url.endswith(self.wheel_name) for url, _ in self.calls), 2)

    def test_archive_paths_are_never_extracted_to_the_filesystem(self):
        bad = archive_bytes({'pyodide/' + n: d for n, d in self.core.items()},
                            extra=[('../escaped.txt', b'bad'), ('/absolute.txt', b'bad')])
        self.responses[runtime.ARCHIVE_URL] = bad
        with patch.object(runtime, 'ARCHIVE_SHA256', digest(bad)), patch.object(
                tarfile.TarFile, 'extractall', side_effect=AssertionError('No extraction')), patch.object(
                tarfile.TarFile, 'extract', side_effect=AssertionError('No extraction')):
            self.assets.asset('pyodide.js')
        self.assertEqual({p.name for p in self.folder.iterdir()}, set(self.core))
        self.assertFalse((self.folder.parent / 'escaped.txt').exists())

    def test_public_get_head_and_preflight_support_opaque_origin_without_credentials(self):
        with self.client() as client:
            for method in ('GET', 'HEAD', 'OPTIONS'):
                with self.subTest(method=method):
                    response = client.request(method, runtime.ASSET_PREFIX + 'pyodide.asm.wasm',
                        headers={'Origin': 'null', 'Cookie': 'access-token=private-cookie'})
                    self.assertEqual(response.status_code, 204 if method == 'OPTIONS' else 200)
                    self.assertEqual(response.headers['access-control-allow-origin'], '*')
                    self.assertNotIn('access-control-allow-credentials', response.headers)
                    self.assertNotIn('set-cookie', response.headers)
                    self.assertEqual(response.headers['x-content-type-options'], 'nosniff')
                    if method != 'OPTIONS':
                        self.assertEqual(response.headers['content-type'], 'application/wasm')
                    if method != 'GET':
                        self.assertEqual(response.content, b'')

    def test_http_errors_do_not_return_private_files_or_unverified_downloads(self):
        with self.client() as client:
            self.assertEqual(client.get(runtime.ASSET_PREFIX + 'private.json').status_code, 404)
            self.responses[runtime.PACKAGE_URL + self.wheel_name] = b'wrong hash'
            response = client.get(runtime.ASSET_PREFIX + self.wheel_name)
            self.assertEqual(response.status_code, 503)
            self.assertNotIn('wrong hash', response.text)
            self.assertFalse((self.folder / self.wheel_name).exists())


if __name__ == '__main__':
    unittest.main()
