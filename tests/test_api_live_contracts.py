"""ASGI API contracts using inert jobs; never import workers or load models."""
import asyncio
import base64
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from modules.openai_api import ChatRequest, create_apps
from modules.web_access import PrivateResponses, require_ui_login

ROOT = Path(__file__).resolve().parents[1]


def job(*events):
    value = SimpleNamespace(events=queue.Queue(), cancelled=threading.Event())
    for event in events:
        value.events.put(event)
    return value


class ApiLiveContracts(unittest.TestCase):
    def setUp(self):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        self.backend = SimpleNamespace(models=Mock(return_value={'image/example': {'kind': 'image'}}),
            capabilities=Mock(return_value={'samplers': ['example']}), resolve_model=Mock(),
            chat=Mock(), images=Mock(), upscale=Mock(), review=Mock())
        def resolve(name, kind):
            if name == 'missing':
                raise ValueError('Unknown local model')
            return kind + '/example', {'kind': kind}
        self.backend.resolve_model.side_effect = resolve
        api, tools = create_apps(self.backend, 'secret')
        app = FastAPI()
        self.api = api
        app.mount('/v1', api)
        app.mount('/tools', tools)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.headers = {'Authorization': 'Bearer secret'}

    def test_auth_models_capabilities_and_schema(self):
        for headers in ({}, {'Authorization': 'Bearer wrong'}):
            response = self.client.get('/v1/models', headers=headers)
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers['www-authenticate'], 'Bearer')
        self.assertEqual(self.client.get('/v1/models', headers=self.headers).json()['data'][0]['id'], 'image/example')
        self.assertEqual(self.client.get('/v1/capabilities', headers=self.headers).json(), {'samplers': ['example']})
        for path in ('/v1/openapi.json', '/tools/openapi.json'):
            response = self.client.get(path, headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['paths'])

    def test_validation_selection_and_queue_failures(self):
        response = self.client.post('/v1/chat/completions', headers=self.headers, json={'messages': []})
        self.assertEqual(response.status_code, 400)
        response = self.client.post('/v1/chat/completions', headers=self.headers,
            json={'model': 'missing', 'messages': [{'role': 'user', 'content': 'hi'}]})
        self.assertEqual(response.status_code, 404)
        self.backend.chat.side_effect = RuntimeError('queue full')
        response = self.client.post('/v1/chat/completions', headers=self.headers,
            json={'messages': [{'role': 'user', 'content': 'hi'}]})
        self.assertEqual(response.status_code, 429)

    def test_chat_reply_and_sse_completion_cancel_consumed_jobs(self):
        reply = {'choices': [{'message': {'content': 'reply'}}]}
        completed = job(('result', reply))
        self.backend.chat.return_value = completed
        payload = {'messages': [{'role': 'user', 'content': 'hello'}]}
        response = self.client.post('/v1/chat/completions', headers=self.headers, json=payload)
        self.assertEqual(response.json(), reply)
        self.assertTrue(completed.cancelled.is_set())
        stream = job(('chunk', {'choices': [{'delta': {'content': 'reply'}}]}), ('result', None))
        self.backend.chat.return_value = stream
        response = self.client.post('/v1/chat/completions', headers=self.headers, json=dict(payload, stream=True))
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/event-stream', response.headers['content-type'])
        self.assertIn('reply', response.text)
        self.assertTrue(response.text.endswith('data: [DONE]\n\n'))
        self.assertTrue(stream.cancelled.is_set())

    def test_disconnected_chat_cancels_job_without_waiting(self):
        pending = job()
        self.backend.chat.return_value = pending
        endpoint = next(route.endpoint for route in self.api.routes if route.path == '/chat/completions')
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(endpoint(ChatRequest(messages=[{'role': 'user', 'content': 'hi'}]), request))
        self.assertEqual(raised.exception.status_code, 499)
        self.assertTrue(pending.cancelled.is_set())

    def test_worker_errors_are_reported_and_cancelled(self):
        for kind, status in (('invalid_request', 400), ('error', 500)):
            with self.subTest(kind=kind):
                failed = job((kind, 'synthetic failure'))
                self.backend.chat.return_value = failed
                response = self.client.post('/v1/chat/completions', headers=self.headers,
                    json={'messages': [{'role': 'user', 'content': 'hi'}]})
                self.assertEqual(response.status_code, status)
                self.assertTrue(failed.cancelled.is_set())

    def test_mocked_image_capability_url_and_base64(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            path = Path(directory) / 'synthetic.png'
            path.write_bytes(b'synthetic image bytes')
            self.backend.images.side_effect = lambda *args: job(('result', [str(path)]))
            response = self.client.post('/v1/images/generations', headers=self.headers, json={'prompt': 'synthetic'})
            self.assertEqual(response.status_code, 200)
            url = response.json()['data'][0]['url']
            self.assertEqual(self.client.get(url).content, path.read_bytes())
            self.assertEqual(self.client.get('/v1/images/unknown').status_code, 404)
            response = self.client.post('/v1/images/generations', headers=self.headers,
                json={'prompt': 'synthetic', 'response_format': 'b64_json'})
            self.assertEqual(base64.b64decode(response.json()['data'][0]['b64_json']), path.read_bytes())

    def test_multipart_validation_never_submits_invalid_edits(self):
        cases = [({}, {}), ({'image': ('image.png', b'not needed')}, {'mask': 'unsupported'}),
                 ({'image': ('image.png', b'not needed')}, {'styles': 'invalid json'})]
        for files, data in cases:
            response = self.client.post('/v1/images/edits', headers=self.headers, files=files or None, data=data)
            self.assertEqual(response.status_code, 400)
        self.backend.images.assert_not_called()
        self.backend.images.return_value = job(('result', []))
        response = self.client.post('/v1/images/edits', headers=self.headers,
            files={'image': ('image.png', b'synthetic')}, data={'prompt': 'edit', 'styles': '["style"]'})
        self.assertEqual(response.status_code, 200)
        payload = self.backend.images.call_args.args[0]
        self.assertEqual(payload['styles'], ['style'])
        self.assertEqual(base64.b64decode(payload['input_image']), b'synthetic')

    def test_guest_routes_and_private_headers(self):
        app = FastAPI()
        @app.get('/healthy')
        def healthy():
            return {'ok': True}
        client = TestClient(PrivateResponses(app, guest=True))
        self.addCleanup(client.close)
        for path in ('/gradio_api/file=secret', '/gradio_api/upload', '/gradio_api/proxy/x', '/gradio_api/stream/x'):
            response = client.get(path)
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.headers['cache-control'], 'no-store')
        response = client.get('/healthy')
        self.assertEqual(response.headers['cache-control'], 'private, no-store')
        self.assertEqual(response.headers['referrer-policy'], 'no-referrer')

    def test_actual_gradio_login_cookie_and_logout_protect_custom_route(self):
        import gradio as gr
        from fastapi import Request
        from starlette.routing import Mount
        with gr.Blocks(analytics_enabled=False) as blocks:
            gr.Textbox(label='Inert fixture')
        self.addCleanup(blocks.close)
        app = gr.mount_gradio_app(FastAPI(), blocks, path='/owner',
                                 auth=('fixture-owner', 'fixture-password'), ssr_mode=False,
                                 mcp_server=False)
        child = next(route.app for route in app.routes if isinstance(route, Mount) and route.path == '/owner')
        @child.get('/protected-contract')
        def protected(request: Request):
            require_ui_login(request)
            return {'ok': True}
        client = TestClient(app)
        self.addCleanup(client.close)
        self.assertEqual(client.get('/owner/protected-contract').status_code, 401)
        login = client.post('/owner/login', data={'username': 'fixture-owner', 'password': 'wrong'})
        self.assertEqual(login.status_code, 400)
        self.assertEqual(client.get('/owner/protected-contract').status_code, 401)
        login = client.post('/owner/login', data={'username': 'fixture-owner', 'password': 'fixture-password'})
        self.assertEqual(login.status_code, 200)
        self.assertTrue(login.json()['success'])
        cookies = dict(client.cookies)
        self.assertTrue(any(name.startswith('access-token-unsecure-') for name in cookies))
        self.assertEqual(client.get('/owner/protected-contract').status_code, 200)
        logout = client.get('/owner/logout', follow_redirects=False)
        self.assertEqual(logout.status_code, 302)
        self.assertEqual(client.get('/owner/protected-contract').status_code, 401)
        # Logging out invalidates the server token, not just the browser cookie.
        stale = '; '.join(name + '=' + value for name, value in cookies.items())
        self.assertEqual(client.get('/owner/protected-contract', headers={'Cookie': stale}).status_code, 401)

    def test_ui_cookie_authentication(self):
        app = FastAPI()
        app.auth = ['configured']
        app.cookie_id = 'test'
        app.tokens = {'valid': 'owner'}
        from fastapi import Request
        @app.get('/private')
        def private(request: Request):
            require_ui_login(request)
            return {'ok': True}
        client = TestClient(app)
        self.addCleanup(client.close)
        self.assertEqual(client.get('/private').status_code, 401)
        self.assertEqual(client.get('/private', headers={'Cookie': 'access-token-test=invalid'}).status_code, 401)
        self.assertEqual(client.get('/private', headers={'Cookie': 'access-token-test=valid'}).status_code, 200)


if __name__ == '__main__':
    unittest.main()
