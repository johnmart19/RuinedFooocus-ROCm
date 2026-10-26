const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const listeners = [];
const messages = [];
const fetched = [];
const child = {postMessage: (message) => messages.push(message)};
const frame = {dataset: {pythonAssets: 'browser-python/314.0.7/'}, contentWindow: child, addEventListener() {}};
const context = {
    element: {querySelector: () => frame},
    window: {location: {href: 'http://localhost:7860/'}, addEventListener: (type, fn) => listeners.push(fn)},
    document: {body: {classList: {contains: () => false}}, documentElement: {}},
    getComputedStyle: () => ({getPropertyValue: () => '#fff'}),
    watch() {}, MutationObserver: class {observe() {}}, URL,
    fetch: async (url, options) => {
        fetched.push([String(url), options]);
        return {ok: true, arrayBuffer: async () => new ArrayBuffer(4)};
    }
};
vm.runInNewContext(fs.readFileSync('html/chat_python_theme.js', 'utf8'), context);
const relay = listeners.at(-1);
(async () => {
    for (const name of ['../config.json', 'https://evil.invalid/code.js', '/private', 'file..js']) {
        await relay({source: child, data: {type: 'python-asset', id: 1, name}});
    }
    await relay({source: {}, data: {type: 'python-asset', id: 1, name: 'pyodide.js'}});
    assert.equal(fetched.length, 0);
    await relay({source: child, data: {type: 'python-asset', id: 2, generation: 9, name: 'pyodide.js'}});
    assert.equal(fetched.length, 1);
    assert.equal(fetched[0][0], 'http://localhost:7860/browser-python/314.0.7/pyodide.js');
    assert.equal(fetched[0][1].credentials, 'omit');
    assert.equal(messages.at(-1).generation, 9);
    assert.equal(messages.at(-1).bytes.byteLength, 4);
    console.log('Browser Python bridge: source, path, credentials and generation checks passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
