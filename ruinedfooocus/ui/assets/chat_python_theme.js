function applyTheme() {
    const frame = element.querySelector('iframe');
    if (!frame) return;
    const send = () => {
        const style = getComputedStyle(element);
        const colors = {};
        for (const name of ['block-background-fill', 'input-background-fill', 'body-text-color',
                            'border-color-primary', 'button-secondary-background-fill']) {
            colors[name] = style.getPropertyValue('--' + name).trim();
        }
        const scheme = document.body.classList.contains('dark') ? 'dark' : 'light';
        const assetBase = new URL(frame.dataset.pythonAssets, window.location.href).href;
        frame.contentWindow.postMessage({type: 'python-theme', colors, scheme, assetBase}, '*');
    };
    frame.addEventListener('load', send, {once: true});
    send();
}
watch('value', applyTheme);
window.addEventListener('message', event => {
    const frame = element.querySelector('iframe');
    if (frame && event.source === frame.contentWindow && event.data?.type === 'python-ready') applyTheme();
});
applyTheme();
const observer = new MutationObserver(applyTheme);
observer.observe(document.body, {attributes: true, attributeFilter: ['class', 'style']});
observer.observe(document.documentElement, {attributes: true, attributeFilter: ['class', 'style']});

// Opaque sandbox workers cannot directly request localhost assets in some browsers.
// Relay only this version's software files; never arbitrary URLs or app responses.
window.addEventListener('message', async event => {
    const frame = element.querySelector('iframe');
    const data = event.data;
    if (!frame || event.source !== frame.contentWindow || data?.type !== 'python-asset') return;
    if (!Number.isSafeInteger(data.id) || typeof data.name !== 'string'
        || !/^[A-Za-z0-9][A-Za-z0-9_.+-]*$/.test(data.name) || data.name.includes('..')) return;
    const reply = {type: 'python-asset-result', id: data.id, generation: data.generation};
    try {
        const base = new URL(frame.dataset.pythonAssets, window.location.href);
        const response = await fetch(new URL(data.name, base), {credentials: 'omit'});
        if (!response.ok) throw new Error(`Python asset unavailable (${response.status}): ${data.name}`);
        reply.bytes = await response.arrayBuffer();
        if (reply.bytes.byteLength > 128 * 1024 * 1024) throw new Error('Python asset exceeds size limit.');
        frame.contentWindow.postMessage(reply, '*', [reply.bytes]);
    } catch (error) {
        frame.contentWindow.postMessage({...reply, bytes: undefined, error: String(error)}, '*');
    }
});
