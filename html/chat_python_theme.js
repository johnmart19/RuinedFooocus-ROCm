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
        frame.contentWindow.postMessage({type: 'python-theme', colors, scheme}, '*');
    };
    frame.addEventListener('load', send, {once: true});
    send();
}
watch('value', applyTheme);
applyTheme();
const observer = new MutationObserver(applyTheme);
observer.observe(document.body, {attributes: true, attributeFilter: ['class', 'style']});
observer.observe(document.documentElement, {attributes: true, attributeFilter: ['class', 'style']});
