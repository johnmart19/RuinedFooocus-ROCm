const handle = element.querySelector('[role="separator"]');
const row = element.closest('#chat-layout');
const setWidth = percent => {
    percent = Math.max(35, Math.min(85, percent));
    row.style.setProperty('--chat-width', percent + '%');
    handle.setAttribute('aria-valuenow', Math.round(percent));
};
handle.addEventListener('pointerdown', event => {
    handle.focus();
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
});
handle.addEventListener('pointermove', event => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    const bounds = row.getBoundingClientRect();
    setWidth((event.clientX - bounds.left) / bounds.width * 100);
});
handle.addEventListener('pointerup', event => handle.releasePointerCapture(event.pointerId));
handle.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return;
    event.preventDefault();
    const current = Number(handle.getAttribute('aria-valuenow'));
    setWidth(event.key === 'Home' ? 72 : current + (event.key === 'ArrowLeft' ? -2 : 2));
});
handle.addEventListener('dblclick', () => setWidth(72));
