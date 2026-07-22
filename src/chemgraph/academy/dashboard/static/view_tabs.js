// === View switcher: Observability / Canvas ===
(function initViewTabs() {
  const VIEWS = {
    observability: 'observabilityView',
    canvas: 'canvasView',
  };
  document.querySelectorAll('.viewTab').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.view;
      document.querySelectorAll('.viewTab').forEach(b => {
        const active = b.dataset.view === target;
        b.classList.toggle('active', active);
        b.style.borderBottom = active ? '2px solid #2563eb' : '2px solid transparent';
        b.style.color = active ? '' : '#6b7280';
      });
      for (const [name, elId] of Object.entries(VIEWS)) {
        const el = document.getElementById(elId);
        if (el) el.style.display = name === target ? '' : 'none';
      }
      // Lazy-init the canvas the first time it's opened.
      if (target === 'canvas' && !window.__canvasInited) {
        window.__canvasInited = true;
        window.__initCanvas && window.__initCanvas();
      }
    });
  });
})();
