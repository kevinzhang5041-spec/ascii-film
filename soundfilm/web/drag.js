// Drag and drop for the dashboard pages (served at /web/drag.js).
//
// SF.pointerDrag: drag things within the page, such as photos between pools. It is built on pointer events,
// so it behaves the same in every browser, including embedded ones where HTML drag and drop is unreliable.
// SF.fileDrops: files and folders dragged in from Finder. It also stops the browser from leaving the page to
// open a file that was dropped somewhere else.
(function () {
  const SF = window.SF = {};

  // o: {root, items, targets, start(el, e) -> drag data with .ghost (HTML) or null, accepts(target, d),
  //     label(target|null, d) -> HTML, drop(target, d), end(d), litClass}
  SF.pointerDrag = function (o) {
    const lit = o.litClass || 'dropok';
    let down = null, drag = null, ghost = null, over = null, swallow = false;
    const finish = () => {
      if (ghost) ghost.remove();
      if (over) over.classList.remove(lit);
      document.body.classList.remove('dragging');
      const d = drag;
      ghost = over = drag = down = null;
      if (d && o.end) o.end(d);
    };
    o.root.addEventListener('pointerdown', e => {
      if (e.button !== 0 || e.pointerType === 'touch' || drag) return;
      const el = e.target.closest(o.items);
      if (el && o.root.contains(el)) down = {el, x: e.clientX, y: e.clientY, e};
    });
    document.addEventListener('pointermove', e => {
      if (!down) return;
      if (!drag) {
        if (Math.hypot(e.clientX - down.x, e.clientY - down.y) < (o.threshold || 6)) return;
        const d = o.start(down.el, down.e);
        if (!d) { down = null; return; }
        drag = d;
        ghost = document.createElement('div');
        ghost.className = 'dragghost';
        ghost.innerHTML = d.ghost || '';
        document.body.append(ghost);
        document.body.classList.add('dragging');
        try { getSelection().removeAllRanges(); } catch {}
      }
      ghost.style.left = e.clientX + 'px';
      ghost.style.top = e.clientY + 'px';
      // near the bottom or right edge, the picture goes above or left of the pointer so it stays in view
      const up = e.clientY + ghost.offsetHeight + 20 > innerHeight, left = e.clientX + ghost.offsetWidth + 20 > innerWidth;
      ghost.style.transform = `translate(${left ? 'calc(-100% - 14px)' : '14px'}, ${up ? 'calc(-100% - 12px)' : '12px'})`;
      const hit = document.elementFromPoint(e.clientX, e.clientY);
      const t = hit && hit.closest(o.targets);
      const ok = t && (!o.accepts || o.accepts(t, drag)) ? t : null;
      if (ok !== over) {
        if (over) over.classList.remove(lit);
        over = ok;
        if (over) over.classList.add(lit);
      }
      const l = ghost.querySelector('.to');
      if (l && o.label) l.innerHTML = o.label(over, drag);
    });
    document.addEventListener('pointerup', () => {
      if (!down) return;
      if (!drag) { down = null; return; }
      swallow = true;                              // the click that follows this release is not a click
      setTimeout(() => { swallow = false; }, 0);
      const t = over, d = drag;
      finish();
      if (t) o.drop(t, d);
    });
    document.addEventListener('pointercancel', () => { if (down) finish(); });
    window.addEventListener('blur', () => { if (down) finish(); });
    document.addEventListener('keydown', e => {
      if (drag && e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); finish(); }
    }, true);
    document.addEventListener('click', e => {
      if (swallow) { e.preventDefault(); e.stopImmediatePropagation(); swallow = false; }
    }, true);
    o.root.addEventListener('dragstart', e => { if (e.target.closest && e.target.closest(o.items)) e.preventDefault(); });
    return {active: () => !!drag, cancel: finish};
  };

  // o: {where(el under the pointer) -> destination name, or null where files can't go;
  //     light(dest) -> element to highlight; show(on); drop(dest, files); refused(el); litClass}
  SF.fileDrops = function (o) {
    const cls = o.litClass || 'dropok';
    let shown = false, depth = 0, timer = null, lit = null;
    const hasFiles = e => !!e.dataTransfer && [...e.dataTransfer.types].includes('Files');
    const light = el => {
      if (el === lit) return;
      if (lit) lit.classList.remove(cls);
      lit = el;
      if (lit) lit.classList.add(cls);
    };
    const hide = () => {
      clearTimeout(timer); depth = 0; light(null);
      if (shown) { shown = false; if (o.show) o.show(false); }
    };
    const dest = e => o.where(e.target && e.target.closest ? e.target : null);
    document.addEventListener('dragenter', e => { if (hasFiles(e)) depth++; });
    document.addEventListener('dragleave', e => { if (hasFiles(e) && --depth <= 0) hide(); });
    document.addEventListener('dragover', e => {
      e.preventDefault();                          // dropped anywhere else, a file would replace this page
      if (!hasFiles(e)) { e.dataTransfer.dropEffect = 'none'; return; }
      if (!shown) { shown = true; if (o.show) o.show(true); }
      clearTimeout(timer);                         // dragover repeats while a drag is over the page
      timer = setTimeout(hide, 1000);
      const to = dest(e);
      light(to && o.light ? o.light(to) : null);
      e.dataTransfer.dropEffect = to ? 'copy' : 'none';
    });
    document.addEventListener('drop', e => {
      e.preventDefault();
      if (!hasFiles(e)) { hide(); return; }
      const to = dest(e);
      const files = to ? SF.collectFiles(e.dataTransfer) : null;   // read now: the drop's data goes away after this
      hide();
      if (!to) { if (o.refused) o.refused(e.target); return; }
      files.then(fs => o.drop(to, fs), err => o.drop(to, [], err));
    });
    document.addEventListener('dragend', hide);
  };

  // The files in a drop. Folders are opened and every file inside them is returned; hidden files are left out.
  SF.collectFiles = function (dt) {
    const items = [...(dt.items || [])].filter(i => i.kind === 'file');
    const plain = [...(dt.files || [])];
    const entries = items.map(i => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null));
    const out = [];
    const walk = async ent => {
      if (ent.name.startsWith('.')) return;
      if (ent.isFile) {
        try { out.push(await new Promise((res, rej) => ent.file(res, rej))); } catch {}
        return;
      }
      if (!ent.isDirectory) return;
      const reader = ent.createReader();
      for (;;) {                                  // a folder is read in batches until one comes back empty
        const batch = await new Promise((res, rej) => reader.readEntries(res, rej)).catch(() => []);
        if (!batch.length) break;
        for (const b of batch) await walk(b);
      }
    };
    return (async () => {
      const n = Math.max(entries.length, plain.length);
      for (let i = 0; i < n; i++) {
        if (entries[i]) await walk(entries[i]);
        else if (plain[i] && !plain[i].name.startsWith('.')) out.push(plain[i]);
      }
      return out;
    })();
  };

  SF.extOf = name => { const m = /\.[^./]+$/.exec(name || ''); return m ? m[0].toLowerCase() : ''; };
})();
