// Small DOM / form / plotting helpers.

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const c of kids.flat(Infinity)) if (c != null && c !== false) el.append(c instanceof Node ? c : String(c));
  return el;
}

// Schema-driven form. schema: [{key, label, type, options, min, max, step, help, show}]
// type: number | select | checkbox | text | color | range | section
export function form(schema, values, onChange) {
  const el = h('div', { class: 'form' });
  const inputs = {};
  const rows = [];
  const refresh = () => { for (const [row, f] of rows) row.style.display = f.show && !f.show(values) ? 'none' : ''; };
  for (const f of schema) {
    if (f.type === 'section') { const r = h('div', { class: 'form-section' }, f.label); rows.push([r, f]); el.append(r); continue; }
    let inp;
    const v = values[f.key];
    if (f.type === 'select') {
      inp = h('select', {}, f.options.map(o => {
        const [val, lab] = Array.isArray(o) ? o : [o, o];
        return h('option', { value: val, selected: String(val) === String(v) }, lab);
      }));
    } else if (f.type === 'checkbox') inp = h('input', { type: 'checkbox', checked: !!v });
    else if (f.type === 'color') inp = h('input', { type: 'color', value: v });
    else if (f.type === 'text') inp = h('input', { type: 'text', value: v ?? '' });
    else inp = h('input', { type: f.type === 'range' ? 'range' : 'number', value: v, min: f.min, max: f.max, step: f.step ?? 'any' });
    const read = () => {
      if (f.type === 'checkbox') return inp.checked;
      if (f.type === 'select') { const o = f.options.map(o => (Array.isArray(o) ? o[0] : o)).find(o => String(o) === inp.value); return o ?? inp.value; }
      if (f.type === 'number' || f.type === 'range' || f.type == null) return parseFloat(inp.value);
      return inp.value;
    };
    inp.addEventListener(f.type === 'number' || f.type == null || f.type === 'text' ? 'change' : 'input', () => {
      values[f.key] = read(); refresh(); onChange?.(f.key, values[f.key], values);
    });
    inputs[f.key] = inp;
    const row = h('label', { class: 'form-row' + (f.type === 'checkbox' ? ' check' : ''), title: f.help || '' },
      h('span', { class: 'lbl' }, f.label), inp, f.unit ? h('span', { class: 'unit' }, f.unit) : null);
    rows.push([row, f]);
    el.append(row);
  }
  refresh();
  el.set = (k, v) => {
    values[k] = v; const inp = inputs[k]; if (!inp) return;
    if (inp.type === 'checkbox') inp.checked = !!v; else inp.value = v;
    refresh();
  };
  el.setAll = obj => { for (const [k, v] of Object.entries(obj)) el.set(k, v); };
  return el;
}

export function button(label, onClick, cls = '') { return h('button', { class: cls, onclick: onClick }, label); }

export function download(name, data, type = 'application/octet-stream') {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const a = h('a', { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
export function downloadCanvas(name, canvas) { canvas.toBlob(b => download(name, b), 'image/png'); }

export function stamp() { return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19); }

let toastTimer;
export function toast(msg, ms = 3500) {
  let t = document.getElementById('toast');
  if (!t) { t = h('div', { id: 'toast' }); document.body.append(t); }
  t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), ms);
}

// Theme-aware colours read from CSS custom properties.
export function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// Line chart. series: [{x:[], y:[], color, width, label}], bands: [{a, b, color}], marks: [{x, color}]
export function lineChart(canvas, { series = [], bands = [], marks = [], hlines = [], xlabel = '', ylabel = '', xmin, xmax, ymin, ymax, logy = false } = {}) {
  const dpr = devicePixelRatio || 1, W = canvas.clientWidth || 600, H = canvas.clientHeight || 220;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const c = canvas.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  const fg = cssVar('--fg') || '#ddd', mute = cssVar('--muted') || '#888', grid = cssVar('--line') || '#333';
  c.clearRect(0, 0, W, H);
  const L = 48, R = 10, T = 10, B = 30, pw = W - L - R, ph = H - T - B;
  const allX = series.flatMap(s => s.x), allY = series.flatMap(s => s.y).filter(Number.isFinite);
  const tf = v => (logy ? Math.log10(Math.max(v, 1e-12)) : v);
  let x0 = xmin ?? Math.min(...allX), x1 = xmax ?? Math.max(...allX);
  let y0 = ymin ?? Math.min(...allY.map(tf)), y1 = ymax ?? Math.max(...allY.map(tf));
  if (!Number.isFinite(x0) || !Number.isFinite(x1)) { x0 = 0; x1 = 1; }
  if (!Number.isFinite(y0) || !Number.isFinite(y1)) { y0 = 0; y1 = 1; }
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) { y1 += 1; y0 -= 1; }
  const X = v => L + (v - x0) / (x1 - x0) * pw, Y = v => T + ph - (tf(v) - y0) / (y1 - y0) * ph;
  c.font = '11px system-ui, sans-serif'; c.fillStyle = mute; c.strokeStyle = grid; c.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yv = y0 + (y1 - y0) * i / 4, yy = T + ph - ph * i / 4;
    c.beginPath(); c.moveTo(L, yy); c.lineTo(L + pw, yy); c.stroke();
    c.fillText(logy ? '1e' + yv.toFixed(1) : fmt(yv), 4, yy + 4);
    const xv = x0 + (x1 - x0) * i / 4; c.fillText(fmt(xv), X(xv) - 10, H - 12);
  }
  for (const b of bands) { c.fillStyle = b.color || 'rgba(230,60,60,.25)'; c.fillRect(X(b.a), T, Math.max(1, X(b.b) - X(b.a)), ph); }
  for (const m of marks) { c.strokeStyle = m.color || '#2a2'; c.lineWidth = 2; c.beginPath(); c.moveTo(X(m.x), T); c.lineTo(X(m.x), T + 8); c.stroke(); }
  for (const hl of hlines) { c.strokeStyle = hl.color || mute; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(L, Y(hl.y)); c.lineTo(L + pw, Y(hl.y)); c.stroke(); c.setLineDash([]); }
  for (const s of series) {
    c.strokeStyle = s.color || fg; c.lineWidth = s.width || 1.5; c.beginPath();
    let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const v = s.y[i];
      if (!Number.isFinite(v)) { pen = false; continue; }
      const px = X(s.x[i]), py = Math.min(T + ph, Math.max(T, Y(v)));
      if (pen) c.lineTo(px, py); else { c.moveTo(px, py); pen = true; }
    }
    c.stroke();
  }
  c.fillStyle = mute; c.fillText(xlabel, L + pw / 2 - 20, H - 1);
  c.save(); c.translate(10, T + ph / 2 + 20); c.rotate(-Math.PI / 2); c.fillText(ylabel, 0, 0); c.restore();
  let lx = L + 8;
  for (const s of series) if (s.label) { c.fillStyle = s.color || fg; c.fillRect(lx, T + 4, 10, 3); c.fillText(s.label, lx + 14, T + 10); lx += c.measureText(s.label).width + 30; }
}
const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toPrecision(2));

// Put a Float32 [0,1] image on a canvas via a colormap (see interp.toImageData).
export function readFile(file, as = 'text') {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result); r.onerror = rej;
    if (as === 'dataurl') r.readAsDataURL(file); else r.readAsText(file);
  });
}
export function loadImage(src) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
}
export const sleep = ms => new Promise(r => setTimeout(r, ms));
