// §III: "The cursor's shape, size, speed, closeness to face, colour, frequency of flashing
// and brightness were optimized for maximal response." Blocks of flicker at each
// frequency × colour pair, with rests; SNR per condition tells you what works for you.

import { app, requireSource, makeSession, STIM_PAIRS as PAIRS } from '../context.js';
import { h, form, toast, sleep } from '../ui.js';
import { Stage, FlickerClock } from '../stage.js';
import { perfNow } from '../eeg-store.js';
import { showResult } from '../results.js';


export default {
  id: 'tuner', nav: 'Stimulus tuner', title: 'Stimulus optimisation', fig: '§III',
  blurb: 'The paper moved from 12 Hz to 15 Hz yellow/blue after optimising the stimulus (and cites Herrmann 2001 and Pastor 2003 on 1–100 Hz flicker). Run blocks of each frequency and colour pair with rests in between; the table ranks them by SNR. Frequencies that divide your refresh rate evenly (60 Hz: 6, 7.5, 10, 12, 15, 20 Hz …) give the cleanest square waves.',
  mount(el) {
    const v = { freqs: '8.57, 10, 12, 15, 20', pairs: 'yellow/blue, white/black', block: 8, rest: 4, repeats: 2, size: 320, shuffle: true };
    const results = h('div');
    el.append(h('div', { class: 'cols' },
      h('div', { class: 'card' }, form([
        { key: 'freqs', label: 'Frequencies', type: 'text', unit: 'Hz, comma-separated' },
        { key: 'pairs', label: 'Colour pairs', type: 'text', unit: Object.keys(PAIRS).join(', ') },
        { key: 'block', label: 'Block', type: 'number', min: 2, unit: 's' },
        { key: 'rest', label: 'Rest', type: 'number', min: 0, unit: 's' },
        { key: 'repeats', label: 'Repeats', type: 'number', min: 1, step: 1 },
        { key: 'size', label: 'Patch size', type: 'number', min: 20, unit: 'px' },
        { key: 'shuffle', label: 'Randomise order', type: 'checkbox' },
      ], v), h('div', { class: 'btns' }, h('button', { class: 'primary big', onclick: () => run().catch(e => toast(e.message, 6000)) }, 'Start'))),
      h('div', { class: 'card' }, h('p', { class: 'muted' }, 'Look at the dot in the middle of the patch throughout. The best condition can be set as your default; enter it in the other experiments\' frequency/colour fields.'))),
    results);

    const run = async () => {
      requireSource();
      const freqs = v.freqs.split(',').map(Number).filter(x => x > 0);
      const pairs = v.pairs.split(',').map(s => s.trim()).filter(s => PAIRS[s]);
      if (!freqs.length || !pairs.length) throw new Error('Need at least one frequency and one known colour pair');
      let conds = [];
      for (let r = 0; r < v.repeats; r++) for (const f of freqs) for (const p of pairs) conds.push({ f, p });
      if (v.shuffle) conds = conds.map(c => [Math.random(), c]).sort((a, b) => a[0] - b[0]).map(x => x[1]);
      const stage = await new Stage().open();
      const blocks = [];
      let cur = null;
      app.probe(() => (cur ? { s: cur.p === 'white/gray' ? 0.4 : cur.p.includes('black') ? 0.8 : 1, f: cur.f } : null));
      const tStart = perfNow();
      try {
        stage.message(`<div>${conds.length} blocks · ≈ ${(conds.length * (v.block + v.rest) / 60).toFixed(1)} min</div><div class="small">Look at the centre dot. SPACE to start · ESC to stop</div>`);
        stage.clear();
        if ((await stage.waitKey([' ', 'Escape'])) === 'Escape') throw new Error('cancelled');
        stage.message('');
        let stop = false;
        const off = stage.onKey(e => { if (e.key === 'Escape') stop = true; });
        for (const [i, cnd] of conds.entries()) {
          if (stop) break;
          const [A, B] = PAIRS[cnd.p];
          const clock = new FlickerClock(cnd.f, stage.refresh);
          const tr = perfNow();
          await stage.loop(() => { stage.clear(); stage.dot(stage.w / 2, stage.h / 2, 3, '#888'); return !stop && perfNow() - tr < v.rest; });
          let t0 = null;
          cur = cnd;
          await stage.loop((t, ts) => {
            if (stop) return false;
            if (t0 == null) t0 = t;
            if (t - t0 > v.block) return false;
            stage.clear();
            const s = v.size, c = stage.ctx;
            c.fillStyle = clock.advance(ts) ? A : B; c.fillRect(stage.w / 2 - s / 2, stage.h / 2 - s / 2, s, s);
            stage.dot(stage.w / 2, stage.h / 2, 3, '#fff');
          });
          cur = null;
          if (!stop) blocks.push({ freq: cnd.f, colors: cnd.p, t0, t1: t0 + v.block });
          stage.message(`<div class="small">${i + 1} / ${conds.length}</div>`);
        }
        off();
        await sleep(800);
      } finally { app.probe(null); await stage.close(); }
      if (!blocks.length) return;
      const session = makeSession('tuner', { freq: blocks[0].freq, block: v.block, blocks }, tStart, perfNow(), { title: 'Stimulus tuning' });
      results.innerHTML = ''; showResult(results, session); results.scrollIntoView({ behavior: 'smooth' });
    };
  },
};
