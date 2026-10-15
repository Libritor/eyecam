"""Ear as a Microphone, listening page.

Writes <session>/player.html: the clip the listener heard, the clip's true
loudness envelope re-sounded, the envelope the brain gave back re-sounded (on
noise and on the clip's own fine structure), one under the other, with the two
envelopes drawn above them and a playhead that follows whichever is playing.
The page is self-contained (no network); open it from the session folder.

    python analysis/ear_mic_player.py runs/ear_mic1

ear_mic_decode.py calls this at the end of a decode.
"""

import html
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ear_mic_decode import song_envelope, band_env, zscore, FS_D  # noqa: E402

TRACKS = [
    ("clip.wav", "1. The clip, as played through the speakers"),
    ("true_envelope_noise.wav", "2. Its true loudness envelope (1-8 Hz) on white noise: the most a perfect read-out could give back"),
    ("brain_envelope_noise.wav", "3. The envelope read from the EEG, on the same white noise: the brain's version of 2"),
    ("brain_envelope_carrier.wav", "4. The brain's envelope on the clip's own fine structure"),
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ear as a Microphone</title>
<style>
:root{--bg:#fbfaf7;--ink:#1d1d1b;--mute:#6b6b66;--line:#d9d6ce;--true:#4a4a46;--brain:#c0392b;--pane:#ffffff}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16181c;--ink:#ecebe6;--mute:#a09f98;--line:#3a3d44;--true:#c9c7bf;--brain:#ff6b57;--pane:#1f2227}}
:root[data-theme=dark]{--bg:#16181c;--ink:#ecebe6;--mute:#a09f98;--line:#3a3d44;--true:#c9c7bf;--brain:#ff6b57;--pane:#1f2227}
body{margin:0;padding:24px 16px 48px;background:var(--bg);color:var(--ink);font:15px/1.45 Georgia,"Times New Roman",serif;max-width:960px;margin-inline:auto}
h1{font-size:22px;font-weight:normal;margin:0 0 4px}
.sub{color:var(--mute);margin:0 0 18px}
.verdict{padding:10px 14px;border-left:3px solid var(--brain);background:var(--pane);margin:0 0 18px}
canvas{width:100%;height:220px;display:block;background:var(--pane);border:1px solid var(--line);border-radius:4px}
.legend{display:flex;gap:18px;color:var(--mute);font-size:13px;margin:6px 0 20px}
.legend i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px}
.track{display:grid;grid-template-columns:1fr;gap:4px;padding:12px 0;border-top:1px solid var(--line)}
.track label{font-size:14px}
audio{width:100%}
table{border-collapse:collapse;font-size:14px;margin:14px 0}
td{padding:3px 14px 3px 0;color:var(--mute)}td:first-child{color:var(--ink)}
img{max-width:100%;border:1px solid var(--line);border-radius:4px;margin-top:18px}
</style></head><body>
<h1>Ear as a Microphone</h1>
<p class="sub">__SESSION__ &middot; __CLIP__</p>
<div class="verdict">__VERDICT__</div>
<canvas id="c" width="1800" height="440"></canvas>
<div class="legend"><span><i style="background:var(--true)"></i>true envelope of the clip</span><span><i style="background:var(--brain)"></i>envelope read from the EEG (mean of __NREPS__ held-out reconstructions, r = __RAVG__)</span></div>
__TRACKS__
<table>__TABLE__</table>
<img src="ear_mic_figure.png" alt="decoder figure">
<script>
const T=__TRUE__,B=__BRAIN__,FS=__FS__;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let head=-1;
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function draw(){
  const W=cv.width,H=cv.height,pad=40,n=T.length;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle=css('--line');ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(pad,H/2);ctx.lineTo(W-10,H/2);ctx.stroke();
  const x=i=>pad+(W-pad-10)*i/(n-1),y=v=>H/2-v*(H/2-20)/3;
  for(const[arr,col]of[[T,css('--true')],[B,css('--brain')]]){
    ctx.strokeStyle=col;ctx.lineWidth=arr===B?3:2.5;ctx.beginPath();
    arr.forEach((v,i)=>i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v)));ctx.stroke();
  }
  ctx.fillStyle=css('--mute');ctx.font='22px Georgia';
  for(let s=0;s<=n/FS;s+=5){const xx=x(s*FS);ctx.fillText(s+' s',xx-12,H-6);}
  if(head>=0){ctx.strokeStyle=css('--ink');ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x(head*FS),10);ctx.lineTo(x(head*FS),H-24);ctx.stroke();}
}
const auds=[...document.querySelectorAll('audio')];
auds.forEach(a=>{
  a.addEventListener('play',()=>auds.forEach(o=>{if(o!==a)o.pause()}));
  a.addEventListener('timeupdate',()=>{head=a.currentTime;draw()});
  a.addEventListener('ended',()=>{head=-1;draw()});
});
function tick(){const p=auds.find(a=>!a.paused);if(p){head=p.currentTime;draw()}requestAnimationFrame(tick)}
draw();tick();
</script></body></html>
"""


def write_player(session):
    res = json.load(open(os.path.join(session, "ear_mic_results.json")))
    meta = json.load(open(os.path.join(session, "stimulus.json")))
    clip, fs = sf.read(os.path.join(session, "clip.wav"), dtype="float64", always_2d=True)
    true_env = band_env(song_envelope(clip.mean(axis=1), fs))
    brain = zscore(np.load(os.path.join(session, "brain_envelope_64hz.npy")))
    n = min(len(true_env), len(brain))
    al = res["aligned"]
    ctrl = res.get("control") or {}
    rows = [("averaged reconstruction r", f"{al['r_avg']:.3f} (p = {al['p_avg']:.3g}, null 95 % {al['null_avg_95']:.3f})"),
            ("single-trial r, mean over repetitions", f"{al['r_single_mean']:.3f} &plusmn; {al['r_single_sd']:.3f} (p = {al['p_single']:.3g})"),
            ("repetitions decoded", f"{len(al['reps_used'])} of {res['reps_logged']}"),
            ("control, EEG rolled off every clip", f"averaged r {ctrl['r_avg']:.3f} (p = {ctrl['p_avg']:.2g})" if ctrl else "not run"),
            ("decoder channels", ", ".join(res["decoder_channels"]))]
    tracks = "".join(f'<div class="track"><label>{html.escape(t)}</label><audio controls preload="auto" src="{f}"></audio></div>'
                     for f, t in TRACKS if os.path.exists(os.path.join(session, f)))
    page = (PAGE.replace("__SESSION__", html.escape(os.path.basename(os.path.normpath(session))))
            .replace("__CLIP__", html.escape(f"{res['clip_s']:.0f} s of {os.path.basename(meta['song'])} from {meta['start_s']:.0f} s, "
                                             f"{meta['reps']} repetitions"))
            .replace("__VERDICT__", html.escape(res["verdict"]))
            .replace("__NREPS__", str(len(al["reps_used"])))
            .replace("__RAVG__", f"{al['r_avg']:.2f}")
            .replace("__TRACKS__", tracks)
            .replace("__TABLE__", "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows))
            .replace("__TRUE__", json.dumps([round(float(v), 3) for v in true_env[:n]]))
            .replace("__BRAIN__", json.dumps([round(float(v), 3) for v in brain[:n]]))
            .replace("__FS__", str(FS_D)))
    out = os.path.join(session, "player.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


if __name__ == "__main__":
    print("wrote", write_player(sys.argv[1]))
