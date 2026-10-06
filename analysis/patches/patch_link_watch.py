"""2026-10-05: pause and tell the subject when the link breaks.

Page:   losing the WebSocket no longer aborts the scan. The position in
        progress is held, the page says CONNECTION LOST, reconnects, and
        redoes that position. Every pause shows its reason.
Driver: watches this computer's own network address; if it changes (the PC
        hopped to another Wi-Fi mid-run) the scan is paused with a message
        naming the new address the phone must stream to. The EEG-silence
        pause now says what to check. The Muse badge flags an Oz electrode
        that is not in contact.
Usage: python analysis/patches/patch_link_watch.py [repo dir]  (default .)
"""
import io
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
NL = chr(10)


def rw(path, fn):
    x = io.open(path, encoding="utf-8").read()
    y = fn(x)
    assert y != x, path
    io.open(path, "w", encoding="utf-8", newline=NL).write(y)


def page(x):
    # Minu's page declares the pause flag together with another one
    x = x.replace("let paused = false, redoNow = false;" + NL + "async function holdWhilePaused() {",
                  "let redoNow = false;" + NL + "let paused = false;" + NL + "async function holdWhilePaused() {")
    # a halt is either a driver pause or a lost link
    a = '''let paused = false;
async function holdWhilePaused() {
  if (!paused) return false;
  black();
  send({ type: "frame", stage: "paused", pt: now(), gx: -1, gy: -1, lum: 0, fl: 0 });
  const keep = sub.textContent;
  while (paused) {
    sub.textContent = "PAUSED - no EEG data from the Muse. Check MuseLog on the phone.";
    await sleep(200);
  }'''
    b = '''let paused = false, linkLost = false, pauseReason = "";
const halted = () => paused || linkLost;
async function holdWhilePaused() {
  if (!halted()) return false;
  black();
  send({ type: "frame", stage: "paused", pt: now(), gx: -1, gy: -1, lum: 0, fl: 0 });
  const keep = sub.textContent;
  title.textContent = "PAUSED"; title.className = "warn";
  while (halted()) {
    sub.textContent = linkLost
      ? "CONNECTION TO THE COMPUTER LOST - keep the headset on; reconnecting. This position will be shown again."
      : (pauseReason || "PAUSED - no EEG data from the Muse. Check MuseLog on the phone.");
    await sleep(200);
  }
  title.textContent = "";'''
    assert x.count(a) == 1, "holdWhilePaused"
    x = x.replace(a, b)
    # every dwell loop and redo check honours a lost link too
    x, n1 = re.subn(r"&& !paused\)", "&& !halted())", x)
    x, n2 = re.subn(r"if \(paused\) \{ (\w+)--; continue; \}", r"if (halted()) { \1--; continue; }", x)
    assert n1 >= 2 and n2 >= 2, (n1, n2)
    # the driver can say why it pauses
    a = '''      case "pause": paused = true; break;
      case "resume": paused = false; break;'''
    b = '''      case "pause": paused = true; pauseReason = m.text || ""; break;
      case "resume": paused = false; pauseReason = ""; break;'''
    assert x.count(a) == 1, "pause cmd"
    x = x.replace(a, b)
    # losing the socket holds the run instead of killing it
    a = '''  ws.onclose = () => { running = null; sub.textContent = "driver disconnected — retrying"; setTimeout(connect, 2000); };'''
    b = '''  ws.onclose = () => {
    linkLost = true;   // hold the run; holdWhilePaused() shows the message
    if (!running) sub.textContent = "connection to the computer lost — reconnecting";
    setTimeout(connect, 2000);
  };'''
    assert x.count(a) == 1, "onclose"
    x = x.replace(a, b)
    a = '''  ws.onopen = () => { sub.textContent = "connected — waiting for driver";'''
    b = '''  ws.onopen = () => { linkLost = false; if (!running) sub.textContent = "connected — waiting for driver";'''
    assert x.count(a) == 1, "onopen"
    x = x.replace(a, b)
    # Oz contact shown in words on the badge
    a = '''  badge.innerHTML = `MUSE ${m.rate.toFixed(0)} Hz   ${cells}`;
}'''
    b = '''  badge.innerHTML = `MUSE ${m.rate.toFixed(0)} Hz   ${cells}`;
  const oz = m.names.findIndex(n => /^AUX/i.test(n) || /^OZ$/i.test(n));
  if (oz >= 0 && !m.q[oz]) {
    badge.innerHTML += `<br><span style="color:#ff6666">Oz electrode: no contact — wet it and press it down</span>`;
  }
}'''
    assert x.count(a) == 1, "badge"
    return x.replace(a, b)


def driver(x):
    a = '''            async def heartbeat():
                while True:
                    tail.poll()'''
    b = '''            ip0 = local_ip()

            async def heartbeat():
                nonlocal ip0
                while True:
                    tail.poll()
                    # this computer hopping to another Wi-Fi mid-run breaks
                    # the phone's stream: say so, with the address to use
                    ip_now = local_ip()
                    if ip_now != ip0 and ip_now != "127.0.0.1":
                        print(f"NETWORK CHANGED: this PC is now {ip_now} (was {ip0}); "
                              f"the phone must stream to {ip_now}")
                        await self.send(cmd="pause", text=f"PAUSED - the computer changed "
                                        f"network: it is now {ip_now}. On the phone, set "
                                        f"MuseLog's target to {ip_now} port 5000; the run "
                                        "continues when EEG returns.")
                        self.paused = True
                        ip0 = ip_now'''
    assert x.count(a) == 1, "heartbeat"
    x = x.replace(a, b)
    a = '''                            self.paused = True
                            print("EEG STREAM SILENT: scan paused")
                            await self.send(cmd="pause")'''
    b = '''                            self.paused = True
                            print("EEG STREAM SILENT: scan paused")
                            await self.send(cmd="pause", text="PAUSED - no EEG for 3 s. On "
                                            "the phone: is MuseLog still connected to the "
                                            f"Muse and streaming to {local_ip()} port 5000? "
                                            "The run continues when data returns.")'''
    assert x.count(a) == 1, "silence pause"
    return x.replace(a, b)


rw(f"{root}/xr_stimulus.html", page)
rw(f"{root}/xr_session.py", driver)
print("patched", root)
