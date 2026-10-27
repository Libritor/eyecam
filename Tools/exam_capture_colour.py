"""Capture the stimulus page's canvas during an ON frame of the colour test
(a screenshot misses the lit half of a 15 Hz flicker as often as not): runs a
colour-only exam against the phantom in Playwright's Chromium and saves the
canvas when the patch is green.

    python Tools/exam_capture_colour.py [out.png]
"""
import base64
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTTP, OSC = 8087, 5005


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Tools", "preview", "exam", "colour_on.png")
    session = os.path.join("runs", "exam_capture")
    os.makedirs(os.path.join(ROOT, session), exist_ok=True)
    cmd = [sys.executable, "xr_session.py", "--mode", "exam", "--exam-eye", "right",
           "--exam-tests", "colour", "--exam-on", "4", "--exam-off", "1", "--exam-reps", "1",
           "--freq", "15", "--exam-ppd", "30", "--session", session, "--http-port", str(HTTP),
           "--osc-port", str(OSC), "--spectator", "", "--signal-timeout", "120", "--linger", "1"]
    procs = []
    try:
        driver = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(driver)
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{HTTP}/", timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        procs.append(subprocess.Popen(
            [sys.executable, "phantom_subject.py", "--session", session, "--host", "127.0.0.1",
             "--port", str(OSC), "--aux", "--max-seconds", "300"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.goto(f"http://127.0.0.1:{HTTP}/?auto=1")
            got = False
            t0 = time.time()
            while driver.poll() is None and time.time() - t0 < 120 and not got:
                r = page.evaluate("""() => {
                    const side = Math.min(cv.width, cv.height) * 0.6;
                    const x = Math.round((cv.width - side) / 2 + 12), y = Math.round((cv.height - side) / 2 + 12);
                    const d = ctx.getImageData(x, y, 1, 1).data;
                    return (d[1] > 200 && d[0] < 50) ? cv.toDataURL('image/png') : '';
                }""")
                if r:
                    with open(out, "wb") as f:
                        f.write(base64.b64decode(r.split(",", 1)[1]))
                    got = True
                time.sleep(0.01)
            browser.close()
        print("saved" if got else "no green frame seen", out)
    finally:
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()
        stop = os.path.join(ROOT, session, "stop_recorder")
        open(stop, "w").close()


if __name__ == "__main__":
    main()
