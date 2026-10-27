"""Screenshots of the exam page while a short exam session runs, to see what the
subject sees: the grating on its grey, a field sector, a colour plane.

    python Tools/exam_screens.py [out_dir]

Starts xr_session.py --mode exam (one eye, one ON/OFF pair per step, short
blocks) on its own ports with the phantom as the EEG, opens the page in
Playwright's Chromium at 1200x800, and saves a PNG every 1.5 s until the
driver exits. Nothing of this touches runs/ except runs/exam_screens.
"""
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTTP, OSC = 8087, 5005


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Tools", "preview", "exam")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".png"):
            os.remove(os.path.join(out, f))
    session = os.path.join("runs", "exam_screens")
    shutil.rmtree(os.path.join(ROOT, session), ignore_errors=True)
    os.makedirs(os.path.join(ROOT, session))
    cmd = [sys.executable, "xr_session.py", "--mode", "exam", "--exam-eye", "right",
           "--exam-tests", "acuity,contrast,field,colour", "--exam-sizes", "40,10,4",
           "--exam-contrasts", "1,0.12", "--exam-field", "4x4", "--exam-on", "2", "--exam-off", "1",
           "--exam-reps", "1", "--freq", "15", "--exam-ppd", "30", "--session", session,
           "--http-port", str(HTTP), "--osc-port", str(OSC), "--spectator", "",
           "--signal-timeout", "120", "--linger", "1"]
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
             "--port", str(OSC), "--aux", "--max-seconds", "600"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.goto(f"http://127.0.0.1:{HTTP}/?auto=1")
            n = 0
            while driver.poll() is None and n < 400:
                page.screenshot(path=os.path.join(out, f"exam_{n:03d}.png"))
                n += 1
                time.sleep(1.5)
            browser.close()
        print(f"{n} screenshots in {out}; driver exit {driver.poll()}")
    finally:
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()


if __name__ == "__main__":
    main()
