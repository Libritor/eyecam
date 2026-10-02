"""Everything on one computer, one command.

    python eyecam.py                 # read "NO" from the EEG
    python eyecam.py --what full     # image, then colour, then tagged music
    python eyecam.py --what alpha    # is the Oz electrode on the scalp?
    python eyecam.py --what sweep    # which flicker frequency suits this person?
    python eyecam.py --what music    # the 40 Hz tagged music on its own
    python eyecam.py --demo          # no hardware at all: synthetic subject

What it starts:
  * the Muse, connected straight over this computer's Bluetooth
    (muse_ble.py). No MuseLog, no Mind Monitor, no phone.
  * the session driver (xr_session.py): recording, gates, scoring, image.
  * the stimulus page, opened in this computer's browser. Press F11 or click
    the page; it goes full screen on the first click.
  * sound, where a stage has any, from this computer's audio output.

Options:
  --headset   show the stimulus in a Meta Quest browser instead (the page is
              pushed over adb if the headset is plugged in; the address is
              printed either way)
  --phone     take the EEG from a phone app over Wi-Fi (the old path)
  --muse X    pick a headband by part of its name or address
Anything after `--` goes to xr_session.py unchanged, for example
    python eyecam.py -- --target pix:HI --passes 2
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

COMMON = ["--freq", "12", "--calib-blocks", "6", "--calib-on", "8", "--calib-off", "8",
          "--calib-style", "bw", "--calib-size", "1"]
WHAT = {
    "image": ["--mode", "visual", "--target", "pix:NO", "--spc", "8", "--passes", "3"],
    "full": ["--mode", "full", "--target", "pix:NO", "--spc", "8", "--passes", "2",
             "--color-reps", "3", "--color-on", "6", "--color-spc", "6",
             "--music-blocks", "6"],
    "alpha": ["--mode", "alpha", "--calib-blocks", "4", "--calib-on", "20",
              "--calib-off", "20"],
    "sweep": ["--mode", "sweep", "--sweep", "7.5,10,12,15,20", "--calib-blocks", "4",
              "--calib-on", "8", "--calib-off", "8", "--calib-style", "bw",
              "--calib-size", "1"],
    "music": ["--mode", "music", "--music-blocks", "10", "--music-on", "10",
              "--music-off", "10"],
}


def lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=list(WHAT), default="image")
    ap.add_argument("--session", default="")
    ap.add_argument("--muse", default="")
    ap.add_argument("--headset", action="store_true")
    ap.add_argument("--phone", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open the page; just print its address")
    ap.add_argument("--http-port", type=int, default=8082)
    args = ap.parse_args(argv)

    session = args.session or os.path.join(
        "runs", f"{'demo' if args.demo else args.what}_{time.strftime('%Y%m%d_%H%M%S')}")
    token = "pc"
    cmd = [PY, "-u", "xr_session.py", "--session", session, "--page-token", token,
           "--http-port", str(args.http_port), "--signal-timeout", "1800"]
    cmd += (WHAT[args.what] if args.what in ("alpha", "sweep", "music")
            else COMMON + WHAT[args.what])
    if not args.headset:
        cmd += ["--color-freqs", "8,10,12"]      # frame-exact on 60/120/240 Hz panels
    if args.demo:
        cmd += ["--osc-port", "5001", "--spc", "2", "--passes", "2", "--spectator", ""]
    elif args.phone:
        cmd += ["--source", "osc"]
    else:
        cmd += ["--source", "ble"] + (["--muse", args.muse] if args.muse else [])
    if not args.demo:
        try:
            import sounddevice  # noqa: F401
            cmd += ["--pc-audio"]
        except Exception:
            print("(sounddevice is not installed: any music plays from the page)")
    cmd += extra

    print("session folder:", session)
    driver = subprocess.Popen(cmd, cwd=ROOT)
    procs = [driver]
    try:
        time.sleep(3.0)
        if driver.poll() is not None:
            return driver.returncode
        if args.demo:
            os.makedirs(os.path.join(ROOT, session), exist_ok=True)
            procs.append(subprocess.Popen(
                [PY, "phantom_subject.py", "--session", session, "--host", "127.0.0.1",
                 "--port", "5001", "--max-seconds", "1800"], cwd=ROOT))
        q = f"?k={token}" + ("&auto=1" if args.demo else "")
        local = f"http://localhost:{args.http_port}/{q}"
        remote = f"http://{lan_ip()}:{args.http_port}/{q}"
        if args.headset:
            print("open this in the headset's browser:", remote)
            if shutil.which("adb"):
                subprocess.run(["adb", "shell", "am start -a android.intent.action.VIEW "
                                f"-d '{remote}' com.oculus.browser"],
                               capture_output=True, timeout=20)
        elif args.no_browser:
            print("stimulus page:", local)
        else:
            print("opening the stimulus page:", local)
            webbrowser.open(local)
        return driver.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    sys.exit(main())
