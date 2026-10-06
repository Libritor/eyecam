"""Resolve the five xr_session.py conflict hunks of the colour merge
(Khalil-Color-Branch + meny's-branch + all-on-pc), keeping every side's
feature. Run once from the repo root while the merge is in progress."""
import io
import re

P = "xr_session.py"
x = io.open(P, encoding="utf-8").read()
pat = re.compile(r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> all-on-pc\n", re.S)
hunks = pat.findall(x)
assert len(hunks) == 5, len(hunks)

R = []
# 1: driver state - both
R.append(hunks[0][0] + hunks[0][1])
# 2: colour plan print (smooth tag) + colour pass rotation - both
a, b = hunks[1]
R.append(a + b.split("\n", 2)[2])        # drop all-on-pc's plain print line
# 3: recorder choice - one switch for all sources
R.append('''        if a.source in ("lsl", "muselsl"):
            # Muse on this PC's Bluetooth through muselsl, or any LSL EEG
            # stream: recorded straight from LSL, no phone app and no OSC
            ports = []
            recorder = Recorder("lsl", self.session, 0,
                                extra=["--no-aux"] if a.no_aux else [])
        elif len(ports) == 1:
            recorder = Recorder(a.source, self.session, ports[0], muse=a.muse)
''')
# 4: arguments - --source picks the path, --muse names the headband
R.append('''    ap.add_argument("--source", choices=["osc", "ble", "muselsl", "lsl"], default="osc",
                    help="osc = EEG from a phone app or the phantom over UDP; "
                         "ble = the Muse straight over this computer's Bluetooth "
                         "(muse_ble.py, no phone app); muselsl = the Muse over "
                         "this computer's Bluetooth through muselsl -> LSL; "
                         "lsl = an EEG stream another program already puts on LSL "
                         "(muselsl stream, mind2motor run_muse.py --live, Petal)")
    ap.add_argument("--muse", default="",
                    help="--source ble: part of the headband's name or address; "
                         "--source muselsl: its address")
    ap.add_argument("--lsl", action="store_true", help="same as --source lsl")
    ap.add_argument("--muse-model", default="auto", choices=["auto", "athena", "legacy"],
                    help="Muse protocol for --source muselsl (athena = Muse S Gen 3)")
    ap.add_argument("--muse-preset", default=None,
                    help="--source muselsl: muselsl preset (default p20 = aux/Oz ON on "
                         "a classic Muse; none with --no-aux or --muse-model athena)")
    ap.add_argument("--no-aux", action="store_true",
                    help="--source muselsl/lsl: drop the aux column (no Oz electrode)")
''')
# 5: start-up - muselsl stream when asked, otherwise print where the EEG comes from
R.append('''        if args.lsl:
            args.source = "lsl"
        muse = None
        if args.source == "muselsl":
            from muse_stream import MuseStream
            preset = args.muse_preset
            if preset is None and not args.no_aux and args.muse_model != "athena":
                preset = "20"           # classic Muse: aux (Oz) input ON
            muse = MuseStream(args.muse or None, args.muse_model, preset=preset).start()
            print("EEG in:         Muse over this PC's Bluetooth (muselsl -> LSL)"
                  + (f", preset {preset}" if preset else "")
                  + ("" if preset or args.no_aux else
                     "; Athena: aux preset not set - check AUX with --mode alpha"))
        elif args.source == "lsl":
            print("EEG in:         LSL EEG stream (start muselsl / mind2motor first)")
        elif args.source == "ble":
            print("EEG in:         Muse over this computer's Bluetooth "
                  "(no phone app)")
        else:
            print(f"EEG OSC in:     {local_ip()}:{args.osc_port}  "
                  "(Mind Monitor / MuseLog / phantom)")
        try:
            code = await driver.run()
        finally:
            if muse:
                muse.stop()
''')
it = iter(R)
out = pat.sub(lambda m: next(it), x)
assert "<<<<<<<" not in out
# the two leftover references to the old flags
out = out.replace('eeg_src = "LSL" if (a.muse or a.lsl) else f"udp:{a.osc_port}"',
                  'eeg_src = "LSL" if a.source in ("lsl", "muselsl") else \\\n'
                  '            ("Bluetooth" if a.source == "ble" else f"udp:{a.osc_port}")')
out = out.replace('("the Muse over this PC\'s Bluetooth" if a.muse else "an LSL EEG stream")',
                  '("the Muse over this PC\'s Bluetooth" if a.source in ("ble", "muselsl")'
                  ' else "an LSL EEG stream")')
io.open(P, "w", encoding="utf-8", newline="\n").write(out)
print("resolved", P)
