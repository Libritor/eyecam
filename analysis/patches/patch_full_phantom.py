"""Patch phantom_subject.py: three frequency-tagged colour drives + audio
(40 Hz) drive read from the extended stimulus log columns."""
import io, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
x = io.open("phantom_subject.py", encoding="utf-8").read()

a = '''    tails = [CsvTail(os.path.join(args.session, "calib_log.csv")),
             CsvTail(os.path.join(args.session, "cursor_log.csv"))]'''
a2 = '''    tails = [CsvTail(os.path.join(args.session, n)) for n in
             ("calib_log.csv", "cursor_log.csv", "sweep_log.csv",
              "ccal_log.csv", "color_log.csv", "assr_log.csv")]'''
assert a in x; x = x.replace(a, a2)

b = '''    # Flicker observed from the logs: rising-edge periods give the actual
    # on-screen alternation rate; a stale toggle means no flicker is shown.
    last_fl = 0
    last_rise_t = None
    last_toggle_pc = -1e9
    periods = deque(maxlen=8)
    phase = 0.0
'''
b2 = '''    # Flicker observed from the logs: rising-edge periods give the actual
    # on-screen alternation rate; a stale toggle means no flicker is shown.
    # Three trackers: the primary flicker (column 4), and the colour-scan
    # G / B taggings (columns 6, 7); plus the audio drive (columns 3 + 11).
    class Tracker:
        def __init__(self):
            self.last_fl = 0
            self.last_rise_t = None
            self.last_toggle_pc = -1e9
            self.periods = deque(maxlen=8)
            self.phase = 0.0
            self.env = 0.0
            self.drive = 0.0

        def see(self, t_row, fl):
            if fl != self.last_fl:
                self.last_toggle_pc = time.perf_counter()
                if fl == 1:
                    if self.last_rise_t is not None:
                        period = t_row - self.last_rise_t
                        if 0.01 < period < 0.5:
                            self.periods.append(period)
                    self.last_rise_t = t_row
                self.last_fl = fl

        def live(self):
            return time.perf_counter() - self.last_toggle_pc < 0.15

        def freq(self, fallback):
            return (1.0 / np.median(self.periods) if len(self.periods) >= 2
                    else fallback)

    trk = [Tracker(), Tracker(), Tracker()]
    comp = [0.0, 0.0, 0.0]          # r, g, b of the current cell
    audio_hz, audio_lum, audio_pc = 0.0, 0.0, -1e9
    audio_phase, audio_env = 0.0, 0.0
    last_fl = 0
    last_rise_t = None
    last_toggle_pc = -1e9
    periods = deque(maxlen=8)
    phase = 0.0
'''
assert b in x; x = x.replace(b, b2)

c = '''                    try:
                        t_row = float(row[0])
                        lum = float(row[3])
                        fl = int(row[4])
                    except (IndexError, ValueError):
                        continue
                    if fl != last_fl:
                        last_toggle_pc = time.perf_counter()
                        if fl == 1:
                            if last_rise_t is not None:
                                period = t_row - last_rise_t
                                if 0.01 < period < 0.5:
                                    periods.append(period)
                            last_rise_t = t_row
                        last_fl = fl

            flicker_live = time.perf_counter() - last_toggle_pc < 0.15
            drive = lum if flicker_live else 0.0
            f_meas = (1.0 / np.median(periods) if len(periods) >= 2
                      else config.STIM_FREQ_HZ)

            for _ in range(burst):
                t = n / fs
                env = a_env * env + (1 - a_env) * drive
                phase += 2 * math.pi * f_meas / fs
                ssvep = env * (math.sin(phase) + 0.4 * math.sin(2 * phase))'''
c2 = '''                    try:
                        t_row = float(row[0])
                        lum = float(row[3])
                        fl = int(row[4])
                    except (IndexError, ValueError):
                        continue
                    trk[0].see(t_row, fl)
                    if len(row) >= 12:
                        try:
                            trk[1].see(t_row, int(row[6]))
                            trk[2].see(t_row, int(row[7]))
                            comp = [float(row[8]), float(row[9]), float(row[10])]
                            hz_a = float(row[11])
                        except ValueError:
                            hz_a = 0.0
                        if hz_a > 0:
                            audio_hz, audio_lum = hz_a, lum
                            audio_pc = time.perf_counter()
                    else:
                        comp = [lum, 0.0, 0.0]

            # primary drive = luminance column gated by a live primary toggle
            trk[0].drive = lum if trk[0].live() else 0.0
            trk[1].drive = comp[1] if trk[1].live() else 0.0
            trk[2].drive = comp[2] if trk[2].live() else 0.0
            f_meas = trk[0].freq(config.STIM_FREQ_HZ)
            f_g = trk[1].freq(0.0)
            f_b = trk[2].freq(0.0)
            audio_live = time.perf_counter() - audio_pc < 0.3
            a_drive = audio_lum if audio_live else 0.0

            for _ in range(burst):
                t = n / fs
                env = a_env * env + (1 - a_env) * trk[0].drive
                phase += 2 * math.pi * f_meas / fs
                ssvep = env * (math.sin(phase) + 0.4 * math.sin(2 * phase))
                for k, f_k in ((1, f_g), (2, f_b)):
                    trk[k].env = a_env * trk[k].env + (1 - a_env) * trk[k].drive
                    if f_k > 0:
                        trk[k].phase += 2 * math.pi * f_k / fs
                        ssvep += trk[k].env * (math.sin(trk[k].phase)
                                               + 0.4 * math.sin(2 * trk[k].phase))
                audio_env = a_env * audio_env + (1 - a_env) * a_drive
                if audio_hz > 0:
                    audio_phase += 2 * math.pi * audio_hz / fs
                    ssvep += 0.6 * audio_env * math.sin(audio_phase)'''
assert c in x; x = x.replace(c, c2)
io.open("phantom_subject.py", "w", encoding="utf-8", newline="\n").write(x)
print("patched phantom_subject.py")
