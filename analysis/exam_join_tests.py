"""Join one eye's tests from several sessions into one exam.json for a joint
read-out: a sitting that lost its EEG stream part-way and was resumed in a
new session (vr_exam8 lost the phone's stream after its sixth test; vr_exam9
ran the rest with its own flash as the reference).

    python analysis/exam_join_tests.py runs/<out> <eye> runs/<a>:flash,temporal,acuity runs/<b>:field,gaze

Each part names a session and the tests taken from it; a test named twice is
taken from the later part. The joined folder has no EEG or logs, so it can be
read out but not re-scored. The flash of the first part is the eye's
reference; a part scored against its own flash keeps its amplitudes, which
were read against a different reference amplitude (the report says so).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main(out, eye, parts):
    import exam
    import exam_readout
    joined, sources, delivery = None, {}, {}
    for spec in parts:
        session, _, tests = spec.partition(":")
        tests = [t.strip() for t in tests.split(",") if t.strip()]
        with open(os.path.join(session, "exam.json")) as f:
            e = json.load(f)
        if joined is None:
            joined = {k: v for k, v in e.items() if k not in ("eyes", "results", "delivery", "tests")}
            joined.update(eyes=[eye], tests=[], results={eye: {}}, delivery={}, sources={})
        res = e["results"].get(eye, {})
        for t in tests:
            if t not in res:
                raise SystemExit(f"{session} has no {eye} {t} (has {list(res)})")
            joined["results"][eye][t] = dict(res[t], eye=eye)
            sources[t] = session
            d = e.get("delivery", {}).get(f"{eye}/{t}")
            if d is not None:
                delivery[f"{eye}/{t}"] = d
    joined["tests"] = [t for t in exam.TESTS if t in joined["results"][eye]]
    joined["delivery"] = delivery
    joined["sources"] = sources
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exam.json"), "w") as f:
        json.dump(joined, f, indent=1)
    r = exam_readout.run(out)
    print(r["text"])
    print("report:", r["report"])
    print("sources:", ", ".join(f"{t} <- {s}" for t, s in sources.items()))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
