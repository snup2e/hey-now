"""KWS trigger recovery — find a high-recall detection operating point.

Root cause of the regression (measured): build_kws applied SpecAugment to the
synth positives, and time/freq masking on a 1 s window erases the short
"이번역은" token -> positives became label noise -> the detector collapsed to a
~0.4 constant output (val 69%, 0 cross-trip dets). Fix: spec_aug=False (now the
default) -> val 98%, and the per-mark PEAK sliding-window prob is 0.6-0.97 on
10-12/12 held-out announcements. The remaining knob is the detection post-
processing: TRIG threshold + MINRUN (consecutive windows) + cooldown.

This sweeps MINRUN x TRIG over a seeded trip-LOO, training the KWS once per fold
and reusing the sliding-window probabilities, and reports recall + false count.
The hybrid uses high RECALL (catch every announcement) and lets the classifier /
a confidence gate reject the few false triggers, so we favour recall.

Run:  python scripts/path2_kws_recover.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D
import path2_poc as P

LIVE_DIR = P.LIVE_DIR
TRIPS = ["20260527_0654_등교", "20260528_0642_등교",
         "20260527_1431_하교", "20260528_2118_하교"]
COOLDOWN_S = 20.0
MATCH_S = 10.0


def sliding_probs(test, kws, kn):
    full = P._full_logmel(test.y)
    fps = D.SR / D.MEL_HOP
    kfr = int(round(D.KWS_WIN * fps)); fhop = max(1, int(round(0.25 * fps)))
    starts = list(range(0, full.shape[1] - kfr + 1, fhop))
    kp = P.predict_batch(kws, [P._win_cmn(full, f, kfr, True) for f in starts], kn)[:, 1]
    return kp, np.array(starts) / fps


def debounce(kp, times, trig, minrun, cooldown_s):
    trg, run, rs, j = [], 0, 0.0, 0
    while j < len(kp):
        if kp[j] > trig:
            if run == 0:
                rs = times[j]
            run += 1; j += 1
        else:
            if run >= minrun:
                trg.append(rs)
                while j < len(times) and times[j] < rs + cooldown_s:
                    j += 1
            else:
                j += 1
            run = 0
    if run >= minrun:
        trg.append(rs)
    return trg


def score(trg, marks):
    used = [False] * len(trg); hit = 0
    for _, idx in marks:
        t0 = idx / D.SR; best = None
        for di, dt in enumerate(trg):
            if used[di] or abs(dt - t0) > MATCH_S:
                continue
            if best is None or abs(dt - t0) < abs(trg[best] - t0):
                best = di
        if best is not None:
            used[best] = True; hit += 1
    return hit, sum(1 for u in used if not u)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    clean = D.load_clean_sources(P.CLEAN_DIR)
    folds = []
    for held in TRIPS:
        trips = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in TRIPS if t != held]
        test = D.load_live_trip(os.path.join(LIVE_DIR, held))
        rng = np.random.default_rng(0)
        Xk, Yk = D.build_kws(clean, trips, rng, snr=(0.0, 25.0), spec_aug=False)
        kws, kn, kacc = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)
        kp, times = sliding_probs(test, kws, kn)
        folds.append((held, test.marks, kp, times, kacc))
        print(f"  trained fold {held[9:13]} (val {kacc*100:.0f}%), {len(test.marks)} marks")

    print(f"\n{'MINRUN':>6s} {'TRIG':>5s} | {'recall':>14s}  {'false(total)':>12s}")
    for minrun in (1, 2, 3):
        for trig in (0.5, 0.6, 0.7):
            h = f = g = 0
            for _, marks, kp, times, _ in folds:
                trg = debounce(kp, times, trig, minrun, COOLDOWN_S)
                hi, fa = score(trg, marks); h += hi; f += fa; g += len(marks)
            print(f"{minrun:6d} {trig:5.1f} | {h:2d}/{g} ({h/g*100:3.0f}%)      {f:3d}")


if __name__ == "__main__":
    main()
