"""KWS learning curve vs number of real channels (trips) — is 10 trips worth it?

The KWS positive is a SINGLE phrase ("이번역은") repeated at every station, so
unlike the 13-class CNN it does NOT split data across classes: k real trips give
~12*k channel-diverse positives of the exact thing to detect. This script asks,
honestly and with the seeded harness, whether held-out detection IMPROVES as we
add real channels — i.e. whether the slope over k=1,2,3 trips projects to a
reliable KWS at ~10 trips, BEFORE asking the friend to collect them.

Config: REAL-ONLY positives (use_synth=False) + real-noise aug, balanced
negatives. Real-only is the regime that scales with trips (it failed at 4 trips
only because ~36 windows is data-starved); this measures whether more channels
cure that. For each held-out trip we train on every size-k subset of the other
three and average, so each k is a fair mean over channel choices.

Output: recall (caught announcements) and false triggers/trip vs k. Rising
recall + falling/flat false over 1->2->3 = 10 trips is worth it for KWS.

Run:  python scripts/path2_kws_lcurve.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D
import path2_poc as P
import path2_kws_recover as R          # sliding_probs / debounce / score

TRIPS = ["20260527_0654_등교", "20260528_0642_등교",
         "20260527_1431_하교", "20260528_2118_하교"]
TRIG, MINRUN, COOLDOWN = 0.6, 2, 20.0
REAL_AUG = 8


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    clean = D.load_clean_sources(P.CLEAN_DIR)            # unused when use_synth=False
    cache = {t: D.load_live_trip(os.path.join(P.LIVE_DIR, t)) for t in TRIPS}
    print(f"KWS learning curve | REAL-ONLY positives, real_pos_aug={REAL_AUG}, "
          f"neg 1x | TRIG={TRIG} MINRUN={MINRUN}")
    print("per k = #real training trips; mean over held-out x subsets "
          "(chance: detector fires ~everywhere)\n")

    agg = {k: {"hit": 0, "mark": 0, "false": 0, "runs": 0} for k in (1, 2, 3)}
    for held in TRIPS:
        test = cache[held]
        others = [t for t in TRIPS if t != held]
        line = {}
        for k in (1, 2, 3):
            hh = mm = ff = rr = 0
            for subset in itertools.combinations(others, k):
                trips = [cache[t] for t in subset]
                rng = np.random.default_rng(0)
                Xk, Yk = D.build_kws(clean, trips, rng, use_synth=False,
                                     real_pos_aug=REAL_AUG, neg_ratio=1.0,
                                     snr=(0.0, 25.0), spec_aug=False)
                kws, kn, _ = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)
                kp, times = R.sliding_probs(test, kws, kn)
                trg = R.debounce(kp, times, TRIG, MINRUN, COOLDOWN)
                hit, false = R.score(trg, test.marks)
                hh += hit; mm += len(test.marks); ff += false; rr += 1
            agg[k]["hit"] += hh; agg[k]["mark"] += mm
            agg[k]["false"] += ff; agg[k]["runs"] += rr
            line[k] = (hh, mm, ff / rr)
        print(f"  held-out {held[9:13]}: " + "  ".join(
            f"k={k}: recall {line[k][0]}/{line[k][1]} ({line[k][0]/line[k][1]*100:3.0f}%) "
            f"false~{line[k][2]:.0f}" for k in (1, 2, 3)))

    print("\n" + "=" * 60)
    print(f"{'#real trips (k)':>16s}  {'recall':>10s}  {'false/trip':>11s}")
    for k in (1, 2, 3):
        a = agg[k]
        print(f"{k:>16d}  {a['hit']}/{a['mark']} ({a['hit']/a['mark']*100:3.0f}%)"
              f"  {a['false']/a['runs']:>10.1f}")
    print("\nrising recall over k=1->2->3  => 10 trips likely lifts KWS recall.")
    print("flat after k=2                => more trips wash (rethink KWS).")


if __name__ == "__main__":
    main()
