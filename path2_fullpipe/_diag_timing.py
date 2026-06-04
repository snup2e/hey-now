# -*- coding: utf-8 -*-
"""Diagnostic: real temporal relationship between GT marks, KWS triggers, door
closes, and stop-state segments — to set the cross-gate window correctly.
Trains only KWS + door (few epochs); skips the encoder. ~3-4 min one fold."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import path2_dataset as D, path2_poc as P, path2_door_poc as DR, path2_pipeline as PL

PL.CLEAN_DIR = P.CLEAN_DIR = "data/processed/wav"
PL.LIVE_DIR = P.LIVE_DIR = "data/raw/line1_live"
DR.load_trip = lambda tid: PL.load_door(os.path.join(PL.LIVE_DIR, tid))
DR.WIN_S = float(os.environ.get("DOOR_WIN_S", "1.0"))

trips = PL.list_trips()
held = trips[int(os.environ.get("HELD", "0"))]
train_ids = [t for t in trips if t != held]
clean = D.load_clean_sources(PL.CLEAN_DIR)
trips_live = [D.load_live_trip(os.path.join(PL.LIVE_DIR, t)) for t in train_ids]

Xk, Yk = D.build_kws(clean, trips_live, np.random.default_rng(0), snr=(0., 25.), spec_aug=False)
kws, kn, _ = P.train(Xk, Yk, 2, epochs=18, lr=5e-4, patience=8)
Xd, Yd = DR.build_pool(train_ids, np.random.default_rng(0), 0)
door, dn, _ = P.train(Xd, Yd, 3, epochs=18, lr=5e-4, patience=8)

test = D.load_live_trip(os.path.join(PL.LIVE_DIR, held))
marks = sorted(test.marks, key=lambda x: x[1])
gt_t = [idx / D.SR for _, idx in marks]
kws_trigs = sorted(PL.kws_triggers(test.y, (kws, kn)))
closes_det, stop_segs = PL.door_detections(held, (door, dn))
closes_det = sorted(closes_det)

print(f"\n=== {held} ({test.direction}) ===")
print(f"GT marks {len(gt_t)} | KWS trig {len(kws_trigs)} | close det {len(closes_det)} | stop segs {len(stop_segs)}")
print("\nper GT mark: nearest KWS Δ | nearest close Δ(signed, +after) | in-stop-seg?")
for (st, _), t in zip(marks, gt_t):
    dk = min((abs(t - k) for k in kws_trigs), default=999)
    # nearest close, signed (close - mark): + means close AFTER the announcement
    cs = sorted(closes_det, key=lambda c: abs(c - t))
    dc = (cs[0] - t) if cs else 999
    seg = next(((a, b) for a, b in stop_segs if a - 12 <= t <= b + 6), None)
    segrel = f"start{t-seg[0]:+.0f}s len{seg[1]-seg[0]:.0f}s" if seg else "NONE"
    print(f"  {st:8s} t={t:6.0f}  KWSΔ={dk:4.0f}  closeΔ={dc:+5.0f}  stop:[{segrel}]")

# distribution of close-after-trigger gaps for MATCHED (real) triggers
print("\nclose gap after each KWS trigger that matches a GT mark (±12s):")
gaps = []
for k in kws_trigs:
    if min((abs(k - t) for t in gt_t), default=999) <= 12:   # real trigger
        after = [c - k for c in closes_det if 0 <= c - k <= 120]
        g = min(after) if after else None
        gaps.append(g)
        print(f"  trig t={k:6.0f}  next close +{g:.0f}s" if g is not None else f"  trig t={k:6.0f}  NO close within 120s")
val = [g for g in gaps if g is not None]
if val:
    print(f"\nmatched-trigger→close gap: n={len(val)} min={min(val):.0f} med={np.median(val):.0f} max={max(val):.0f}")
print(f"stop seg lengths: {[round(b-a) for a,b in stop_segs][:30]}")
