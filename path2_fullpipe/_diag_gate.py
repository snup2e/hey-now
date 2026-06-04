# -*- coding: utf-8 -*-
"""Evaluate candidate cross-gate rules (recall / false / precision) on one fold,
using the real KWS-trigger / close / stop-segment timing. Trains KWS+door only."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import path2_dataset as D, path2_poc as P, path2_door_poc as DR, path2_pipeline as PL

PL.CLEAN_DIR = P.CLEAN_DIR = "data/processed/wav"
PL.LIVE_DIR = P.LIVE_DIR = "data/raw/line1_live"
DR.load_trip = lambda tid: PL.load_door(os.path.join(PL.LIVE_DIR, tid))
DR.WIN_S = float(os.environ.get("DOOR_WIN_S", "1.0"))
# stricter stop-state detection (reduce spurious mid-travel "stops")
DR.MOVING_THR = float(os.environ.get("MOVING_THR", "0.5"))
DR.MIN_STOP_S = float(os.environ.get("MIN_STOP_S", "3.0"))
DR.MERGE_GAP_S = float(os.environ.get("MERGE_GAP_S", "4.0"))
DWELL_MAX = 70.0   # reject spurious over-merged "stops" longer than a real dwell
MATCH = 12.0

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
gt_t = sorted(idx / D.SR for _, idx in test.marks)
trigs = sorted(PL.kws_triggers(test.y, (kws, kn)))
closes, segs = PL.door_detections(held, (door, dn))
closes = sorted(closes)
segs = [(a, b) for a, b in segs if (b - a) <= DWELL_MAX]     # drop over-merged
seg_starts = sorted(a for a, b in segs)

def recall_false(events):
    used = [False] * len(gt_t); hit = 0
    for e in sorted(events):
        best = None
        for j, t in enumerate(gt_t):
            if used[j] or abs(e - t) > MATCH: continue
            if best is None or abs(e - t) < abs(e - gt_t[best]): best = j
        if best is not None: used[best] = True; hit += 1
    return hit, len(events) - hit   # recall_hits, false_events

def has_stop_start(t, lo, hi):
    return any(t - lo <= a <= t + hi for a in seg_starts)
def has_close(t, lo, hi):
    return any(t + lo <= c <= t + hi for c in closes)

rules = {
    "kws (none)":            trigs,
    "and_close[+20,+120]":   [t for t in trigs if has_close(t, 20, 120)],
    "and_close[+30,+110]":   [t for t in trigs if has_close(t, 30, 110)],
    "and_stopstart[-12,+45]":[t for t in trigs if has_stop_start(t, 12, 45)],
    "and_stopstart[-8,+30]": [t for t in trigs if has_stop_start(t, 8, 30)],
    "stopstart AND close":   [t for t in trigs if has_stop_start(t, 12, 45) and has_close(t, 20, 120)],
}
print(f"\n=== {held} ({test.direction}) win={DR.WIN_S} MOVING_THR={DR.MOVING_THR} "
      f"MIN_STOP_S={DR.MIN_STOP_S} ===")
print(f"GT={len(gt_t)} trig={len(trigs)} close={len(closes)} segs(≤{DWELL_MAX:.0f}s)={len(segs)}")
print(f"\n  {'rule':26s} {'events':>6s} {'recall':>10s} {'false':>6s}")
for name, ev in rules.items():
    h, f = recall_false(ev)
    print(f"  {name:26s} {len(ev):6d} {h:3d}/{len(gt_t)} ({h/len(gt_t)*100:3.0f}%) {f:6d}")
