"""Crux PoC: can the 출발 차임(삐리리리) be detected cross-trip, so we can count
stops → station? 3-class acoustic-event detector (주행 / 정차 / 출발), trip-LOO.

Marked by hand (path2_event_mark.py) into A_train/'audio (N).door_events.json',
perfect open/close alternation. Acoustic facts from listening (memory
project_door_event_acoustics):
  - 출발 "삐리리리" chime is CLOSE-only and the one reliable, characteristic
    marker → the countable. Mark = chime onset; the event spans [close, +~2 s].
  - 열림 is weak (esp. middle-of-car), so 정차's start boundary is fuzzy; we
    guard it. The dwell STATE is still distinct (no running rumble).
  - Seating position is a channel axis: trips 1,2 near-door (loud), 3,4 middle.
  - CHIME VARIES BY TRAIN MODEL: trips 1-3 share one chime; trip 4's chime is a
    different pitch. So holding out trip 4 = an UNSEEN chime variant (genuinely
    harder, not a channel effect). PITCH_AUG pitch-shifts training chimes to
    cover variants (physically motivated; on-board cost 0).

Primary metric = 출발 detection recall + false count on the held-out trip
(matched to its marked closes), reported PER FOLD (trip 4 flagged = variant).

Run:  python scripts/path2_door_poc.py
      DOOR_PITCH_AUG=6 python scripts/path2_door_poc.py     # add pitch-shift aug
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D
import path2_poc as P

import librosa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A_TRAIN = os.path.join(ROOT, "A_train")
TRIPS = [1, 2, 3, 4]                     # A_train audio (N).wav
VARIANT_TRIP = 4                         # different-pitch chime (per listening)

SR = D.SR
# Window length covers the 출발 set. 1.0s = chime-only (weak: chime pitch varies
# by train model + faint close mid-car). 3.0s spans 삐리리리 chime + 치익/덜컥
# close together, so the model leans on whichever cue is clearer per trip.
WIN_S = float(os.environ.get("DOOR_WIN_S", "1.0"))
HOP_S = 0.25
DEP_S = 2.0           # 출발 region after a close mark (삐리리리+치익+close)
GUARD_S = 1.0         # keep 정차 windows off the (weak) open + (loud) close edges
RUN_GUARD_S = 3.0     # keep 주행 windows away from stops
RUN_HOP_S = 2.0
RUN_CAP = 400         # subsample 주행 per trip (else it swamps the other classes)

PITCH_AUG = int(os.environ.get("DOOR_PITCH_AUG", "0"))
PITCH_STEPS = (-3.0, 3.0)
CLASSES = ["주행", "정차", "출발"]
TRIG, MINRUN, COOLDOWN_S, MATCH_S = 0.6, 2, 10.0, 3.0
# state-based stop counting: stopped = P(주행) low; a real dwell is ~20 s, so
# require a min stopped-run and merge short gaps (filters brief mis-class).
MOVING_THR, MIN_STOP_S, MERGE_GAP_S = 0.5, 3.0, 4.0


def load_trip(n):
    apath = os.path.join(A_TRAIN, f"audio ({n}).wav")
    dpath = os.path.join(A_TRAIN, f"audio ({n}).door_events.json")
    y = D.load_wav_float(apath)
    ev = sorted(json.load(open(dpath, encoding="utf-8"))["events"],
                key=lambda e: e["sample_index"])
    opens = [e["sample_index"] for e in ev if e["type"] == "open"]
    closes = [e["sample_index"] for e in ev if e["type"] == "close"]
    # pair each close with the nearest preceding open for dwell windows
    dwells, last_open = [], None
    for e in ev:
        if e["type"] == "open":
            last_open = e["sample_index"]
        elif e["type"] == "close" and last_open is not None and last_open < e["sample_index"]:
            dwells.append((last_open, e["sample_index"])); last_open = None
    # running = between a close and the next open
    runs = []
    for i, e in enumerate(ev[:-1]):
        if e["type"] == "close" and ev[i + 1]["type"] == "open":
            runs.append((e["sample_index"], ev[i + 1]["sample_index"]))
    return y, opens, closes, dwells, runs


def _win(y, a, n):
    w = y[max(0, a):a + n]
    if len(w) < n:
        w = np.pad(w, (0, n - len(w)))
    return w


def build_pool(trip_ids, rng, pitch_aug):
    n = int(WIN_S * SR)
    X, Y = [], []
    for t in trip_ids:
        y, opens, closes, dwells, runs = load_trip(t)
        # 출발 (label 2): windows overlapping [close, close+DEP_S]
        for c in closes:
            for off in (-0.2, 0.0, 0.2, 0.4, 0.6):
                w = _win(y, int(c + off * SR), n)
                X.append(D.to_logmel(w)); Y.append(2)
                for _ in range(pitch_aug):                  # pitch-variant chimes
                    ws = librosa.effects.pitch_shift(
                        y=w, sr=SR, n_steps=float(rng.uniform(*PITCH_STEPS)))
                    X.append(D.to_logmel(ws.astype(np.float32))); Y.append(2)
        # 정차 (label 1): dwell interior, guarded off both (weak) edges
        for o, c in dwells:
            a0, a1 = int(o + GUARD_S * SR), int(c - GUARD_S * SR)
            for a in range(a0, max(a0, a1 - n), int(WIN_S * SR)):
                X.append(D.to_logmel(_win(y, a, n))); Y.append(1)
        # 주행 (label 0): running interior, capped
        runw = []
        for a0, a1 in runs:
            a0, a1 = int(a0 + RUN_GUARD_S * SR), int(a1 - RUN_GUARD_S * SR)
            for a in range(a0, max(a0, a1 - n), int(RUN_HOP_S * SR)):
                runw.append(a)
        if len(runw) > RUN_CAP:
            runw = list(rng.choice(runw, RUN_CAP, replace=False))
        for a in runw:
            X.append(D.to_logmel(_win(y, a, n))); Y.append(0)
    X = np.asarray(X, np.float32)[..., None]
    return X, np.asarray(Y, np.int32)


def window_probs(y, model, norm):
    """Per-window 3-class probabilities + window-start times (s) over the trip."""
    full = P._full_logmel(y)
    fps = SR / D.MEL_HOP
    kfr = int(round(WIN_S * fps)); fhop = max(1, int(round(HOP_S * fps)))
    starts = list(range(0, full.shape[1] - kfr + 1, fhop))
    probs = P.predict_batch(model, [P._win_cmn(full, f, kfr, True) for f in starts], norm)
    return probs, np.array(starts) / fps


def detect_departures(probs, times):
    """Threshold 출발 prob, debounce+cooldown → detection times (s)."""
    dep = probs[:, 2]
    trg, run, rs, j = [], 0, 0.0, 0
    while j < len(dep):
        if dep[j] > TRIG:
            if run == 0:
                rs = times[j]
            run += 1; j += 1
        else:
            if run >= MINRUN:
                trg.append(rs)
                while j < len(times) and times[j] < rs + COOLDOWN_S:
                    j += 1
            else:
                j += 1
            run = 0
    if run >= MINRUN:
        trg.append(rs)
    return trg


def score_dep(trg, closes):
    used = [False] * len(trg); hit = 0
    for c in closes:
        t0 = c / SR; best = None
        for di, dt in enumerate(trg):
            if used[di] or abs(dt - t0) > MATCH_S:
                continue
            if best is None or abs(dt - t0) < abs(trg[best] - t0):
                best = di
        if best is not None:
            used[best] = True; hit += 1
    return hit, sum(1 for u in used if not u)


def detect_stops_state(probs, times):
    """주행/정차 STATE → stop segments. stopped = P(주행) < MOVING_THR; form runs,
    merge gaps < MERGE_GAP_S, keep runs ≥ MIN_STOP_S (a real dwell is ~20 s, so a
    few mis-classified windows don't break it — variant/seating robust)."""
    stopped = probs[:, 0] < MOVING_THR
    runs, i, n = [], 0, len(stopped)
    while i < n:
        if stopped[i]:
            j = i
            while j < n and stopped[j]:
                j += 1
            runs.append([times[i], times[min(j, n - 1)]]); i = j
        else:
            i += 1
    merged = []
    for s in runs:
        if merged and s[0] - merged[-1][1] < MERGE_GAP_S:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
    return [(a, b) for a, b in merged if b - a >= MIN_STOP_S]


def score_segments(segs, closes):
    """A stop is hit if a close mark falls within a segment (±MATCH_S)."""
    used = [False] * len(segs); hit = 0
    for c in closes:
        t = c / SR
        for k, (a, b) in enumerate(segs):
            if not used[k] and a - MATCH_S <= t <= b + MATCH_S:
                used[k] = True; hit += 1; break
    return hit, sum(1 for u in used if not u)


def fuse(trg, segs):
    """Stop = a 정차 segment OR a 출발 detection not already inside one."""
    stops = [tuple(s) for s in segs]
    for t in trg:
        if not any(a - MATCH_S <= t <= b + MATCH_S for a, b in segs):
            stops.append((t, t))
    return sorted(stops)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"3-class door-event detector | win={WIN_S}s pitch_aug={PITCH_AUG} "
          f"(trip {VARIANT_TRIP} = different-pitch chime variant)")
    print(f"counting compared: [출발]=chime event | [정차]=stop-state run | "
          f"[융합]=출발 OR 정차\n")
    tot = {k: [0, 0, 0] for k in ("dep", "state", "fuse")}   # hit, total, false
    for held in TRIPS:
        train_ids = [t for t in TRIPS if t != held]
        rng = np.random.default_rng(0)
        Xtr, Ytr = build_pool(train_ids, rng, PITCH_AUG)
        model, norm, val = P.train(Xtr, Ytr, 3, epochs=40, lr=5e-4, patience=12)
        y, opens, closes, _, _ = load_trip(held)
        nc = len(closes)
        probs, times = window_probs(y, model, norm)
        trg = detect_departures(probs, times)
        segs = detect_stops_state(probs, times)
        fused = fuse(trg, segs)
        res = {}
        for k, (h, f) in (("dep", score_dep(trg, closes)),
                          ("state", score_segments(segs, closes)),
                          ("fuse", score_segments(fused, closes))):
            tot[k][0] += h; tot[k][1] += nc; tot[k][2] += f
            res[k] = (h, f)
        flag = "  ⟵ variant" if held == VARIANT_TRIP else ""
        print(f"  trip {held} (stops={nc:2d}): "
              f"출발 {res['dep'][0]:2d}/{nc} (f{res['dep'][1]:2d})  "
              f"정차 {res['state'][0]:2d}/{nc} (f{res['state'][1]:2d})  "
              f"융합 {res['fuse'][0]:2d}/{nc} (f{res['fuse'][1]:2d})  "
              f"val{val*100:3.0f}%{flag}")
    print("\n" + "=" * 60)
    for k, name in (("dep", "출발 차임 이벤트"), ("state", "정차 상태 런"),
                    ("fuse", "융합(OR)")):
        h, n, f = tot[k]
        print(f"  LOO {name:14s}: {h}/{n} ({h/n*100:3.0f}%)  false {f} ({f/4:.1f}/trip)")
    print("\n  state/fusion이 출발-이벤트보다 recall↑·robust면 → state 카운팅 채택.")


if __name__ == "__main__":
    main()
