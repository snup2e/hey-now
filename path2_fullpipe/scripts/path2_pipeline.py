"""Path 2 — FULL end-to-end pipeline (KWS ∧ door cross-gate → 분류 → 시퀀스 prior).

This is the §11-D option-3 experiment: instead of using any single cue to count
stops (each has too many false triggers cross-trip — KWS 83%/56fp, 닫힘 81%/15fp,
열림 92%/34fp), FUSE them and read the station name end-to-end.

Pipeline per held-out trip (8-trip leave-one-out):

  1. KWS  "이번역은"      sliding detector  → candidate trigger times
  2. door 3-class (주행/정차/출발)          → close(출발 차임) dets + stop-state segs
  3. CROSS-GATE: a real station fires only where INDEPENDENT cues agree. KWS
     false (chatter/KTX) ⟂ door false (random rumble), so an AND gate multiplies
     the precisions and the union false rate collapses (the §11-C hypothesis).
  4. CLASSIFY the gated announcement window with the GRL metric encoder
     (cosine-to-prototype) → a per-event emission vector over the 13 stations.
  5. SEQUENCE PRIOR: the train moves monotonically along the known 1-D line in a
     known direction, and the boarding station is known → anchored Viterbi turns
     the noisy per-event emissions into the correct station sequence
     (seqprior/anchor: per-mark 33% → 75–100%).

Reported: end-to-end cross-trip station accuracy (LOO), plus a fusion-mode table
(KWS-only / KWS∧close / +stop-state) showing the false-suppression tradeoff.

Everything downstream of the 3 small models is pure post-processing → on-board
(F411) cost 0; only the encoder/KWS/door tflite ship.

Designed to be imported from the Colab notebook (paths set there). Reads every
trip uniformly by trip_id from LIVE_DIR/<trip_id>/{audio.wav,marks.json,
door_events.json}; the door windowing/detection is reused verbatim from
path2_door_poc (one source of truth) via a trip-dir loader shim.
"""
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import path2_dataset as D
import path2_poc as P
import path2_metric_poc as MP
import path2_grl_poc as G
import path2_door_poc as DR

# --------------------------------------------------------------------------- #
# paths (the notebook overrides these before calling run_loo)
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(ROOT, "data", "processed", "wav")
LIVE_DIR = os.path.join(ROOT, "data", "raw", "line1_live")


def list_trips():
    """All synced live trips (8 as of 2026-06), trip_id sorted = chronological."""
    return sorted(d for d in os.listdir(LIVE_DIR)
                  if os.path.isdir(os.path.join(LIVE_DIR, d)) and not d.startswith("_"))


# --------------------------------------------------------------------------- #
# physical line topology (== path2_seqprior_poc / path2_anchor_decode)
# 등교 travels +1 toward 성균관대, 하교 travels -1 toward 구로.
# --------------------------------------------------------------------------- #
ROUTE = ["구로", "가산디지털단지", "독산", "금천구청", "석수", "관악", "안양",
         "명학", "금정", "군포", "당정", "의왕", "성균관대"]
R_OF = {s: i for i, s in enumerate(ROUTE)}
LAB_OF_ROUTE = np.array([D.LABEL_IDX[s] for s in ROUTE])      # route pos -> TARGET13 col
BOARDING_ROUTE = {"등교": 0, "하교": 12}                       # boarding station route pos
TEMP = 0.1
NEG = -1e9


def direction_of(s):
    return "등교" if "등" in str(s) else "하교"


# --------------------------------------------------------------------------- #
# door loader keyed by trip dir, so path2_door_poc's array-based build_pool /
# window_probs / detect_* all work on the live trip_id layout unchanged.
# door_events.json sample indices were marked on the SAME audio (verified by md5),
# so they align to LIVE_DIR/<trip_id>/audio.wav exactly.
# --------------------------------------------------------------------------- #
def load_door(trip_dir):
    """Return (y, opens, closes, dwells, runs) — identical shape to DR.load_trip."""
    y = D.load_wav_float(os.path.join(trip_dir, "audio.wav"))
    ev = sorted(json.load(open(os.path.join(trip_dir, "door_events.json"),
                               encoding="utf-8"))["events"],
                key=lambda e: e["sample_index"])
    opens = [e["sample_index"] for e in ev if e["type"] == "open"]
    closes = [e["sample_index"] for e in ev if e["type"] == "close"]
    dwells, last_open = [], None
    for e in ev:
        if e["type"] == "open":
            last_open = e["sample_index"]
        elif e["type"] == "close" and last_open is not None and last_open < e["sample_index"]:
            dwells.append((last_open, e["sample_index"])); last_open = None
    runs = []
    for i, e in enumerate(ev[:-1]):
        if e["type"] == "close" and ev[i + 1]["type"] == "open":
            runs.append((e["sample_index"], ev[i + 1]["sample_index"]))
    return y, opens, closes, dwells, runs


# point path2_door_poc's loader at the live trip dirs (id-keyed)
DR.load_trip = lambda tid: load_door(os.path.join(LIVE_DIR, tid))


# --------------------------------------------------------------------------- #
# detection operating points (high-recall: the gate, not the threshold, kills fp)
# --------------------------------------------------------------------------- #
KWS_TRIG, KWS_MINRUN, KWS_COOLDOWN = 0.6, 2, 20.0
MATCH_S = 12.0          # detection<->GT-mark match tolerance (== path2_poc.score)

# cross-gate timing (MEASURED on the live trips, scripts _diag_timing/_diag_gate):
#   "이번역은 OO역" is announced ~40-90 s BEFORE the train actually stops (played
#   while still travelling from the previous station), then dwell ~20 s, then the
#   출발 차임(close). So trigger→close gap is large: median ~66 s (47-99 s) — NOT
#   "a few seconds". The right pairing windows:
CLOSE_MIN_S, CLOSE_MAX_S = 20.0, 115.0   # a real trigger's departure close lands here
# A real announcement is followed by the train STOPPING: a stop-state segment
# whose START falls just after (or at) the trigger. False triggers (chatter/KTX
# while moving) are NOT followed by a stop. Window the stop-segment START:
STOP_LEAD_S, STOP_LAG_S = 12.0, 45.0     # stop start within [t-LEAD, t+LAG]
DWELL_MAX_S = 70.0                        # drop over-merged spurious "stops" (real dwell ≤ ~60 s)


# --------------------------------------------------------------------------- #
# training — 3 models per fold (held-in trips only)
# --------------------------------------------------------------------------- #
def _metric_pool(clean, trips, rng, augment):
    """real-only + heavy in-domain aug (train) / clean un-aug (prototype reg)."""
    return D.build_metric_pool(
        clean, trips, rng, use_clean=False, snr=(-5.0, 25.0),
        real_noise_aug=(16 if augment else 0),
        real_jitter=(12 if augment else 3),
        jitter_s=(0.3 if augment else 0.1), spec_aug=augment)


def train_fold(train_ids, grl_lambda=0.3, verbose=True):
    """Train KWS + door + GRL-encoder on the held-in trips; register prototypes."""
    clean = D.load_clean_sources(CLEAN_DIR)
    trips = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in train_ids]

    # 1) KWS "이번역은" (spec_aug=False + lr 5e-4 — §7 collapse fix)
    Xk, Yk = D.build_kws(clean, trips, np.random.default_rng(0),
                         snr=(0.0, 25.0), spec_aug=False)
    kws, kn, kacc = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)

    # 2) door 3-class (주행/정차/출발). DR.WIN_S / DR.PITCH_AUG set by the notebook.
    Xd, Yd = DR.build_pool(train_ids, np.random.default_rng(0), DR.PITCH_AUG)
    door, dn, dacc = P.train(Xd, Yd, 3, epochs=40, lr=5e-4, patience=12)

    # 3) GRL metric encoder (real-only + aug); prototypes from a clean un-aug pass
    X, Y, src = _metric_pool(clean, trips, np.random.default_rng(0), True)
    if grl_lambda > 0:
        dom, n_dom, mask = G.build_domain_labels(src, with_synth=False)
        enc, (m, s) = G.train_grl(X, Y, src, dom, n_dom, grl_lambda, mask)
    else:
        enc, (m, s) = MP.train_encoder(X, Y, src)
    Xp, Yp, srcp = _metric_pool(clean, trips, np.random.default_rng(1), False)
    protos = MP.register_protos(enc, Xp, Yp, srcp, m, s, include_real=True)

    if verbose:
        print(f"    trained: KWS val {kacc*100:.0f}%  door val {dacc*100:.0f}%  "
              f"| KWS pool {len(Yk)}  door pool {len(Yd)}  enc pool {len(Y)}")
    return {"kws": (kws, kn), "door": (door, dn), "enc": (enc, protos, m, s)}


# --------------------------------------------------------------------------- #
# inference on the held-out trip
# --------------------------------------------------------------------------- #
def kws_triggers(y, kws_pack):
    """Sliding "이번역은" detector → debounced trigger times (s)."""
    kws, kn = kws_pack
    full = P._full_logmel(y)
    fps = D.SR / D.MEL_HOP
    kfr = int(round(D.KWS_WIN * fps)); fhop = max(1, int(round(0.25 * fps)))
    starts = list(range(0, full.shape[1] - kfr + 1, fhop))
    kp = P.predict_batch(kws, [P._win_cmn(full, f, kfr, True) for f in starts], kn)[:, 1]
    times = np.array(starts) / fps
    return _debounce(kp, times)


def _debounce(kp, times, trig=KWS_TRIG, minrun=KWS_MINRUN, cooldown=KWS_COOLDOWN):
    trg, run, rs, j = [], 0, 0.0, 0
    while j < len(kp):
        if kp[j] > trig:
            if run == 0:
                rs = times[j]
            run += 1; j += 1
        else:
            if run >= minrun:
                trg.append(rs)
                while j < len(times) and times[j] < rs + cooldown:
                    j += 1
            else:
                j += 1
            run = 0
    if run >= minrun:
        trg.append(rs)
    return trg


def door_detections(trip_id, door_pack):
    """close(출발 차임) detection times + stop-state segments on the held-out trip."""
    door, dn = door_pack
    y, opens, closes, _, _ = load_door(os.path.join(LIVE_DIR, trip_id))
    probs, times = DR.window_probs(y, door, dn)
    closes_det = DR.detect_departures(probs, times)        # 출발 차임 events
    stop_segs = DR.detect_stops_state(probs, times)        # 정차 dwell segments
    return closes_det, stop_segs


# --------------------------------------------------------------------------- #
# the cross-gate — where independent cues agree = a station event
# returns time-ordered event anchor times (s) used to cut the classify window
# --------------------------------------------------------------------------- #
def _has_close(t, closes_det):
    """A departure close in this trigger's expected post-announcement window."""
    return any(t + CLOSE_MIN_S <= c <= t + CLOSE_MAX_S for c in closes_det)


def _has_stop_start(t, stop_segs):
    """The train STOPS shortly after the announcement: a (non-spurious) stop-state
    segment whose START falls in [t-LEAD, t+LAG]."""
    return any((b - a) <= DWELL_MAX_S and t - STOP_LEAD_S <= a <= t + STOP_LAG_S
               for a, b in stop_segs)


def gate(kws_trigs, closes_det, stop_segs, mode):
    """A station event = a KWS trigger that an INDEPENDENT door cue corroborates.
    KWS false (chatter/KTX) ⟂ door cues, so the AND collapses false (§11-C).
    Windows are MEASURED (see _diag_gate): announcement leads the stop/close.
        'kws'        — every KWS trigger (no fusion; the cascade baseline)
        'and_close'  — trigger whose departure 차임 lands in [+20,+115] s
        'and_state'  — trigger followed by the train STOPPING (stop-state start)
        'and_both'   — both cues agree (strictest; lowest false)
    """
    kws_trigs = sorted(kws_trigs)
    if mode == "kws":
        return list(kws_trigs)
    if mode == "and_close":
        return [t for t in kws_trigs if _has_close(t, closes_det)]
    if mode == "and_state":
        return [t for t in kws_trigs if _has_stop_start(t, stop_segs)]
    if mode == "and_both":
        return [t for t in kws_trigs
                if _has_stop_start(t, stop_segs) and _has_close(t, closes_det)]
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# classify gated windows → emission matrix; sequence-prior decode
# --------------------------------------------------------------------------- #
def emissions_at(times, y, enc_pack):
    """Per-event cosine-to-prototype emission rows [N,13] (TARGET13 order)."""
    enc, protos, m, s = enc_pack
    E = []
    for t in times:
        idx = int(round(t * D.SR))
        wins = [D.window_feature(y, idx, d) for d in (-0.1, 0.0, 0.1)]
        e = MP.embed(enc, np.asarray(wins)[..., None], m, s)
        q = e.mean(0); q /= (np.linalg.norm(q) + 1e-9)
        E.append(protos @ q)
    return np.asarray(E, np.float32) if E else np.zeros((0, len(D.TARGET13)), np.float32)


def _log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True) + 1e-12)


def per_mark_route(E):
    """Argmax station per event → route position (no prior)."""
    inv = {int(l): r for r, l in enumerate(LAB_OF_ROUTE)}
    return np.array([inv[int(l)] for l in E.argmax(1)]) if len(E) else np.array([], int)


def anchored_viterbi(E, direction, alpha=6.0, beta=8.0, gamma=3.0, max_skip=3):
    """Skip- AND duplicate-tolerant monotone decode, pinned near the boarding
    anchor (== anchor decode + a 'stay' transition):
      k=+1   advance one station        free      (normal consecutive marks)
      k in [2,max_skip]  forward skip    -alpha·(k-1)  (a missed gate)
      k=0    stay at same station        -gamma    (a gate FALSE POSITIVE — two
                                                     events on one station — so a
                                                     spurious event no longer
                                                     forces the route to advance)
      k<0    backward                     forbidden
    k = (cur-prev)·d, d = travel direction. Robustness to both gate misses and
    gate false-positives is what lets the noisy cross-gate feed a clean decode."""
    if len(E) == 0:
        return np.array([], int)
    Er = E[:, LAB_OF_ROUTE]
    N = len(Er)
    d = +1 if direction == "등교" else -1
    logE = _log_softmax(Er / TEMP)
    a0 = BOARDING_ROUTE[direction] + d
    j = np.arange(13)
    k = (j[None, :] - j[:, None]) * d                       # steps forward [prev,cur]
    logT = np.full((13, 13), NEG)
    logT[k == 0] = -gamma                                   # stay (absorb a false event)
    adv = (k >= 1) & (k <= max_skip)
    logT[adv] = -alpha * (k[adv] - 1)                       # +1 free, skips penalised
    dp = logE[0] - beta * np.abs(j - a0)
    back = []
    for t in range(1, N):
        sc = dp[:, None] + logT + logE[t][None, :]
        back.append(sc.argmax(0)); dp = sc.max(0)
    path = [int(dp.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return np.array(path[::-1])


# --------------------------------------------------------------------------- #
# end-to-end scoring
# --------------------------------------------------------------------------- #
def match_events_to_marks(times, marks):
    """Greedy nearest match each event-time to a GT mark within ±MATCH_S.
    Returns (matched pairs [(event_i, mark_j)], n_false_events)."""
    mt = [(R_OF[st], idx / D.SR) for st, idx in marks]      # (route_pos, time)
    used = [False] * len(mt)
    pairs = []
    for ei, t in enumerate(times):
        best = None
        for mj, (rp, t0) in enumerate(mt):
            if used[mj] or abs(t - t0) > MATCH_S:
                continue
            if best is None or abs(t - t0) < abs(t - mt[best][1]):
                best = mj
        if best is not None:
            used[best] = True; pairs.append((ei, best))
    n_false = len(times) - len(pairs)
    return pairs, n_false


def run_fold(held, train_ids, packs=None, grl_lambda=0.3, modes=("kws", "and_close", "and_state"),
             verbose=True):
    """Train (or reuse packs) → detect → gate → classify → decode → score one fold."""
    if packs is None:
        packs = train_fold(train_ids, grl_lambda, verbose)
    test = D.load_live_trip(os.path.join(LIVE_DIR, held))
    direction = test.direction
    marks = sorted(test.marks, key=lambda x: x[1])
    n_gt = len(marks)
    y = test.y

    kws_trigs = kws_triggers(y, packs["kws"])
    closes_det, stop_segs = door_detections(held, packs["door"])

    out = {"held": held, "direction": direction, "n_gt": n_gt, "modes": {}}
    for mode in modes:
        ev_times = gate(kws_trigs, closes_det, stop_segs, mode)
        pairs, n_false = match_events_to_marks(ev_times, marks)
        # emissions over ALL gated events (false ones included — the gate must
        # keep the sequence clean enough for the monotone decode)
        E = emissions_at(ev_times, y, packs["enc"])
        dec_pm = per_mark_route(E)
        dec_vit = anchored_viterbi(E, direction)
        # station accuracy: a GT mark is correct iff matched to an event whose
        # decoded route position equals the GT route position.
        gt_rp = {mj: R_OF[marks[mj][0]] for mj in range(n_gt)}
        corr_pm = sum(1 for ei, mj in pairs if len(dec_pm) and dec_pm[ei] == gt_rp[mj])
        corr_vit = sum(1 for ei, mj in pairs if len(dec_vit) and dec_vit[ei] == gt_rp[mj])
        out["modes"][mode] = {
            "n_event": len(ev_times), "matched": len(pairs), "false": n_false,
            "acc_permark": corr_pm, "acc_viterbi": corr_vit,
        }
    if verbose:
        ks = len(kws_trigs)
        print(f"  {held[9:13]} ({direction}) GT={n_gt:2d} | KWS trig={ks} "
              f"close={len(closes_det)} stop={len(stop_segs)}")
        for mode in modes:
            r = out["modes"][mode]
            print(f"      {mode:10s}: events={r['n_event']:2d} matched={r['matched']:2d}/{n_gt} "
                  f"false={r['false']:2d} | acc per-mark {r['acc_permark']:2d}/{n_gt} "
                  f"→ Viterbi {r['acc_viterbi']:2d}/{n_gt}")
    return out, packs


def run_loo(grl_lambda=0.3, modes=("kws", "and_close", "and_state"), trips=None):
    """8-trip leave-one-out over the full pipeline. Returns aggregate per mode."""
    trips = trips or list_trips()
    print(f"FULL pipeline LOO | {len(trips)} trips | GRL λ={grl_lambda} | "
          f"door win={DR.WIN_S}s | episodes={MP.EPISODES}")
    print(f"modes: {modes}\n")
    agg = {m: {"event": 0, "matched": 0, "false": 0, "pm": 0, "vit": 0, "gt": 0}
           for m in modes}
    for held in trips:
        train_ids = [t for t in trips if t != held]
        t0 = time.time()
        out, _ = run_fold(held, train_ids, grl_lambda=grl_lambda, modes=modes)
        for m in modes:
            r = out["modes"][m]
            a = agg[m]
            a["event"] += r["n_event"]; a["matched"] += r["matched"]
            a["false"] += r["false"]; a["pm"] += r["acc_permark"]
            a["vit"] += r["acc_viterbi"]; a["gt"] += out["n_gt"]
        print(f"      [fold {time.time()-t0:.0f}s]\n")

    print("=" * 72)
    print(f"  {'mode':10s} | {'gate recall':>12s} | {'false/trip':>10s} | "
          f"{'per-mark acc':>13s} | {'Viterbi acc':>12s}")
    for m in modes:
        a = agg[m]; n = a["gt"]; T = len(trips)
        print(f"  {m:10s} | {a['matched']:3d}/{n} ({a['matched']/n*100:3.0f}%) | "
              f"{a['false']/T:10.1f} | {a['pm']:3d}/{n} ({a['pm']/n*100:3.0f}%) | "
              f"{a['vit']:3d}/{n} ({a['vit']/n*100:3.0f}%)")
    print("\n  gate recall = gated events matched to a true station (caps accuracy).")
    print("  AND-gate should cut false/trip hard vs 'kws' while keeping recall;")
    print("  Viterbi (boarding anchor + monotone route) lifts accuracy over per-mark.")
    return agg


# --------------------------------------------------------------------------- #
# open / close detection experiment — PER-EVENT window (사용자 음향 관찰):
#   open  = "탁" 한 번      → 짧은 1.0 s 윈도우면 충분
#   close = "삐리리 차임+탁" → 3.0 s 윈도우가 있어야 둘 다 들어옴
# 한 CNN은 입력 길이 고정이라 둘을 한 모델로 못 함 → 이벤트별 독립 이진 검출기.
# 이게 이벤트-순서 상태기계(방송→open→close)와 직접 맞물림. Colab GPU에서 실행.
# --------------------------------------------------------------------------- #
# 음향 관찰 기반 자연 윈도우(초). chime=삐리리(톤,크지만 변종), clunk=탁(충격,불변추정).
# 'close'(=차임+탁 3초 blob)는 분리 전 기준선; 'chime'·'clunk'가 분리 실험.
# chime win = 3.0s: the 출발 chime sustains ~3-4 s, so the detector window should
# cover the whole melody (more distinctive → fewer false) — HPSS strips the
# broadband 치이익 / transient 탁 that the longer window also catches.
EVENT_WIN = {"open": 1.0, "close": 3.0, "chime": 3.0, "clunk": 1.0}
EVENT_REGION_S = {"open": 1.0, "close": 2.0, "chime": 1.2, "clunk": 0.5}
# 검출기 event → door_events.json 의 어떤 type 마크에서 학습하나.
# 'chime'은 'close' 마크(=차임 onset)를 짧은 윈도우로, 'clunk'은 새 '탁' 마크.
EVENT_MARK_TYPE = {"open": "open", "close": "close", "chime": "close", "clunk": "clunk"}


def load_event_marks(trip_dir, raw_type):
    """door_events.json 에서 raw_type('open'|'close'|'clunk') 샘플 인덱스 정렬 리스트."""
    p = os.path.join(trip_dir, "door_events.json")
    if not os.path.exists(p):
        return []
    ev = json.load(open(p, encoding="utf-8")).get("events", [])
    return sorted(e["sample_index"] for e in ev if e.get("type") == raw_type)


def build_event_pool(trip_ids, event, rng):
    """이진 풀: 1 = 해당 이벤트 구간, 0 = 그 외(주행·정차·다른 이벤트). 윈도우 =
    EVENT_WIN[event]. event∈{open,close,chime,clunk}; 마크 출처 = EVENT_MARK_TYPE."""
    win = EVENT_WIN[event]; n = int(win * D.SR); reg = EVENT_REGION_S[event]
    src_type = EVENT_MARK_TYPE[event]
    X, Y = [], []
    for tid in trip_ids:
        tdir = os.path.join(LIVE_DIR, tid)
        y, opens, closes, dwells, runs = load_door(tdir)
        tgt = load_event_marks(tdir, src_type)
        other = [m for t in ("open", "close", "clunk") if t != src_type
                 for m in load_event_marks(tdir, t)]
        # positives: windows covering [mark, mark+reg]
        offs = [k * reg for k in (-0.2, 0.0, 0.2, 0.4, 0.6)]
        for e in tgt:
            for off in offs:
                X.append(D.to_logmel(DR._win(y, int(e + off * D.SR), n))); Y.append(1)
        # negatives: the OTHER event (so it doesn't confuse) + dwell + running
        other_neg = [int(e + off * D.SR) for e in other for off in offs]
        dwell_neg = []
        for o, c in dwells:
            a0, a1 = int(o + DR.GUARD_S * D.SR), int(c - DR.GUARD_S * D.SR)
            dwell_neg += list(range(a0, max(a0, a1 - n), n))
        run_neg = []
        for a0, a1 in runs:
            a0, a1 = int(a0 + DR.RUN_GUARD_S * D.SR), int(a1 - DR.RUN_GUARD_S * D.SR)
            run_neg += list(range(a0, max(a0, a1 - n), int(DR.RUN_HOP_S * D.SR)))
        # keep the other-event negatives (cheap, important), fill the rest from
        # dwell+running, capped so neg ≈ 3× this trip's positives (balanced train).
        n_pos = len(tgt) * len(offs)
        fill = dwell_neg + run_neg
        budget = max(0, 3 * n_pos - len(other_neg))
        if len(fill) > budget:
            fill = list(rng.choice(fill, budget, replace=False)) if budget else []
        for a in other_neg + fill:
            X.append(D.to_logmel(DR._win(y, a, n))); Y.append(0)
    return np.asarray(X, np.float32)[..., None], np.asarray(Y, np.int32)


def _detect_col(col, times, trig=0.6, minrun=2, cooldown=10.0):
    """Threshold an event-prob column + debounce/cooldown → event times (s)."""
    trg, run, rs, j = [], 0, 0.0, 0
    while j < len(col):
        if col[j] > trig:
            if run == 0: rs = times[j]
            run += 1; j += 1
        else:
            if run >= minrun:
                trg.append(rs)
                while j < len(times) and times[j] < rs + cooldown:
                    j += 1
            else:
                j += 1
            run = 0
    if run >= minrun: trg.append(rs)
    return trg


def _score_events(dets, gt_samples, match_s=5.0):
    used = [False] * len(dets); hit = 0
    for g in gt_samples:
        t0 = g / D.SR; best = None
        for di, dt in enumerate(dets):
            if used[di] or abs(dt - t0) > match_s: continue
            if best is None or abs(dt - t0) < abs(dets[best] - t0): best = di
        if best is not None: used[best] = True; hit += 1
    return hit, sum(1 for u in used if not u)


def run_event_detect_loo(event, trips=None):
    """8-fold LOO for ONE event detector at its natural window. recall/false per trip."""
    trips = trips or list_trips()
    win = EVENT_WIN[event]
    DR.WIN_S = win                       # window_probs slides at this length
    print(f"[{event}] detector | win={win}s | {len(trips)} trips, trip-LOO")
    tot = [0, 0, 0]                       # hit, total, false
    for held in trips:
        train_ids = [t for t in trips if t != held]
        Xtr, Ytr = build_event_pool(train_ids, event, np.random.default_rng(0))
        model, norm, val = P.train(Xtr, Ytr, 2, epochs=40, lr=5e-4, patience=12)
        held_dir = os.path.join(LIVE_DIR, held)
        y, _, _, _, _ = load_door(held_dir)
        gt = load_event_marks(held_dir, EVENT_MARK_TYPE[event])
        probs, times = DR.window_probs(y, model, norm)
        dets = _detect_col(probs[:, 1], times)
        h, f = _score_events(dets, gt)
        tot[0] += h; tot[1] += len(gt); tot[2] += f
        print(f"  {held[9:13]:9s}: {h:2d}/{len(gt):2d} (f{f:2d})  val{val*100:3.0f}%")
    h, n, f = tot
    print(f"  → LOO {event}: {h}/{n} ({h/n*100:3.0f}%)  false {f} ({f/len(trips):.1f}/trip)\n")
    return tot


def run_openclose_loo(trips=None):
    """open@1s and close@3s detectors, each at its natural window. For deciding
    whether the event-ordered gate requires open (strong) or close-only."""
    trips = trips or list_trips()
    res = {}
    for event in ("open", "close"):
        res[event] = run_event_detect_loo(event, trips)
    print("=" * 52)
    for event in ("open", "close"):
        h, n, f = res[event]
        print(f"  {event:5s} (win={EVENT_WIN[event]}s): {h}/{n} ({h/n*100:3.0f}%)  "
              f"false {f/len(trips):.1f}/trip")
    print("\n  open이 close만큼 잡히면 게이트 'open+close', 약하면 'close 필수+open 선택'.")
    return res


# --------------------------------------------------------------------------- #
# chime + clunk 분리 실험 (사용자 아이디어, '탁' 마킹 필요):
#   chime(삐리리, win=1.5s) 와 clunk(탁, win=1.0s)를 각각 검출 → 둘이 ORDINAL로
#   "연속"(어떤 chime 다음, 다음 chime 전에 clunk가 옴)일 때만 '닫힘(역바뀜)'으로 인정.
#   시간 길이/간격 가정 없음 — 순서만. clunk가 차임보다 채널-강건한지 + 연속 규칙이
#   false를 줄이는지 측정. door_events.json 에 type:'clunk' 마크가 있어야 동작.
# --------------------------------------------------------------------------- #
def chime_clunk_sequence(chime_dets, clunk_dets):
    """ORDINAL pairing (no timing): a confirmed 닫힘 = a chime detection that has a
    clunk detection AFTER it and BEFORE the next chime detection."""
    chime, clunk = sorted(chime_dets), sorted(clunk_dets)
    out = []
    for k, t in enumerate(chime):
        nxt = chime[k + 1] if k + 1 < len(chime) else float("inf")
        if any(t < c < nxt for c in clunk):
            out.append(t)
    return out


def run_chime_clunk_loo(trips=None):
    """chime@1.5s, clunk@1.0s 각각 검출 + 연속(chime→clunk) 규칙. 각각의 cross-trip
    recall/false 와 연속 규칙 결과를 close(닫힘) GT 기준으로 비교. ('탁' 마크 필요)"""
    trips = trips or list_trips()
    # 데이터 점검: clunk 마크가 있나?
    n_clunk = sum(len(load_event_marks(os.path.join(LIVE_DIR, t), "clunk")) for t in trips)
    if n_clunk == 0:
        print("⚠ '탁'(clunk) 마크가 없음 — path2_event_mark.py 로 't' 키로 마킹 후 다시.")
        return None
    print(f"chime+clunk 분리 | chime@{EVENT_WIN['chime']}s clunk@{EVENT_WIN['clunk']}s | "
          f"{len(trips)} trips, trip-LOO  (clunk 마크 {n_clunk}개)\n")
    tot = {"chime": [0, 0, 0], "clunk": [0, 0, 0], "seq": [0, 0, 0]}
    for held in trips:
        train_ids = [t for t in trips if t != held]
        held_dir = os.path.join(LIVE_DIR, held)
        gt_close = load_event_marks(held_dir, "close")     # 닫힘 = 차임 onset GT
        y, _, _, _, _ = load_door(held_dir)
        dets, per = {}, {}
        for event in ("chime", "clunk"):
            DR.WIN_S = EVENT_WIN[event]
            Xtr, Ytr = build_event_pool(train_ids, event, np.random.default_rng(0))
            model, norm, _ = P.train(Xtr, Ytr, 2, epochs=40, lr=5e-4, patience=12)
            probs, times = DR.window_probs(y, model, norm)
            dets[event] = _detect_col(probs[:, 1], times)
            gt = load_event_marks(held_dir, EVENT_MARK_TYPE[event])
            h, f = _score_events(dets[event], gt)
            per[event] = (h, len(gt), f)
            tot[event][0] += h; tot[event][1] += len(gt); tot[event][2] += f
        seq = chime_clunk_sequence(dets["chime"], dets["clunk"])
        hs, fs = _score_events(seq, gt_close)
        tot["seq"][0] += hs; tot["seq"][1] += len(gt_close); tot["seq"][2] += fs
        c, k = per["chime"], per["clunk"]
        print(f"  {held[9:13]:9s}: chime {c[0]:2d}/{c[1]:2d}(f{c[2]:2d})  "
              f"clunk {k[0]:2d}/{k[1]:2d}(f{k[2]:2d})  seq {hs:2d}/{len(gt_close):2d}(f{fs:2d})")
    print("\n" + "=" * 56)
    for k, name in (("chime", "차임 단독"), ("clunk", "탁 단독"), ("seq", "연속(chime→clunk)")):
        h, n, f = tot[k]
        r = f"{h}/{n} ({h/n*100:3.0f}%)" if n else "  -"
        print(f"  {name:18s}: {r}  false {f} ({f/len(trips):.1f}/trip)")
    print("\n  탁이 차임보다 false↓·recall 유지면 채널-강건 입증. 연속이 단독보다 false↓면 규칙 유효.")
    return tot


# --------------------------------------------------------------------------- #
# 차임 검출기 — HPSS-harmonic 톤 분리 front-end (사용자 아이디어)
# 닫힘 차임(삐리리, 톤)을 카운터로. 좌석따라 섞이는 치이익(광대역)·탁(충격)을 HPSS로
# 제거해 채널-강건한 차임 특징을 학습. 학습·추론 모두 D.to_logmel(=FEATURE_MODE) 한
# 함수로 처리 → train/infer 바이트 일치(0 트리거 mismatch 방지).
# --------------------------------------------------------------------------- #
def _event_slide_probs(y, model, norm, win_s, hop_s=0.25):
    """Per-window slide using the SAME D.to_logmel as training (so whatever
    FEATURE_MODE — incl. HPSS-harmonic — is applied identically at inference)."""
    n = int(win_s * D.SR); hop = int(hop_s * D.SR)
    starts = list(range(0, max(1, len(y) - n + 1), hop))
    feats = [D.to_logmel(y[s:s + n]) for s in starts]
    probs = P.predict_batch(model, feats, norm)
    return probs, np.array(starts) / D.SR


def run_chime_loo(feature_mode="logmel_cmn_harmonic", trips=None, verbose=True):
    """차임 검출기 cross-trip LOO under a given FEATURE_MODE. GT = 'close'(차임) 마크.
    feature_mode='logmel_cmn'(베이스라인) vs 'logmel_cmn_harmonic'(HPSS 톤분리)."""
    trips = trips or list_trips()
    win = EVENT_WIN["chime"]
    prev = D.FEATURE_MODE
    D.FEATURE_MODE = feature_mode
    try:
        tot = [0, 0, 0]
        for held in trips:
            train_ids = [t for t in trips if t != held]
            Xtr, Ytr = build_event_pool(train_ids, "chime", np.random.default_rng(0))
            model, norm, val = P.train(Xtr, Ytr, 2, epochs=40, lr=5e-4, patience=12)
            held_dir = os.path.join(LIVE_DIR, held)
            y, _, _, _, _ = load_door(held_dir)
            gt = load_event_marks(held_dir, "close")
            probs, times = _event_slide_probs(y, model, norm, win)
            dets = _detect_col(probs[:, 1], times)
            h, f = _score_events(dets, gt)
            tot[0] += h; tot[1] += len(gt); tot[2] += f
            if verbose:
                print(f"    {held[9:13]:9s}: {h:2d}/{len(gt):2d} (f{f:2d})  val{val*100:3.0f}%")
    finally:
        D.FEATURE_MODE = prev
    h, n, f = tot
    print(f"  → [{feature_mode}] chime LOO: {h}/{n} ({h/n*100:3.0f}%)  "
          f"false {f} ({f/len(trips):.1f}/trip)\n")
    return tot


def run_chime_compare(trips=None):
    """베이스라인(logmel_cmn) vs HPSS 톤분리(logmel_cmn_harmonic) 차임 검출 비교.
    HPSS가 cross-trip recall↑(특히 하교 붕괴 회복)·false↓면 톤 분리가 채널변이를 이긴 것."""
    trips = trips or list_trips()
    print(f"차임 검출 HPSS 비교 | win={EVENT_WIN['chime']}s | {len(trips)} trips, trip-LOO\n")
    res = {}
    for mode in ("logmel_cmn", "logmel_cmn_harmonic"):
        print(f"  [{mode}]")
        res[mode] = run_chime_loo(mode, trips)
    print("=" * 52)
    for mode in ("logmel_cmn", "logmel_cmn_harmonic"):
        h, n, f = res[mode]
        print(f"  {mode:22s}: {h}/{n} ({h/n*100:3.0f}%)  false {f/len(trips):.1f}/trip")
    return res


# --------------------------------------------------------------------------- #
# 이벤트-순서 상태기계 end-to-end (차임 카운터 + KWS 이번역 + 인코더 + 노선순서 + 앵커)
#   상태:  "이번역은 K"(KWS) ─ … ─ 차임(출발) ─→ 주행(K→다음) ─ "이번역은 L" ─ …
#   - 진짜 안내 = 다음 안내 전에 차임(출발)이 따라오는 KWS 트리거 (순서만, 시간 미사용)
#     → KWS 오발화(잡담, 차임 없음)를 걸러냄.
#   - 확정된 안내들의 인코더 emission → anchored Viterbi(노선 단조 + 탑승역 앵커)로 역 이름.
#   FEATURE_MODE 전환: KWS·인코더 = logmel_cmn, 차임 = logmel_cmn_harmonic(HPSS).
#   trip4(차임 +4st 변종)는 차임 학습/LOO에서 제외(KWS·인코더엔 유지).
# --------------------------------------------------------------------------- #
VARIANT_TRIP_SUFFIX = "2118_하교"          # trip4 = chime pitch variant


def confirm_announcements(ann_times, chime_times):
    """ORDINAL: keep a KWS announcement iff a chime(출발) occurs after it and before
    the next announcement (= the train actually departed that station). No timing."""
    ann = sorted(ann_times); ch = sorted(chime_times)
    keep = []
    for i, t in enumerate(ann):
        nxt = ann[i + 1] if i + 1 < len(ann) else float("inf")
        if any(t < c < nxt for c in ch):
            keep.append(i)
    return keep, ann


def train_sm_fold(train_ids, chime_train_ids, grl_lambda=0.3, verbose=True):
    """KWS + 인코더(GRL, logmel_cmn) + 차임 검출기(HPSS) 학습."""
    clean = D.load_clean_sources(CLEAN_DIR)
    trips = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in train_ids]
    prev = D.FEATURE_MODE
    D.FEATURE_MODE = "logmel_cmn"
    Xk, Yk = D.build_kws(clean, trips, np.random.default_rng(0), snr=(0.0, 25.0), spec_aug=False)
    kws, kn, kacc = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)
    X, Y, src = _metric_pool(clean, trips, np.random.default_rng(0), True)
    dom, n_dom, mask = G.build_domain_labels(src, with_synth=False)
    enc, (m, s) = G.train_grl(X, Y, src, dom, n_dom, grl_lambda, mask)
    Xp, Yp, srcp = _metric_pool(clean, trips, np.random.default_rng(1), False)
    protos = MP.register_protos(enc, Xp, Yp, srcp, m, s, include_real=True)
    D.FEATURE_MODE = "logmel_cmn_harmonic"         # chime detector on HPSS tones
    ctrips = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in chime_train_ids]  # noqa: F841
    Xc, Yc = build_event_pool(chime_train_ids, "chime", np.random.default_rng(0))
    chime, cn, cacc = P.train(Xc, Yc, 2, epochs=40, lr=5e-4, patience=12)
    D.FEATURE_MODE = prev
    if verbose:
        print(f"    trained: KWS {kacc*100:.0f}%  chime {cacc*100:.0f}%  enc pool {len(Y)}")
    return {"kws": (kws, kn), "chime": (chime, cn), "enc": (enc, protos, m, s)}



# --------------------------------------------------------------------------- #
# 융합 디코더 — 검출 포함 end-to-end (사용자 설계: 흐름 전체를 한꺼번에)
#   KWS 후보(시간순) 중 노선 순서대로 N개를 고르는 DP(Viterbi). 각 후보의 점수 =
#   인코더 emission(그 위치 역 이름일 확률) + 차임 동반 보너스(진짜 정차였다는 증거).
#   고르지 않은 후보(KWS 헛검출)는 자동 폐기. greedy가 아니라 전체최적이라 한 번
#   잘못 세도 cascade 안 됨. 위치(역 이름)는 노선순서+탑승역 앵커로 결정.
# --------------------------------------------------------------------------- #
def _travel_col(p, direction):
    """여행 순서 p(0..)번째 역의 TARGET13 emission 컬럼 (탑승역 앵커 + 노선 순서)."""
    d = +1 if direction == "등교" else -1
    rp = BOARDING_ROUTE[direction] + d * (1 + p)
    return int(LAB_OF_ROUTE[int(np.clip(rp, 0, 12))])


def fuse_decode(kws_times, E, chime_times, direction, nstop,
                lam_chime=2.0, lam_time=0.015, cycle=165.0, temp=0.1):
    """KWS 후보 중 노선 순서대로 nstop개를 고르는 DP. 반환: 고른 (시간, route_pos) 리스트.
    점수(i를 여행순서 p에) = log p(역_p | 후보 i) + lam_chime·(차임 동반?)
    + transition 시간 prior: 고른 이웃끼리 ~cycle 간격이도록 -lam_time·|Δt - cycle|.
    → 너무 붙은 가짜 후보 배제(카운트 아니라 간격 정합, cascade 없음)."""
    M = len(kws_times)
    if M == 0 or nstop == 0:
        return []
    order = np.argsort(kws_times)
    t = np.asarray(kws_times)[order]
    logE = _log_softmax(np.asarray(E)[order] / temp)
    sup = np.zeros(M)
    for i in range(M):
        nxt = t[i + 1] if i + 1 < M else float("inf")
        sup[i] = 1.0 if any(t[i] < c < nxt for c in chime_times) else 0.0
    nstop = min(nstop, M)
    cols = [_travel_col(p, direction) for p in range(nstop)]

    def emit(i, p):
        return float(logE[i, cols[p]] + lam_chime * sup[i])

    NEG = -1e18
    dp = np.full((nstop, M), NEG)
    back = np.full((nstop, M), -1, int)
    for i in range(M):
        dp[0][i] = emit(i, 0)
    for p in range(1, nstop):
        for i in range(M):
            best, barg = NEG, -1
            for j in range(i):                       # 이웃 j→i, 시간간격 ~cycle 선호
                if dp[p - 1][j] <= NEG:
                    continue
                sc = dp[p - 1][j] - lam_time * abs((t[i] - t[j]) - cycle)
                if sc > best:
                    best, barg = sc, j
            if barg >= 0:
                dp[p][i] = emit(i, p) + best
                back[p][i] = barg
    last = int(np.argmax(dp[nstop - 1]))
    chosen = [last]
    for p in range(nstop - 1, 0, -1):
        last = int(back[p][last]); chosen.append(last)
    chosen = chosen[::-1]
    d = +1 if direction == "등교" else -1
    return [(float(t[i]), int(np.clip(BOARDING_ROUTE[direction] + d * (1 + p), 0, 12)))
            for p, i in enumerate(chosen)]


def _score_routes(pred, marks, match_s=MATCH_S):
    """pred=[(time,route_pos)] → GT marks 시간매칭 후 route 정답 수."""
    mt = [(R_OF[st], idx / D.SR) for st, idx in marks]
    used = [False] * len(mt); correct = 0; matched = 0
    for pt, pr in sorted(pred):
        best = None
        for mj, (rp, t0) in enumerate(mt):
            if used[mj] or abs(pt - t0) > match_s:
                continue
            if best is None or abs(pt - t0) < abs(pt - mt[best][1]):
                best = mj
        if best is not None:
            used[best] = True; matched += 1
            correct += int(pr == mt[best][0])
    return correct, matched


def run_fusion_loo(trips=None, grl_lambda=0.3, lam_chime=2.0, lam_time=0.015, cycle=165.0):
    """검출 포함 end-to-end LOO. KWS·차임·인코더 학습 → 검출 → 융합 DP(인코더+차임+시간간격
    prior+단조) → 역정확도. 이게 '쓸 수 있나'의 최종 답. trip4 차임 변종 제외 7트립."""
    all_trips = list_trips()
    non_variant = [t for t in all_trips if not t.endswith(VARIANT_TRIP_SUFFIX)]
    trips = trips or non_variant
    print(f"융합 end-to-end LOO | {len(trips)} trips | λ_chime={lam_chime} λ_time={lam_time} "
          f"cycle={cycle} | episodes={MP.EPISODES}")
    print("KWS 검출 → 인코더 emission + 차임 보너스 + 시간간격 prior + 단조 → 노선순서 DP\n")
    agg = {"correct": 0, "matched": 0, "gt": 0}
    for held in trips:
        train_ids = [t for t in all_trips if t != held]
        chime_train = [t for t in non_variant if t != held]
        t0 = time.time()
        packs = train_sm_fold(train_ids, chime_train, grl_lambda)
        test = D.load_live_trip(os.path.join(LIVE_DIR, held))
        marks = sorted(test.marks, key=lambda x: x[1]); n_gt = len(marks)
        y, direction = test.y, test.direction
        prev = D.FEATURE_MODE
        D.FEATURE_MODE = "logmel_cmn"
        kws_t = kws_triggers(y, packs["kws"])
        E = emissions_at(kws_t, y, packs["enc"])
        D.FEATURE_MODE = "logmel_cmn_harmonic"
        cprobs, ctimes = _event_slide_probs(y, packs["chime"][0], packs["chime"][1], EVENT_WIN["chime"])
        chimes = _detect_col(cprobs[:, 1], ctimes)
        D.FEATURE_MODE = prev
        pred = fuse_decode(kws_t, E, chimes, direction, n_gt, lam_chime, lam_time, cycle)
        correct, matched = _score_routes(pred, marks)
        agg["correct"] += correct; agg["matched"] += matched; agg["gt"] += n_gt
        print(f"  {held[9:13]} ({direction}) GT={n_gt:2d} | KWS후보={len(kws_t)} chime={len(chimes)} "
              f"→ 고른역 {matched}/{n_gt} 매칭, 역정확도 {correct:2d}/{n_gt}  [{time.time()-t0:.0f}s]")
    n = agg["gt"]
    print("\n" + "=" * 60)
    print(f"  검출-포함 역정확도: {agg['correct']}/{n} ({agg['correct']/n*100:.0f}%)  ← 최종 '쓸 수 있나'")
    return agg


# --------------------------------------------------------------------------- #
# 타이밍-정렬 디코더 (인코더 X, announce+close만) — 측정된 시간 문법 사용
#   Δac(announce→close) med~74s, range[46,124]  → close를 직전 announce로 confirm
#   사이클(close→close) ~165s  → gap÷사이클로 급행 skip 역수 역산
#   위치 = 탑승역 앵커 + d×(1 + confirm된 close 수[skip 보정])
# --------------------------------------------------------------------------- #
AC_LO, AC_HI = 45.0, 130.0     # announce→close 윈도우 (close confirm용; 측정 range)
CYCLE_S = 165.0                # close→close 정상 사이클 (Δac+Δca); skip 추론 기준


def timing_decode(announce_t, close_t, direction, mark_t,
                  ac_lo=AC_LO, ac_hi=AC_HI, cycle=CYCLE_S):
    """announce-backed close만 채택 → close grid 카운트(+skip) → 각 mark 시점 역(route).
    인코더 미사용. announce/close 검출만으로 위치."""
    d = +1 if direction == "등교" else -1
    a0 = BOARDING_ROUTE[direction]
    ann = sorted(announce_t)
    conf = [c for c in sorted(close_t)
            if any(c - ac_hi <= a <= c - ac_lo for a in ann)]   # 직전 announce 있는 close만

    def advances_before(t):
        adv, prev = 0, None
        for c in conf:
            if c >= t:
                break
            adv += 1 if prev is None else max(1, int(round((c - prev) / cycle)))
            prev = c
        return adv

    return [int(np.clip(a0 + d * (1 + advances_before(t)), 0, 12)) for t in mark_t]


def run_timing_loo(trips=None, ac_lo=AC_LO, ac_hi=AC_HI, cycle=CYCLE_S):
    """검출-포함 타이밍-정렬 LOO (인코더 X). KWS·차임 검출 → timing_decode → 역정확도.
    가벼움(2모델/fold). trip4 차임 변종 제외 7트립."""
    all_trips = list_trips()
    non_variant = [t for t in all_trips if not t.endswith(VARIANT_TRIP_SUFFIX)]
    trips = trips or non_variant
    print(f"타이밍-정렬 LOO (인코더 X) | {len(trips)} trips | Δac[{ac_lo},{ac_hi}] cycle={cycle}")
    print("announce-backed close만 채택 → grid 카운트(+skip) → 위치\n")
    agg = {"correct": 0, "gt": 0}
    for held in trips:
        train_ids = [t for t in all_trips if t != held]
        chime_train = [t for t in non_variant if t != held]
        t0 = time.time()
        clean = D.load_clean_sources(CLEAN_DIR)
        tr = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in train_ids]
        prev = D.FEATURE_MODE
        D.FEATURE_MODE = "logmel_cmn"
        Xk, Yk = D.build_kws(clean, tr, np.random.default_rng(0), snr=(0.0, 25.0), spec_aug=False)
        kws, kn, _ = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)
        D.FEATURE_MODE = "logmel_cmn_harmonic"
        Xc, Yc = build_event_pool(chime_train, "chime", np.random.default_rng(0))
        chime, cn, _ = P.train(Xc, Yc, 2, epochs=40, lr=5e-4, patience=12)
        test = D.load_live_trip(os.path.join(LIVE_DIR, held))
        marks = sorted(test.marks, key=lambda x: x[1]); n_gt = len(marks)
        mark_t = [idx / D.SR for _, idx in marks]; gt = [R_OF[s] for s, _ in marks]
        y, direction = test.y, test.direction
        D.FEATURE_MODE = "logmel_cmn"
        anns = kws_triggers(y, (kws, kn))
        D.FEATURE_MODE = "logmel_cmn_harmonic"
        cp, ct = _event_slide_probs(y, chime, cn, EVENT_WIN["chime"])
        closes = _detect_col(cp[:, 1], ct)
        D.FEATURE_MODE = prev
        pred = timing_decode(anns, closes, direction, mark_t, ac_lo, ac_hi, cycle)
        corr = sum(int(p == g) for p, g in zip(pred, gt))
        agg["correct"] += corr; agg["gt"] += n_gt
        print(f"  {held[9:13]} ({direction}) GT={n_gt:2d} | KWS={len(anns)} chime={len(closes)} "
              f"→ 역정확도 {corr:2d}/{n_gt}  [{time.time()-t0:.0f}s]")
    n = agg["gt"]
    print("\n" + "=" * 56)
    print(f"  타이밍-정렬 역정확도(인코더 X): {agg['correct']}/{n} ({agg['correct']/n*100:.0f}%)")
    return agg


# --------------------------------------------------------------------------- #
# KWS hard-negative mining — false positive 억제 (학습트립서 FP 채굴 → neg 재학습)
# --------------------------------------------------------------------------- #
def _kws_recall_false(kws_pack, test, match_s=10.0):
    """held-out 트립에서 KWS recall(매칭된 마크) + false(매칭 안된 트리거)."""
    trg = kws_triggers(test.y, kws_pack)
    mt = [idx / D.SR for _, idx in test.marks]
    used = [False] * len(trg); hit = 0
    for m in mt:
        best = None
        for i, t in enumerate(trg):
            if used[i] or abs(t - m) > match_s:
                continue
            if best is None or abs(t - m) < abs(trg[best] - m):
                best = i
        if best is not None:
            used[best] = True; hit += 1
    return hit, len(mt), sum(1 for u in used if not u)


def _mine_kws_fp(kws_pack, trips_live):
    """학습 트립에서 KWS 헛발화 윈도우(진짜 마크와 >10s 떨어진 트리거) 수집 → logmel neg."""
    n = int(D.KWS_WIN * D.SR)
    negs = []
    for t in trips_live:
        mt = [idx / D.SR for _, idx in t.marks]
        for tt in kws_triggers(t.y, kws_pack):
            if all(abs(tt - m) > 10.0 for m in mt):
                a = int(tt * D.SR); w = t.y[a:a + n]
                if len(w) == n:
                    negs.append(D.to_logmel(w))
    return negs


def train_kws_hardneg(clean, trips_live, rng, rounds=1):
    """KWS 학습 → 학습트립서 FP 채굴 → negative 추가 재학습 (rounds회). 반환 ((kws,kn), n_mined)."""
    prev = D.FEATURE_MODE; D.FEATURE_MODE = "logmel_cmn"
    Xk, Yk = D.build_kws(clean, trips_live, rng, snr=(0.0, 25.0), spec_aug=False)
    kws, kn, _ = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)
    n_mined = 0
    for _ in range(rounds):
        negs = _mine_kws_fp((kws, kn), trips_live)
        if not negs:
            break
        n_mined += len(negs)
        Xk = np.concatenate([Xk, np.asarray(negs, np.float32)[..., None]])
        Yk = np.concatenate([Yk, np.zeros(len(negs), np.int32)])
        kws, kn, _ = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)
    D.FEATURE_MODE = prev
    return (kws, kn), n_mined


def run_kws_hardneg_loo(trips=None, rounds=1):
    """KWS false: hard-negative mining 전 vs 후 (cross-trip LOO)."""
    trips = trips or list_trips()
    clean = D.load_clean_sources(CLEAN_DIR)
    print(f"KWS hard-negative mining LOO | {len(trips)} trips | rounds={rounds}\n")
    base = {"hit": 0, "n": 0, "false": 0}; hard = {"hit": 0, "n": 0, "false": 0}
    for held in trips:
        tr = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in trips if t != held]
        test = D.load_live_trip(os.path.join(LIVE_DIR, held))
        prev = D.FEATURE_MODE; D.FEATURE_MODE = "logmel_cmn"
        Xk, Yk = D.build_kws(clean, tr, np.random.default_rng(0), snr=(0.0, 25.0), spec_aug=False)
        m0, n0_, _ = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)
        D.FEATURE_MODE = prev
        h0, n0, f0 = _kws_recall_false((m0, n0_), test)
        (kws1, kn1), nm = train_kws_hardneg(clean, tr, np.random.default_rng(0), rounds)
        h1, n1, f1 = _kws_recall_false((kws1, kn1), test)
        base["hit"] += h0; base["n"] += n0; base["false"] += f0
        hard["hit"] += h1; hard["n"] += n1; hard["false"] += f1
        print(f"  {held[9:13]} ({test.direction}): baseline {h0}/{n0} f{f0}  →  "
              f"+mined({nm}) {h1}/{n1} f{f1}")
    T = len(trips)
    print("\n" + "=" * 56)
    print(f"  baseline : recall {base['hit']}/{base['n']} ({base['hit']/base['n']*100:.0f}%)  "
          f"false {base['false']/T:.0f}/trip")
    print(f"  +hardneg : recall {hard['hit']}/{hard['n']} ({hard['hit']/hard['n']*100:.0f}%)  "
          f"false {hard['false']/T:.0f}/trip")
    print("  → false↓·recall 유지면 mining 성공.")
    return base, hard


# --------------------------------------------------------------------------- #
# 진단: 각 모델 따로 (인코더 7트립 per-mark + KWS/차임 학습곡선)
# --------------------------------------------------------------------------- #
def run_encoder_permark_loo(trips=None, grl_lambda=0.3):
    """인코더 단독 per-mark cross-trip LOO (노선prior·앵커 없이 순수 분류).
    7트립으로 재서 4트립 baseline(clean-synth 33% / GRL 44% / chance 8%)과 비교."""
    trips = trips or list_trips()
    clean = D.load_clean_sources(CLEAN_DIR)
    prev = D.FEATURE_MODE; D.FEATURE_MODE = "logmel_cmn"
    tot = [0, 0]
    print(f"인코더 per-mark LOO | {len(trips)} trips | GRL λ={grl_lambda} | episodes={MP.EPISODES}")
    print("(노선prior·앵커 없이 인코더 단독 분류. 4트립: clean-synth 33%, GRL 44%, chance 8%)\n")
    for held in trips:
        tr = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in trips if t != held]
        test = D.load_live_trip(os.path.join(LIVE_DIR, held))
        X, Y, src = _metric_pool(clean, tr, np.random.default_rng(0), True)
        dom, n_dom, mask = G.build_domain_labels(src, with_synth=False)
        enc, (m, s) = G.train_grl(X, Y, src, dom, n_dom, grl_lambda, mask)
        Xp, Yp, srcp = _metric_pool(clean, tr, np.random.default_rng(1), False)
        protos = MP.register_protos(enc, Xp, Yp, srcp, m, s, include_real=True)
        ok, n, _ = MP.proto_score(enc, protos, test, m, s)
        tot[0] += ok; tot[1] += n
        print(f"  {held[9:13]} ({test.direction}): {ok}/{n} ({ok/n*100:3.0f}%)")
    D.FEATURE_MODE = prev
    print("\n" + "=" * 50)
    print(f"  인코더 per-mark LOO: {tot[0]}/{tot[1]} ({tot[0]/tot[1]*100:.0f}%)  "
          f"← 4트립 GRL 44% 대비 ({'개선' if tot[0]/tot[1] > 0.44 else '비슷/미달'})")
    return tot


def plot_training_curves(held=None):
    """KWS·차임 검출기의 학습곡선(val_accuracy/epoch)을 따로 그림 — 각자 학습 확인용."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    import tensorflow as tf
    trips = list_trips()
    held = held or trips[0]
    tr = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in trips if t != held]
    clean = D.load_clean_sources(CLEAN_DIR)

    def fit_hist(X, Y, label):
        m_, s_ = float(X.mean()), float(X.std()) + 1e-6
        Xn = (X - m_) / s_
        Xtr, Xva, Ytr, Yva = train_test_split(Xn, Y, test_size=0.15, stratify=Y, random_state=42)
        P.set_seeds(0)
        mdl = P.small_cnn(X.shape[1:], 2)
        mdl.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        w = compute_class_weight("balanced", classes=np.unique(Ytr), y=Ytr)
        h = mdl.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=40, batch_size=64,
                    class_weight={int(c): float(wi) for c, wi in zip(np.unique(Ytr), w)},
                    verbose=0)
        print(f"  {label}: final val {h.history['val_accuracy'][-1]*100:.0f}%")
        return h.history["val_accuracy"]

    prev = D.FEATURE_MODE
    D.FEATURE_MODE = "logmel_cmn"
    Xk, Yk = D.build_kws(clean, tr, np.random.default_rng(0), snr=(0., 25.), spec_aug=False)
    kws_v = fit_hist(Xk, Yk, "KWS")
    D.FEATURE_MODE = "logmel_cmn_harmonic"
    Xc, Yc = build_event_pool([t for t in trips if t != held and not t.endswith(VARIANT_TRIP_SUFFIX)],
                              "chime", np.random.default_rng(0))
    ch_v = fit_hist(Xc, Yc, "chime")
    D.FEATURE_MODE = prev
    plt.figure(figsize=(8, 4.5))
    plt.plot(kws_v, "-o", label="KWS (이번역은)")
    plt.plot(ch_v, "-s", label="chime (HPSS)")
    plt.xlabel("epoch"); plt.ylabel("val accuracy"); plt.ylim(0.4, 1.02)
    plt.title(f"separate training curves (held-out {held[9:13]})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("model_training_curves.png", dpi=110)
    print("saved model_training_curves.png  (인코더는 episodic이라 별도 — run_encoder_permark_loo)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    DR.WIN_S = float(os.environ.get("DOOR_WIN_S", "1.0"))
    MP.EPISODES = G.EPISODES = int(os.environ.get("PROTO_EPISODES", "600"))
    run_loo()
