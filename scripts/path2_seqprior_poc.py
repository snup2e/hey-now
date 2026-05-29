"""Sequence-prior decoding on top of the metric encoder (Path 2 station ID).

Per-mark classification tops out at ~42% cross-trip (4 channels = the ceiling).
But the train moves MONOTONICALLY along a known 1-D line in a known direction
(등교 = toward 성균관대, 하교 = toward 구로), so consecutive announcements are
consecutive stations. That is real physical structure, not leakage: the capture
UI already knows the direction.

We keep the encoder's per-mark cosine-to-prototype scores as Viterbi EMISSIONS
and add a TRANSITION prior over the line topology (advance ~+1 in the travel
direction; skips penalised, backward heavily penalised). Decoding the whole
announcement sequence lets confident detections fix the wrong ones.

Honest go/no-go: same seeded trip-LOO. Encoder = real-only + heavy aug + GRL
(λ=0.3) — the current best emission model. Report per-mark argmax vs Viterbi.

Run:  python scripts/path2_seqprior_poc.py
"""
import collections
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D
import path2_metric_poc as MP
import path2_grl_poc as G

# Physical line order (구로 end -> 성균관대 end). 등교 travels +1, 하교 travels -1.
ROUTE = ["구로", "가산디지털단지", "독산", "금천구청", "석수", "관악", "안양",
         "명학", "금정", "군포", "당정", "의왕", "성균관대"]
R_OF = {s: i for i, s in enumerate(ROUTE)}                  # station -> route position
LAB_OF_ROUTE = np.array([D.LABEL_IDX[s] for s in ROUTE])    # route pos -> TARGET13 label

LIVE_DIR = MP.LIVE_DIR
TRIPS = MP.TRIPS
ALPHAS = [0.0, 3.0, 6.0, 10.0]    # 0.0 = no prior (per-mark argmax); sweep prior strength
TEMP = 0.1                        # emission softmax temperature (cosines ~ [0.3,1])
EMIT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "processed", "seqprior_emits.npz")


def build_pool(clean, trips, rng, augment):
    return D.build_metric_pool(
        clean, trips, rng, use_clean=False, snr=(-5.0, 25.0),
        real_noise_aug=(16 if augment else 0), real_jitter=(12 if augment else 3),
        jitter_s=(0.3 if augment else 0.1), spec_aug=augment)


def emissions(enc, protos, test, m, s):
    """Time-ordered marks + per-mark cosine-to-prototype emission matrix [N,13]."""
    marks = sorted(test.marks, key=lambda x: x[1])
    E = []
    for _, idx in marks:
        wins = [D.window_feature(test.y, idx, d) for d in (-0.1, 0.0, 0.1)]
        e = MP.embed(enc, np.asarray(wins)[..., None], m, s)
        q = e.mean(0); q /= (np.linalg.norm(q) + 1e-9)
        E.append(protos @ q)                       # cosine to each station prototype
    return marks, np.asarray(E, np.float32)


def _log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True) + 1e-12)


def viterbi(E_label, direction, alpha, temp):
    """E_label[N,13] cosines (TARGET13 order) -> decoded route positions [N]."""
    N = len(E_label)
    Er = E_label[:, LAB_OF_ROUTE]                  # reorder cols to ROUTE positions
    logE = _log_softmax(Er / temp)
    d = +1 if direction == "등교" else -1
    j = np.arange(13)
    logT = -alpha * np.abs((j[None, :] - j[:, None]) - d)   # [prev,cur]; advance by d is free
    dp = logE[0].copy(); back = []
    for t in range(1, N):
        sc = dp[:, None] + logT + logE[t][None, :]
        back.append(sc.argmax(0)); dp = sc.max(0)
    path = [int(dp.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return path[::-1]


def offset_decode(E_label, direction):
    """Strongest legitimate prior: the marks are N CONSECUTIVE stations in the
    known direction; only the start offset is unknown. Pick the offset whose
    strict +d run maximises total emission. (Assumes no missed detections; the
    Viterbi above is the skip-tolerant version for deployment.)"""
    Er = E_label[:, LAB_OF_ROUTE]
    N = len(Er); d = +1 if direction == "등교" else -1
    best = None
    for s0 in range(13):
        pos = s0 + d * np.arange(N)
        if pos.min() < 0 or pos.max() > 12:
            continue
        score = float(Er[np.arange(N), pos].sum())
        if best is None or score > best[0]:
            best = (score, pos)
    return best[1]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    clean = D.load_clean_sources(MP.CLEAN_DIR)
    print(f"encoder = real-only + heavy aug + GRL(λ=0.3) | episodes={MP.EPISODES}")
    print(f"alphas={ALPHAS} (0=per-mark argmax)  temp={TEMP}\n")
    tot = {a: [0, 0] for a in ALPHAS}
    tot["offset"] = [0, 0]
    cache = {}
    for held in TRIPS:
        trips = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in TRIPS if t != held]
        test = D.load_live_trip(os.path.join(LIVE_DIR, held))
        t0 = time.time()
        X, Y, src = build_pool(clean, trips, np.random.default_rng(0), True)
        dom, n_dom, mask = G.build_domain_labels(src, with_synth=False)
        enc, (m, s) = G.train_grl(X, Y, src, dom, n_dom, 0.3, mask)
        Xp, Yp, srcp = build_pool(clean, trips, np.random.default_rng(1), False)
        protos = MP.register_protos(enc, Xp, Yp, srcp, m, s, include_real=True)
        marks, E = emissions(enc, protos, test, m, s)
        gt_route = np.array([R_OF[st] for st, _ in marks])
        cache[held] = E; cache[held + "_dir"] = test.direction; cache[held + "_gt"] = gt_route
        print(f"held-out {held[9:13]} ({test.direction}) [{time.time()-t0:.0f}s]:")
        for a in ALPHAS:
            dec = np.array(viterbi(E, test.direction, a, TEMP))
            ok = int((dec == gt_route).sum())
            tot[a][0] += ok; tot[a][1] += len(marks)
            tag = "per-mark" if a == 0 else f"viterbi α={a}"
            print(f"    {tag:13s}: {ok}/{len(marks)} ({ok/len(marks)*100:3.0f}%)")
        dec = offset_decode(E, test.direction)
        ok = int((dec == gt_route).sum())
        tot["offset"][0] += ok; tot["offset"][1] += len(marks)
        print(f"    offset-run   : {ok}/{len(marks)} ({ok/len(marks)*100:3.0f}%)  "
              f"dec={list(dec)}")
        print(f"    GT route pos : {list(gt_route)}\n")
    np.savez(EMIT_CACHE, **{k: np.asarray(v) for k, v in cache.items()})
    print(f"(emissions cached -> {EMIT_CACHE})")
    print("=" * 56)
    for a in ALPHAS:
        o, n = tot[a]
        tag = "per-mark (no prior)" if a == 0 else f"viterbi α={a}"
        print(f"LOO {tag:20s}: {o}/{n} ({o/n*100:.0f}%)")
    o, n = tot["offset"]
    print(f"LOO {'offset-run (consec)':20s}: {o}/{n} ({o/n*100:.0f}%)")


if __name__ == "__main__":
    main()
