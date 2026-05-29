"""Embedded-final station decoder: boarding-station ANCHOR + monotone route.

Strictly on-board cost 0 (pure arithmetic over a length-N decision stream, no
model change), so this is the F411 deployment decoder. It evaluates on the
emissions cached by path2_seqprior_poc.py (the current best GRL encoder), so no
retraining / GPU is needed -- run it any time.

Three pieces of free physical structure, in increasing strength:
  1. monotone route   -- consecutive announcements are consecutive stations in
                         the known travel direction (seqprior_poc: 33% -> 75%).
  2. boarding ANCHOR  -- the capture UI knows the direction, so the FIRST real
                         announcement is fixed: 등교 boards 구로 -> first = 가산
                         (route 1); 하교 boards 성대 -> first = 의왕 (route 11).
                         The start offset is KNOWN, not searched -> fixes the
                         off-by-one that broke the 2118 fold (75% -> 100%).
  3. emissions        -- the classifier cosines, used to stay robust when KWS
                         MISSES or FALSE-fires (the real deployment failure: a
                         missed trigger breaks the "N consecutive" assumption).

Decoders compared (per-mark, searched-offset, anchored-offset, anchored-Viterbi)
and -- the point of this script -- their graceful degradation under simulated
missed detections, which counting-based tracking cannot survive.

Run:  python scripts/path2_anchor_decode.py
"""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "processed", "seqprior_emits.npz")

# Physical line order (구로 end -> 성균관대 end). 등교 travels +1, 하교 travels -1.
ROUTE = ["구로", "가산디지털단지", "독산", "금천구청", "석수", "관악", "안양",
         "명학", "금정", "군포", "당정", "의왕", "성균관대"]
TARGET13 = sorted(ROUTE)                                   # == path2_dataset.TARGET13
LABEL_IDX = {s: i for i, s in enumerate(TARGET13)}
LAB_OF_ROUTE = np.array([LABEL_IDX[s] for s in ROUTE])     # route pos -> TARGET13 col
BOARDING_ROUTE = {"등교": 0, "하교": 12}                    # boarding station route pos
TEMP = 0.1
NEG = -1e9


def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True) + 1e-12)


def direction_of(dstr):
    return "등교" if "등" in str(dstr) else "하교"


# --------------------------------------------------------------------------- #
# decoders : E[N,13] cosines in TARGET13 order -> decoded route positions [N]
# --------------------------------------------------------------------------- #
def per_mark(E):
    return LAB_OF_ROUTE_inv(E.argmax(1))


def LAB_OF_ROUTE_inv(labels):
    """TARGET13 label -> route position."""
    inv = {int(l): r for r, l in enumerate(LAB_OF_ROUTE)}
    return np.array([inv[int(l)] for l in labels])


def offset_run(E, direction, anchored):
    """N consecutive route positions in direction d. anchored=True pins the start
    at boarding+d (known); else searches the best start by total emission."""
    Er = E[:, LAB_OF_ROUTE]
    N = len(Er)
    d = +1 if direction == "등교" else -1
    if anchored:
        s0 = BOARDING_ROUTE[direction] + d                 # first announcement
        starts = [s0]
    else:
        starts = range(13)
    best = None
    for s in starts:
        pos = s + d * np.arange(N)
        if pos.min() < 0 or pos.max() > 12:
            continue
        score = float(Er[np.arange(N), pos].sum())
        if best is None or score > best[0]:
            best = (score, pos)
    if best is None:                                       # anchor + N too long: clip
        s = BOARDING_ROUTE[direction] + d
        pos = np.clip(s + d * np.arange(N), 0, 12)
        best = (0.0, pos)
    return best[1]


def anchored_viterbi(E, direction, alpha=6.0, beta=8.0, max_skip=3):
    """Skip-tolerant: pins mark 0 near the boarding anchor (soft, weight beta),
    forces monotone advance in direction d (forward skips = missed detections,
    penalised alpha*extra-steps; stay/backward forbidden). Robust to misses."""
    Er = E[:, LAB_OF_ROUTE]
    N = len(Er)
    d = +1 if direction == "등교" else -1
    logE = log_softmax(Er / TEMP)
    a0 = BOARDING_ROUTE[direction] + d
    j = np.arange(13)
    # transition[p,c]: advance by k=(c-p)*d steps; k=1 free, k in [2,max_skip]
    # penalised, k<1 (stay/backward) forbidden.
    k = (j[None, :] - j[:, None]) * d                      # [prev,cur] steps forward
    logT = np.where((k >= 1) & (k <= max_skip), -alpha * (k - 1), NEG).astype(float)
    dp = logE[0] - beta * np.abs(j - a0)                   # anchor prior on first mark
    back = []
    for t in range(1, N):
        sc = dp[:, None] + logT + logE[t][None, :]
        back.append(sc.argmax(0))
        dp = sc.max(0)
    path = [int(dp.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return np.array(path[::-1])


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def load_cache():
    d = np.load(CACHE, allow_pickle=True)
    trips = sorted({k[:-4] for k in d.files if k.endswith("_dir")})
    out = []
    for t in trips:
        out.append((t, np.asarray(d[t], np.float32),
                    direction_of(d[t + "_dir"]),
                    np.asarray(d[t + "_gt"], int)))
    return out


def score(dec, gt):
    return int((dec == gt).sum()), len(gt)


def drop_combos(N, k, cap=20, rng=None):
    """Up to `cap` index-subsets to drop (all C(N,k) if small, else sampled)."""
    allc = list(itertools.combinations(range(N), k))
    if len(allc) <= cap:
        return allc
    rng = rng or np.random.default_rng(0)
    return [tuple(sorted(rng.choice(N, k, replace=False))) for _ in range(cap)]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    data = load_cache()
    print(f"cache = {CACHE}  ({len(data)} trips, GRL encoder emissions)\n")

    decoders = {
        "per-mark argmax":      lambda E, dr: per_mark(E),
        "offset-run (search)":  lambda E, dr: offset_run(E, dr, anchored=False),
        "anchored offset-run":  lambda E, dr: offset_run(E, dr, anchored=True),
        "anchored Viterbi":     lambda E, dr: anchored_viterbi(E, dr),
    }
    print("== full detections (no misses) ==")
    tot = {name: [0, 0] for name in decoders}
    for trip, E, dr, gt in data:
        line = []
        for name, fn in decoders.items():
            ok, n = score(fn(E, dr), gt)
            tot[name][0] += ok; tot[name][1] += n
            line.append(f"{ok:2d}/{n}")
        print(f"  {trip[9:13]} ({dr}): " +
              "  ".join(f"{nm}={l}" for nm, l in zip(decoders, line)))
    print("  " + "-" * 60)
    for name in decoders:
        o, n = tot[name]
        print(f"  LOO {name:22s}: {o}/{n} ({o/n*100:3.0f}%)")

    # --- robustness to missed detections (KWS misses k triggers) --------------
    print("\n== graceful degradation under k missed detections "
          "(avg over which-dropped) ==")
    print(f"  {'k missed':10s}  {'offset-run(search)':>20s}  {'anchored offset':>16s}"
          f"  {'anchored Viterbi':>17s}")
    for k in range(0, 4):
        agg = {"search": [0, 0], "anchor_off": [0, 0], "anchor_vit": [0, 0]}
        for trip, E, dr, gt in data:
            N = len(gt)
            for drop in drop_combos(N, k):
                keep = [i for i in range(N) if i not in drop]
                Ek, gtk = E[keep], gt[keep]
                for tag, fn in (("search", lambda: offset_run(Ek, dr, False)),
                                ("anchor_off", lambda: offset_run(Ek, dr, True)),
                                ("anchor_vit", lambda: anchored_viterbi(Ek, dr))):
                    ok, n = score(fn(), gtk)
                    agg[tag][0] += ok; agg[tag][1] += n
        def pct(t):
            o, n = agg[t]; return f"{o/n*100:3.0f}%" if n else "  -"
        print(f"  {k:^10d}  {pct('search'):>20s}  {pct('anchor_off'):>16s}"
              f"  {pct('anchor_vit'):>17s}")
    print("\n  (counting-based tracking would cascade to ~0 after the first miss;")
    print("   the classifier emissions are what keep anchored Viterbi recoverable.)")


if __name__ == "__main__":
    main()
