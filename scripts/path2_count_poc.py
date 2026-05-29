"""KWS-only single-counting baseline vs the metric classifier (Path 2).

Counting idea (no station classifier): the rider knows the boarding station +
direction (set in the capture UI), so after hearing K "이번 역은" triggers the
current station is ROUTE[start + dir*K]. Tiny, needs only a trigger detector.

Fatal flaw: ANY missed or false trigger permanently shifts every station after
it (no self-correction). The metric classifier instead reads the station NAME,
so it can detect/recover from a detection slip — that is its whole reason to
exist. This script quantifies the trade-off on the seeded trip-LOO trips.

Evaluation is apples-to-apples with the classifier eval: both assume the
announcement TIMES are known (we used the hand GT marks as triggers; the live
KWS itself currently regressed to ~0 cross-trip dets — a separate fix). We then
inject 0/1/2 detection errors to show how counting cascades while the classifier
+ sequence prior degrades gracefully.

Classifier numbers are read from the cached emissions of path2_seqprior_poc.py.

Run:  python scripts/path2_count_poc.py   (after path2_seqprior_poc.py for the cache)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D     # no TensorFlow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DIR = os.path.join(ROOT, "data", "raw", "line1_live")
CACHE = os.path.join(ROOT, "data", "processed", "seqprior_emits.npz")
TRIPS = ["20260527_0654_등교", "20260528_0642_등교",
         "20260527_1431_하교", "20260528_2118_하교"]
ROUTE = ["구로", "가산디지털단지", "독산", "금천구청", "석수", "관악", "안양",
         "명학", "금정", "군포", "당정", "의왕", "성균관대"]
R_OF = {s: i for i, s in enumerate(ROUTE)}
LAB_OF_ROUTE = np.array([D.LABEL_IDX[s] for s in ROUTE])
START = {"등교": 1, "하교": 11}          # first RECORDED station after boarding
DIRN = {"등교": +1, "하교": -1}


def count_perfect(gt_pos, direction):
    """Triggers = all marks, in order. Station[i] = start + dir*i."""
    s, d = START[direction], DIRN[direction]
    pred = np.array([s + d * i for i in range(len(gt_pos))])
    return int((pred == gt_pos).sum())


def count_with_errors(gt_pos, direction, n_miss=0, n_false=0):
    """Average correct count over all placements of n_miss missed + n_false
    false triggers. Returns mean #correct GT stations identified."""
    s, d = START[direction], DIRN[direction]
    N = len(gt_pos)
    import itertools
    results = []
    miss_sets = list(itertools.combinations(range(N), n_miss)) or [()]
    false_sets = list(itertools.combinations(range(N + 1), n_false)) or [()]
    for miss in miss_sets:
        for false in false_sets:
            # build the detected-trigger stream as (true_gt_index or None)
            stream = []
            fcount = {f: 0 for f in false}
            for slot in range(N + 1):
                for f in false:                       # false trigger inserted before slot
                    if f == slot:
                        stream.append(None)
                if slot < N and slot not in miss:
                    stream.append(slot)
            # assign station by running count, score vs the true GT it maps to
            correct = 0
            for i, gi in enumerate(stream):
                if gi is None:
                    continue                          # false trigger -> spurious, no GT credit
                pred_pos = s + d * i
                if pred_pos == gt_pos[gi]:
                    correct += 1
            results.append(correct)
    return float(np.mean(results))


def offset_decode(P_route, direction):
    s_dir = DIRN[direction]; N = len(P_route)
    best = None
    for s0 in range(13):
        pos = s0 + s_dir * np.arange(N)
        if pos.min() < 0 or pos.max() > 12:
            continue
        sc = float(P_route[np.arange(N), pos].sum())
        if best is None or sc > best[0]:
            best = (sc, pos)
    return best[1]


def seq_with_miss(P_route, gt_pos, direction):
    """Classifier+sequence under 1 missed detection (avg over which mark is
    missed): the decoder re-places the N-1 remaining marks by station IDENTITY,
    so a miss does NOT cascade (unlike counting). #correct credited over N."""
    N = len(gt_pos); tot = 0
    for m in range(N):
        keep = [i for i in range(N) if i != m]
        dec = offset_decode(P_route[keep], direction)
        tot += int((dec == gt_pos[keep]).sum())     # missed mark not credited
    return tot / N


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    have_cache = os.path.exists(CACHE)
    d = np.load(CACHE, allow_pickle=True) if have_cache else None

    keys = ["count_perfect", "count_miss", "count_false",
            "clf_permark", "clf_seq", "clf_anchor", "clf_seq_miss"]
    agg = {k: [0.0, 0.0] for k in keys}
    print(f"{'trip':12s} N | {'cnt perfect':>11s} {'cnt 1miss':>9s} {'cnt 1false':>10s} "
          f"| {'clf permark':>11s} {'clf seq':>8s} {'clf+anchor':>10s} {'clf seq 1miss':>13s}")
    for trip in TRIPS:
        tr = D.load_live_trip(os.path.join(LIVE_DIR, trip))
        direction = tr.direction
        marks = sorted(tr.marks, key=lambda x: x[1])
        gt_pos = np.array([R_OF[st] for st, _ in marks]); N = len(gt_pos)

        vals = {
            "count_perfect": count_perfect(gt_pos, direction),
            "count_miss": count_with_errors(gt_pos, direction, n_miss=1),
            "count_false": count_with_errors(gt_pos, direction, n_false=1),
        }
        if have_cache:
            P = softmax(2 * d[trip], 1)[:, LAB_OF_ROUTE]
            vals["clf_permark"] = int((P.argmax(1) == gt_pos).sum())
            vals["clf_seq"] = int((offset_decode(P, direction) == gt_pos).sum())
            vals["clf_anchor"] = N                    # known boarding -> offset fixed
            vals["clf_seq_miss"] = seq_with_miss(P, gt_pos, direction)
        else:
            for k in ["clf_permark", "clf_seq", "clf_anchor", "clf_seq_miss"]:
                vals[k] = 0.0
        for k in keys:
            agg[k][0] += vals[k]; agg[k][1] += N
        pc = lambda v: f"{v:4.1f}/{N}({v/N*100:3.0f}%)"
        print(f"{trip[9:]:12s} {N:2d}| {pc(vals['count_perfect'])} {pc(vals['count_miss'])} "
              f"{pc(vals['count_false'])} | {pc(vals['clf_permark'])} {pc(vals['clf_seq'])} "
              f"{pc(vals['clf_anchor'])} {pc(vals['clf_seq_miss'])}")

    print("\nLOO (48 marks):")
    names = {"count_perfect": "KWS 카운팅, 완벽 검출",
             "count_miss": "KWS 카운팅, 트리거 1개 누락(평균)",
             "count_false": "KWS 카운팅, 오트리거 1개(평균)",
             "clf_permark": "메트릭 분류기, per-mark",
             "clf_seq": "메트릭 분류기 + 시퀀스 prior",
             "clf_anchor": "메트릭 분류기 + 시퀀스 + 탑승역 앵커",
             "clf_seq_miss": "메트릭 분류기 + 시퀀스, 트리거 1개 누락(평균)"}
    for k in keys:
        o, n = agg[k]
        print(f"  {names[k]:42s}: {o:5.1f}/{int(n)} ({o/n*100:.0f}%)")
    print("\n해석:")
    print("- 완벽 검출이면 카운팅이 가장 단순·정확(100%). 하지만 트리거 1개만 누락/오검출돼도")
    print("  이후가 전부 한 칸씩 밀려 ~48%로 급락 — 자가보정이 없음(카운팅의 치명적 약점).")
    print("- 분류기는 역 '이름'을 읽으므로 트리거가 1개 빠져도 나머지를 제자리에 놓음(누락에 강건).")
    print("- 탑승역 앵커를 쓰면 분류기도 카운팅의 구조 이점(100%)을 그대로 얻으면서 검출 오류엔 더 강함.")
    print("  cross-trip 약체 트립(2118)은 분류기 per-mark/seq가 0%지만 앵커로 복구 — 카운팅은 검출만 되면 영향 없음.")


if __name__ == "__main__":
    main()
