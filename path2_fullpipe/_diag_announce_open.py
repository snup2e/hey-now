# -*- coding: utf-8 -*-
"""announce→open 변동 원인 분해: 역/방향별로 일정한가(=고정위치 트리거) vs 랜덤인가.
within-station std(같은 역 반복)와 across-station spread(역마다 다름)를 비교."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import path2_dataset as D, path2_pipeline as PL
PL.LIVE_DIR = "data/raw/line1_live"
SR = D.SR
R_OF = PL.R_OF
PAIR_MAX = 130.0


def visits(tid):
    d = os.path.join(PL.LIVE_DIR, tid)
    info = json.loads(open(os.path.join(d, "marks.json"), encoding="utf-8").read())
    segs = info.get("segments") or [{"direction": info.get("direction", "?"), "marks": info.get("marks", [])}]
    direction = "등교" if "등" in segs[0].get("direction", tid) else "하교"
    raw = sorted(((m["station"], m["sample_index"] / SR)
                  for s in segs for m in s.get("marks", []) if m["station"] in R_OF), key=lambda x: x[1])
    marks = []
    for st, t in raw:
        if not marks or t - marks[-1][1] > 10:
            marks.append((st, t))
    _, opens, closes, dwells, runs = PL.load_door(d)
    V = []
    for o, c in dwells:
        to = o / SR
        cand = [(st, t) for st, t in marks if 0 <= to - t <= PAIR_MAX]
        if cand:
            st, ta = cand[-1]; V.append((st, ta, to, c / SR))
    # 직전 주행시간(접근 속도 proxy): 이 open 직전 close→이 open
    runs_s = [(a / SR, b / SR) for a, b in runs]
    return direction, V, runs_s


by = {}        # (direction, station) -> [announce→open]
ao_vs_prevtravel = []   # (announce→open, preceding travel time) 상관용
for tid in PL.list_trips():
    direction, V, runs_s = visits(tid)
    for st, ta, to, tc in V:
        by.setdefault((direction, st), []).append(to - ta)
        prev = [b - a for a, b in runs_s if b <= to + 1]   # 직전 주행
        if prev:
            ao_vs_prevtravel.append((to - ta, prev[-1]))

print("=== announce→open : 역·방향별 (고정위치 트리거면 역별로 일정) ===")
print(f"  {'방향 역':26s} {'n':>2s} {'mean':>5s} {'std':>4s}")
within = []
for k in sorted(by, key=lambda x: (x[0], R_OF[x[1]])):
    v = np.array(by[k])
    if len(v) >= 2:
        within.append(v.std())
    print(f"  {k[0]+' '+k[1]:26s} {len(v):2d} {v.mean():5.0f} {v.std():4.0f}")

allao = np.array([x for v in by.values() for x in v])
station_means = np.array([np.mean(v) for v in by.values()])
print(f"\n전체 announce→open std        = {allao.std():4.0f}s  (전체 변동)")
print(f"  역내(within-station) 평균 std = {np.mean(within):4.0f}s  (같은 역 반복 변동)")
print(f"  역간(across-station) mean std = {station_means.std():4.0f}s  (역마다 다른 정도)")
if ao_vs_prevtravel:
    a = np.array(ao_vs_prevtravel)
    r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    print(f"\nannounce→open vs 직전 주행시간 상관 r = {r:+.2f}  "
          f"(+면 '느린 접근=긴 announce→open' = 속도 효과)")
print("\n해석: within<<across면 → 역마다 고정거리 트리거(역별 체계적). within도 크면 → 속도/운영 변동.")
