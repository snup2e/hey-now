# -*- coding: utf-8 -*-
"""실전용 타이밍 테이블 — announce(KWS)·close(차임) 둘만 사용 (open 실전 불가).
역×방향별로 여러 트립 걸쳐:
  Δac = announce→close (역 내부; KWS↔차임 페어링 윈도우)
  Δca = close→다음 announce (역간 이동; 다음 역 예측)
분포를 면밀히 조사하고, 필터가 쓸 테이블을 JSON으로 저장."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import path2_dataset as D, path2_pipeline as PL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PL.LIVE_DIR = "data/raw/line1_live"
SR = D.SR
ROUTE, R_OF = PL.ROUTE, PL.R_OF
PAIR_MAX = 130.0


def visits(tid):
    """(station, t_announce, t_close) — close를 직전 announce(≤PAIR_MAX)와 페어링."""
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
    _, _, closes, _, _ = PL.load_door(d)
    V = []
    for c in sorted(closes):
        tc = c / SR
        cand = [(st, t) for st, t in marks if 0 <= tc - t <= PAIR_MAX]
        if cand:
            st, ta = cand[-1]; V.append((st, ta, tc))
    # 한 역 중복 close 제거(같은 역 라벨 연속이면 첫 것)
    dedup = []
    for st, ta, tc in V:
        if not dedup or dedup[-1][0] != st:
            dedup.append((st, ta, tc))
    return direction, dedup


AC = {}   # (dir, station) -> [Δac]
CA = {}   # (dir, station, next_station) -> [Δca]
allac, allca = [], []
for tid in PL.list_trips():
    direction, V = visits(tid)
    for st, ta, tc in V:
        if st in ("구로", "성균관대"):     # 엔드포인트 절단 제외
            continue
        AC.setdefault((direction, st), []).append(tc - ta)
        allac.append(tc - ta)
    for i in range(len(V) - 1):
        st, _, tc = V[i]; nst, nta, _ = V[i + 1]
        if st in ("구로", "성균관대"):
            continue
        CA.setdefault((direction, st, nst), []).append(nta - tc)
        allca.append(nta - tc)

allac, allca = np.array(allac), np.array(allca)


def line(v):
    return f"n={len(v):2d} mean={np.mean(v):5.0f} std={np.std(v):4.0f} med={np.median(v):5.0f} [{np.min(v):.0f},{np.max(v):.0f}]"


for direction in ("등교", "하교"):
    print(f"\n================= {direction} =================")
    print("  [Δac] announce→close (KWS↔차임 페어링 윈도우)")
    keys = sorted([k for k in AC if k[0] == direction], key=lambda k: R_OF[k[1]])
    for k in keys:
        print(f"    {k[1]:14s} {line(np.array(AC[k]))}")
    print("  [Δca] close→다음 announce (다음 역 예측)")
    keys = sorted([k for k in CA if k[0] == direction], key=lambda k: R_OF[k[1]])
    for k in keys:
        print(f"    {k[1]+'→'+k[2]:24s} {line(np.array(CA[k]))}")

print("\n================= 종합 =================")
print(f"  Δac announce→close : {line(allac)}")
print(f"  Δca close→announce : {line(allca)}")
wstd_ac = np.mean([np.std(v) for v in AC.values() if len(v) >= 2])
wstd_ca = np.mean([np.std(v) for v in CA.values() if len(v) >= 2])
print(f"  Δac 역내 평균 std={wstd_ac:.0f}s (전체 std={allac.std():.0f}) → 역별 일관성")
print(f"  Δca 역내 평균 std={wstd_ca:.0f}s (전체 std={allca.std():.0f})")

# JSON 테이블 저장 (필터가 로드)
table = {"dir_station_ac": {f"{d}|{s}": [float(np.mean(v)), float(np.std(v)), len(v)] for (d, s), v in AC.items()},
         "dir_seg_ca": {f"{d}|{s}|{n}": [float(np.mean(v)), float(np.std(v)), len(v)] for (d, s, n), v in CA.items()},
         "global": {"ac_med": float(np.median(allac)), "ac_std": float(allac.std()),
                    "ca_med": float(np.median(allca)), "ca_std": float(allca.std())}}
out_json = "reports/path2_align/timing_table.json"
os.makedirs(os.path.dirname(out_json), exist_ok=True)
json.dump(table, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nsaved {out_json}")

# 분포 plot
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for a, data, ttl in zip(ax, (allac, allca), ("announce->close (Dac)", "close->next announce (Dca)")):
    a.hist(data, bins=20, color="#4a8aa0", edgecolor="white")
    a.axvline(np.median(data), color="red", ls="--", label=f"med {np.median(data):.0f}s")
    a.set_title(ttl); a.set_xlabel("seconds"); a.legend(); a.grid(alpha=0.3)
fig.tight_layout(); out = "reports/path2_align/timing_table.png"
fig.savefig(out, dpi=110); print(f"saved {out}")
