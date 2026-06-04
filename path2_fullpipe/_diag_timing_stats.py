# -*- coding: utf-8 -*-
"""세 신호 시간 문법 종합 — 이번역은(KWS) → 문열림(open) → 문닫힘(close).
한 역 방문의 내부 타이밍(Δ 안내→열림, dwell 열림→닫힘)과 역간 타이밍(닫힘→다음 안내)을
측정해, "진짜 역 = 이 패턴이 이 간격으로 줄줄이" 라는 false 필터 template을 만든다.
엔드포인트(성대/구로)는 절단이라 양끝은 빠질 수 있음."""
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
R_OF = PL.R_OF
trips = PL.list_trips()
PAIR_MAX = 130.0   # 안내는 문열림보다 최대 이만큼 전 (그 이상이면 짝 아님)


def visits(tid):
    """한 역 방문 = (station, t_announce, t_open, t_close). 각 dwell(open,close)을
    직전 안내(가장 가까운, PAIR_MAX 이내)와 짝지음."""
    d = os.path.join(PL.LIVE_DIR, tid)
    info = json.loads(open(os.path.join(d, "marks.json"), encoding="utf-8").read())
    segs = info.get("segments") or [{"direction": info.get("direction", "?"), "marks": info.get("marks", [])}]
    direction = "등교" if "등" in segs[0].get("direction", tid) else "하교"
    raw = sorted(((m["station"], m["sample_index"] / SR)
                  for s in segs for m in s.get("marks", []) if m["station"] in R_OF),
                 key=lambda x: x[1])
    marks = []                                   # dedup (환승역 2번 방송 등 <10s 중복 제거)
    for st, t in raw:
        if not marks or t - marks[-1][1] > 10:
            marks.append((st, t))
    _, opens, closes, dwells, _ = PL.load_door(d)
    V = []
    for o, c in dwells:
        to, tc = o / SR, c / SR
        cand = [(st, t) for st, t in marks if 0 <= to - t <= PAIR_MAX]
        if cand:
            st, ta = cand[-1]                    # 문열림 직전, 가장 가까운 안내
            V.append((st, ta, to, tc))
    V.sort(key=lambda x: x[2])
    return direction, V


AO, DW, MOVE = [], [], []     # 안내→열림, 열림→닫힘(dwell), 닫힘→다음안내
print("=== 역 방문 내부 타이밍 (트립별 평균) ===")
print(f"  {'trip':22s} {'n역':>3s} {'안내→열림':>8s} {'dwell':>6s} {'닫힘→다음안내':>10s}")
for tid in trips:
    direction, V = visits(tid)
    ao = [to - ta for _, ta, to, _ in V]
    dw = [tc - to for _, _, to, tc in V]
    mv = [V[i + 1][1] - V[i][3] for i in range(len(V) - 1)]   # close[i] → announce[i+1]
    AO += ao; DW += dw; MOVE += mv
    print(f"  {tid:22s} {len(V):3d} {np.mean(ao):8.0f} {np.mean(dw):6.0f} "
          f"{(np.mean(mv) if mv else 0):10.0f}")

AO, DW, MOVE = np.array(AO), np.array(DW), np.array(MOVE)
def stat(x): return f"mean={x.mean():.0f} med={np.median(x):.0f} std={x.std():.0f} range[{x.min():.0f},{x.max():.0f}]"
print("\n=== 종합 시간 문법 (n=%d 역방문) ===" % len(AO))
print(f"  ① 이번역은 → 문열림 : {stat(AO)}")
print(f"  ② 문열림 → 문닫힘(dwell): {stat(DW)}")
print(f"  ③ 문닫힘 → 다음 이번역은: {stat(MOVE)}")
print(f"\n  ▶ 한 역 template:  [이번역은] +{np.median(AO):.0f}s→ [열림] +{np.median(DW):.0f}s→ [닫힘]"
      f"   ─{np.median(MOVE):.0f}s→ 다음 [이번역은]")
print(f"  ▶ 한 사이클(닫힘→닫힘) ≈ {np.median(MOVE)+np.median(AO)+np.median(DW):.0f}s")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for a, data, ttl in zip(ax, (AO, DW, MOVE),
                        ("(1) announce -> open", "(2) open -> close (dwell)", "(3) close -> next announce")):
    a.hist(data, bins=18, color="#4a8aa0", edgecolor="white")
    a.axvline(np.median(data), color="red", ls="--", label=f"med {np.median(data):.0f}s")
    a.set_title(ttl); a.set_xlabel("seconds"); a.legend(); a.grid(alpha=0.3)
fig.suptitle("3-signal timing grammar:  announce -> open -> close -> (move) -> announce")
fig.tight_layout()
out = "reports/path2_align/timing_grammar.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=110)
print(f"\nsaved {out}")
