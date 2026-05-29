"""Evaluation report for the Path 2 station classifier (matplotlib, local).

Uses the cached per-mark cosine-to-prototype scores from path2_seqprior_poc.py
(data/processed/seqprior_emits.npz) — no retraining. Produces, for the seeded
trip-LOO held-out trips:

  1) per-trip probability heatmaps  [12 marks x 13 stations]  -> "추론당 확률값"
     (예: 금천구청 마크를 금천구청 52% / 구로 28% ... 로 추정)
  2) confusion matrix (per-mark argmax) AND sequence-decoded, route-ordered so
     adjacent-station (off-by-one) confusion sits next to the diagonal
  3) per_mark_topk.csv + summary.txt  (which live audio, each mark's top-3)

Probability = softmax(2*cosine): for L2-normalised embeddings the ProtoNet
logit is -||q-p||^2 = 2*cos - 2, so the model's implied probability is
softmax(2*cos). (Temperature is principled, not hand-picked.)

Run:  python scripts/path2_eval_report.py     (after path2_seqprior_poc.py)
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D     # no TensorFlow; librosa only

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

# Korean font (Windows: Malgun Gothic). Fall back gracefully.
for _f in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim", "Batang"]:
    if any(_f.lower() == x.name.lower() for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "processed", "seqprior_emits.npz")
LIVE_DIR = os.path.join(ROOT, "data", "raw", "line1_live")
OUT = os.path.join(ROOT, "reports", "path2_eval")
TRIPS = ["20260527_0654_등교", "20260528_0642_등교",
         "20260527_1431_하교", "20260528_2118_하교"]

# Physical line order (구로 end -> 성균관대 end); makes adjacency = near-diagonal.
ROUTE = ["구로", "가산디지털단지", "독산", "금천구청", "석수", "관악", "안양",
         "명학", "금정", "군포", "당정", "의왕", "성균관대"]
R_OF = {s: i for i, s in enumerate(ROUTE)}
LAB_OF_ROUTE = np.array([D.LABEL_IDX[s] for s in ROUTE])   # route pos -> TARGET13 label
SHORT = {"가산디지털단지": "가산"}                          # shorten long label for plots


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def offset_decode(P_route, direction):
    """Best consecutive run (start-offset search) over route-ordered scores."""
    N = len(P_route); d = +1 if direction == "등교" else -1
    best = None
    for s0 in range(13):
        pos = s0 + d * np.arange(N)
        if pos.min() < 0 or pos.max() > 12:
            continue
        score = float(P_route[np.arange(N), pos].sum())
        if best is None or score > best[0]:
            best = (score, pos)
    return best[1]


def mark_times(trip_id):
    """Time-ordered (station, t_sec) — same order emissions() used."""
    trip = D.load_live_trip(os.path.join(LIVE_DIR, trip_id))
    ms = sorted(trip.marks, key=lambda x: x[1])
    return trip.direction, [(st, idx / D.SR) for st, idx in ms]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not os.path.exists(CACHE):
        print(f"missing {CACHE} — run: python scripts/path2_seqprior_poc.py")
        return
    os.makedirs(OUT, exist_ok=True)
    d = np.load(CACHE, allow_pickle=True)

    rlabels = [SHORT.get(s, s) for s in ROUTE]
    cm_mark = np.zeros((13, 13), int)        # rows=GT route pos, cols=pred route pos
    cm_seq = np.zeros((13, 13), int)
    rows_csv = [["trip", "time_s", "GT", "pred(per-mark)", "correct",
                 "top1", "p1", "top2", "p2", "top3", "p3"]]
    summary = []

    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    for ax, trip in zip(axes.ravel(), TRIPS):
        E = d[trip]                          # (N,13) cosine, TARGET13 order
        direction = str(d[trip + "_dir"])
        gt_route = d[trip + "_gt"].astype(int)
        _, mt = mark_times(trip)
        N = len(E)

        P = softmax(2.0 * E, axis=1)                 # ProtoNet implied prob, TARGET13 order
        P_route = P[:, LAB_OF_ROUTE]                 # reorder cols -> route order
        pred_route = P_route.argmax(1)
        seq_route = offset_decode(P_route, direction)

        ok_m = int((pred_route == gt_route).sum())
        ok_s = int((seq_route == gt_route).sum())
        summary.append(f"{trip}  ({direction}) — held-out live audio: "
                       f"data/raw/line1_live/{trip}/audio.wav, {N} announcements; "
                       f"per-mark {ok_m}/{N} ({ok_m/N*100:.0f}%), "
                       f"sequence {ok_s}/{N} ({ok_s/N*100:.0f}%)")
        for t in range(N):
            cm_mark[gt_route[t], pred_route[t]] += 1
            cm_seq[gt_route[t], seq_route[t]] += 1
            order = np.argsort(P_route[t])[::-1]
            gt_st, ts = mt[t]
            top = [(rlabels[order[k]], P_route[t, order[k]]) for k in range(3)]
            rows_csv.append([trip[9:], f"{ts:.1f}", gt_st, ROUTE[pred_route[t]],
                             int(pred_route[t] == gt_route[t]),
                             ROUTE[order[0]], f"{top[0][1]:.2f}",
                             ROUTE[order[1]], f"{top[1][1]:.2f}",
                             ROUTE[order[2]], f"{top[2][1]:.2f}"])

        # ---- per-trip probability heatmap ----
        im = ax.imshow(P_route, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(13)); ax.set_xticklabels(rlabels, rotation=90, fontsize=8)
        ylab = [f"{ts:5.0f}s {SHORT.get(st, st)}" for st, ts in mt]
        ax.set_yticks(range(N)); ax.set_yticklabels(ylab, fontsize=8)
        for t in range(N):
            ax.add_patch(Rectangle((gt_route[t] - .5, t - .5), 1, 1,   # GT = white box
                                   fill=False, edgecolor="white", lw=2))
            pr = pred_route[t]
            ax.text(pr, t, "✓" if pr == gt_route[t] else "✗",
                    ha="center", va="center",
                    color=("lime" if pr == gt_route[t] else "red"), fontsize=9, fontweight="bold")
        ax.set_title(f"{trip[9:]} ({direction})  per-mark {ok_m}/{N}={ok_m/N*100:.0f}%, "
                     f"seq {ok_s}/{N}={ok_s/N*100:.0f}%\n흰박스=정답역  ✓/✗=모델 top1",
                     fontsize=10)
        ax.set_xlabel("예측 확률 (노선 순서: 구로→성균관대)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.025).set_label("softmax(2·cos)")
    fig.suptitle("Path 2 추론별 역 확률 (트립단위 LOO held-out, encoder=real-only+aug+GRL)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "prob_heatmaps.png"), dpi=130)
    plt.close(fig)

    # ---- confusion matrices ----
    tot = cm_mark.sum()
    for cm, name, title in [(cm_mark, "confusion_permark.png", "per-mark argmax"),
                            (cm_seq, "confusion_sequence.png", "sequence-decoded (offset-run)")]:
        acc = np.trace(cm) / tot * 100
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(13)); ax.set_xticklabels(rlabels, rotation=90, fontsize=9)
        ax.set_yticks(range(13)); ax.set_yticklabels(rlabels, fontsize=9)
        ax.set_xlabel("예측 역"); ax.set_ylabel("실제 역 (GT)")
        for i in range(13):
            for j in range(13):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                            color="white" if cm[i, j] > cm.max() * 0.5 else "black")
        ax.set_title(f"Confusion — {title}\nLOO 48 marks, 정확도 {np.trace(cm)}/{tot} ({acc:.0f}%)  "
                     f"(노선 순서: 대각선=정답, 대각선 인접=옆역 혼동)", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name), dpi=130)
        plt.close(fig)

    with open(os.path.join(OUT, "per_mark_topk.csv"), "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows_csv)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Path 2 평가 — 재생한 live 음원 & 결과 (트립단위 LOO)\n" + "=" * 60 + "\n")
        f.write("\n".join(summary))
        f.write(f"\n\nper-mark LOO: {np.trace(cm_mark)}/{tot} ({np.trace(cm_mark)/tot*100:.0f}%)")
        f.write(f"\nsequence  LOO: {np.trace(cm_seq)}/{tot} ({np.trace(cm_seq)/tot*100:.0f}%)\n")

    print("저장 완료 ->", OUT)
    for fn in ["prob_heatmaps.png", "confusion_permark.png", "confusion_sequence.png",
               "per_mark_topk.csv", "summary.txt"]:
        print("  ", fn)
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
