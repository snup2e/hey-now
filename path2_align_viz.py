#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""path2_align_viz.py — cross-trip window-alignment diagnostic.

목적
----
같은 역의 안내방송은 4트립 모두 *동일한 KORAIL 녹음*이다. 따라서 분류 윈도우
[onset, +2.0s]가 트립마다 정확히 같은 지점을 잡았다면, 4트립의 CMN log-mel은
거의 포개져야 한다. 포개지지 않는다면 원인은 둘 중 하나다:
  (A) 마킹/윈도우 정렬 오차  -> 정밀 마킹으로 고칠 수 있음
  (B) 실제 채널(PA·반향·마이크) 차이 -> 마킹해도 안 고쳐짐

이 스크립트는 (A)와 (B)를 *구분*해 준다:
  1) 역별로 4트립의 voice-band 에너지 엔벨로프를 겹쳐 그려 어긋남을 보여주고,
  2) cross-correlation으로 정렬을 자동 보정한 뒤, 트립 간 분산이 얼마나
     줄어드는지를 역별 막대그래프로 수치화한다.
  분산이 크게 줄면 = 정렬이 범인(정밀 마킹하면 정확도 오름).
  거의 안 줄면 = 진짜 채널차(마킹으로는 한계).

산출물 (reports/path2_align/)
  overview_envelopes.png       13역 엔벨로프 겹쳐보기 (한눈에 스캔)
  align_reduction.png          역별 "정렬 보정 후 분산 감소율" 막대
  suggested_onset_shifts.csv   트립×역 추천 onset 보정량(ms/sample) -> recheck에 사용
  station_<name>.png           역 상세 (--station / --all-details)

실행
  python scripts/path2_align_viz.py                  # overview + 감소율 막대 + csv
  python scripts/path2_align_viz.py --station 의왕    # 한 역 상세
  python scripts/path2_align_viz.py --all-details     # 모든 역 상세

주의: 아래 CONFIG의 LIVE_DIR / TRIPS / STATIONS / marks.json 스키마를 본인 환경에
맞게 확인할 것. 처음 실행 시 각 트립에서 읽은 마크 개수와 역 이름을 출력하니,
들리는 역과 이름이 맞는지 먼저 확인하면 안전하다.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# ---- 비대화형 백엔드 + 한글 폰트 (Windows: Malgun Gothic) --------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    import librosa
except ImportError:
    sys.exit("librosa가 필요합니다:  pip install librosa soundfile")

# 저장소 정식 로더 재사용 (one source of truth): segments 스키마 파싱 + 탑승역
# 가짜 마크 drop + 풀네임(station) 반환. 이 스크립트 원래의 load_marks 는
# {segments:[{marks:[...]}]} 스키마를 못 읽어 마크 0개가 나왔다.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "scripts"))
import path2_dataset as D

# ============================ CONFIG (환경에 맞게 확인) ========================
# cwd 와 무관하게 동작하도록 스크립트 위치 기준 절대경로.
LIVE_DIR = os.path.join(_HERE, "data", "raw", "line1_live")  # <trip_id>/audio.wav + marks.json
OUT_DIR  = os.path.join(_HERE, "reports", "path2_align")

# 트립 정의: id 폴더명, 표시 이름, 방향. 방향은 탑승역(가짜 마크) 제외에만 쓰임.
TRIPS = [
    {"id": "20260527_0654_등교", "label": "0654 등교", "dir": "등교"},
    {"id": "20260528_0642_등교", "label": "0642 등교", "dir": "등교"},
    {"id": "20260527_1431_하교", "label": "1431 하교", "dir": "하교"},
    {"id": "20260528_2118_하교", "label": "2118 하교", "dir": "하교"},
]

# 노선 순서 (구로 -> 성균관대). 역 이름은 marks.json / TARGET13 과 동일한 풀네임.
STATIONS = ["구로", "가산디지털단지", "독산", "금천구청", "석수", "관악",
            "안양", "명학", "금정", "군포", "당정", "의왕", "성균관대"]

# 방향별 탑승역(미녹음·가짜 마크) -> 정렬 분석에서 제외
BOARDING = {"등교": "구로", "하교": "성균관대"}

# 신호처리 (CLAUDE.md와 동일)
SR        = 16000
N_FFT     = 512
HOP       = 256
N_MELS    = 40

# 윈도우/컨텍스트 (초)
WIN_S     = 2.0      # 분류 윈도우 [onset, onset+2.0]
CTX_PRE   = 0.4      # 보기용으로 onset 앞을 더 보여줌
CTX_POST  = 0.4      # 윈도우 끝 뒤를 더 보여줌
MAX_LAG_S = 0.5      # 정렬 보정 탐색 범위 (±0.5s)

# voice-band: 저역 열차 럼블을 피하고 음성 포먼트 강조 (40 mel 중 대략 mid)
VBAND = slice(8, 32)

# 정렬 감소율 임계 (막대 색): 이 이상이면 "정렬이 범인(고칠 수 있음)"
REDUCTION_GREEN = 0.30
# =============================================================================


def frames(n_samples):
    return 1 + n_samples // HOP


def logmel_cmn(y):
    """log-mel + per-window CMN (mel-bin별 시간평균 제거). 모델이 보는 특징."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    lm = librosa.power_to_db(mel, ref=1.0)          # 절대 ref (레벨은 CMN이 처리)
    lm = lm - lm.mean(axis=1, keepdims=True)        # CMN
    return lm.astype(np.float32)                    # [N_MELS, T]


def voice_env(lm):
    """voice-band 평균 -> 1D 엔벨로프 (정렬 비교용, 내용 중심)."""
    e = lm[VBAND, :].mean(axis=0)
    e = e - e.min()
    return e


def load_marks(trip_dir):
    """{station_name: sample_index} 반환 (저장소 정식 로더 위임).

    D.load_live_trip 이 segments 스키마를 파싱하고 방향별 탑승역(가짜 마크)을
    이미 drop 하므로, 풀네임 역 이름과 정밀 sample_index 만 남는다. (원래의 generic
    파서는 이 repo 의 {segments:[...]} 스키마를 못 읽어 마크 0개를 냈다.)
    """
    trip = D.load_live_trip(trip_dir)
    return {station: int(idx) for station, idx in trip.marks}


def extract(y, onset):
    """컨텍스트 클립과 윈도우 클립을 반환. onset은 샘플 인덱스."""
    pre  = int(CTX_PRE * SR)
    win  = int(WIN_S * SR)
    post = int(CTX_POST * SR)
    a = max(0, onset - pre)
    b = min(len(y), onset + win + post)
    ctx = y[a:b]
    # 윈도우 시작이 컨텍스트 내 어디인지 (초)
    win_start_s = (onset - a) / SR
    return ctx, win_start_s


def best_lag(env_ref, env, max_lag_frames):
    """env를 env_ref에 맞추는 lag(frame). +면 env가 늦음 -> 앞으로 당겨야."""
    n = min(len(env_ref), len(env))
    a = env_ref[:n] - env_ref[:n].mean()
    b = env[:n] - env[:n].mean()
    full = np.correlate(a, b, mode="full")
    center = n - 1
    lo = center - max_lag_frames
    hi = center + max_lag_frames + 1
    seg = full[max(0, lo):min(len(full), hi)]
    k = np.argmax(seg) + max(0, lo)
    return k - center  # frame lag


def build():
    """모든 트립·역의 클립과 특징을 모은다."""
    db = {st: [] for st in STATIONS}   # st -> list of dict(trip, ctx, win_start_s, lm_win, env_ctx, onset)
    for tr in TRIPS:
        tdir = os.path.join(LIVE_DIR, tr["id"])
        wav = os.path.join(tdir, "audio.wav")
        if not os.path.isfile(wav):
            print(f"  [skip] {wav} 없음")
            continue
        y, _ = librosa.load(wav, sr=SR, mono=True)
        marks = load_marks(tdir)
        boarding = BOARDING.get(tr["dir"])
        print(f"  {tr['label']}: 마크 {len(marks)}개 -> {list(marks.keys())}")
        for st, onset in marks.items():
            if st == boarding:
                continue  # 가짜 탑승역 마크 제외
            if st not in db:
                print(f"    [warn] 알 수 없는 역 '{st}' (STATIONS에 없음) — 건너뜀")
                continue
            ctx, ws = extract(y, onset)
            lm_ctx = logmel_cmn(ctx)
            env_ctx = voice_env(lm_ctx)
            # 윈도우 부분만 잘라 분산 비교에 사용
            f0 = int(round(ws * SR / HOP))
            f1 = f0 + frames(int(WIN_S * SR))
            lm_win = lm_ctx[:, f0:min(f1, lm_ctx.shape[1])]
            db[st].append({
                "trip": tr["label"], "ctx": ctx, "win_start_s": ws,
                "lm_ctx": lm_ctx, "env_ctx": env_ctx, "lm_win": lm_win,
                "onset": onset, "f0": f0,
            })
    return db


def trip_var(lms):
    """길이를 맞춰 [n, mel, T] 스택 후 트립 간 분산의 평균."""
    if len(lms) < 2:
        return np.nan, None
    T = min(x.shape[1] for x in lms)
    stack = np.stack([x[:, :T] for x in lms], axis=0)
    vmap = stack.var(axis=0)          # [mel, T]
    return float(vmap.mean()), vmap


def refine(db, st):
    """역 st에서 첫 트립을 기준으로 나머지를 정렬, lag(frame) 리스트 반환."""
    items = db[st]
    if len(items) < 2:
        return [0] * len(items)
    ref = items[0]["env_ctx"]
    max_lag_frames = int(round(MAX_LAG_S * SR / HOP))
    lags = [0]
    for it in items[1:]:
        lags.append(best_lag(ref, it["env_ctx"], max_lag_frames))
    return lags


def windows_after_lag(db, st, lags):
    """lag만큼 보정해 다시 자른 윈도우 log-mel 리스트."""
    out = []
    for it, lag in zip(db[st], lags):
        f0 = it["f0"] + lag
        f0 = max(0, f0)
        f1 = f0 + frames(int(WIN_S * SR))
        lm = it["lm_ctx"][:, f0:min(f1, it["lm_ctx"].shape[1])]
        out.append(lm)
    return out


# ----------------------------- plotting --------------------------------------
def plot_overview(db):
    fig, axes = plt.subplots(4, 4, figsize=(18, 12))
    axes = axes.ravel()
    for i, st in enumerate(STATIONS):
        ax = axes[i]
        items = db[st]
        for it in items:
            t = np.arange(len(it["env_ctx"])) * HOP / SR - CTX_PRE
            ax.plot(t, it["env_ctx"], lw=1.2, label=it["trip"], alpha=0.85)
        ax.axvspan(0, WIN_S, color="0.85", zorder=0)   # 분류 윈도우
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{st}  (트립 {len(items)})", fontsize=10)
        ax.set_xlim(-CTX_PRE, WIN_S + CTX_POST)
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for j in range(len(STATIONS), len(axes)):
        axes[j].axis("off")
    fig.suptitle("역별 voice-band 엔벨로프 겹쳐보기 — 포개지면 정렬OK, 어긋나면 마킹 의심\n"
                 "(회색 = 분류 윈도우 [onset, +2.0s], 점선 = onset)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT_DIR, "overview_envelopes.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  저장: {p}")


def plot_reduction(db):
    names, reds = [], []
    rows = []
    for st in STATIONS:
        items = db[st]
        if len(items) < 2:
            names.append(st); reds.append(0.0); continue
        v0, _ = trip_var([it["lm_win"] for it in items])
        lags = refine(db, st)
        v1, _ = trip_var(windows_after_lag(db, st, lags))
        red = 0.0 if (v0 is None or np.isnan(v0) or v0 == 0) else (v0 - v1) / v0
        names.append(st); reds.append(max(0.0, red))
        for it, lag in zip(items, lags):
            rows.append({
                "station": st, "trip": it["trip"],
                "lag_ms": round(lag * HOP / SR * 1000, 1),
                "lag_samples": int(lag * HOP),
                "var_before": round(v0, 3), "var_after": round(v1, 3),
            })

    # csv
    cp = os.path.join(OUT_DIR, "suggested_onset_shifts.csv")
    with open(cp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["station", "trip", "lag_ms",
                                          "lag_samples", "var_before", "var_after"])
        w.writeheader(); w.writerows(rows)
    print(f"  저장: {cp}")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2e7d32" if r >= REDUCTION_GREEN else "#b0bec5" for r in reds]
    ax.bar(names, [r * 100 for r in reds], color=colors)
    ax.axhline(REDUCTION_GREEN * 100, color="#2e7d32", ls="--", lw=0.8)
    ax.set_ylabel("정렬 보정 후 트립간 분산 감소율 (%)")
    ax.set_title("역별: 윈도우만 정밀 정렬하면 트립간 차이가 얼마나 사라지나\n"
                 "초록(높음)=정렬이 범인, 정밀 마킹하면 오름 / 회색(낮음)=진짜 채널차")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "align_reduction.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  저장: {p}")


def plot_station(db, st):
    items = db[st]
    if not items:
        print(f"  [skip] {st}: 클립 없음"); return
    lags = refine(db, st)
    n = len(items)
    fig = plt.figure(figsize=(16, 3.2 * n))
    gs = fig.add_gridspec(n, 3, width_ratios=[2, 1, 1])

    # 왼쪽: 트립별 컨텍스트 스펙트로그램 + 윈도우 박스
    for r, it in enumerate(items):
        ax = fig.add_subplot(gs[r, 0])
        lm = it["lm_ctx"]
        extent = [-CTX_PRE, lm.shape[1] * HOP / SR - CTX_PRE, 0, N_MELS]
        ax.imshow(lm, origin="lower", aspect="auto", extent=extent, cmap="magma")
        ax.add_patch(plt.Rectangle((0, 0), WIN_S, N_MELS, fill=False,
                                   edgecolor="cyan", lw=1.5))
        ax.axvline(0, color="white", lw=0.8, ls="--")
        ax.set_ylabel(it["trip"], fontsize=9)
        if r == 0:
            ax.set_title(f"{st} — CMN log-mel (청록 박스=분류 윈도우)")
        if r == n - 1:
            ax.set_xlabel("시간 (s, onset=0)")

    # 오른쪽 위: 엔벨로프 as-marked
    axa = fig.add_subplot(gs[0, 1])
    for it in items:
        t = np.arange(len(it["env_ctx"])) * HOP / SR - CTX_PRE
        axa.plot(t, it["env_ctx"], lw=1.2, label=it["trip"], alpha=0.85)
    axa.axvspan(0, WIN_S, color="0.88"); axa.axvline(0, color="k", lw=0.7, ls="--")
    axa.set_title("엔벨로프 (보정 전)"); axa.set_yticks([]); axa.legend(fontsize=7)

    # 오른쪽 가운데: 엔벨로프 refined
    axb = fig.add_subplot(gs[1, 1]) if n > 1 else fig.add_subplot(gs[0, 2])
    for it, lag in zip(items, lags):
        t = np.arange(len(it["env_ctx"])) * HOP / SR - CTX_PRE - lag * HOP / SR
        axb.plot(t, it["env_ctx"], lw=1.2, alpha=0.85)
    axb.axvspan(0, WIN_S, color="0.88"); axb.axvline(0, color="k", lw=0.7, ls="--")
    axb.set_title("엔벨로프 (정렬 보정 후)"); axb.set_yticks([])

    # 오른쪽: 분산맵 before/after
    v0, vm0 = trip_var([it["lm_win"] for it in items])
    v1, vm1 = trip_var(windows_after_lag(db, st, lags))
    if vm0 is not None:
        ax0 = fig.add_subplot(gs[min(2, n - 1), 1])
        ax0.imshow(vm0, origin="lower", aspect="auto", cmap="viridis")
        ax0.set_title(f"트립간 분산 (보정 전)  mean={v0:.2f}", fontsize=9)
        ax0.set_xticks([]); ax0.set_yticks([])
    if vm1 is not None:
        ax1 = fig.add_subplot(gs[min(2, n - 1), 2])
        ax1.imshow(vm1, origin="lower", aspect="auto", cmap="viridis")
        red = 0 if not v0 else (v0 - v1) / v0 * 100
        ax1.set_title(f"분산 (보정 후)  mean={v1:.2f}  (-{red:.0f}%)", fontsize=9)
        ax1.set_xticks([]); ax1.set_yticks([])

    # lag 안내
    lag_txt = "  ".join(f"{it['trip']}:{lag*HOP/SR*1000:+.0f}ms"
                        for it, lag in zip(items, lags))
    fig.text(0.5, 0.005, f"추천 onset 보정: {lag_txt}", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    p = os.path.join(OUT_DIR, f"station_{st}.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  저장: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", help="한 역만 상세 (예: 의왕)")
    ap.add_argument("--all-details", action="store_true", help="모든 역 상세")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("트립 로드:")
    db = build()

    if args.station:
        plot_station(db, args.station)
    elif args.all_details:
        for st in STATIONS:
            plot_station(db, st)
        plot_overview(db); plot_reduction(db)
    else:
        plot_overview(db); plot_reduction(db)
    print("완료.")


if __name__ == "__main__":
    main()
