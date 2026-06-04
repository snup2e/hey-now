# -*- coding: utf-8 -*-
"""trip4 차임 변종 분석 v2 — 차임 톤만 HPSS 격리 → 시간-주파수 패널 + 톤 주파수 정량.
라벨은 ASCII(N1..N8)로(한글 폰트 회피)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import path2_dataset as D, path2_pipeline as PL
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PL.LIVE_DIR = "data/raw/line1_live"
SR = D.SR
CHIME_S = 1.3
LO, HI = 400, 5000          # chime tonal band (skip low-freq door rumble)
trips = PL.list_trips()


def best_chime(tid):
    """가장 깨끗한(차임 대역 에너지 최대) close 구간 1개의 HPSS-harmonic STFT."""
    d = os.path.join(PL.LIVE_DIR, tid)
    y = D.load_wav_float(os.path.join(d, "audio.wav"))
    closes = PL.load_event_marks(d, "close")
    best, bestE = None, -1
    for c in closes:
        w = y[c:c + int(CHIME_S * SR)]
        if len(w) < int(CHIME_S * SR):
            continue
        S = np.abs(librosa.stft(w, n_fft=2048, hop_length=256))
        H, _ = librosa.decompose.hpss(S, margin=3.0)         # isolate tones
        f = librosa.fft_frequencies(sr=SR, n_fft=2048)
        band = (f >= LO) & (f <= HI)
        E = H[band].sum()
        if E > bestE:
            bestE, best = E, (H, f)
    return best


fig, axes = plt.subplots(2, 4, figsize=(16, 7))
print(f"{'N  trip':26s} {'chime tone freqs (Hz, HPSS-isolated, top by band-energy)'}")
note = {}
f = None
for i, tid in enumerate(trips):
    H, f = best_chime(tid)
    band = (f >= LO) & (f <= HI)
    fb = f[band]
    Hb = H[band]
    # dominant tone per frame, then the set of notes (peaks of time-summed band spectrum)
    spec = Hb.sum(axis=1)
    order = np.argsort(spec)[::-1]
    peaks, used = [], []
    for j in order:
        if all(abs(fb[j] - fb[u]) > 120 for u in used):
            used.append(j); peaks.append(fb[j])
        if len(peaks) == 4:
            break
    note[tid] = sorted(peaks)
    is4 = tid.endswith("2118_하교")
    tag = f"N{i+1} {tid[9:13]}{'(하교 변종)' if is4 else ''}"
    print(f"N{i+1} {tid:22s} {[round(p) for p in sorted(peaks)]}{'   <-- trip4' if is4 else ''}")
    ax = axes[i // 4, i % 4]
    ax.imshow(librosa.amplitude_to_db(H[band], ref=np.max), origin="lower",
              aspect="auto", extent=[0, CHIME_S, LO, HI], cmap="magma")
    ax.set_title(f"N{i+1} {'TRIP4 variant' if is4 else tid[9:13]}",
                 color="red" if is4 else "black", fontsize=10)
    ax.set_ylabel("Hz", fontsize=7)
fig.suptitle("Chime (HPSS-harmonic) spectrogram per trip  —  N4 = trip4 variant", fontsize=12)
fig.tight_layout()
out = "reports/path2_align/chime_variant.png"
fig.savefig(out, dpi=110)
print(f"\nsaved {out}")

# pairwise: trip4's note set vs others — pitch ratio
print("\ntrip4(N4) 최강 톤 vs 다른 트립 최강 톤 (Hz) + 비율:")
t4 = note[trips[3]]
ref = np.median([note[t][-1] for t in trips if not t.endswith('2118_하교')])  # others' top tone
print(f"  others top-tone median ~{ref:.0f} Hz | trip4 top-tone {t4[-1]:.0f} Hz | "
      f"ratio {t4[-1]/ref:.2f}x ({12*np.log2(t4[-1]/ref):+.1f} semitones)")
