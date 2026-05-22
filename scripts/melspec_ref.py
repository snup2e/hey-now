"""log-mel reference — numpy reimplementation verified against librosa.

The firmware computes log-mel on-device, so it must match librosa
(used in training) numerically. This script reimplements the pipeline
with plain numpy, then checks it against librosa. The verified numpy
code is the exact spec for firmware/melspec.c.

Run:  python scripts/melspec_ref.py
"""
import json
from pathlib import Path

import numpy as np
import librosa

ROOT = Path(__file__).resolve().parent.parent
meta = json.loads((ROOT / "models" / "path1_meta.json").read_text(encoding="utf-8"))
SR = meta["sample_rate"]
N_FFT = meta["n_fft"]
N_MELS = meta["n_mels"]
MEL_HOP = meta["mel_hop"]
TOP_DB = 80.0

# librosa mel filterbank — same matrix exported to firmware/mel_filterbank.h
MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)


def hann_periodic(n):
    """librosa/scipy 'hann' window (fftbins=True, periodic)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


WINDOW = hann_periodic(N_FFT).astype(np.float32)


def logmel_numpy(y):
    """Reimplements librosa.power_to_db(melspectrogram(y), ref=1.0).

    This is the algorithm firmware/melspec.c must reproduce:
      1. center-pad by N_FFT/2 (zeros)
      2. frame (N_FFT, hop MEL_HOP), apply Hann window
      3. rfft -> power spectrum |X|^2
      4. mel filterbank matmul
      5. 10*log10, clip to (max - TOP_DB)
    """
    pad = N_FFT // 2
    yp = np.pad(y, pad, mode="constant")
    n_frames = 1 + (len(yp) - N_FFT) // MEL_HOP
    power = np.empty((N_FFT // 2 + 1, n_frames), dtype=np.float64)
    for t in range(n_frames):
        frame = yp[t * MEL_HOP: t * MEL_HOP + N_FFT] * WINDOW
        spec = np.fft.rfft(frame, n=N_FFT)
        power[:, t] = (spec.real ** 2 + spec.imag ** 2)
    mel = MEL_FB @ power
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db = np.maximum(db, db.max() - TOP_DB)
    return db.astype(np.float32)


def logmel_librosa(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=MEL_HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=1.0).astype(np.float32)


def main():
    rng = np.random.default_rng(0)
    # test on a few real announcement windows + noise
    wav_dir = ROOT / "data" / "processed" / "wav"
    samples = []
    for name in ["성균관대.wav", "의왕.wav"]:
        p = wav_dir / name
        if p.exists():
            y, _ = librosa.load(str(p), sr=SR, mono=True)
            samples.append(y[:SR])              # 1 s window (KWS)
            samples.append(y[:2 * SR])          # 2 s window (CNN)
    samples.append(rng.standard_normal(SR).astype(np.float32) * 0.1)

    print(f"{'입력':<22}{'shape':>12}{'max|diff|':>12}{'mean|diff|':>12}")
    print("-" * 58)
    worst = 0.0
    for i, y in enumerate(samples):
        a = logmel_numpy(y)
        b = logmel_librosa(y)
        if a.shape != b.shape:
            print(f"sample {i}: shape 불일치 {a.shape} vs {b.shape}")
            continue
        diff = np.abs(a - b)
        worst = max(worst, diff.max())
        print(f"{'sample '+str(i):<22}{str(a.shape):>12}"
              f"{diff.max():>12.2e}{diff.mean():>12.2e}")
    print("-" * 58)
    if worst < 1e-2:
        print(f"OK — numpy 구현이 librosa와 일치 (최대 오차 {worst:.2e} dB)")
    else:
        print(f"불일치 — 최대 오차 {worst:.2e} dB, melspec 알고리즘 점검 필요")


if __name__ == "__main__":
    main()
