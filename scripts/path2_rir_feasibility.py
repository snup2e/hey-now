"""Lever 3 feasibility probe -- can we estimate a REAL per-trip channel (RIR)
from the live recordings, to interpolate virtual channels for augmentation?

Two decisive, fast checks before building anything (no training):

  A. Does per-window CMN already kill a static spectral channel (EQ)?
     A channel's magnitude response is a static per-frequency gain on the power
     spectrum = a constant per-mel-bin offset in log-mel = removed exactly by
     CMN (subtracting each bin's temporal mean). If so, ONLY the reverb / time-
     smearing part of a channel can survive CMN, so an EQ-only "RIR" is wash
     (matches CLAUDE.md: random reverb + CMVN/EQ both washed).

  B. Are the clean (서울교통공사) and live (KORAIL) 1st-calling segments the SAME
     underlying recording? RIR/deconvolution needs a valid input->output pair.
     We align [onset,+2s] clean vs [mark,+2s] live by short-time energy-envelope
     cross-correlation and report the peak correlation + lag, plus the CMN'd
     log-mel correlation at best lag. High, sharp, lag~0 -> same recording (RIR
     estimable). Low / scattered -> different content + heavy noise -> a clean
     channel cannot be deconvolved, so real-RIR interpolation is infeasible.

Run:  python scripts/path2_rir_feasibility.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D

CLEAN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "processed", "wav")
LIVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "raw", "line1_live")
TRIP = "20260527_0654_등교"


def envelope(y, frame=320, hop=160):
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + np.arange(n)[:, None] * hop
    e = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    return (e - e.mean()) / (e.std() + 1e-9)


def xcorr_peak(a, b, max_lag):
    """Best normalised cross-correlation of equal-length envelopes within +-lag."""
    best, blag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:len(b) - lag]
        else:
            x, y = a[:len(a) + lag], b[-lag:]
        m = min(len(x), len(y))
        if m < 10:
            continue
        c = float(np.dot(x[:m], y[:m]) / m)
        if c > best:
            best, blag = c, lag
    return best, blag


def cmn_logmel(y):
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=D.SR, n_fft=D.N_FFT,
                                         hop_length=D.MEL_HOP, n_mels=D.N_MELS)
    lm = librosa.power_to_db(mel, ref=1.0)
    return lm - lm.mean(axis=1, keepdims=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import librosa

    # ---- Check A: does CMN remove a static EQ channel? ----
    print("=== Check A: CMN vs a static spectral channel (EQ) ===")
    rng = np.random.default_rng(0)
    clean = D.load_clean_sources(CLEAN_DIR)
    src0 = clean[0]
    clip = src0.name_clip()
    mel = librosa.feature.melspectrogram(y=clip, sr=D.SR, n_fft=D.N_FFT,
                                         hop_length=D.MEL_HOP, n_mels=D.N_MELS)
    eq = np.exp(rng.normal(0, 0.7, size=(D.N_MELS, 1)))          # random static per-bin gain
    f_base = librosa.power_to_db(mel, ref=1.0)
    f_eq = librosa.power_to_db(mel * eq, ref=1.0)
    f_base_cmn = f_base - f_base.mean(axis=1, keepdims=True)
    f_eq_cmn = f_eq - f_eq.mean(axis=1, keepdims=True)
    print(f"  EQ gain spread (dB): {np.ptp(10*np.log10(eq)):.1f} dB across mel bins")
    print(f"  max |logmel diff| BEFORE CMN: {np.abs(f_base - f_eq).max():.3f} dB")
    print(f"  max |logmel diff| AFTER  CMN: {np.abs(f_base_cmn - f_eq_cmn).max():.2e} dB")
    print("  => static EQ is removed by CMN; only reverb/time-smearing can survive.\n")

    # ---- Check B: are clean and live the same recording? ----
    print(f"=== Check B: clean vs live 1st-calling, same recording? (trip {TRIP[9:]}) ===")
    cmap = {}
    for c in clean:
        cmap.setdefault(c.station, c)
    trip = D.load_live_trip(os.path.join(LIVE_DIR, TRIP))
    cw = int(D.CNN_WIN * D.SR)
    max_lag = int(0.4 * D.SR / 160)        # +-0.4 s in envelope frames
    rows = []
    for station, idx in trip.marks:
        if station not in cmap:
            continue
        c = cmap[station]
        clean_clip = c.name_clip()
        live_clip = trip.y[idx:idx + cw]
        if len(live_clip) < cw:
            live_clip = np.pad(live_clip, (0, cw - len(live_clip)))
        ec, el = envelope(clean_clip), envelope(live_clip)
        peak, lag = xcorr_peak(ec, el, max_lag)
        # CMN'd log-mel correlation at best lag (shift live by lag*hop samples)
        shift = lag * 160
        lc = trip.y[max(0, idx + shift):max(0, idx + shift) + cw]
        if len(lc) < cw:
            lc = np.pad(lc, (0, cw - len(lc)))
        a, b = cmn_logmel(clean_clip).ravel(), cmn_logmel(lc).ravel()
        melcorr = float(np.corrcoef(a, b)[0, 1])
        rows.append((station, peak, lag * 160 / D.SR, melcorr))

    print(f"  {'station':14s} {'env_xcorr':>9s} {'lag(s)':>7s} {'logmel_corr(CMN)':>17s}")
    for st, pk, lg, mc in rows:
        print(f"  {st:14s} {pk:9.3f} {lg:7.2f} {mc:17.3f}")
    pks = np.array([r[1] for r in rows]); mcs = np.array([r[3] for r in rows])
    print(f"\n  env_xcorr  mean={pks.mean():.3f}  (1.0=identical envelope, 0=unrelated)")
    print(f"  logmel_corr mean={mcs.mean():.3f}  (high=same spectro-temporal content)")
    print("\n  Interpretation: high+consistent => same recording, RIR estimable;")
    print("  low/scattered => content mismatch + noise => real-RIR deconvolution infeasible.")


if __name__ == "__main__":
    main()
