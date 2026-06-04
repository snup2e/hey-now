"""Shared dataset builder for the 13-class Path 2 (live cabin) KWS + CNN models.

Measured fact: the Path-1 model trained on clean Seoul-Metro audio fires on
*zero* announcements in the recorded live trips (domain gap: cabin reverb,
mic response, train noise, PA EQ).  Level-normalisation alone does not help.
So a live-domain model must be trained, and we bridge the gap with three
sources mixed at the *real* live amplitude:

  1. CLEAN station announcements (Path-1 wavs) -- well localised, correctly
     labelled, plentiful -- synthesised into the cabin domain by adding real
     train-noise at a target SNR (+ optional light reverb).
  2. REAL live cabin-PA announcements from the trips -- true domain character
     but few and only roughly marked (friend taps ~+-3 s, sometimes worse).
  3. REAL train-noise windows as KWS negatives -- the thing the clean model
     never saw, and the cheapest part of the gap to close.

Both scripts/path2_poc.py (local CPU sanity check) and the Colab training
notebook import from here, so there is exactly one source of truth.

Key amplitude rule: mel features use power_to_db(ref=1.0), i.e. *absolute*
level matters.  We therefore synthesise at the real noise level (clean scaled
to noise_rms * 10**(snr/20)) and feed RAW audio at inference -- no level
normalisation anywhere -- so training and inference see the same dB range.
"""
from __future__ import annotations

import glob
import json
import os
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:                                   # librosa only needed at build time
    import librosa
except ImportError:                    # pragma: no cover
    librosa = None

SR = 16000
N_MELS, N_FFT, MEL_HOP = 40, 512, 256
KWS_WIN = 1.0
CNN_WIN = 2.0

# 13 target stations (Path-1's 14 minus 신도림 -- live route ends at 구로).
# Sorted so the CNN label index is deterministic and matches path1_meta.json.
TARGET13 = sorted([
    "가산디지털단지", "관악", "구로", "군포", "금정", "금천구청", "당정",
    "독산", "명학", "석수", "성균관대", "안양", "의왕",
])
LABEL_IDX = {s: i for i, s in enumerate(TARGET13)}

# Guard radius around a mark when harvesting "pure" train noise (s).
NOISE_GUARD_S = 18.0

# Direction -> boarding station, whose mark is BOGUS: the rider is already
# aboard, so the "이번역은 ~역" announcement played on the platform before
# boarding and is NOT in the recording. The capture UI listed all 13 stations,
# so the friend tapped an arbitrary point. Drop it from positives/labels; its
# region falls into the noise pool. The terminal/arrival mark at the far end IS
# real -> 구로 has real onboard samples only from 하교, 성대 only from 등교.
BOARDING = {"등교": "구로", "하교": "성균관대"}

# CNN classification uses only the 1st calling "이번역은 [primary name]" =
# [trigger_onset, +CNN_WIN]. Live KORAIL audio adds a secondary name (부역명,
# e.g. 마리오아울렛) absent from the 서울교통공사 clean source; the primary name
# alone separates all 13 stations. Measured clean structure (trigger onset = 0):
#   "이번역은" ~[0,0.7] | pause | primary name ~[0.9,1.8] | long pause | "역입니다".
# A fixed 2.0 s window captures "이번역은 [primary]" and ends before 부역명; the
# longest name (가산디지털단지) is truncated to "가산디지" but consistently in both
# training and inference, and stays unique. Fixed beats per-station-variable
# because inference does not yet know the station.


# --------------------------------------------------------------------------- #
# low-level audio / feature helpers
# --------------------------------------------------------------------------- #
def load_wav_float(path: str | Path) -> np.ndarray:
    """16 kHz mono float32 in [-1, 1] (matches librosa.load used at inference)."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR, f"sr != {SR}: {path}"
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# Per-window cepstral/spectral mean normalisation.  power_to_db(ref=1.0) is an
# ABSOLUTE scale, so low-level live audio lands far from clean training data
# (this is why the clean model fired on 0 live announcements).  Subtracting
# each window's per-mel-bin temporal mean removes the stationary train-rumble
# profile and the absolute level, leaving the announcement's spectral *change*
# -- level-invariant and domain-robust.  The firmware melspec must replicate
# this (subtract each mel bin's mean across the window's frames).
USE_CMN = True

# Feature front-end selector (Lever experiments).
#   "logmel_cmn" -- baseline: power_to_db(ref=1.0) + per-window CMN.
#   "pcen"       -- Per-Channel Energy Normalisation: an IIR-smoothed AGC that
#                   divides each mel bin by a slow running average of its own
#                   energy, then root-compresses. The division is a per-channel
#                   gain control with a time constant, so it suppresses the
#                   per-trip PA colour / level (the cross-trip channel) more
#                   aggressively than a static per-window mean subtraction.
#   "logmel_cmn_harmonic" -- HPSS-harmonic isolation for the 출발 CHIME (삐리리).
#                   The chime is a sustained TONE (horizontal lines in the mel
#                   spectrogram); the door-slide 치이익 is broadband noise and the
#                   탁 is a transient (vertical). librosa.decompose.hpss with a
#                   median filter keeps the harmonic (tonal) component; margin>1
#                   pushes the broadband 치이익 into a dropped residual. So the
#                   chime survives but the seat-position-dependent contamination
#                   (치이익 present near doors / absent mid-car) is removed -> a
#                   channel-robust chime feature. Run on the MEL power (cheap: two
#                   median filters on a 40xF array, ~µs; laptop inference per
#                   CLAUDE.md, so cost is a non-issue). For an eventual F411 port
#                   the time-axis median needs a small lookahead frame buffer.
# CRITICAL: this selects BOTH the training feature (to_logmel) and the inference
# feature (window_feature) at once -- never change one without the other, or the
# old "0 live triggers" train/inference mismatch returns. librosa.pcen defaults,
# identical params for train and inference. The IIR smoother + pow are CMSIS-DSP
# portable (see melspec.c snippet).
FEATURE_MODE = os.environ.get("PATH2_FEATURE", "logmel_cmn")
PCEN_GAIN, PCEN_BIAS, PCEN_POWER = 0.98, 2.0, 0.5
PCEN_TIME_CONST, PCEN_EPS = 0.4, 1e-6
# librosa's PCEN defaults (eps, phantom init=1.0) assume the input is scaled to
# ~[0, 2**31] (see its docstring); our float [-1,1] audio yields mel power with
# median ~1e-4 (clean) / ~3e-6 (live), so eps and the phantom init dominate and
# the AGC never enters its intended regime. PCEN_SCALE lifts mel power into that
# regime (M >> eps) before PCEN. Identical for train and inference.
PCEN_SCALE = float(os.environ.get("PATH2_PCEN_SCALE", "1.0"))
# HPSS (harmonic isolation). margin>1 drops the broadband residual (치이익);
# kernel = (harmonic time-median, percussive freq-median); freq kernel < N_MELS.
HPSS_MARGIN = float(os.environ.get("PATH2_HPSS_MARGIN", "3.0"))
HPSS_KERNEL = (31, 17)


def _mel_power(y: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=MEL_HOP, n_mels=N_MELS)


def feat_from_mel_power(mel: np.ndarray, mode: str | None = None) -> np.ndarray:
    """Mel power spectrogram -> model-input feature (FEATURE_MODE selected)."""
    mode = mode or FEATURE_MODE
    if mode == "pcen":
        return librosa.pcen(
            mel * PCEN_SCALE, sr=SR, hop_length=MEL_HOP, gain=PCEN_GAIN,
            bias=PCEN_BIAS, power=PCEN_POWER, time_constant=PCEN_TIME_CONST,
            eps=PCEN_EPS
        ).astype(np.float32)
    if mode == "logmel_cmn_harmonic":
        # keep tonal chime; drop transient 탁 (percussive) + broadband 치이익 (residual)
        mel, _ = librosa.decompose.hpss(mel, kernel_size=HPSS_KERNEL, margin=HPSS_MARGIN)
    logmel = librosa.power_to_db(mel, ref=1.0).astype(np.float32)
    if USE_CMN:
        logmel = logmel - logmel.mean(axis=1, keepdims=True)
    return logmel


def to_logmel(y: np.ndarray, cmn: bool = USE_CMN) -> np.ndarray:
    """Raw audio window -> model-input feature (see FEATURE_MODE)."""
    return feat_from_mel_power(_mel_power(y))


def window_feature(y: np.ndarray, center_sample: int, jitter_sec: float = 0.0,
                   mode: str | None = None) -> np.ndarray:
    """CNN_WIN feature at a sample index (inference path).

    Cuts the SAME [mark, mark+CNN_WIN] window from raw audio that build_cnn /
    build_metric_pool cut at training time, then applies the SAME feature, so
    train and inference are byte-for-byte consistent under either FEATURE_MODE.
    """
    cw = int(CNN_WIN * SR)
    a = max(0, int(round(center_sample + jitter_sec * SR)))
    clip = y[a:a + cw]
    if len(clip) < cw:
        clip = np.pad(clip, (0, cw - len(clip)))
    return feat_from_mel_power(_mel_power(clip), mode)


def make_windows(y: np.ndarray, win_sec: float, hop_sec: float) -> list[np.ndarray]:
    w, h = int(win_sec * SR), int(hop_sec * SR)
    if len(y) < w:
        y = np.pad(y, (0, w - len(y)))
    return [y[i:i + w] for i in range(0, len(y) - w + 1, h)]


def slice_sec(y: np.ndarray, t0: float, t1: float) -> np.ndarray:
    a, b = int(t0 * SR), int(t1 * SR)
    seg = y[max(0, a):min(len(y), b)]
    if len(seg) < b - a:
        seg = np.pad(seg, (0, b - a - len(seg)))
    return seg


def find_onset(y: np.ndarray, t_start: float = 3.0, t_end: float = 6.5) -> float:
    """Energy-onset finder for clean 환승 sources (same as the Path-1 notebook)."""
    frame, hop = 400, 160
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + np.arange(n)[:, None] * hop
    env = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-10)
    thr = env.max() * 0.08
    s, e = int(t_start * SR / hop), min(int(t_end * SR / hop), n)
    for i in range(max(s, 1), e):
        if env[i] > thr and env[i - 1] <= thr:
            return i * hop / SR
    return t_start


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #
def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2) + 1e-12))


def add_noise_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix real train `noise` onto `clean` at `snr_db`, keeping `clean`'s level.

    We scale the noise (not the announcement) so the speech stays intelligible
    and learnable; CMN in to_logmel removes the absolute level afterwards, so
    matching the live audio's low amplitude is unnecessary -- only the spectral
    texture / SNR of the real train noise matters.
    """
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]
    target_n = rms(clean) / (10 ** (snr_db / 20.0))   # desired noise RMS
    noise = noise * (target_n / rms(noise))
    return (clean + noise).astype(np.float32)


def light_reverb(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap exponential-decay reverb (RT60 ~0.15-0.45 s) for cabin colour."""
    rt60 = rng.uniform(0.15, 0.45)
    n = int(rt60 * SR)
    ir = (rng.standard_normal(n) * np.exp(-np.arange(n) / (rt60 * SR / 3))).astype(np.float32)
    ir[0] = 1.0
    wet = np.convolve(y, ir, mode="full")[:len(y)]
    mix = rng.uniform(0.15, 0.45)
    out = (1 - mix) * y + mix * wet
    return out.astype(np.float32)


def spec_augment(mel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mel = mel.copy()
    f, t = mel.shape
    floor = float(mel.min())
    for _ in range(int(rng.integers(0, 3))):
        wv = int(rng.integers(1, max(2, t // 8)))
        s = int(rng.integers(0, max(1, t - wv)))
        mel[:, s:s + wv] = floor
    for _ in range(int(rng.integers(0, 3))):
        wv = int(rng.integers(1, max(2, f // 6)))
        s = int(rng.integers(0, max(1, f - wv)))
        mel[s:s + wv, :] = floor
    return mel


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
@dataclass
class CleanSource:
    """One clean station wav, aligned to its "이번역은" onset.

    All spans are measured forward from `onset` so clean and live clips are cut
    by the SAME rule (live mark = "이번역은" onset). See the windowing note above.
    """
    station: str
    y: np.ndarray
    onset: float       # "이번역은" onset (s)
    transfer: bool

    def kws_pos(self) -> np.ndarray:        # "이번역은" trigger phrase (label 1)
        return slice_sec(self.y, self.onset - 0.3, self.onset + 1.3)

    def kws_neg(self) -> np.ndarray:        # name/"역입니다" tail = NOT a trigger
        return slice_sec(self.y, self.onset + 1.2, self.onset + 4.2)

    def name_clip(self) -> np.ndarray:      # 1st calling [onset, onset+CNN_WIN]
        return slice_sec(self.y, self.onset, self.onset + CNN_WIN)


def load_clean_sources(clean_dir: str | Path,
                       target: list[str] = TARGET13) -> list[CleanSource]:
    """One CleanSource per target-station wav, aligned to its "이번역은" onset."""
    out: list[CleanSource] = []
    for path in sorted(glob.glob(str(Path(clean_dir) / "*.wav"))):
        stem = Path(path).stem
        station = stem.split("_")[0]
        if station not in target:
            continue
        y = load_wav_float(path)
        transfer = "환승" in stem
        # Transfer clips carry a preceding announcement; "이번역은" starts ~4 s in.
        onset = find_onset(y, 3.0, 6.5) if transfer else find_onset(y, 0.0, 2.0)
        out.append(CleanSource(station, y, onset, transfer))
    return out


@dataclass
class LiveTrip:
    trip_id: str
    direction: str
    y: np.ndarray
    marks: list[tuple[str, int]]                      # (station, mark=onset); boarding dropped
    noise_segs: list[np.ndarray] = field(repr=False)  # contiguous pure-noise segments


def load_live_trip(trip_dir: str | Path) -> LiveTrip:
    trip_dir = Path(trip_dir)
    info = json.loads((trip_dir / "marks.json").read_text(encoding="utf-8"))
    y = load_wav_float(trip_dir / "audio.wav")
    if "segments" in info:
        segs_meta = info["segments"]
    else:
        segs_meta = [{"direction": info.get("direction", "?"), "marks": info.get("marks", [])}]
    direction = segs_meta[0].get("direction", "?")
    boarding = BOARDING.get(direction)
    marks: list[tuple[str, int]] = []
    for s in segs_meta:
        for m in s["marks"]:
            st = m["station"]
            if st in LABEL_IDX and st != boarding:     # drop bogus boarding mark
                marks.append((st, int(m["sample_index"])))
    # pure-noise = everything outside +-NOISE_GUARD_S of a (real) announcement.
    guard = int(NOISE_GUARD_S * SR)
    keep = np.ones(len(y), bool)
    for _, idx in marks:
        keep[max(0, idx - guard):min(len(y), idx + guard)] = False
    # Contiguous True runs -> segments (sample within one segment to avoid
    # artificial joins, and to keep "same signal + different noise" clean).
    noise_segs: list[np.ndarray] = []
    minlen = int(CNN_WIN * SR)
    i, n = 0, len(keep)
    while i < n:
        if keep[i]:
            j = i
            while j < n and keep[j]:
                j += 1
            if j - i >= minlen:
                noise_segs.append(y[i:j])
            i = j
        else:
            i += 1
    return LiveTrip(info.get("trip_id", trip_dir.name), direction, y, marks, noise_segs)


def rand_noise(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(pool) <= n:
        return np.tile(pool, int(np.ceil(n / max(1, len(pool)))))[:n]
    s = int(rng.integers(0, len(pool) - n))
    return pool[s:s + n]


def sample_noise(segs: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """A length-n real-noise window from a random segment at a random offset."""
    cands = [s for s in segs if len(s) >= n]
    if not cands:
        pool = np.concatenate(segs) if segs else np.zeros(n, np.float32)
        return rand_noise(pool, n, rng)
    seg = cands[int(rng.integers(len(cands)))]
    off = int(rng.integers(0, len(seg) - n + 1))
    return seg[off:off + n]


# --------------------------------------------------------------------------- #
# dataset builders
# --------------------------------------------------------------------------- #
def _synth(clip: np.ndarray, segs: list[np.ndarray], rng: np.random.Generator,
           snr, reverb_p: float, cmn: bool) -> np.ndarray:
    """Fixed clean signal -> cabin domain: (opt reverb) + REAL noise at SNR -> mel.

    The noise is a real driving-noise segment (matched statistics), so the model
    latches onto the fixed announcement signal rather than synthetic-noise
    residue. CMN then removes the absolute level so amplitude need not match.
    """
    y = clip.astype(np.float32)
    if reverb_p and rng.random() < reverb_p:
        y = light_reverb(y, rng)
    if segs:
        noise = sample_noise(segs, len(y), rng)
        y = add_noise_snr(y, noise, rng.uniform(*snr))
    return to_logmel(y, cmn)


def build_kws(clean: list[CleanSource], trips: list[LiveTrip], rng,
              n_synth_pos: int = 24, n_synth_neg: int = 10,
              use_real_pos: bool = True, use_synth: bool = True,
              real_pos_aug: int = 0, neg_ratio: float = 1.0,
              snr=(0.0, 25.0), reverb_p: float = 0.0, cmn: bool = USE_CMN,
              spec_aug: bool = False):
    """KWS dataset. label 1 = "이번역은" (1st calling) present, 0 = not.

    Negatives draw on real train-noise segments, which already contain the
    "다음 역은 ~" next-station preview, so the model learns that is NOT a trigger.
    The 2nd "이번역" of transfer/express stations sits inside the +-guard (not in
    the noise pool) and is handled at inference by the post-trigger cooldown,
    not trained as a negative (too close acoustically to the real trigger).

    spec_aug=False by default: SpecAugment time/freq masking on a 1 s window
    often erases the short "이번역은" token, turning positives into label noise
    and collapsing the detector (measured: val 69%, ~0.4 constant output, 0
    cross-trip dets). The trigger needs its signal intact; noise/SNR augmentation
    already gives robustness.
    """
    segs = [s for t in trips for s in t.noise_segs]
    X, Y = [], []
    aug = (lambda m: spec_augment(m, rng)) if spec_aug else (lambda m: m)

    # use_synth=False -> real-only positives (drop the 서울교통공사 clean "이번역은",
    # a DIFFERENT recording from the 코레일 live PA -- same clean!=live mismatch
    # that made the classifier real-only; with synth on, ~90% of positives are the
    # wrong recording, so the KWS learns a fuzzy "announcement voice" and fires on
    # chatter/door/KTX cross-trip). real_pos_aug>0 then expands the few real
    # positives with extra real-noise mixes. neg_ratio scales negatives vs
    # positives (>1 = more hard non-keyword negatives for the chatter problem).
    if use_synth:
        for src in clean:
            pos, neg = src.kws_pos(), src.kws_neg()
            for _ in range(n_synth_pos):
                for w in make_windows(pos, KWS_WIN, 0.25):
                    X.append(aug(_synth(w, segs, rng, snr, reverb_p, cmn))); Y.append(1)
            for _ in range(n_synth_neg):
                for w in make_windows(neg, KWS_WIN, 0.5):
                    X.append(aug(_synth(w, segs, rng, snr, reverb_p, cmn))); Y.append(0)

    kw = int(KWS_WIN * SR)
    if use_real_pos:
        for t in trips:
            for _, idx in t.marks:
                span = t.y[max(0, idx - int(0.3 * SR)): idx + int(1.3 * SR)]
                for w in make_windows(span, KWS_WIN, 0.25):
                    X.append(to_logmel(w, cmn)); Y.append(1)
                    for _ in range(real_pos_aug):
                        yn = add_noise_snr(w, sample_noise(segs, len(w), rng),
                                           rng.uniform(*snr))
                        X.append(to_logmel(yn, cmn)); Y.append(1)

    # Balance: draw neg_ratio x (positives) real-noise negatives total.
    n_pos = int(np.sum(Y))
    n_neg = len(Y) - n_pos
    for _ in range(max(0, int(neg_ratio * n_pos) - n_neg)):
        X.append(to_logmel(sample_noise(segs, kw, rng), cmn)); Y.append(0)

    X = np.asarray(X, np.float32)[..., None]
    Y = np.asarray(Y, np.int32)
    return X, Y


def build_cnn(clean: list[CleanSource], trips: list[LiveTrip], rng,
              n_synth: int = 60, real_jitter: int = 8, use_real: bool = True,
              snr=(0.0, 25.0), reverb_p: float = 0.0, cmn: bool = USE_CMN):
    """CNN dataset: 2 s "이번역은 [primary name]" window -> 13-class.

    Per fixed clean signal, many real-noise mixes (K=n_synth); real live windows
    (small +-jitter) add true PA/cabin character. Returns (X, Y, pairs) where
    pairs[i]=(label, signal_id) so a later metric-learning stage can pull
    positive pairs (same signal_id, different noise) straight from this builder.
    """
    segs = [s for t in trips for s in t.noise_segs]
    cw = int(CNN_WIN * SR)
    X, Y, pairs = [], [], []

    for src in clean:
        lab, sig, sid = LABEL_IDX[src.station], src.name_clip(), f"clean:{src.station}"
        for _ in range(n_synth):
            X.append(_synth(sig, segs, rng, snr, reverb_p, cmn)); Y.append(lab); pairs.append((lab, sid))

    if use_real:
        for t in trips:
            for station, idx in t.marks:
                lab, sid = LABEL_IDX[station], f"real:{t.trip_id}:{station}"
                for d in np.linspace(-0.2, 0.2, real_jitter):
                    a = max(0, idx + int(d * SR))
                    clip = t.y[a:a + cw]
                    if len(clip) < cw:
                        clip = np.pad(clip, (0, cw - len(clip)))
                    X.append(to_logmel(clip, cmn)); Y.append(lab); pairs.append((lab, sid))

    X = np.asarray(X, np.float32)[..., None]
    Y = np.asarray(Y, np.int32)
    return X, Y, pairs


def build_metric_pool(clean: list[CleanSource], trips: list[LiveTrip], rng,
                      n_synth: int = 60, real_jitter: int = 8,
                      snr=(0.0, 25.0), reverb_p: float = 0.0, cmn: bool = USE_CMN,
                      use_clean: bool = True, real_noise_aug: int = 0,
                      spec_aug: bool = False, jitter_s: float = 0.2):
    """Sample pool for metric learning (encoder + prototype).

    Same 2 s "이번역은 [primary name]" windows as build_cnn, but returns a per-
    sample SOURCE tag so the episode sampler can force cross-source positive
    pairs: "synth" (clean signal + a random real-noise mix -> teaches NOISE
    invariance) vs "real:<trip_id>" (true cabin/PA channel -> teaches CHANNEL
    invariance, the scarce-but-crucial signal for cross-trip generalisation).

    use_clean=False drops the synthesised clean source entirely (live-only). The
    clean 서울교통공사 source is a DIFFERENT recording from the live 코레일 PA
    (verified: clean<->live envelope xcorr ~0.25, CMN log-mel corr ~-0.09), so
    clean+noise asks the encoder to bridge a recording gap, not just a channel,
    and dilutes the episodes with synth<->synth (noise-only) positive pairs.
    Live-only makes EVERY positive pair cross-trip (channel-invariance). To
    offset the smaller pool, real_noise_aug>0 adds that many extra copies of each
    real window mixed with more real cabin noise (stays in the live domain).

    Returns (X[N,40,F,1], Y[N], src[list[str]]).
    """
    segs = [s for t in trips for s in t.noise_segs]
    cw = int(CNN_WIN * SR)
    X, Y, src = [], [], []

    def feat(mel):                       # optional SpecAugment (time/freq masking)
        return spec_augment(mel, rng) if spec_aug else mel

    if use_clean:
        for source in clean:
            lab, sig = LABEL_IDX[source.station], source.name_clip()
            for _ in range(n_synth):
                X.append(feat(_synth(sig, segs, rng, snr, reverb_p, cmn))); Y.append(lab); src.append("synth")

    for t in trips:
        for station, idx in t.marks:
            lab = LABEL_IDX[station]
            for d in np.linspace(-jitter_s, jitter_s, real_jitter):
                a = max(0, idx + int(d * SR))
                clip = t.y[a:a + cw]
                if len(clip) < cw:
                    clip = np.pad(clip, (0, cw - len(clip)))
                X.append(feat(to_logmel(clip, cmn))); Y.append(lab); src.append(f"real:{t.trip_id}")
                for _ in range(real_noise_aug):
                    yn = add_noise_snr(clip, sample_noise(segs, cw, rng), rng.uniform(*snr))
                    X.append(feat(to_logmel(yn, cmn))); Y.append(lab); src.append(f"real:{t.trip_id}")

    X = np.asarray(X, np.float32)[..., None]
    Y = np.asarray(Y, np.int32)
    return X, Y, src
