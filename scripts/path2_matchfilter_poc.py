"""Matched-filter "이번역은" detector — cross-trip LOO vs the trained KWS.

Premise (the lever the CNN-KWS throws away): "이번역은" is the SAME 코레일
deterministic recording replayed at every station (12 marks/trip x 8 trips).
Only the additive noise and the channel (distance/reverb/car) change. So this is
not "learn a keyword" but "detect a KNOWN fixed signal in noise" -- the textbook
matched filter, the optimal linear detector for a known template in additive
noise. Unlike the CNN (which learned ~"voice-band energy present" and fires on
잡담 / "다음역은" / KTX, ~56 false/trip), a template correlation measures match to
THIS specific phrase, so phrases with a different spectro-temporal shape should
score low -> fewer false.

Why the old "auto-mark 불가" verdict (archive §, NCC/DTW 4종 실패) does NOT apply:
that used the CLEAN 서울교통공사 template (a DIFFERENT recording, corr ~0.25 with
live) and chased the VARIABLE station name. Here the template is the FIXED prefix
"이번역은", built from the LIVE domain (the trips' own marks).

Method (per fold, trip-leave-one-out):
  1. Template = aligned average of the CMN-log-mel "이번역은" windows at the other
     trips' marks. Averaging denoises (noise averages down, the fixed signal
     stays); a small +-ALIGN_S offset search per instance removes mark jitter
     before averaging. Built in the SAME logmel_cmn front-end the KWS uses.
  2. Detect on the held-out trip: slide the template (0.25 s hop), score each
     window by cosine similarity in the CMN-log-mel space (= normalized cross-
     correlation, level/static-EQ invariant via CMN).
  3. Score with the EXACT same post-processing as the KWS (debounce MINRUN +
     cosine threshold + COOLDOWN, then MATCH_S greedy match to marks) by
     importing path2_kws_recover.{debounce,score,COOLDOWN_S,MATCH_S} -> the
     recall/false numbers are directly comparable to kws_8trip.log.

Streaming-safe (CLAUDE.md rule 6): the template is fixed offline (like weights);
each window's cosine uses no future audio. No training -> the whole 8-fold LOO
runs in well under 10 min locally (only the per-trip librosa mel costs anything).

Knobs (env): MF_TEMPL_S (template length s, default 1.0 = KWS window),
MF_T0_S (template start rel. onset), MF_ALIGN_S (+/- align search for averaging).

Run:  python scripts/path2_matchfilter_poc.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D
import path2_poc as P
import path2_kws_recover as K          # reuse debounce/score/COOLDOWN_S/MATCH_S/TRIPS/LIVE_DIR

FPS = D.SR / D.MEL_HOP                  # 62.5 frames/s
TEMPL_S = float(os.environ.get("MF_TEMPL_S", "1.0"))   # template length (s)
T0_S = float(os.environ.get("MF_T0_S", "0.0"))         # template start rel. "이번역은" onset
ALIGN_S = float(os.environ.get("MF_ALIGN_S", "0.30"))  # +/- offset search when averaging
SLIDE_S = 0.25                                          # detection hop (= KWS)


def _cmn_flat(full, f0, nfr, wbin=None):
    """Unit-norm flattened (optionally noise-whitened) CMN-log-mel window.

    wbin: per-mel-bin weight (40,1) = 1/noise_std -> emphasizes bins where the
    announcement stands out from the train rumble (whitened matched filter, the
    GCC-PHAT idea: don't let loud broadband noise dominate the correlation).
    """
    w = P._win_cmn(full, f0, nfr, True)        # 40 x nfr, per-mel-bin mean removed
    if wbin is not None:
        w = w * wbin
    v = w.reshape(-1).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _mark_frames(marks):
    t0fr = int(round(T0_S * FPS))
    return [int(round(idx / D.SR * FPS)) + t0fr for _, idx in marks]


def _aligned_avg(insts, nfr, wbin):
    """Aligned average of windows. insts = [(full, base_frame), ...]."""
    afr = int(round(ALIGN_S * FPS))
    ref = _cmn_flat(*insts[0], nfr, wbin)      # seed = first instance
    for _ in range(2):                         # align -> average -> repeat
        acc = np.zeros_like(ref)
        for full, base in insts:
            best_c, best_v = -2.0, None
            for d in range(-afr, afr + 1):
                f0 = base + d
                if f0 < 0 or f0 + nfr > full.shape[1]:
                    continue
                v = _cmn_flat(full, f0, nfr, wbin)
                if (c := float(v @ ref)) > best_c:
                    best_c, best_v = c, v
            if best_v is not None:
                acc += best_v
        ref = acc / (np.linalg.norm(acc) + 1e-8)
    return ref


def build_templates(fulls, marks_per_trip, nfr, mode, wbin):
    """mode='avg': one aligned-average template over all training marks.
       mode='maxtrip': one template per training trip (score = max cosine)."""
    if mode == "maxtrip":
        out = []
        for full, marks in zip(fulls, marks_per_trip):
            insts = [(full, b) for b in _mark_frames(marks)
                     if b >= 0 and b + nfr <= full.shape[1]]
            if insts:
                out.append(_aligned_avg(insts, nfr, wbin))
        return out
    insts = [(full, b) for full, marks in zip(fulls, marks_per_trip)
             for b in _mark_frames(marks) if b >= 0 and b + nfr <= full.shape[1]]
    return [_aligned_avg(insts, nfr, wbin)]


def noise_whiten(fulls, marks_per_trip, nfr, rng):
    """Per-mel-bin weight 1/std estimated from training-trip noise windows."""
    guard = int(round(D.NOISE_GUARD_S * FPS))
    cols = []
    for full, marks in zip(fulls, marks_per_trip):
        bad = np.zeros(full.shape[1], bool)
        for b in _mark_frames(marks):
            bad[max(0, b - guard):b + guard] = True
        ok = [f for f in range(0, full.shape[1] - nfr, max(1, nfr // 2))
              if not bad[f:f + nfr].any()]
        for f in rng.choice(ok, size=min(80, len(ok)), replace=False):
            cols.append(P._win_cmn(full, int(f), nfr, True))   # 40 x nfr
    allcols = np.concatenate(cols, axis=1)                      # 40 x (.)
    std = allcols.std(axis=1, keepdims=True) + 1e-6             # 40 x 1
    return (1.0 / std).astype(np.float32)


def mf_scores(full, nfr, templs, wbin):
    """Sliding score series (max cosine over templates) + window start times (s)."""
    fhop = max(1, int(round(SLIDE_S * FPS)))
    starts = list(range(0, full.shape[1] - nfr + 1, fhop))
    W = np.stack([P._win_cmn(full, f, nfr, True) for f in starts])   # n x 40 x nfr
    if wbin is not None:
        W = W * wbin
    V = W.reshape(len(starts), -1)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    sc = np.max([V @ t for t in templs], axis=0)               # max over templates
    return sc, np.array(starts) / FPS


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    nfr = int(round(TEMPL_S * FPS))
    trips = K.TRIPS
    print(f"matched-filter PoC | {len(trips)} trips | template {TEMPL_S}s ({nfr} fr), "
          f"align +-{ALIGN_S}s | slide {SLIDE_S}s\n")

    t_load = time.time()
    data = {}                                  # trip -> (full_logmel, marks)
    for t in trips:
        lt = D.load_live_trip(os.path.join(K.LIVE_DIR, t))
        data[t] = (P._full_logmel(lt.y), lt.marks)
    print(f"  loaded + mel'd {len(trips)} trips in {time.time()-t_load:.0f}s")

    # MINRUN=1 only: the matched-filter peak is sharp (one alignment), so a
    # "consecutive windows" debounce collapses recall (measured). Peak threshold
    # + COOLDOWN is the right post-proc here; vary the cosine threshold.
    variants = [("avg", False), ("avg", True), ("maxtrip", False), ("maxtrip", True)]
    for mode, whiten in variants:
        folds, peakdiag = [], []
        for held in trips:
            tr_f = [data[t][0] for t in trips if t != held]
            tr_m = [data[t][1] for t in trips if t != held]
            wbin = noise_whiten(tr_f, tr_m, nfr, np.random.default_rng(0)) if whiten else None
            templs = build_templates(tr_f, tr_m, nfr, mode, wbin)
            full, marks = data[held]
            scores, times = mf_scores(full, nfr, templs, wbin)
            for _, idx in marks:               # peak cosine within +-5s of each mark
                sel = np.abs(times - idx / D.SR) <= 5.0
                peakdiag.append(float(scores[sel].max()) if sel.any() else 0.0)
            folds.append((marks, scores, times))
        pk = np.array(peakdiag)
        allsc = np.concatenate([s for _, s, _ in folds])
        print(f"\n=== mode={mode:7s} whiten={int(whiten)} | "
              f"true-mark peak-cos mean {pk.mean():.2f} (min {pk.min():.2f}) "
              f"vs noise p99 {np.percentile(allsc,99):.2f} ===")
        print(f"{'COS':>5s} | {'recall':>14s}  {'false':>6s}  {'false/trip':>10s}")
        for trig in (0.30, 0.40, 0.50, 0.60, 0.70):
            h = f = g = 0
            for marks, scores, times in folds:
                trg = K.debounce(scores, times, trig, 1, K.COOLDOWN_S)
                hi, fa = K.score(trg, marks)
                h += hi; f += fa; g += len(marks)
            print(f"{trig:5.2f} | {h:2d}/{g} ({h/g*100:3.0f}%)      {f:4d}    {f/len(trips):6.1f}")

    print("\nKWS reference (kws_8trip.log, SAME debounce+score):")
    print("  MINRUN1 TRIG0.5  83% (80/96)  false 447  (~56/trip)")
    print("  MINRUN2 TRIG0.6  71% (68/96)  false 287  (~36/trip)")
    print("  MINRUN3 TRIG0.7  65% (62/96)  false 145  (~18/trip)")


if __name__ == "__main__":
    main()
