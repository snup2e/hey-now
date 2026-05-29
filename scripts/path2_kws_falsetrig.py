"""Is the KWS healthy, and what are its FALSE triggers? (the user heard
"이번 역에서 내리시기 바랍니다" — a non-target phrase that also starts "이번 역").

The KWS positive is "이번역은" (the arrival/boarding call). But several other
cabin announcements share the identical "이번 역" onset (~0.4-0.6 s), e.g. the
transfer line's "환승하실 분은 이번 역에서 내리시기 바랍니다". A 1 s small-CNN
detector keying on that shared onset can fire on them too -> false triggers that
become spurious marks feeding the decoder. The window alone does NOT separate
them (the discriminative "-은" vs "-에서" comes after the shared part).

This trains the KWS per trip-LOO fold (high-recall operating point), slides over
the held-out trip, and for every detection reports whether it matches a true
station mark or is FALSE, plus each false trigger's signed gap to the nearest
mark. Clustering of falses at +20-90 s after a mark (same station, after the
arrival call) is the signature of the door/transfer announcement; scattered
falses are train noise.

It also EXPORTS each false-trigger window to wav (reports/path2_kws_falsetrig/)
so a human can listen and confirm whether it is indeed "이번 역에서 ..." — the
model cannot transcribe, so this is the load-bearing manual check.

Run:  python scripts/path2_kws_falsetrig.py
      KWS_TRIG=0.5 KWS_MINRUN=2 KWS_COOLDOWN=20 python scripts/path2_kws_falsetrig.py
"""
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D
import path2_poc as P

LIVE_DIR = P.LIVE_DIR
TRIPS = ["20260527_0654_등교", "20260528_0642_등교",
         "20260527_1431_하교", "20260528_2118_하교"]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reports", "path2_kws_falsetrig")

TRIG = float(os.environ.get("KWS_TRIG", "0.6"))
MINRUN = int(os.environ.get("KWS_MINRUN", "2"))
COOLDOWN_S = float(os.environ.get("KWS_COOLDOWN", "20"))
# KWS_USE_SYNTH=0 -> real-only positives (drop the clean 서울교통공사 synth, a
# different recording than the 코레일 live PA). REAL_AUG = extra real-noise mixes
# per real positive; NEG_RATIO = negatives per positive (>1 for the chatter problem).
USE_SYNTH = os.environ.get("KWS_USE_SYNTH", "1") == "1"
REAL_AUG = int(os.environ.get("KWS_REAL_AUG", "0"))
NEG_RATIO = float(os.environ.get("KWS_NEG_RATIO", "1.0"))
MATCH_S = 10.0                 # a trigger within this of a mark = that station's call
MAX_WAV_PER_TRIP = 6


def write_wav(path, y, sr=D.SR):
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def detect(test, kws, kn):
    """All debounced trigger onset times (s) over the held-out trip."""
    full = P._full_logmel(test.y)
    fps = D.SR / D.MEL_HOP
    kfr = int(round(D.KWS_WIN * fps)); fhop = max(1, int(round(0.25 * fps)))
    starts = list(range(0, full.shape[1] - kfr + 1, fhop))
    kp = P.predict_batch(kws, [P._win_cmn(full, f, kfr, True) for f in starts], kn)[:, 1]
    times = np.array(starts) / fps
    trg, run, rs, j = [], 0, 0.0, 0
    while j < len(kp):
        if kp[j] > TRIG:
            if run == 0:
                rs = times[j]
            run += 1; j += 1
        else:
            if run >= MINRUN:
                trg.append(rs)
                while j < len(times) and times[j] < rs + COOLDOWN_S:
                    j += 1
            else:
                j += 1
            run = 0
    if run >= MINRUN:
        trg.append(rs)
    return trg


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.makedirs(OUT_DIR, exist_ok=True)
    clean = D.load_clean_sources(P.CLEAN_DIR)
    print(f"operating point: TRIG={TRIG} MINRUN={MINRUN} COOLDOWN={COOLDOWN_S}s  "
          f"(match window ±{MATCH_S}s)")
    print(f"KWS positives: {'synth+real' if USE_SYNTH else 'REAL-ONLY'}  "
          f"real_pos_aug={REAL_AUG}  neg_ratio={NEG_RATIO}\n")

    tot_hit = tot_mark = tot_false = 0
    near_false = 0          # falses within [+3,+100]s AFTER a mark (likely door/transfer)
    gaps_all = []
    for held in TRIPS:
        trips = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in TRIPS if t != held]
        test = D.load_live_trip(os.path.join(LIVE_DIR, held))
        rng = np.random.default_rng(0)
        Xk, Yk = D.build_kws(clean, trips, rng, snr=(0.0, 25.0), spec_aug=False,
                             use_synth=USE_SYNTH, real_pos_aug=REAL_AUG,
                             neg_ratio=NEG_RATIO)
        npos = int(Yk.sum())
        kws, kn, kacc = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)
        trg = detect(test, kws, kn)
        mark_t = sorted(idx / D.SR for _, idx in test.marks)

        # classify each trigger: matched (nearest mark within MATCH_S) or false
        hit_marks = set(); falses = []
        for t in trg:
            gaps = [(t - m, mi) for mi, m in enumerate(mark_t)]
            dt, mi = min(gaps, key=lambda g: abs(g[0]))
            if abs(dt) <= MATCH_S:
                hit_marks.add(mi)
            else:
                falses.append((t, dt))           # dt = signed gap to nearest mark
        hit = len(hit_marks)
        tot_hit += hit; tot_mark += len(mark_t); tot_false += len(falses)

        tdir = os.path.join(OUT_DIR, held)
        os.makedirs(tdir, exist_ok=True)
        for k, (t, dt) in enumerate(falses):
            gaps_all.append(dt)
            if 3 <= dt <= 100:
                near_false += 1
            if k < MAX_WAV_PER_TRIP:             # export window for human listening
                a = max(0, int((t - 0.5) * D.SR)); b = int((t + 2.5) * D.SR)
                write_wav(os.path.join(tdir, f"false_{t:06.1f}s_gap{dt:+05.0f}.wav"),
                          test.y[a:b])
        print(f"  {held[9:13]} ({test.direction}) val{kacc*100:3.0f}% "
              f"pool{npos}+{len(Yk)-npos} | "
              f"marks {hit}/{len(mark_t)} hit, triggers {len(trg)}, "
              f"FALSE {len(falses)}  gaps(s)={[round(d,0) for _, d in falses][:10]}")

    print("\n" + "=" * 60)
    print(f"recall          : {tot_hit}/{tot_mark} ({tot_hit/tot_mark*100:.0f}%)")
    print(f"false triggers  : {tot_false} total over 4 held-out trips "
          f"({tot_false/4:.1f}/trip)")
    if gaps_all:
        g = np.array(gaps_all)
        print(f"false→nearest-mark gap: median {np.median(np.abs(g)):+.0f}s | "
              f"within [+3,+100]s after a mark: {near_false}/{len(g)} "
              f"({near_false/len(g)*100:.0f}%)  <- likely door/transfer 이번역에서")
    print(f"\nwav of false triggers exported under {OUT_DIR}\\<trip>\\  -- LISTEN to")
    print("confirm whether they are '이번 역에서 ...' (then we can target them).")


if __name__ == "__main__":
    main()
