"""Materialise EVERY training cut as a listenable .wav, for human audit.

Model experiments are only meaningful if the input clips actually contain the
right speech. The cuts are currently chosen by an automatic energy onset
(clean) or a hand mark (live) plus FIXED offsets -- never verified by ear. This
script writes the exact spans the model sees, plus a wider CONTEXT clip so you
can hear the full announcement and judge whether the cut is right.

For each CLEAN station wav and each LIVE trip mark it writes:
  CONTEXT : [t0-1.0, t0+4.5]  -- the whole 1st announcement around the cut
  NAME    : [t0,     t0+2.0]  -- what the CNN / metric encoder sees  (이번역은 + 역명)
  KWS     : [t0-0.3, t0+1.3]  -- what the KWS trigger sees           (이번역은)
where t0 = detected onset (clean) or hand mark (live).

Output: data/processed/clip_audit/  (gitignored). Also prints an onset table
with SUSPECT flags (onset stuck at the search-window start = detection failed).

Run:  python scripts/path2_export_clips.py
"""
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(ROOT, "data", "processed", "wav")
LIVE_DIR = os.path.join(ROOT, "data", "raw", "line1_live")
OUT = os.path.join(ROOT, "data", "processed", "clip_audit")
TRIPS = ["20260527_0654_등교", "20260528_0642_등교",
         "20260527_1431_하교", "20260528_2118_하교"]

CONTEXT = (-1.0, 4.5)
NAME = (0.0, D.CNN_WIN)        # [onset, +2.0]  -> CNN / metric input
KWS = (-0.3, 1.3)             # [onset-0.3, +1.3] -> KWS positive


def write_wav16(path, y):
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767.0).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(D.SR)
        w.writeframes(pcm)


def cut(y, t0, span):
    a = int((t0 + span[0]) * D.SR)
    b = int((t0 + span[1]) * D.SR)
    a, b = max(0, a), min(len(y), b)
    return y[a:b]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.makedirs(os.path.join(OUT, "clean"), exist_ok=True)
    manifest = ["source\tstation\tt0_sec\tclip\tspan_sec\texpected_content"]

    # ---------- CLEAN ----------
    print("=== CLEAN sources: detected onset table ===")
    print(f"{'station':16s} {'file':28s} {'transfer':8s} {'onset(s)':>9s}  flag")
    clean = D.load_clean_sources(CLEAN_DIR)
    # re-read filenames to show which wav each came from
    import glob
    files = {}
    for p in sorted(glob.glob(os.path.join(CLEAN_DIR, "*.wav"))):
        st = os.path.splitext(os.path.basename(p))[0].split("_")[0]
        files.setdefault(st, []).append(os.path.basename(p))
    seen = {}
    for c in clean:
        seen.setdefault(c.station, 0)
        fname = files.get(c.station, ["?"])[min(seen[c.station], len(files.get(c.station, ["?"]))-1)]
        seen[c.station] += 1
        sstart = 3.0 if c.transfer else 0.0
        flag = "SUSPECT(onset=search start)" if c.onset <= sstart + 0.06 else ""
        print(f"{c.station:16s} {fname:28s} {str(c.transfer):8s} {c.onset:9.2f}  {flag}")
        base = os.path.join(OUT, "clean", f"{c.station}_{seen[c.station]}")
        write_wav16(base + "__CONTEXT.wav", cut(c.y, c.onset, CONTEXT))
        write_wav16(base + "__NAME_이번역은+역명.wav", cut(c.y, c.onset, NAME))
        write_wav16(base + "__KWS_이번역은.wav", cut(c.y, c.onset, KWS))
        for tag, span, exp in (("CONTEXT", CONTEXT, "full announcement around onset"),
                               ("NAME", NAME, f"이번역은 {c.station}(역) -- CNN input"),
                               ("KWS", KWS, "이번역은 -- KWS trigger")):
            manifest.append(f"clean\t{c.station}\t{c.onset:.2f}\t{tag}\t{span[0]:+.1f}..{span[1]:+.1f}\t{exp}")

    # ---------- LIVE ----------
    print("\n=== LIVE trips: marks (hand-placed onset) ===")
    for trip_id in TRIPS:
        trip = D.load_live_trip(os.path.join(LIVE_DIR, trip_id))
        tdir = os.path.join(OUT, "live", trip_id)
        os.makedirs(tdir, exist_ok=True)
        ms = sorted(trip.marks, key=lambda x: x[1])
        print(f"  {trip_id} ({trip.direction}): {len(ms)} marks")
        for i, (station, idx) in enumerate(ms):
            t0 = idx / D.SR
            base = os.path.join(tdir, f"{i:02d}_{station}_{t0:.0f}s")
            write_wav16(base + "__CONTEXT.wav", cut(trip.y, t0, CONTEXT))
            write_wav16(base + "__NAME.wav", cut(trip.y, t0, NAME))
            write_wav16(base + "__KWS.wav", cut(trip.y, t0, KWS))
            for tag, span, exp in (("CONTEXT", CONTEXT, "full announcement around mark"),
                                   ("NAME", NAME, f"이번역은 {station}(역) + cabin noise"),
                                   ("KWS", KWS, "이번역은 + cabin noise")):
                manifest.append(f"{trip_id}\t{station}\t{t0:.2f}\t{tag}\t{span[0]:+.1f}..{span[1]:+.1f}\t{exp}")

    with open(os.path.join(OUT, "manifest.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest))
    n = len(manifest) - 1
    print(f"\nWrote {n} clip entries to: {OUT}")
    print("  clean/  : <station>_<n>__CONTEXT/NAME/KWS.wav")
    print("  live/<trip>/ : <NN>_<station>_<t>s__CONTEXT/NAME/KWS.wav")
    print("  manifest.tsv : source, station, t0, clip, span, expected content")


if __name__ == "__main__":
    main()
