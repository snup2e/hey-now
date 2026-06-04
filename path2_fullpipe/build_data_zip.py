"""(Re)build the 8-trip data zip for the Colab full-pipeline notebook.

The committed heynow_path2_data.zip is STALE (4 trips, no door_events). The full
pipeline needs all 8 trips WITH door_events.json inside each trip dir. Run this
from the repo root (E:\\imsisul) once; upload the output to Drive.

  python path2_fullpipe/build_data_zip.py

Output: path2_fullpipe/heynow_path2_data_8trip.zip
  processed/wav/*.wav                                         (clean — KWS positives)
  raw/line1_live/<trip_id>/audio.wav, marks.json, door_events.json   (8 trips)

door_events.json sample indices were marked on the SAME audio (md5-verified), so
they align to audio.wav exactly. If a trip dir is missing door_events.json, it is
pulled from path2_fullpipe/door_events/<trip_id>.door_events.json.
"""
import glob
import hashlib
import os
import shutil
import sys
import wave
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLEAN = os.path.join(ROOT, "data", "processed", "wav")
LIVE = os.path.join(ROOT, "data", "raw", "line1_live")
A_TRAIN = os.path.join(ROOT, "A_train")
PKG_DOOR = os.path.join(HERE, "door_events")
OUT = os.path.join(HERE, "heynow_path2_data_8trip.zip")


def _audio_md5(path):
    with wave.open(path, "rb") as w:
        return hashlib.md5(w.readframes(w.getnframes())).hexdigest()[:12]


def sync_from_atrain():
    """A_train/'audio (N).door_events.json' → live/<trip_id>/door_events.json,
    matched by audio md5 (the same recording). So re-marking 탁 in the GUI on
    A_train propagates into the zip. No-op if A_train missing."""
    if not os.path.isdir(A_TRAIN):
        return
    at = {}
    for n in range(1, 9):
        ap = os.path.join(A_TRAIN, f"audio ({n}).wav")
        de = os.path.join(A_TRAIN, f"audio ({n}).door_events.json")
        if os.path.exists(ap) and os.path.exists(de):
            at[_audio_md5(ap)] = de
    synced = 0
    for d in glob.glob(os.path.join(LIVE, "*")):
        ap = os.path.join(d, "audio.wav")
        if not os.path.isdir(d) or not os.path.exists(ap):
            continue
        src = at.get(_audio_md5(ap))
        if src:
            shutil.copy(src, os.path.join(d, "door_events.json")); synced += 1
    print(f"  synced door_events from A_train → live: {synced} trips")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not os.path.isdir(CLEAN) or not os.path.isdir(LIVE):
        sys.exit(f"run from repo root; missing {CLEAN} or {LIVE}")

    sync_from_atrain()   # pull latest hand-marks (incl. 탁) from A_train

    trips = sorted(d for d in os.listdir(LIVE)
                   if os.path.isdir(os.path.join(LIVE, d)) and not d.startswith("_"))
    n_clean = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for w in sorted(glob.glob(os.path.join(CLEAN, "*.wav"))):
            z.write(w, f"processed/wav/{os.path.basename(w)}")
            n_clean += 1
        for tid in trips:
            d = os.path.join(LIVE, tid)
            # ensure door_events.json present (pull from package if needed)
            de = os.path.join(d, "door_events.json")
            if not os.path.exists(de):
                src = os.path.join(PKG_DOOR, f"{tid}.door_events.json")
                if os.path.exists(src):
                    shutil.copy(src, de)
                else:
                    print(f"  ⚠ {tid}: no door_events.json (skipping door for this trip)")
            for fn in ("audio.wav", "marks.json", "door_events.json"):
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    z.write(p, f"raw/line1_live/{tid}/{fn}")

    size = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT}")
    print(f"  clean wav: {n_clean} | trips: {len(trips)} | size: {size:.0f} MB")
    print("  trips:", trips)


if __name__ == "__main__":
    main()
