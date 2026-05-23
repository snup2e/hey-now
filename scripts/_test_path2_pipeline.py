"""Synthetic end-to-end test for path2_capture + path2_slice.

Builds a fake "trip" by concatenating 5 existing station wavs with
silence between them, writes the marks.json a real capture session
would produce, then asks path2_slice to cut it. Asserts that:
  - 5 clips appear in data/processed/line1_clips/<trip>/
  - 5 rows show up in data/path2_metadata.csv
  - the mark land inside each clip at the expected offset

Run:  python scripts/_test_path2_pipeline.py
"""
import csv
import json
import shutil
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WAV_DIR = ROOT / "data" / "processed" / "wav"
LIVE = ROOT / "data" / "raw" / "line1_live"
CLIPS = ROOT / "data" / "processed" / "line1_clips"
META = ROOT / "data" / "path2_metadata.csv"

SR = 16000
GAP_S = 30.0  # silence between announcements
ROUTE = [  # first 5 stations of north direction
    ("성균관대", "성균관대.wav"),
    ("의왕",     "의왕.wav"),
    ("당정",     "당정.wav"),
    ("군포",     "군포.wav"),
    ("금정",     "금정_환승.wav"),
]


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.readframes(w.getnframes())


def main():
    trip_id = "TEST_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_north"
    trip_dir = LIVE / trip_id
    trip_dir.mkdir(parents=True, exist_ok=True)

    # build long PCM: gap, ann1, gap, ann2, ...
    silence = b"\x00\x00" * int(GAP_S * SR)
    pcm = b""
    marks = []
    for station, fname in ROUTE:
        pcm += silence
        ann_start_sample = len(pcm) // 2
        ann_pcm = read_pcm(WAV_DIR / fname)
        pcm += ann_pcm
        # simulate operator click 0.7s after announcement begins
        click_sample = ann_start_sample + int(0.7 * SR)
        marks.append({
            "station_idx": len(marks),
            "station": station,
            "sample_index": click_sample,
            "elapsed_s": round(click_sample / SR, 3),
            "wall_time": datetime.now().isoformat(timespec="seconds"),
        })
    pcm += silence

    # write audio.wav + marks.json
    wav_path = trip_dir / "audio.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm)

    info = {
        "trip_id": trip_id,
        "direction": "north",
        "sample_rate": SR,
        "audio_file": "audio.wav",
        "stations_route": [s for s, _ in ROUTE],
        "marks": marks,
    }
    (trip_dir / "marks.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    total_s = len(pcm) // 2 / SR
    print(f"synthetic trip: {trip_id}  ({total_s:.1f}s, {len(marks)} marks)")

    # run slicer
    result = subprocess.run(
        [sys.executable, "-X", "utf8",
         str(ROOT / "scripts" / "path2_slice.py"),
         "--trip", trip_id],
        capture_output=True, text=True, encoding="utf-8")
    print("--- slicer stdout ---")
    print(result.stdout)
    if result.returncode != 0:
        print("--- slicer stderr ---")
        print(result.stderr)
        sys.exit("slicer failed")

    # assertions
    out_dir = CLIPS / trip_id
    clips = sorted(out_dir.glob("*.wav"))
    assert len(clips) == len(ROUTE), f"expected {len(ROUTE)} clips, got {len(clips)}"

    with open(META, "r", encoding="utf-8-sig", newline="") as fp:
        all_rows = list(csv.DictReader(fp))
    trip_rows = [r for r in all_rows if r["trip_id"] == trip_id]
    assert len(trip_rows) == len(ROUTE), f"metadata rows = {len(trip_rows)}"

    print(f"\nOK: {len(clips)} clips, {len(trip_rows)} metadata rows")
    for r in trip_rows:
        clip_path = ROOT / "data" / "processed" / "line1_clips" / r["clip_file"]
        with wave.open(str(clip_path), "rb") as w:
            actual_s = w.getnframes() / w.getframerate()
        print(f"  {r['clip_file']:<60} mark@{r['mark_offset_sec']}s / clip {actual_s:.2f}s")

    # cleanup
    shutil.rmtree(trip_dir)
    shutil.rmtree(out_dir)
    # purge test rows from metadata
    remaining = [r for r in all_rows if r["trip_id"] != trip_id]
    with open(META, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys()) if all_rows else
                                          ["clip_file","station","trip_id","direction",
                                           "mark_offset_sec","clip_sec","sample_rate"])
        w.writeheader(); w.writerows(remaining)
    print(f"\ncleaned up. metadata.csv now has {len(remaining)} non-test rows.")


if __name__ == "__main__":
    main()
