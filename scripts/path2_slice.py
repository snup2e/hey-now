"""Slice a Path-2 long trip recording into per-station clips.

For each mark in marks.json, cut a window [mark - pre, mark + post]
from audio.wav. The mark is roughly when the recorder *heard* the
announcement begin — clicks are not surgical, so we keep a wide window
(default 4s before, 12s after = 16s clip). Training code does its
own short windowing later.

Reads:
  data/raw/line1_live/<trip_id>/audio.wav
  data/raw/line1_live/<trip_id>/marks.json

Writes:
  data/processed/line1_clips/<trip_id>/<station>_<seq>.wav
  data/path2_metadata.csv   (one row per clip; re-slicing overwrites)

Run:
  python scripts/path2_slice.py                       # every trip
  python scripts/path2_slice.py --trip 20260524_0742_north
  python scripts/path2_slice.py --pre 4 --post 12
"""
import argparse
import csv
import json
import sys
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "data" / "raw" / "line1_live"
CLIPS = ROOT / "data" / "processed" / "line1_clips"
META = ROOT / "data" / "path2_metadata.csv"

FIELDS = ["clip_file", "station", "trip_id", "direction",
          "mark_offset_sec", "clip_sec", "sample_rate"]


def slice_trip(trip_dir: Path, pre_s: float, post_s: float) -> list[dict]:
    marks_path = trip_dir / "marks.json"
    wav_path = trip_dir / "audio.wav"
    if not (marks_path.exists() and wav_path.exists()):
        print(f"  스킵 (미완성): {trip_dir.name}")
        return []

    info = json.loads(marks_path.read_text(encoding="utf-8"))
    sr = info["sample_rate"]
    trip_id = info["trip_id"]

    # marks.json may be in the new "segments" format (round-trip aware) or the
    # original flat format. Normalise to a list of (direction, marks) pairs.
    if "segments" in info:
        segs = [(s["direction"], s["marks"]) for s in info["segments"]]
    else:
        segs = [(info.get("direction", "north"), info.get("marks", []))]

    total_marks = sum(len(m) for _, m in segs)
    if total_marks == 0:
        print(f"  스킵 (마크 0개): {trip_id}")
        return []

    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getframerate() == sr, f"sr mismatch in {trip_id}"
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2
        total = wf.getnframes()
        pcm = wf.readframes(total)

    out_dir = CLIPS / trip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_n = int(pre_s * sr)
    post_n = int(post_s * sr)

    rows = []
    seq = Counter()
    for direction, marks in segs:
        for m in sorted(marks, key=lambda x: x["sample_index"]):
            center = m["sample_index"]
            start = max(0, center - pre_n)
            end = min(total, center + post_n)
            clip = pcm[start * 2 : end * 2]
            if not clip:
                print(f"    경고: {m['station']} 마크가 오디오 범위 밖 (skip)")
                continue

            station = m["station"]
            seq[station] += 1
            name = f"{station}_{seq[station]}.wav"
            out_path = out_dir / name
            with wave.open(str(out_path), "wb") as ow:
                ow.setnchannels(1)
                ow.setsampwidth(2)
                ow.setframerate(sr)
                ow.writeframes(clip)

            rows.append({
                "clip_file": f"{trip_id}/{name}",
                "station": station,
                "trip_id": trip_id,
                "direction": direction,
                "mark_offset_sec": round((center - start) / sr, 3),
                "clip_sec": round((end - start) / sr, 3),
                "sample_rate": sr,
            })

    print(f"  {trip_id}: {len(rows)} 클립  →  {out_dir.relative_to(ROOT)}")
    return rows


def upsert_metadata(new_rows: list[dict]):
    """Merge new rows into path2_metadata.csv, keyed by clip_file."""
    existing = []
    if META.exists():
        with open(META, "r", encoding="utf-8-sig", newline="") as fp:
            existing = list(csv.DictReader(fp))
    by_key = {r["clip_file"]: r for r in existing}
    for r in new_rows:
        by_key[r["clip_file"]] = r
    all_rows = sorted(by_key.values(), key=lambda r: r["clip_file"])
    META.parent.mkdir(parents=True, exist_ok=True)
    with open(META, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n메타데이터: {len(all_rows)} 행  →  {META.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trip", help="단일 trip_id만 처리 (예: 20260524_0742_north)")
    ap.add_argument("--pre", type=float, default=4.0, help="마크 전 포함 초")
    ap.add_argument("--post", type=float, default=12.0, help="마크 후 포함 초")
    args = ap.parse_args()

    if not LIVE.exists():
        sys.exit(f"녹음 폴더 없음: {LIVE}")

    if args.trip:
        targets = [LIVE / args.trip]
        if not targets[0].is_dir():
            sys.exit(f"trip 없음: {targets[0]}")
    else:
        targets = sorted(p for p in LIVE.iterdir() if p.is_dir())
        if not targets:
            sys.exit(f"trip 폴더가 없음: {LIVE}")

    print(f"처리 대상: {len(targets)} trip(s)")
    all_rows = []
    for t in targets:
        all_rows += slice_trip(t, args.pre, args.post)

    if all_rows:
        upsert_metadata(all_rows)
    else:
        print("\n새로 생성된 클립 없음.")


if __name__ == "__main__":
    main()
