"""Build dataset metadata from announcement filenames.

Filenames look like:  성균관대.mp3 / 광운대_종착.mp3 / 구로_환승_경부.mp3
The part before the first '_' is the station name (the class label);
the rest is a variant tag (환승 / 종착 / 출발 / 상행 / 하행 / line info).

Writes data/metadata.csv  (utf-8-sig so Excel opens it cleanly).

Run:  python scripts/build_metadata.py
"""
import csv
import io
import subprocess
import sys
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "seoul_metro"
OUT = ROOT / "data" / "metadata.csv"


def duration_sec(mp3: Path) -> float:
    """Clip duration via ffprobe (seconds)."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(mp3),
    ]
    try:
        return round(float(subprocess.run(cmd, capture_output=True, text=True,
                                          check=True).stdout.strip()), 2)
    except Exception:
        return 0.0


def parse(stem: str):
    """Return (station, variant). Station = text before first underscore."""
    parts = stem.split("_")
    station = parts[0]
    variant = "_".join(parts[1:]) if len(parts) > 1 else "기본"
    return station, variant


def main():
    if not SRC.is_dir():
        sys.exit(f"음원 폴더 없음: {SRC}")
    files = sorted(SRC.glob("*.mp3"))
    if not files:
        sys.exit(f"mp3 파일이 없음: {SRC}")

    rows = []
    for f in files:
        station, variant = parse(f.stem)
        rows.append({
            "file": f.name,
            "station": station,
            "variant": variant,
            "duration_sec": duration_sec(f),
        })

    with open(OUT, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=["file", "station", "variant", "duration_sec"])
        w.writeheader()
        w.writerows(rows)

    stations = sorted({r["station"] for r in rows})
    counts = Counter(r["station"] for r in rows)
    multi = {k: v for k, v in sorted(counts.items()) if v > 1}
    total_dur = sum(r["duration_sec"] for r in rows)

    print(f"파일 {len(rows)}개  →  역(클래스) {len(stations)}개")
    print(f"총 음원 길이: {total_dur/60:.1f}분  (평균 {total_dur/len(rows):.1f}초/파일)")
    print(f"음원 2개 이상인 역 {len(multi)}개: {multi}")
    print(f"\n메타데이터 저장: {OUT}")


if __name__ == "__main__":
    main()
