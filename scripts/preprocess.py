"""Convert raw announcement mp3s to 16 kHz mono wav.

The Path-1 model works at 16 kHz mono. Converting locally keeps the
Colab notebook simple (no ffmpeg step there) and shrinks the upload.

Output: data/processed/wav/<name>.wav

Run:  python scripts/preprocess.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "seoul_metro"
DST = ROOT / "data" / "processed" / "wav"
SR = 16000


def main():
    if not SRC.is_dir():
        sys.exit(f"음원 폴더 없음: {SRC}")
    files = sorted(SRC.glob("*.mp3"))
    if not files:
        sys.exit(f"mp3 파일이 없음: {SRC}")

    DST.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        out = DST / (f.stem + ".wav")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(f),
             "-ac", "1", "-ar", str(SR), str(out)],
            check=True,
        )
        if i % 20 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] 변환 중...")

    total_mb = sum(p.stat().st_size for p in DST.glob("*.wav")) / 1e6
    print(f"\n완료: {len(files)}개 wav  ({total_mb:.1f} MB)  →  {DST}")


if __name__ == "__main__":
    main()
