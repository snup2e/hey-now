"""Cut KWS / station-name clips from target announcements.

Target line: Seoul Metro Line 1, 성균관대 → 신도림 (14 stations).

Announcement structure (from listening):
  plain station :  "이번 역은"  0.0 ~ 1.5 s   |  station name  1.5 ~ 5.0 s
  transfer stn  :  preceded by other audio; "이번 역은" starts ~4 s,
                   so we detect the speech onset after 3 s.

Outputs:
  data/processed/clips/kws/<file>.wav      -- "이번 역은" trigger clip
  data/processed/clips/station/<file>.wav  -- station-name clip

Run:  python scripts/split_clips.py
"""
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WAV_DIR = ROOT / "data" / "processed" / "wav"
KWS_DIR = ROOT / "data" / "processed" / "clips" / "kws"
NAME_DIR = ROOT / "data" / "processed" / "clips" / "station"
SR = 16000

# Line 1: 성균관대 → 신도림
TARGET = ["성균관대", "의왕", "당정", "군포", "금정", "명학", "안양",
          "관악", "석수", "금천구청", "독산", "가산디지털단지", "구로", "신도림"]

# plain-station split (seconds)
KWS_START, KWS_END = 0.0, 1.5
NAME_START, NAME_END = 1.5, 5.0
# transfer-station: relative to detected onset
TR_KWS_LEN = 1.5
TR_NAME_LEN = 3.5


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, audio: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def find_onset(audio: np.ndarray, search_start=3.0, search_end=6.5) -> float:
    """First silence→speech transition within the search window (seconds)."""
    frame, hop = 400, 160
    n = 1 + (len(audio) - frame) // hop
    idx = np.arange(frame)[None, :] + np.arange(n)[:, None] * hop
    env = np.sqrt(np.mean(audio[idx] ** 2, axis=1) + 1e-10)
    thr = env.max() * 0.08
    s = int(search_start * SR / hop)
    e = min(int(search_end * SR / hop), n)
    for i in range(max(s, 1), e):
        if env[i] > thr and env[i - 1] <= thr:
            return i * hop / SR
    return search_start


def slice_sec(audio: np.ndarray, t0: float, t1: float) -> np.ndarray:
    a, b = int(t0 * SR), int(t1 * SR)
    seg = audio[max(0, a):min(len(audio), b)]
    want = b - a
    if len(seg) < want:
        seg = np.pad(seg, (0, want - len(seg)))
    return seg


def main():
    if not WAV_DIR.is_dir():
        sys.exit(f"wav 폴더 없음: {WAV_DIR}  (먼저 preprocess.py 실행)")

    files = sorted(WAV_DIR.glob("*.wav"))
    targets = [f for f in files if f.stem.split("_")[0] in TARGET]
    if not targets:
        sys.exit("타겟 음원을 찾지 못함")

    print(f"타겟 음원 {len(targets)}개  (성균관대 → 신도림 구간)\n")
    for f in targets:
        is_transfer = "환승" in f.stem
        audio = read_wav(f)
        if is_transfer:
            t0 = find_onset(audio)
            kws = slice_sec(audio, t0, t0 + TR_KWS_LEN)
            name = slice_sec(audio, t0 + TR_KWS_LEN, t0 + TR_KWS_LEN + TR_NAME_LEN)
            tag = f"환승  · onset {t0:.2f}s → KWS [{t0:.2f}, {t0+TR_KWS_LEN:.2f}]"
        else:
            kws = slice_sec(audio, KWS_START, KWS_END)
            name = slice_sec(audio, NAME_START, NAME_END)
            tag = f"일반  · KWS [0.0, 1.5]  역이름 [1.5, 5.0]"
        write_wav(KWS_DIR / f.name, kws)
        write_wav(NAME_DIR / f.name, name)
        print(f"  {f.stem:<22} {tag}")

    print(f"\nKWS 클립    → {KWS_DIR}")
    print(f"역이름 클립 → {NAME_DIR}")
    print("\n클립을 직접 들어보고 잘림이 어색하면 알려주세요 (특히 환승역).")


if __name__ == "__main__":
    main()
