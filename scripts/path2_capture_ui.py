"""Path 2 live recording UI: capture USB-CDC PCM + mark stations.

STM32 streams 16-bit mono PCM over USB-CDC (raw bytes, no framing).
This writes the stream to a long wav while the recorder marks each
station announcement as it plays.

Marking UX (kept dead-simple for use on a moving train):
  - Spacebar  → mark the next expected station (auto-advances)
  - Click     → mark any station out of order / re-mark
  - The "next expected" station is highlighted; marked ones turn green.

Output per recording session:
  data/raw/line1_live/<trip_id>/audio.wav
  data/raw/line1_live/<trip_id>/marks.json

Mock mode (no board yet) — replays an existing wav at real-time pace
so the UI is fully testable on a laptop:
  python scripts/path2_capture_ui.py --mock-wav data/processed/wav/성균관대.wav --direction north

Real run:
  python scripts/path2_capture_ui.py --port COM5 --direction north
"""
import argparse
import json
import sys
import threading
import time
import tkinter as tk
import wave
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1호선 성균관대 → 구로 (상행 = north). south = reversed.
STATIONS_NORTH = [
    "성균관대", "의왕", "당정", "군포", "금정", "명학", "안양",
    "관악", "석수", "금천구청", "독산", "가산디지털단지", "구로",
]


# ---------- audio sources ----------

class SerialSource:
    """Read raw 16-bit PCM bytes from STM32 over USB-CDC."""
    def __init__(self, port: str, sr: int):
        import serial  # imported lazily so mock mode has no dep
        # USB-CDC on STM32 ignores baud, but pyserial wants something.
        self.ser = serial.Serial(port, baudrate=1_000_000, timeout=0.1)
        self.sr = sr

    def chunks(self):
        while True:
            data = self.ser.read(4096)
            if data:
                yield data


class MockSource:
    """Replay a wav file at real-time pace — for UI dev without a board.

    Loops the file so the operator can keep clicking past the file end.
    """
    def __init__(self, path: Path, sr: int):
        wf = wave.open(str(path), "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            sys.exit(f"mock wav must be mono 16-bit PCM: {path}")
        if wf.getframerate() != sr:
            sys.exit(f"mock wav sr {wf.getframerate()} ≠ requested {sr}")
        self.wf = wf
        self.sr = sr

    def chunks(self):
        n = self.sr // 10  # 100 ms per chunk
        while True:
            data = self.wf.readframes(n)
            if not data:
                self.wf.rewind()
                continue
            yield data
            time.sleep(n / self.sr)


# ---------- recorder (audio thread) ----------

class Recorder:
    """Drains a source into a wav file; exposes the live sample counter."""
    def __init__(self, source, wav_path: Path, sr: int):
        self.source = source
        self.sr = sr
        self.wav = wave.open(str(wav_path), "wb")
        self.wav.setnchannels(1)
        self.wav.setsampwidth(2)
        self.wav.setframerate(sr)
        self._count = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        for chunk in self.source.chunks():
            if not self._running:
                break
            self.wav.writeframes(chunk)
            with self._lock:
                self._count += len(chunk) // 2  # 16-bit → 2 bytes/sample

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.wav.close()

    def current_sample(self) -> int:
        with self._lock:
            return self._count


# ---------- UI ----------

class CaptureUI:
    def __init__(self, recorder: Recorder, stations, out_dir: Path,
                 trip_id: str, direction: str):
        self.recorder = recorder
        self.stations = stations
        self.out_dir = out_dir
        self.trip_id = trip_id
        self.direction = direction
        self.marks = {}  # station_idx -> mark dict
        self.next_idx = 0

        self.root = tk.Tk()
        self.root.title(f"Path 2 capture — {trip_id}")
        self.root.geometry("460x740")
        self._build()
        self.root.bind("<space>", self._on_space)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._tick)

    def _build(self):
        head = tk.Frame(self.root)
        head.pack(fill="x", pady=4)
        tk.Label(head, text=f"{self.direction.upper()}  ·  {self.trip_id}",
                 font=("", 10)).pack(side="left", padx=8)
        self.elapsed_var = tk.StringVar(value="0:00")
        tk.Label(head, textvariable=self.elapsed_var,
                 font=("Consolas", 14, "bold")).pack(side="right", padx=8)

        tk.Label(self.root,
                 text="Space = 다음 역 마크    ·    클릭 = 임의 역 마크/재마크",
                 fg="#555").pack(pady=(0, 6))

        self.btns = []
        for i, name in enumerate(self.stations):
            b = tk.Button(self.root, text="", font=("", 14), height=2,
                          anchor="w",
                          command=lambda i=i: self._mark(i))
            b.pack(fill="x", padx=8, pady=2)
            self.btns.append(b)
        self._refresh()

        tk.Button(self.root, text="종료 & 저장 (창 X 도 동일)",
                  font=("", 12), bg="#d44", fg="white",
                  command=self._on_close).pack(fill="x", padx=8, pady=10)

    def _refresh(self):
        for i, b in enumerate(self.btns):
            name = self.stations[i]
            if i in self.marks:
                m = self.marks[i]
                b.config(text=f"✓ {i+1:2d}. {name}   ({m['elapsed_s']:.1f}s)",
                         bg="#c8f0c8", font=("", 14))
            elif i == self.next_idx:
                b.config(text=f"▶ {i+1:2d}. {name}",
                         bg="#fff2a8", font=("", 14, "bold"))
            else:
                b.config(text=f"   {i+1:2d}. {name}",
                         bg="SystemButtonFace", font=("", 14))

    def _mark(self, idx: int):
        s = self.recorder.current_sample()
        self.marks[idx] = {
            "station_idx": idx,
            "station": self.stations[idx],
            "sample_index": s,
            "elapsed_s": round(s / self.recorder.sr, 3),
            "wall_time": datetime.now().isoformat(timespec="seconds"),
        }
        # advance "next" past everything already marked
        while self.next_idx < len(self.stations) and self.next_idx in self.marks:
            self.next_idx += 1
        self._refresh()

    def _on_space(self, _evt):
        if self.next_idx < len(self.stations):
            self._mark(self.next_idx)

    def _tick(self):
        s = self.recorder.current_sample()
        sec = s / self.recorder.sr
        self.elapsed_var.set(f"{int(sec)//60}:{int(sec)%60:02d}")
        self.root.after(100, self._tick)

    def _on_close(self):
        self.recorder.stop()
        marks_sorted = sorted(self.marks.values(), key=lambda m: m["sample_index"])
        (self.out_dir / "marks.json").write_text(json.dumps({
            "trip_id": self.trip_id,
            "direction": self.direction,
            "sample_rate": self.recorder.sr,
            "audio_file": "audio.wav",
            "stations_route": self.stations,
            "marks": marks_sorted,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        total_s = self.recorder.current_sample() / self.recorder.sr
        print(f"\n저장 완료: {self.out_dir}/")
        print(f"  audio.wav  {total_s:.1f}s")
        print(f"  marks.json  {len(marks_sorted)}/{len(self.stations)} 역")
        self.root.destroy()

    def run(self):
        self.recorder.start()
        self.root.mainloop()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="USB-CDC serial port (e.g. COM5, /dev/ttyACM0)")
    src.add_argument("--mock-wav", type=Path,
                     help="replay an existing wav at real-time pace")
    ap.add_argument("--direction", choices=["north", "south"], required=True,
                    help="north = 성균관대→구로 (상행), south = reverse")
    ap.add_argument("--sr", type=int, default=16000, help="sample rate (Hz)")
    ap.add_argument("--out-root", type=Path,
                    default=ROOT / "data" / "raw" / "line1_live")
    args = ap.parse_args()

    stations = STATIONS_NORTH if args.direction == "north" \
                              else list(reversed(STATIONS_NORTH))

    trip_id = datetime.now().strftime("%Y%m%d_%H%M") + "_" + args.direction
    out_dir = args.out_root / trip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    source = (MockSource(args.mock_wav, args.sr) if args.mock_wav
              else SerialSource(args.port, args.sr))
    recorder = Recorder(source, out_dir / "audio.wav", args.sr)
    CaptureUI(recorder, stations, out_dir, trip_id, args.direction).run()


if __name__ == "__main__":
    main()
