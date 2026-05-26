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

On start, a small dialog asks 등교 / 하교 — the choice goes into the
trip_id and seeds the initial station order. Use the ↻ 방향 전환 button
inside the UI to flip mid-trip for a round-trip recording.

Mock mode (no board yet) — replays an existing wav at real-time pace
so the UI is fully testable on a laptop:
  python scripts/path2_capture_ui.py --mock-wav data/processed/wav/성균관대.wav

Real run:
  python scripts/path2_capture_ui.py --port COM5
"""
import argparse
import array
import json
import math
import sys
import threading
import time
import tkinter as tk
import wave
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1호선 친구 통학 구간 (13역).
# 하교: 성균관대 → 구로 (이 순서가 baseline)
# 등교: 구로 → 성균관대 (= reversed)
STATIONS_HAGYO = [
    "성균관대", "의왕", "당정", "군포", "금정", "명학", "안양",
    "관악", "석수", "금천구청", "독산", "가산디지털단지", "구로",
]


def stations_for(direction: str) -> list[str]:
    """Return the station order for a given direction label."""
    return STATIONS_HAGYO if direction == "하교" else list(reversed(STATIONS_HAGYO))


# ---------- audio sources ----------

class SerialSource:
    """Read raw 16-bit PCM bytes from STM32 over USART2 / ST-Link VCP."""
    def __init__(self, port: str, sr: int):
        import serial  # imported lazily so mock mode has no dep
        # ST-Link VCP forwards a real UART, so the baud must match the STM32
        # firmware (USART2 at 921600 baud in our Path 2 capture firmware).
        self.ser = serial.Serial(port, baudrate=921600, timeout=0.1)
        # Drop any stale buffered bytes so the very first sample we read is
        # aligned to a 2-byte little-endian boundary.
        self.ser.reset_input_buffer()
        self.sr = sr
        self._tail = b""  # carry-over orphan byte to keep sample alignment

    def chunks(self):
        while True:
            data = self.ser.read(4096)
            if not data:
                continue
            data = self._tail + data
            if len(data) % 2:
                # Hold the trailing byte for the next read so frombytes()
                # never sees an odd-length buffer (which would crash the thread).
                self._tail = data[-1:]
                data = data[:-1]
            else:
                self._tail = b""
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
    """Drains a source into a wav file; exposes live sample count, RMS, peak."""
    def __init__(self, source, wav_path: Path, sr: int):
        self.source = source
        self.sr = sr
        self.wav = wave.open(str(wav_path), "wb")
        self.wav.setnchannels(1)
        self.wav.setsampwidth(2)
        self.wav.setframerate(sr)
        self._count = 0
        self._rms = 0.0       # RMS of the most recently received chunk
        self._peak = 0        # |max| over chunk
        self._peak_hold = 0   # decays toward current peak
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
            # Compute level stats on the just-received chunk
            samples = array.array("h")  # signed 16-bit
            samples.frombytes(chunk)
            n = len(samples)
            if n > 0:
                # RMS — use sum of squares as int to avoid float in tight loop
                ssq = 0
                pk = 0
                for s in samples:
                    ssq += s * s
                    a = -s if s < 0 else s
                    if a > pk:
                        pk = a
                rms = math.sqrt(ssq / n)
                with self._lock:
                    self._count += n
                    self._rms = rms
                    self._peak = pk
                    # Peak hold: rises instantly, decays slowly
                    if pk > self._peak_hold:
                        self._peak_hold = pk
                    else:
                        self._peak_hold = int(self._peak_hold * 0.92)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.wav.close()

    def current_sample(self) -> int:
        with self._lock:
            return self._count

    def level(self):
        """Return (rms, peak, peak_hold) of the most recent chunk."""
        with self._lock:
            return self._rms, self._peak, self._peak_hold


# ---------- UI ----------

class CaptureUI:
    def __init__(self, recorder: Recorder, stations, out_dir: Path,
                 trip_id: str, direction: str):
        self.recorder = recorder
        self.out_dir = out_dir
        self.trip_id = trip_id

        # Current segment state
        self.current_direction = direction
        self.current_stations = list(stations)
        self.marks = {}                # station_idx -> mark dict
        self.next_idx = 0
        self.segment_start_sample = 0  # sample index when this segment began

        # Completed segments (filled on flip / on close)
        self.segments = []

        self.root = tk.Tk()
        self.root.title(f"Path 2 capture — {trip_id}")
        self.root.geometry("520x820")
        self._build()
        self.root.bind("<space>", self._on_space)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._tick)

    # ---------- layout ----------
    def _build(self):
        # ===== Top monitor panel: trip header + large level meter =====
        top = tk.Frame(self.root, bg="#1a1a1a")
        top.pack(fill="x")

        head = tk.Frame(top, bg="#1a1a1a")
        head.pack(fill="x", padx=10, pady=(8, 4))
        self.direction_var = tk.StringVar(value=self.current_direction)
        tk.Label(head, textvariable=self.direction_var, fg="#fc4", bg="#1a1a1a",
                 font=("", 16, "bold")).pack(side="left")
        tk.Label(head, text=f"  ·  {self.trip_id}", fg="#aaa", bg="#1a1a1a",
                 font=("", 10)).pack(side="left")
        self.elapsed_var = tk.StringVar(value="0:00")
        tk.Label(head, textvariable=self.elapsed_var, fg="#7e7", bg="#1a1a1a",
                 font=("Consolas", 22, "bold")).pack(side="right")

        self.meter_canvas = tk.Canvas(top, height=60, bg="#0a0a0a",
                                      highlightthickness=0)
        self.meter_canvas.pack(fill="x", padx=10, pady=(2, 4))

        self.level_var = tk.StringVar(value="rms     0    pk     0    -∞ dBFS")
        tk.Label(top, textvariable=self.level_var, fg="#bbb", bg="#1a1a1a",
                 font=("Consolas", 11)).pack(padx=10, pady=(0, 6), anchor="w")

        # ===== Hint =====
        tk.Label(self.root,
                 text="Space = 다음 역 마크    ·    클릭 = 임의 역 마크/재마크",
                 fg="#555").pack(pady=(6, 2))

        # ===== Scrollable station list =====
        list_outer = tk.Frame(self.root)
        list_outer.pack(fill="both", expand=True, padx=8, pady=4)

        self.list_canvas = tk.Canvas(list_outer, highlightthickness=0)
        scrollbar = tk.Scrollbar(list_outer, orient="vertical",
                                 command=self.list_canvas.yview)
        self.list_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.list_canvas.pack(side="left", fill="both", expand=True)

        self.list_inner = tk.Frame(self.list_canvas)
        self.list_window = self.list_canvas.create_window(
            (0, 0), window=self.list_inner, anchor="nw")

        def _on_inner_configure(_e):
            self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all"))
        self.list_inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(e):
            # Make inner frame width track the canvas width (for full-width buttons)
            self.list_canvas.itemconfig(self.list_window, width=e.width)
        self.list_canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling
        def _on_wheel(e):
            self.list_canvas.yview_scroll(-int(e.delta / 120), "units")
        self.list_canvas.bind_all("<MouseWheel>", _on_wheel)

        self.btns = []
        self._rebuild_station_buttons()

        # ===== Bottom buttons =====
        bottom = tk.Frame(self.root)
        bottom.pack(fill="x", padx=8, pady=(4, 10))
        tk.Button(bottom, text="↻ 방향 전환 (왕복 다음 구간)", font=("", 11),
                  bg="#345", fg="white",
                  command=self._flip_direction).pack(fill="x", pady=(0, 4))
        tk.Button(bottom, text="종료 & 저장 (창 X 도 동일)", font=("", 12),
                  bg="#d44", fg="white",
                  command=self._on_close).pack(fill="x")

    def _rebuild_station_buttons(self):
        for b in self.btns:
            b.destroy()
        self.btns = []
        for i, name in enumerate(self.current_stations):
            b = tk.Button(self.list_inner, text="", font=("", 14), height=2,
                          anchor="w",
                          command=lambda i=i: self._mark(i))
            b.pack(fill="x", padx=4, pady=2)
            self.btns.append(b)
        self._refresh()

    # ---------- live updates ----------
    def _draw_meter(self):
        c = self.meter_canvas
        c.delete("all")
        w = c.winfo_width() or 480
        h = c.winfo_height() or 60
        rms, peak, hold = self.recorder.level()

        # log-scale (dBFS). Clamp to [-60, 0].
        def to_x(level):
            if level <= 1:
                db = -60.0
            else:
                db = 20.0 * math.log10(level / 32767.0)
            if db < -60.0: db = -60.0
            if db > 0.0: db = 0.0
            return int((db + 60.0) / 60.0 * w), db

        rms_x, rms_db = to_x(rms)
        hold_x, _ = to_x(hold)

        # Background zones: green (-60..-12), yellow (-12..-3), red (-3..0)
        g_end = int((48 / 60.0) * w)
        y_end = int((57 / 60.0) * w)
        c.create_rectangle(0,     6, g_end, h-6, fill="#143", outline="")
        c.create_rectangle(g_end, 6, y_end, h-6, fill="#430", outline="")
        c.create_rectangle(y_end, 6, w,     h-6, fill="#410", outline="")

        # RMS bar (lit portion)
        if rms_x > 0:
            color = "#7e7" if rms_x < g_end else ("#fc4" if rms_x < y_end else "#f55")
            c.create_rectangle(0, 8, rms_x, h-8, fill=color, outline="")

        # Peak-hold tick
        if hold_x > 0:
            c.create_line(hold_x, 0, hold_x, h, fill="#fff", width=2)

        # dB scale ticks (every 6 dB from -60 to 0)
        for db_tick in range(-60, 1, 6):
            x = int((db_tick + 60) / 60.0 * w)
            c.create_line(x, h-4, x, h, fill="#888")
            if db_tick % 12 == 0:
                c.create_text(x + 2, h - 12, text=str(db_tick),
                              anchor="sw", fill="#888",
                              font=("Consolas", 7))

        db_str = "-∞" if rms < 1 else f"{rms_db:5.1f}"
        self.level_var.set(f"rms {int(rms):5d}    pk {peak:5d}    {db_str} dBFS")

    def _refresh(self):
        for i, b in enumerate(self.btns):
            name = self.current_stations[i]
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
            "station": self.current_stations[idx],
            "sample_index": s,
            "elapsed_s": round(s / self.recorder.sr, 3),
            "wall_time": datetime.now().isoformat(timespec="seconds"),
        }
        while self.next_idx < len(self.current_stations) and self.next_idx in self.marks:
            self.next_idx += 1
        self._refresh()
        # Auto-scroll the list so the next station is visible
        if self.next_idx < len(self.btns):
            try:
                btn = self.btns[self.next_idx]
                self.list_canvas.update_idletasks()
                # fraction of next button from top of inner frame
                inner_h = self.list_inner.winfo_height() or 1
                y_top = btn.winfo_y() / inner_h
                self.list_canvas.yview_moveto(max(0.0, y_top - 0.1))
            except Exception:
                pass

    def _on_space(self, _evt):
        if self.next_idx < len(self.current_stations):
            self._mark(self.next_idx)

    def _tick(self):
        s = self.recorder.current_sample()
        sec = s / self.recorder.sr
        self.elapsed_var.set(f"{int(sec)//60}:{int(sec)%60:02d}")
        self._draw_meter()
        self.root.after(50, self._tick)  # 20 Hz refresh

    # ---------- direction flip (round-trip support) ----------
    def _snapshot_segment(self):
        """Save the current direction's marks as a completed segment."""
        if not self.marks:
            return None
        end_sample = self.recorder.current_sample()
        sorted_marks = sorted(self.marks.values(), key=lambda m: m["sample_index"])
        seg = {
            "direction": self.current_direction,
            "stations_route": list(self.current_stations),
            "start_sample": self.segment_start_sample,
            "end_sample": end_sample,
            "start_elapsed_s": round(self.segment_start_sample / self.recorder.sr, 3),
            "end_elapsed_s": round(end_sample / self.recorder.sr, 3),
            "marks": sorted_marks,
        }
        self.segments.append(seg)
        return seg

    def _flip_direction(self):
        self._snapshot_segment()
        # Swap 등교 ↔ 하교 (reverses the station order).
        new_dir = "하교" if self.current_direction == "등교" else "등교"
        self.current_direction = new_dir
        self.current_stations = list(reversed(self.current_stations))
        self.marks = {}
        self.next_idx = 0
        self.segment_start_sample = self.recorder.current_sample()
        self.direction_var.set(self.current_direction)
        self._rebuild_station_buttons()
        self.list_canvas.yview_moveto(0.0)

    # ---------- shutdown ----------
    def _on_close(self):
        self.recorder.stop()
        self._snapshot_segment()
        out = {
            "trip_id": self.trip_id,
            "sample_rate": self.recorder.sr,
            "audio_file": "audio.wav",
            "segments": self.segments,
        }
        (self.out_dir / "marks.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        total_s = self.recorder.current_sample() / self.recorder.sr
        n_marks = sum(len(s["marks"]) for s in self.segments)
        n_stations = sum(len(s["stations_route"]) for s in self.segments)
        print(f"\n저장 완료: {self.out_dir}/")
        print(f"  audio.wav  {total_s:.1f}s")
        print(f"  marks.json  {len(self.segments)} 구간, "
              f"{n_marks}/{n_stations} 역")
        self.root.destroy()

    def run(self):
        self.recorder.start()
        self.root.mainloop()


# ---------- main ----------

def ask_direction() -> str:
    """Pop a small picker so the user chooses 등교 / 하교 at start.

    Returns "등교" or "하교", or None if the user closed the dialog.
    """
    win = tk.Tk()
    win.title("Path 2 — 방향 선택")
    win.geometry("420x230")
    win.resizable(False, False)
    picked = {"v": None}

    def choose(v):
        picked["v"] = v
        win.destroy()

    tk.Label(win, text="이번 트립은?", font=("", 16, "bold")).pack(pady=(16, 8))
    tk.Button(win, text="📚  등교   (구로 → 성균관대)",
              font=("", 14), height=2, bg="#fc4",
              command=lambda: choose("등교")).pack(fill="x", padx=24, pady=4)
    tk.Button(win, text="🏫  하교   (성균관대 → 구로)",
              font=("", 14), height=2, bg="#7e7",
              command=lambda: choose("하교")).pack(fill="x", padx=24, pady=4)
    tk.Label(win, text="(왕복이면 트립 중간에 ↻ 방향 전환 버튼으로 전환)",
             fg="#666", font=("", 9)).pack(pady=(8, 0))
    win.mainloop()
    return picked["v"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="USB-CDC serial port (e.g. COM5, /dev/ttyACM0)")
    src.add_argument("--mock-wav", type=Path,
                     help="replay an existing wav at real-time pace")
    ap.add_argument("--sr", type=int, default=16000, help="sample rate (Hz)")
    ap.add_argument("--out-root", type=Path,
                    default=ROOT / "data" / "raw" / "line1_live")
    args = ap.parse_args()

    # Pick starting direction via small Tk dialog (mock mode defaults to 하교).
    if args.mock_wav:
        direction = "하교"
    else:
        direction = ask_direction()
        if direction is None:
            sys.exit("방향 선택 취소.")

    stations = stations_for(direction)
    trip_id = datetime.now().strftime("%Y%m%d_%H%M") + "_" + direction
    out_dir = args.out_root / trip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    source = (MockSource(args.mock_wav, args.sr) if args.mock_wav
              else SerialSource(args.port, args.sr))
    recorder = Recorder(source, out_dir / "audio.wav", args.sr)
    CaptureUI(recorder, stations, out_dir, trip_id, direction).run()


if __name__ == "__main__":
    main()
