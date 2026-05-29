"""Path 2 post-trip mark recheck + correction UI (matplotlib edition).

Open a recorded trip (audio.wav + marks.json), see all marks overlaid on
the full-trip waveform, listen to short windows, and snap any mis-timed
mark to the actual announcement moment.

The waveform is rendered by matplotlib (FigureCanvasTkAgg) instead of
raw Tk Canvas — Canvas rendering came up blank in our env, matplotlib
is mature and known-good.

Controls:
  Click waveform     → place playhead there & play from there (snaps to other marks)
  Click a table row  → select that station, seek to its mark
  Space              → play / stop window from playhead (cursor stays put)
  ← / →              → nudge playhead 0.2s
  Shift + ← / →      → nudge 1.0s
  Enter              → set selected mark's sample_index to playhead
  Ctrl+S             → save (.bak first, then overwrite)

Run:
  python scripts/path2_recheck.py                  # defaults to repo-root "audio (1).wav" + "marks (1).json"
  python scripts/path2_recheck.py --trip 20260527_0654_등교
  python scripts/path2_recheck.py --audio path/to/audio.wav --marks path/to/marks.json
"""
import argparse
import atexit
import bisect
import json
import sys
import tempfile
import time
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
import traceback
import uuid
import wave
import winsound
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    sys.exit("matplotlib not installed. run: pip install matplotlib")


ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "data" / "raw" / "line1_live"
A_TRAIN = ROOT / "A_train"  # raw trip audio + marks being hand-corrected
ENV_BUCKETS = 8000  # waveform downsample target; high enough to stay smooth when zoomed in


# ---------- I/O ----------

def load_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, "mono only"
        assert wf.getsampwidth() == 2, "16-bit only"
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16), sr


def envelope(pcm: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sample_index_centers, abs-max-per-bucket)."""
    if n <= 0 or len(pcm) == 0:
        return np.zeros(0), np.zeros(0)
    bucket = max(1, len(pcm) // n)
    trimmed = pcm[: bucket * n]
    env = np.abs(trimmed.reshape(n, bucket)).max(axis=1).astype(np.float32)
    centers = (np.arange(n) + 0.5) * bucket
    return centers, env


def find_marks_for(audio_path: Path) -> Path | None:
    """Heuristic: marks.json that pairs with audio.wav, looking in the same dir.

    Order of attempts (first match wins):
      1. Same stem with 'audio' → 'marks' (e.g. 'audio (1).wav' → 'marks (1).json')
      2. Plain 'marks.json' next to it
      3. Any marks*.json in the same dir (alphabetical)
    """
    parent = audio_path.parent
    stem = audio_path.stem
    lowered = stem.lower()
    if "audio" in lowered:
        i = lowered.index("audio")
        cand = parent / (stem[:i] + "marks" + stem[i + len("audio"):] + ".json")
        if cand.exists():
            return cand
    cand = parent / "marks.json"
    if cand.exists():
        return cand
    others = sorted(parent.glob("marks*.json"))
    return others[0] if others else None


def pick_audio_via_dialog() -> Path | None:
    """Open a file picker. Returns selected audio path, or None on cancel."""
    # Prefer the hand-marking A_train dir, then the live-trip dir, else repo root.
    initial = A_TRAIN if A_TRAIN.exists() else (LIVE if LIVE.exists() else ROOT)
    # Use a hidden Tk root so the dialog can render; mainloop hasn't started.
    chooser = tk.Tk()
    chooser.withdraw()
    chooser.attributes("-topmost", True)
    picked = filedialog.askopenfilename(
        parent=chooser,
        title="Pick a trip audio.wav",
        initialdir=str(initial),
        filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
    )
    chooser.destroy()
    return Path(picked) if picked else None


def resolve_inputs(args) -> tuple[Path, Path]:
    if args.audio and args.marks:
        return Path(args.audio), Path(args.marks)
    if args.trip:
        d = LIVE / args.trip
        return d / "audio.wav", d / "marks.json"
    audio = pick_audio_via_dialog()
    if not audio:
        sys.exit("no audio picked")
    marks = find_marks_for(audio)
    if marks is None:
        # Let the user pick the marks file manually.
        chooser = tk.Tk()
        chooser.withdraw()
        chooser.attributes("-topmost", True)
        picked = filedialog.askopenfilename(
            parent=chooser,
            title=f"No marks.json paired with {audio.name} — pick one",
            initialdir=str(audio.parent),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        chooser.destroy()
        if not picked:
            sys.exit("no marks picked")
        marks = Path(picked)
    return audio, marks


# ---------- mark bookkeeping ----------

def flag_out_of_order(marks: list[dict]) -> set[int]:
    """Return station_idx values whose chronological order disagrees with route order.

    Method: sort marks by sample_index, take the longest non-decreasing subsequence
    of station_idx as the 'trusted' set, flag the rest as suspects.
    """
    if not marks:
        return set()
    s = sorted(marks, key=lambda m: m["sample_index"])
    idx = [m["station_idx"] for m in s]
    n = len(idx)
    tails: list[int] = []
    tail_idx: list[int] = []
    parent = [-1] * n
    for i, v in enumerate(idx):
        pos = bisect.bisect_right(tails, v)
        if pos == len(tails):
            tails.append(v)
            tail_idx.append(i)
        else:
            tails[pos] = v
            tail_idx[pos] = i
        parent[i] = tail_idx[pos - 1] if pos > 0 else -1
    keep: set[int] = set()
    cur = tail_idx[-1] if tail_idx else -1
    while cur >= 0:
        keep.add(cur)
        cur = parent[cur]
    return {s[i]["station_idx"] for i in range(n) if i not in keep}


# ---------- app ----------

class App:
    WINDOW_S = 10.0
    PREVIEW_S = 2.0  # short snip for fine-tune (bound to 'p')
    NUDGE_S = 0.2
    NUDGE_BIG_S = 1.0
    TICK_MS = 100  # 10 fps playback tick

    def __init__(self, root: tk.Tk, audio_path: Path, marks_path: Path):
        self.audio_path = audio_path
        self.marks_path = marks_path
        self.pcm, self.sr = load_wav(audio_path)
        self.total_s = len(self.pcm) / self.sr
        self.info = json.loads(marks_path.read_text(encoding="utf-8"))
        self.segment = self.info["segments"][0]
        self.marks: list[dict] = sorted(self.segment["marks"],
                                        key=lambda m: m["station_idx"])
        self.suspect: set[int] = flag_out_of_order(self.marks)

        self.playhead = 0
        self.selected_idx: int | None = None
        self.window_s = self.WINDOW_S
        self.playing = False
        self._play_path: Path | None = None
        self._play_anchor_sample = 0
        self._play_anchor_time = 0.0
        self._play_end_sample = 0
        self._dirty = False
        self._suppress_select = False

        atexit.register(self._cleanup_temp)

        self.root = root
        root.title(f"Path 2 recheck — {self.info.get('trip_id', audio_path.name)}")
        root.geometry("1400x800")
        self._build_ui()
        self._draw_initial_plot()
        self._refresh_table()
        if self.suspect:
            self.jump_to_next_suspect()
        elif self.marks:
            first = self.marks[0]
            self._select(first["station_idx"], first["sample_index"])

    # ---- layout ----
    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=4)
        self.header = tk.Label(top, text=self._header_text(),
                               font=("TkDefaultFont", 10))
        self.header.pack(side="left")

        self.fig = Figure(figsize=(13, 3), dpi=100, facecolor="#101418")
        self.ax = self.fig.add_subplot(111)
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.fig_canvas.get_tk_widget().pack(fill="x", padx=8, pady=(4, 0))
        self.fig_canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.fig_canvas.mpl_connect("scroll_event", self._on_scroll)

        # Horizontal scrollbar driving xlim so the user can pan when zoomed.
        self.scrollbar = ttk.Scrollbar(self.root, orient="horizontal",
                                       command=self._on_scrollbar)
        self.scrollbar.pack(fill="x", padx=8, pady=(0, 4))
        self.scrollbar.set(0.0, 1.0)

        self.time_label = tk.Label(self.root, text="0:00.00 / 0:00.00",
                                   font=("Consolas", 10))
        self.time_label.pack(anchor="w", padx=8)

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=8, pady=6)

        left = tk.Frame(body)
        left.pack(side="left", fill="y")
        cols = ("idx", "station", "elapsed", "status")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=14,
                                 selectmode="browse")
        for c, w, anchor in (("idx", 40, "e"), ("station", 140, "w"),
                             ("elapsed", 90, "e"), ("status", 80, "center")):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor=anchor)
        self.tree.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_table_select)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        sb.pack(side="left", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.tag_configure("ok", background="#1a3a1a", foreground="#cfe9cf")
        self.tree.tag_configure("bad", background="#3a1a1a", foreground="#f0c0c0")
        self.tree.tag_configure("sel", background="#3a3a14", foreground="#fff2a8")

        right = tk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        ctl = tk.LabelFrame(right, text="Playback / Edit")
        ctl.pack(fill="x")

        row = tk.Frame(ctl); row.pack(fill="x", padx=6, pady=4)
        tk.Button(row, text="▶ Play window (Space)", width=22,
                  command=self.toggle_play).pack(side="left")
        tk.Button(row, text="▶▶ Preview 2s (p)", width=18,
                  command=self.preview).pack(side="left", padx=4)
        tk.Button(row, text="■ Stop", width=8,
                  command=self.stop_play).pack(side="left", padx=4)
        tk.Label(row, text="window:").pack(side="left", padx=(12, 2))
        self.win_var = tk.StringVar(value=str(self.WINDOW_S))
        tk.Spinbox(row, from_=2, to=60, increment=1, width=4,
                   textvariable=self.win_var).pack(side="left")
        tk.Label(row, text="s").pack(side="left")

        row2 = tk.Frame(ctl); row2.pack(fill="x", padx=6, pady=4)
        tk.Button(row2, text="⏮ -1s", command=lambda: self.nudge(-1.0)).pack(side="left")
        tk.Button(row2, text="◀ -0.2s", command=lambda: self.nudge(-0.2)).pack(side="left", padx=2)
        tk.Button(row2, text="+0.2s ▶", command=lambda: self.nudge(0.2)).pack(side="left", padx=2)
        tk.Button(row2, text="+1s ⏭", command=lambda: self.nudge(1.0)).pack(side="left")

        row3 = tk.Frame(ctl); row3.pack(fill="x", padx=6, pady=6)
        tk.Button(row3, text="↳ Next suspect", width=18,
                  command=self.jump_to_next_suspect).pack(side="left")
        tk.Button(row3, text="Set selected → playhead (Enter)", width=28,
                  command=self.commit_selected).pack(side="left", padx=8)
        tk.Button(row3, text="🔍 Reset view",
                  command=self.reset_view).pack(side="left", padx=8)

        row4 = tk.Frame(ctl); row4.pack(fill="x", padx=6, pady=(0, 6))
        tk.Button(row4, text="💾 Save (Ctrl+S)", width=18,
                  command=self.save).pack(side="left")
        self.dirty_label = tk.Label(row4, text="", fg="#d99")
        self.dirty_label.pack(side="left", padx=8)

        info = tk.LabelFrame(right, text="Log")
        info.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text = tk.Text(info, height=10, bg="#181c20", fg="#cfe2d8",
                                font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("p", lambda e: self.preview())
        self.root.bind("P", lambda e: self.preview())
        self.root.bind("<Left>", lambda e: self.nudge(-self.NUDGE_S))
        self.root.bind("<Right>", lambda e: self.nudge(self.NUDGE_S))
        self.root.bind("<Shift-Left>", lambda e: self.nudge(-self.NUDGE_BIG_S))
        self.root.bind("<Shift-Right>", lambda e: self.nudge(self.NUDGE_BIG_S))
        self.root.bind("<Return>", lambda e: self.commit_selected())
        self.root.bind("<Control-s>", lambda e: self.save())

    def _header_text(self) -> str:
        return (f"Trip: {self.info.get('trip_id','?')}   "
                f"Audio: {self.audio_path.name}   "
                f"Duration: {self.total_s:.1f}s   "
                f"Marks: {len(self.marks)}   "
                f"Suspect: {len(self.suspect)}")

    # ---- plot ----
    def _draw_initial_plot(self):
        centers, env = envelope(self.pcm, ENV_BUCKETS)
        t = centers / self.sr
        peak = float(env.max()) if env.size else 1.0

        ax = self.ax
        ax.clear()
        ax.set_facecolor("#101418")
        for spine in ax.spines.values():
            spine.set_color("#3a3e44")
        ax.tick_params(colors="#aab", labelsize=8)
        ax.fill_between(t, -env, env, color="#4a8aa0", linewidth=0)
        ax.set_xlim(0, self.total_s)
        ax.set_ylim(-peak * 1.15, peak * 1.15)
        ax.set_yticks([])
        ax.set_xlabel("seconds", color="#aab", fontsize=8)

        self._label_y = peak * 1.08
        self.mark_lines: dict[int, object] = {}
        self.mark_labels: dict[int, object] = {}
        for m in self.marks:
            x = m["sample_index"] / self.sr
            color = self._mark_color(m["station_idx"])
            ln = ax.axvline(x, color=color, lw=1.5, alpha=0.9)
            txt = ax.text(x, self._label_y, f"{m['station_idx']}",
                          color=color, fontsize=8, ha="left", va="bottom")
            self.mark_lines[m["station_idx"]] = ln
            self.mark_labels[m["station_idx"]] = txt

        self.playhead_line = ax.axvline(0, color="#ff8a00", lw=1.2,
                                        linestyle="--", alpha=0.9)
        # Separate transient cursor that animates during playback, so the edit
        # playhead (above) stays exactly where the user clicked / nudged it.
        self.playback_line = ax.axvline(0, color="#46c8ff", lw=1.0, alpha=0.85)
        self.playback_line.set_visible(False)
        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.20)
        self.fig_canvas.draw_idle()

    def _mark_color(self, idx: int) -> str:
        if idx == self.selected_idx:
            return "#ffe85c"
        if idx in self.suspect:
            return "#ff5555"
        return "#5cff8c"

    def _refresh_marks_visuals(self):
        for idx, ln in self.mark_lines.items():
            color = self._mark_color(idx)
            ln.set_color(color)
            self.mark_labels[idx].set_color(color)
        self.fig_canvas.draw_idle()

    def _move_mark(self, idx: int, sample: int):
        x = sample / self.sr
        self.mark_lines[idx].set_xdata([x, x])
        self.mark_labels[idx].set_x(x)

    # ---- table ----
    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for m in self.marks:
            idx = m["station_idx"]
            tag = "sel" if idx == self.selected_idx else ("bad" if idx in self.suspect else "ok")
            status = "⚠" if idx in self.suspect else "✓"
            self.tree.insert("", "end", iid=str(idx),
                             values=(idx, m["station"], f"{m['elapsed_s']:.2f}", status),
                             tags=(tag,))
        if self.selected_idx is not None:
            self._suppress_select = True
            try:
                self.tree.selection_set(str(self.selected_idx))
                self.tree.see(str(self.selected_idx))
            except tk.TclError:
                pass
            finally:
                self._suppress_select = False

    def _on_table_select(self, _evt):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        # Tk fires <<TreeviewSelect>> asynchronously when we delete+reinsert
        # rows in _refresh_table. A naive sync flag does NOT suppress those
        # queued events — they arrive after the flag is cleared and trigger
        # _select → _refresh_table → more queued events → infinite recursion
        # inside update_idletasks(). Bail when the row already matches state.
        if idx == self.selected_idx:
            return
        m = next((mk for mk in self.marks if mk["station_idx"] == idx), None)
        if m:
            self._select(idx, m["sample_index"])

    # ---- plot interaction ----
    def _on_plot_click(self, evt):
        if evt.inaxes != self.ax or evt.xdata is None or evt.button != 1:
            return
        # Pixel-based snap so distance feels the same at any zoom level:
        # if the click lands within ~8 px of a mark line, snap to it.
        snap_idx = None
        snap_dx_px = 8
        for m in self.marks:
            # Don't snap to the mark you're currently editing — clicking near it
            # must reposition freely, not jump back to its old spot.
            if m["station_idx"] == self.selected_idx:
                continue
            mx_s = m["sample_index"] / self.sr
            mx_px, _ = self.ax.transData.transform((mx_s, 0))
            d = abs(evt.x - mx_px)
            if d < snap_dx_px:
                snap_dx_px = d
                snap_idx = m["station_idx"]
        if snap_idx is not None:
            m = next(mk for mk in self.marks if mk["station_idx"] == snap_idx)
            self._select(snap_idx, m["sample_index"])
        else:
            self._set_playhead(int(float(evt.xdata) * self.sr))
        # Click = "play from here". Uses the spinbox window length; press 'p'
        # for a short 2s snip instead.
        self.play_window()

    # ---- zoom / pan ----
    MIN_VIEW_S = 0.5
    ZOOM_IN_FACTOR = 0.7
    ZOOM_OUT_FACTOR = 1.4

    def _on_scroll(self, evt):
        if evt.inaxes != self.ax:
            return
        xlim = self.ax.get_xlim()
        width = xlim[1] - xlim[0]
        cx = evt.xdata if evt.xdata is not None else (xlim[0] + xlim[1]) / 2
        factor = self.ZOOM_IN_FACTOR if evt.button == "up" else self.ZOOM_OUT_FACTOR
        new_left = cx - (cx - xlim[0]) * factor
        new_right = cx + (xlim[1] - cx) * factor
        new_left = max(0.0, new_left)
        new_right = min(self.total_s, new_right)
        if new_right - new_left < self.MIN_VIEW_S:
            return
        self._apply_xlim(new_left, new_right)

    def _on_scrollbar(self, *args):
        cmd = args[0]
        xlim = self.ax.get_xlim()
        width = xlim[1] - xlim[0]
        if cmd == "moveto":
            frac = float(args[1])
            new_left = max(0.0, min(self.total_s - width, frac * self.total_s))
        elif cmd == "scroll":
            units = int(args[1])
            step_what = args[2] if len(args) > 2 else "units"
            step = width * (0.1 if step_what == "units" else 0.9) * units
            new_left = max(0.0, min(self.total_s - width, xlim[0] + step))
        else:
            return
        self._apply_xlim(new_left, new_left + width)

    def _apply_xlim(self, left: float, right: float):
        self.ax.set_xlim(left, right)
        self._sync_scrollbar()
        self.fig_canvas.draw_idle()

    def _sync_scrollbar(self):
        xlim = self.ax.get_xlim()
        lo = max(0.0, xlim[0]) / self.total_s
        hi = min(self.total_s, xlim[1]) / self.total_s
        self.scrollbar.set(lo, hi)

    def reset_view(self):
        self._apply_xlim(0.0, self.total_s)

    # ---- selection / playhead ----
    def _select(self, idx: int, sample: int):
        self.selected_idx = idx
        self.playhead = max(0, min(len(self.pcm) - 1, sample))
        self._refresh_marks_visuals()
        self._update_playhead_only()
        self._refresh_table()

    def _set_playhead(self, sample: int):
        self.playhead = max(0, min(len(self.pcm) - 1, sample))
        self._update_playhead_only()

    def _set_playback_cursor(self, sample: int):
        """Move the transient playback cursor (does NOT touch the edit playhead)."""
        s = sample / self.sr
        self.playback_line.set_xdata([s, s])
        self.playback_line.set_visible(True)
        self.fig_canvas.draw_idle()

    def _hide_playback_cursor(self):
        if self.playback_line.get_visible():
            self.playback_line.set_visible(False)
            self.fig_canvas.draw_idle()

    def _update_playhead_only(self):
        s = self.playhead / self.sr
        self.playhead_line.set_xdata([s, s])
        self.fig_canvas.draw_idle()
        self.time_label.config(text=f"{s/60:.0f}:{s%60:05.2f} / "
                                    f"{self.total_s/60:.0f}:{self.total_s%60:05.2f}")

    # ---- playback ----
    def toggle_play(self):
        if self.playing:
            self.stop_play()
        else:
            self.play_window()

    def play_window(self):
        try:
            self.window_s = float(self.win_var.get())
        except ValueError:
            pass
        n = int(self.window_s * self.sr)
        start = self.playhead  # play from exactly where the user clicked, no pre-roll
        end = min(len(self.pcm), start + n)
        self._play_range(start, end)

    def preview(self):
        """Play a short snip (PREVIEW_S) from the playhead forward.

        Use when fine-tuning a mark: nudge the playhead, hit 'p' to confirm
        the announcement start landed under it, repeat. Faster than waiting
        for a full 10s window."""
        start = self.playhead
        end = min(len(self.pcm), start + int(self.PREVIEW_S * self.sr))
        self._play_range(start, end)

    def _play_range(self, start: int, end: int):
        slice_ = self.pcm[start:end]
        winsound.PlaySound(None, 0)
        self._cleanup_temp()
        path = Path(tempfile.gettempdir()) / f"path2_recheck_{uuid.uuid4().hex}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(slice_.tobytes())
        self._play_path = path
        try:
            winsound.PlaySound(str(path),
                               winsound.SND_FILENAME | winsound.SND_ASYNC)
        except RuntimeError as exc:
            self._log(f"playback failed: {exc}")
            return
        self.playing = True
        self._play_anchor_sample = start
        self._play_anchor_time = time.monotonic()
        self._play_end_sample = end
        self._log(f"▶ {start/self.sr:.2f}s → {end/self.sr:.2f}s "
                  f"({(end-start)/self.sr:.1f}s)")
        self.root.after(int((end - start) / self.sr * 1000) + 50,
                        self._playback_done)
        self._tick_playhead()

    def _tick_playhead(self):
        if not self.playing:
            self._hide_playback_cursor()
            return
        elapsed = time.monotonic() - self._play_anchor_time
        new_head = self._play_anchor_sample + int(elapsed * self.sr)
        if new_head >= self._play_end_sample:
            self._hide_playback_cursor()
            return
        self._set_playback_cursor(new_head)
        self.root.after(self.TICK_MS, self._tick_playhead)

    def _playback_done(self):
        self.playing = False

    def stop_play(self):
        winsound.PlaySound(None, 0)
        self.playing = False
        self._hide_playback_cursor()
        self._cleanup_temp()

    def _cleanup_temp(self):
        if self._play_path and self._play_path.exists():
            try:
                self._play_path.unlink()
            except OSError:
                pass
        self._play_path = None

    # ---- editing ----
    def nudge(self, ds: float):
        self._set_playhead(self.playhead + int(ds * self.sr))

    def commit_selected(self):
        if self.selected_idx is None:
            self._log("(no mark selected)")
            return
        m = next((mk for mk in self.marks if mk["station_idx"] == self.selected_idx), None)
        if not m:
            return
        old = m["sample_index"]
        m["sample_index"] = int(self.playhead)
        m["elapsed_s"] = round(self.playhead / self.sr, 3)
        self.suspect = flag_out_of_order(self.marks)
        self._move_mark(self.selected_idx, m["sample_index"])
        self._refresh_marks_visuals()
        self._refresh_table()
        self.header.config(text=self._header_text())
        self._dirty = True
        self.dirty_label.config(text="● unsaved", fg="#d99")
        self._log(f"set [{self.selected_idx}] {m['station']}: "
                  f"{old/self.sr:.2f}s → {self.playhead/self.sr:.2f}s")

    def jump_to_next_suspect(self):
        if not self.suspect:
            self._log("(no suspect marks)")
            return
        suspect_marks = sorted(
            [m for m in self.marks if m["station_idx"] in self.suspect],
            key=lambda m: m["station_idx"])
        nxt = None
        for m in suspect_marks:
            if self.selected_idx is None or m["station_idx"] > self.selected_idx:
                nxt = m
                break
        if nxt is None:
            nxt = suspect_marks[0]
        lo, hi = self._search_window(nxt["station_idx"])
        self.selected_idx = nxt["station_idx"]
        self.playhead = (lo + hi) // 2
        # Zoom the waveform to the search window with a small margin so the
        # user can see the candidate region clearly — pan/wheel still works
        # to adjust afterwards.
        lo_s, hi_s = lo / self.sr, hi / self.sr
        pad = max(2.0, (hi_s - lo_s) * 0.1)
        self._apply_xlim(max(0.0, lo_s - pad),
                         min(self.total_s, hi_s + pad))
        self._refresh_marks_visuals()
        self._update_playhead_only()
        self._refresh_table()
        self._log(f"→ [{nxt['station_idx']}] {nxt['station']}: "
                  f"search {lo/self.sr:.1f}s … {hi/self.sr:.1f}s")

    def _search_window(self, target_idx: int) -> tuple[int, int]:
        good = sorted(
            [m for m in self.marks if m["station_idx"] not in self.suspect],
            key=lambda m: m["station_idx"])
        before = [m for m in good if m["station_idx"] < target_idx]
        after = [m for m in good if m["station_idx"] > target_idx]
        lo = before[-1]["sample_index"] if before else 0
        hi = after[0]["sample_index"] if after else len(self.pcm) - 1
        return lo, hi

    def save(self):
        bak = self.marks_path.with_suffix(self.marks_path.suffix + ".bak")
        if not bak.exists():
            bak.write_bytes(self.marks_path.read_bytes())
            self._log(f"backup → {bak.name}")
        self.segment["marks"] = sorted(self.marks, key=lambda m: m["sample_index"])
        self.marks_path.write_text(
            json.dumps(self.info, ensure_ascii=False, indent=2),
            encoding="utf-8")
        self._dirty = False
        self.dirty_label.config(text="saved ✓", fg="#9d9")
        self.root.after(1500, lambda: self.dirty_label.config(text=""))
        self._log(f"saved → {self.marks_path}")

    # ---- log ----
    def _log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")


def _run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio")
    ap.add_argument("--marks")
    ap.add_argument("--trip")
    args = ap.parse_args()
    audio, marks = resolve_inputs(args)
    if not audio.exists():
        sys.exit(f"audio not found: {audio}")
    if not marks.exists():
        sys.exit(f"marks not found: {marks}")
    print(f"loading {audio}")
    print(f"        {marks}")
    root = tk.Tk()
    App(root, audio, marks)
    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.focus_force()
    print("UI ready")
    root.mainloop()


def main():
    # Surface unhandled errors via messagebox so the .pyw launcher (no
    # console) doesn't fail silently.
    try:
        _run()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        try:
            r = tk.Tk()
            r.withdraw()
            messagebox.showerror("path2_recheck crashed", tb)
            r.destroy()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
