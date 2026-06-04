"""Path 2 door-event marking UI — mark 문열림 / 문닫힘(출발) per stop.

Sibling of path2_recheck.py (UI patterns reused verbatim: matplotlib waveform,
winsound playback, wheel-zoom, scrollbar, crash dialog). path2_recheck is left
untouched — this writes a SEPARATE door_events.json and never modifies marks.json.

Purpose: hand-label the two door events at every station stop so we can derive
the 3 acoustic-event classes for the station-by-counting idea:
    주행  = between a 닫힘 and the next 열림 (running)
    정차  = [열림, 닫힘]  (dwell, doors open, ~20 s)
    출발  = around 닫힘   (삐리리 chime + 덜컥, the distinctive countable marker)
Door OPEN is quiet ("안전문이 열립니다" + soft 달칵); door CLOSE/departure is the
loud, characteristic event — mark both, CLOSE precisely.

The existing announcement marks (marks.json) are drawn as faint reference lines,
and 'n' jumps to the next one + zooms there, because each stop sits shortly after
its arrival announcement — fastest way to find each stop.

Controls:
  Click waveform      → place playhead there & play from there
  Click near an event → select it (for delete / re-seek)
  o                   → add 열림 (door open) at playhead
  c                   → add 닫힘=차임 (출발 삐리리 chime onset) at playhead
  t                   → add 탁 (door-close clunk, the mechanical thunk) at playhead
  Delete / x          → delete selected event
  Space               → play / stop window from playhead
  p                   → 2 s preview from playhead
  ← / →   (Shift = 1s)→ nudge playhead
  n                   → jump to next announcement mark (+zoom) ; N = previous
  Ctrl+S              → save door_events.json (.bak first)

Run:
  python scripts/path2_event_mark.py                      # file picker
  python scripts/path2_event_mark.py --trip 20260527_0654_등교
  python scripts/path2_event_mark.py --audio path/to/audio.wav
"""
import argparse
import atexit
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

# Reuse the proven, identical waveform/IO helpers from the recheck app.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from path2_recheck import load_wav, envelope, ENV_BUCKETS, find_marks_for  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "data" / "raw" / "line1_live"
A_TRAIN = ROOT / "A_train"   # raw trip audio being hand-marked (same default as recheck)

OPEN, CLOSE, CLUNK = "open", "close", "clunk"
TYPE_KR = {OPEN: "열림", CLOSE: "닫힘", CLUNK: "탁"}
# open=cyan, close(차임)=orange, clunk(탁)=green
TYPE_COLOR = {OPEN: "#46c8ff", CLOSE: "#ff8a00", CLUNK: "#5cff7a"}
LABEL_CHAR = {OPEN: "O", CLOSE: "C", CLUNK: "T"}


# ---------- input resolution ----------
def pick_audio_via_dialog() -> Path | None:
    initial = A_TRAIN if A_TRAIN.exists() else (LIVE if LIVE.exists() else ROOT)
    chooser = tk.Tk()
    chooser.withdraw()
    chooser.attributes("-topmost", True)
    picked = filedialog.askopenfilename(
        parent=chooser, title="Pick a trip audio.wav",
        initialdir=str(initial),
        filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
    chooser.destroy()
    return Path(picked) if picked else None


def resolve_inputs(args) -> Path:
    if args.audio:
        return Path(args.audio)
    if args.trip:
        return LIVE / args.trip / "audio.wav"
    audio = pick_audio_via_dialog()
    if not audio:
        sys.exit("no audio picked")
    return audio


def load_announcement_marks(marks_path: Path | None) -> list[tuple[str, int]]:
    """(station, sample_index) reference marks from the paired marks file.

    Display only. A_train uses 'audio (N).wav' / 'marks (N).json' naming, so the
    path is resolved by recheck's find_marks_for (stem swap), not a fixed name.
    """
    if not marks_path or not marks_path.exists():
        return []
    info = json.loads(marks_path.read_text(encoding="utf-8"))
    segs = info.get("segments") or [{"marks": info.get("marks", [])}]
    out = []
    for s in segs:
        for m in s.get("marks", []):
            if "sample_index" in m:
                out.append((m.get("station", "?"), int(m["sample_index"])))
    return sorted(out, key=lambda x: x[1])


# ---------- app ----------
class App:
    WINDOW_S = 10.0
    PREVIEW_S = 2.0
    NUDGE_S = 0.2
    NUDGE_BIG_S = 1.0
    TICK_MS = 100
    SNAP_PX = 8
    JUMP_PRE_S = 3.0          # zoom margin before an announcement mark
    JUMP_POST_S = 95.0        # ... and after (announcement -> stop -> dwell -> 출발)

    def __init__(self, root: tk.Tk, audio_path: Path):
        self.audio_path = audio_path
        self.trip_dir = audio_path.parent
        self.trip_id = audio_path.stem                       # e.g. "audio (1)"
        self.marks_path = find_marks_for(audio_path)         # paired marks (N).json
        # per-audio output so A_train's 4 audio files don't clobber one file
        self.out_path = self.trip_dir / (audio_path.stem + ".door_events.json")
        self.pcm, self.sr = load_wav(audio_path)
        self.total_s = len(self.pcm) / self.sr
        self.ann_marks = load_announcement_marks(self.marks_path)
        self._ann_cursor = -1

        self.events: list[dict] = self._load_events()   # {id,type,sample_index}
        self._next_id = (max((e["id"] for e in self.events), default=0) + 1)
        self.selected_id: int | None = None

        self.playhead = 0
        self.window_s = self.WINDOW_S
        self.playing = False
        self._play_path: Path | None = None
        self._play_anchor_sample = 0
        self._play_anchor_time = 0.0
        self._play_end_sample = 0
        self._continuous = False         # keep chaining chunks until stopped
        self._play_gen = 0               # invalidates stale after()-callbacks
        self._dirty = False
        atexit.register(self._cleanup_temp)

        self.root = root
        root.title(f"Path 2 door-event mark — {self.audio_path.name}")
        root.geometry("1400x820")
        self._build_ui()
        self._draw_initial_plot()
        self._refresh_table()
        self._update_playhead_only()

    # ---- I/O ----
    def _load_events(self) -> list[dict]:
        if not self.out_path.exists():
            return []
        try:
            data = json.loads(self.out_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        out = []
        for i, e in enumerate(data.get("events", [])):
            if e.get("type") in (OPEN, CLOSE, CLUNK) and "sample_index" in e:
                out.append({"id": i + 1, "type": e["type"],
                            "sample_index": int(e["sample_index"])})
        return sorted(out, key=lambda e: e["sample_index"])

    def save(self):
        if self.out_path.exists():
            bak = self.out_path.with_suffix(".json.bak")
            bak.write_bytes(self.out_path.read_bytes())
            self._log(f"backup → {bak.name}")
        ev = sorted(self.events, key=lambda e: e["sample_index"])
        payload = {
            "trip_id": self.trip_id,
            "audio_file": self.audio_path.name,
            "sample_rate": self.sr,
            "events": [{"type": e["type"], "sample_index": e["sample_index"],
                        "elapsed_s": round(e["sample_index"] / self.sr, 3)}
                       for e in ev],
        }
        self.out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._dirty = False
        self.dirty_label.config(text="saved ✓", fg="#9d9")
        self.root.after(1500, lambda: self.dirty_label.config(text=""))
        self._log(f"saved {len(ev)} events → {self.out_path.name}")

    # ---- layout ----
    def _build_ui(self):
        top = tk.Frame(self.root); top.pack(fill="x", padx=8, pady=4)
        self.header = tk.Label(top, text=self._header_text(),
                               font=("TkDefaultFont", 10))
        self.header.pack(side="left")

        self.fig = Figure(figsize=(13, 3), dpi=100, facecolor="#101418")
        self.ax = self.fig.add_subplot(111)
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.fig_canvas.get_tk_widget().pack(fill="x", padx=8, pady=(4, 0))
        self.fig_canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.fig_canvas.mpl_connect("scroll_event", self._on_scroll)

        self.scrollbar = ttk.Scrollbar(self.root, orient="horizontal",
                                       command=self._on_scrollbar)
        self.scrollbar.pack(fill="x", padx=8, pady=(0, 4))
        self.scrollbar.set(0.0, 1.0)

        self.time_label = tk.Label(self.root, text="0:00.00 / 0:00.00",
                                   font=("Consolas", 10))
        self.time_label.pack(anchor="w", padx=8)

        body = tk.Frame(self.root); body.pack(fill="both", expand=True, padx=8, pady=6)

        left = tk.Frame(body); left.pack(side="left", fill="y")
        cols = ("time", "type")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=16,
                                 selectmode="browse")
        for c, w, anchor in (("time", 90, "e"), ("type", 70, "center")):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor=anchor)
        self.tree.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_table_select)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        sb.pack(side="left", fill="y"); self.tree.configure(yscrollcommand=sb.set)
        self.tree.tag_configure("open", background="#10303a", foreground="#bfe9ff")
        self.tree.tag_configure("close", background="#3a2a10", foreground="#ffd9a0")
        self.tree.tag_configure("clunk", background="#103a18", foreground="#bfffce")
        self.tree.tag_configure("sel", background="#3a3a14", foreground="#fff2a8")

        right = tk.Frame(body); right.pack(side="left", fill="both", expand=True,
                                           padx=(12, 0))
        ctl = tk.LabelFrame(right, text="Mark / Playback"); ctl.pack(fill="x")

        row = tk.Frame(ctl); row.pack(fill="x", padx=6, pady=4)
        tk.Button(row, text="＋ 열림 (o)", width=12, bg="#103040",
                  command=lambda: self.add_event(OPEN)).pack(side="left")
        tk.Button(row, text="＋ 닫힘=차임 (c)", width=14, bg="#3a2a10",
                  command=lambda: self.add_event(CLOSE)).pack(side="left", padx=4)
        tk.Button(row, text="＋ 탁 (t)", width=10, bg="#103a18",
                  command=lambda: self.add_event(CLUNK)).pack(side="left", padx=4)
        tk.Button(row, text="🗑 삭제 (Del)", width=12,
                  command=self.delete_selected).pack(side="left", padx=4)

        row2 = tk.Frame(ctl); row2.pack(fill="x", padx=6, pady=4)
        tk.Button(row2, text="▶ Play window (Space)", width=20,
                  command=self.toggle_play).pack(side="left")
        tk.Button(row2, text="▶▶ 2s (p)", width=10,
                  command=self.preview).pack(side="left", padx=4)
        tk.Button(row2, text="■ Stop", width=7,
                  command=self.stop_play).pack(side="left", padx=4)
        tk.Label(row2, text="win:").pack(side="left", padx=(10, 2))
        self.win_var = tk.StringVar(value=str(self.WINDOW_S))
        tk.Spinbox(row2, from_=2, to=60, increment=1, width=4,
                   textvariable=self.win_var).pack(side="left")

        row3 = tk.Frame(ctl); row3.pack(fill="x", padx=6, pady=4)
        tk.Button(row3, text="◀ -1s", command=lambda: self.nudge(-1.0)).pack(side="left")
        tk.Button(row3, text="◀ -0.2", command=lambda: self.nudge(-0.2)).pack(side="left", padx=2)
        tk.Button(row3, text="+0.2 ▶", command=lambda: self.nudge(0.2)).pack(side="left", padx=2)
        tk.Button(row3, text="+1s ▶", command=lambda: self.nudge(1.0)).pack(side="left")
        tk.Button(row3, text="↳ 다음 방송 (n)", width=14,
                  command=lambda: self.jump_announcement(+1)).pack(side="left", padx=(10, 0))
        tk.Button(row3, text="🔍 전체보기",
                  command=self.reset_view).pack(side="left", padx=6)

        row4 = tk.Frame(ctl); row4.pack(fill="x", padx=6, pady=(0, 6))
        tk.Button(row4, text="💾 Save (Ctrl+S)", width=16,
                  command=self.save).pack(side="left")
        self.dirty_label = tk.Label(row4, text="", fg="#d99")
        self.dirty_label.pack(side="left", padx=8)

        info = tk.LabelFrame(right, text="Log"); info.pack(fill="both", expand=True,
                                                            pady=(8, 0))
        self.log_text = tk.Text(info, height=10, bg="#181c20", fg="#cfe2d8",
                                font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("o", lambda e: self.add_event(OPEN))
        self.root.bind("c", lambda e: self.add_event(CLOSE))
        self.root.bind("t", lambda e: self.add_event(CLUNK))
        self.root.bind("p", lambda e: self.preview())
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("x", lambda e: self.delete_selected())
        self.root.bind("<Left>", lambda e: self.nudge(-self.NUDGE_S))
        self.root.bind("<Right>", lambda e: self.nudge(self.NUDGE_S))
        self.root.bind("<Shift-Left>", lambda e: self.nudge(-self.NUDGE_BIG_S))
        self.root.bind("<Shift-Right>", lambda e: self.nudge(self.NUDGE_BIG_S))
        self.root.bind("n", lambda e: self.jump_announcement(+1))
        self.root.bind("N", lambda e: self.jump_announcement(-1))
        self.root.bind("<Control-s>", lambda e: self.save())

    def _header_text(self) -> str:
        n_open = sum(1 for e in self.events if e["type"] == OPEN)
        n_close = sum(1 for e in self.events if e["type"] == CLOSE)
        n_clunk = sum(1 for e in self.events if e["type"] == CLUNK)
        return (f"Audio: {self.audio_path.name}   Dur: {self.total_s:.0f}s   "
                f"열림: {n_open}   닫힘(차임): {n_close}   탁: {n_clunk}   "
                f"(이번역은 refs: {len(self.ann_marks)})   → {self.out_path.name}")

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
        self._label_y = peak * 1.06
        self._peak = peak

        # "이번역은" announcement reference lines (display only) — violet, clearly
        # visible and distinct from open(cyan)/close(orange)/playhead(yellow), so
        # you can navigate by them ('n' jumps + zooms to the next one).
        for st, idx in self.ann_marks:
            x = idx / self.sr
            ax.axvline(x, color="#b18cff", lw=1.1, ls=(0, (5, 3)), alpha=0.8)
            ax.text(x, -peak * 1.10, st, color="#c9b3ff", fontsize=8,
                    ha="center", va="top", rotation=0)

        self.playhead_line = ax.axvline(0, color="#ffe85c", lw=1.3, ls="--", alpha=0.95)
        self.playback_line = ax.axvline(0, color="#46ffa0", lw=1.0, alpha=0.85)
        self.playback_line.set_visible(False)
        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.22)
        self._event_artists: list = []
        self._redraw_events()

    def _redraw_events(self):
        for a in self._event_artists:
            a.remove()
        self._event_artists = []
        for e in self.events:
            x = e["sample_index"] / self.sr
            sel = e["id"] == self.selected_id
            color = "#fff2a8" if sel else TYPE_COLOR[e["type"]]
            lw = 2.2 if sel else 1.4
            ln = self.ax.axvline(x, color=color, lw=lw, alpha=0.95)
            txt = self.ax.text(x, self._label_y, LABEL_CHAR[e["type"]],
                               color=color, fontsize=9, fontweight="bold",
                               ha="center", va="bottom")
            self._event_artists += [ln, txt]
        self.fig_canvas.draw_idle()

    # ---- table ----
    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for e in sorted(self.events, key=lambda e: e["sample_index"]):
            t = e["sample_index"] / self.sr
            tag = "sel" if e["id"] == self.selected_id else e["type"]
            self.tree.insert("", "end", iid=str(e["id"]),
                             values=(f"{t:7.2f}", TYPE_KR[e["type"]]), tags=(tag,))
        if self.selected_id is not None:
            try:
                self.tree.selection_set(str(self.selected_id))
                self.tree.see(str(self.selected_id))
            except tk.TclError:
                pass
        self.header.config(text=self._header_text())

    def _on_table_select(self, _evt):
        sel = self.tree.selection()
        if not sel:
            return
        eid = int(sel[0])
        if eid == self.selected_id:
            return
        e = next((x for x in self.events if x["id"] == eid), None)
        if e:
            self.selected_id = eid
            self._set_playhead(e["sample_index"])
            self._redraw_events()

    # ---- plot interaction ----
    def _on_plot_click(self, evt):
        if evt.inaxes != self.ax or evt.xdata is None or evt.button != 1:
            return
        # snap to an existing event line if the click is within SNAP_PX → select it
        snap_id, best = None, self.SNAP_PX
        for e in self.events:
            ex_px, _ = self.ax.transData.transform((e["sample_index"] / self.sr, 0))
            d = abs(evt.x - ex_px)
            if d < best:
                best, snap_id = d, e["id"]
        if snap_id is not None:
            self.selected_id = snap_id
            e = next(x for x in self.events if x["id"] == snap_id)
            self._set_playhead(e["sample_index"])
            self._redraw_events()
            self._refresh_table()
        else:
            self.selected_id = None
            self._set_playhead(int(float(evt.xdata) * self.sr))
            self._redraw_events()
        self.play_window()

    # ---- zoom / pan (identical behaviour to recheck) ----
    MIN_VIEW_S = 0.5
    ZOOM_IN_FACTOR = 0.7
    ZOOM_OUT_FACTOR = 1.4

    def _on_scroll(self, evt):
        if evt.inaxes != self.ax:
            return
        xlim = self.ax.get_xlim()
        cx = evt.xdata if evt.xdata is not None else (xlim[0] + xlim[1]) / 2
        factor = self.ZOOM_IN_FACTOR if evt.button == "up" else self.ZOOM_OUT_FACTOR
        new_left = max(0.0, cx - (cx - xlim[0]) * factor)
        new_right = min(self.total_s, cx + (xlim[1] - cx) * factor)
        if new_right - new_left < self.MIN_VIEW_S:
            return
        self._apply_xlim(new_left, new_right)

    def _on_scrollbar(self, *args):
        cmd = args[0]
        xlim = self.ax.get_xlim()
        width = xlim[1] - xlim[0]
        if cmd == "moveto":
            new_left = max(0.0, min(self.total_s - width, float(args[1]) * self.total_s))
        elif cmd == "scroll":
            units = int(args[1])
            step_what = args[2] if len(args) > 2 else "units"
            step = width * (0.1 if step_what == "units" else 0.9) * units
            new_left = max(0.0, min(self.total_s - width, xlim[0] + step))
        else:
            return
        self._apply_xlim(new_left, new_left + width)

    def _apply_xlim(self, left, right):
        self.ax.set_xlim(left, right)
        self._sync_scrollbar()
        self.fig_canvas.draw_idle()

    def _sync_scrollbar(self):
        xlim = self.ax.get_xlim()
        self.scrollbar.set(max(0.0, xlim[0]) / self.total_s,
                           min(self.total_s, xlim[1]) / self.total_s)

    def reset_view(self):
        self._apply_xlim(0.0, self.total_s)

    # ---- announcement navigation ----
    def jump_announcement(self, step: int):
        if not self.ann_marks:
            self._log("(no announcement marks to navigate)")
            return
        self._ann_cursor = (self._ann_cursor + step) % len(self.ann_marks)
        st, idx = self.ann_marks[self._ann_cursor]
        self._set_playhead(idx)
        lo = max(0.0, idx / self.sr - self.JUMP_PRE_S)
        hi = min(self.total_s, idx / self.sr + self.JUMP_POST_S)
        self._apply_xlim(lo, hi)
        self._log(f"→ announcement [{self._ann_cursor}] {st} @ {idx/self.sr:.1f}s "
                  f"(stop should follow within ~{self.JUMP_POST_S:.0f}s)")

    # ---- playhead ----
    def _set_playhead(self, sample: int):
        self.playhead = max(0, min(len(self.pcm) - 1, sample))
        self._update_playhead_only()

    def _update_playhead_only(self):
        s = self.playhead / self.sr
        self.playhead_line.set_xdata([s, s])
        self.fig_canvas.draw_idle()
        self.time_label.config(text=f"{s/60:.0f}:{s%60:05.2f} / "
                                    f"{self.total_s/60:.0f}:{self.total_s%60:05.2f}")

    def _set_playback_cursor(self, sample: int):
        s = sample / self.sr
        self.playback_line.set_xdata([s, s])
        self.playback_line.set_visible(True)
        self.fig_canvas.draw_idle()

    def _hide_playback_cursor(self):
        if self.playback_line.get_visible():
            self.playback_line.set_visible(False)
            self.fig_canvas.draw_idle()

    # ---- editing ----
    def nudge(self, ds: float):
        self._set_playhead(self.playhead + int(ds * self.sr))

    def add_event(self, etype: str):
        e = {"id": self._next_id, "type": etype, "sample_index": int(self.playhead)}
        self._next_id += 1
        self.events.append(e)
        self.selected_id = e["id"]
        self._mark_dirty()
        self._redraw_events()
        self._refresh_table()
        self._log(f"+ {TYPE_KR[etype]} @ {self.playhead/self.sr:.2f}s")

    def delete_selected(self):
        if self.selected_id is None:
            self._log("(no event selected)")
            return
        e = next((x for x in self.events if x["id"] == self.selected_id), None)
        if not e:
            return
        self.events.remove(e)
        self._log(f"- {TYPE_KR[e['type']]} @ {e['sample_index']/self.sr:.2f}s")
        self.selected_id = None
        self._mark_dirty()
        self._redraw_events()
        self._refresh_table()

    def _mark_dirty(self):
        self._dirty = True
        self.dirty_label.config(text="● unsaved", fg="#d99")

    # ---- playback (verbatim winsound approach from recheck) ----
    def toggle_play(self):
        self.stop_play() if self.playing else self.play_window()

    def play_window(self, continuous: bool = True):
        """Continuous by default: play in window_s chunks, chaining chunk→chunk so
        the cursor keeps sweeping until Stop / Space / a new click. (Click and
        Space use this; 'p' preview is a one-shot 2 s snip.)"""
        try:
            self.window_s = float(self.win_var.get())
        except ValueError:
            pass
        start = self.playhead
        end = min(len(self.pcm), start + int(self.window_s * self.sr))
        self._play_range(start, end, continuous=continuous)

    def preview(self):
        start = self.playhead
        end = min(len(self.pcm), start + int(self.PREVIEW_S * self.sr))
        self._play_range(start, end, continuous=False)

    def _play_range(self, start: int, end: int, continuous: bool = False):
        self._continuous = continuous
        self._play_gen += 1                      # supersede any in-flight playback
        gen = self._play_gen
        slice_ = self.pcm[start:end]
        winsound.PlaySound(None, 0)
        self._cleanup_temp()
        path = Path(tempfile.gettempdir()) / f"path2_event_{uuid.uuid4().hex}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.sr)
            wf.writeframes(slice_.tobytes())
        self._play_path = path
        try:
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except RuntimeError as exc:
            self._log(f"playback failed: {exc}")
            return
        self.playing = True
        self._play_anchor_sample = start
        self._play_anchor_time = time.monotonic()
        self._play_end_sample = end
        self.root.after(int((end - start) / self.sr * 1000) + 40,
                        lambda: self._playback_done(gen))
        self._tick_playhead(gen)

    def _tick_playhead(self, gen: int):
        if gen != self._play_gen or not self.playing:
            return
        elapsed = time.monotonic() - self._play_anchor_time
        new_head = self._play_anchor_sample + int(elapsed * self.sr)
        if new_head >= self._play_end_sample:
            # In continuous mode the next chunk (started by _playback_done) keeps
            # the cursor going, so don't hide it at a chunk boundary.
            if not (self._continuous and self._play_end_sample < len(self.pcm)):
                self._hide_playback_cursor()
            return
        self._set_playback_cursor(new_head)
        self.root.after(self.TICK_MS, lambda: self._tick_playhead(gen))

    def _playback_done(self, gen: int):
        if gen != self._play_gen:
            return                                # stale callback, ignore
        if self._continuous and self._play_end_sample < len(self.pcm):
            nxt_start = self._play_end_sample
            nxt_end = min(len(self.pcm), nxt_start + int(self.window_s * self.sr))
            self._play_range(nxt_start, nxt_end, continuous=True)   # chain
        else:
            self.playing = False
            self._hide_playback_cursor()

    def stop_play(self):
        winsound.PlaySound(None, 0)
        self.playing = False
        self._continuous = False
        self._play_gen += 1                       # cancel pending chained chunks
        self._hide_playback_cursor()
        self._cleanup_temp()

    def _cleanup_temp(self):
        if self._play_path and self._play_path.exists():
            try:
                self._play_path.unlink()
            except OSError:
                pass
        self._play_path = None

    # ---- log ----
    def _log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")


def _run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio")
    ap.add_argument("--trip")
    args = ap.parse_args()
    audio = resolve_inputs(args)
    if not audio.exists():
        sys.exit(f"audio not found: {audio}")
    print(f"loading {audio}")
    root = tk.Tk()
    App(root, audio)
    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.focus_force()
    print("UI ready")
    root.mainloop()


def main():
    try:
        _run()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        try:
            r = tk.Tk(); r.withdraw()
            messagebox.showerror("path2_event_mark crashed", tb)
            r.destroy()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
