"""Path 2 — KWS inspector (보고 + 듣고 점검).

현재 KWS가 한 트립에서 *무엇을* 트리거했는지 눈/귀로 검사하는 read-only 뷰어.
recheck/doormark과 같은 matplotlib 파형 + winsound 재생 + 줌/스크롤 UX. 마크는
절대 수정하지 않는다(뷰어). 두 단계로 나뉜다:

  1) BUILD (무거움, tensorflow): held-out LOO — 보는 트립을 빼고 나머지 7트립으로
     baseline KWS를 학습한 뒤 그 트립에 슬라이딩 검출. 트리거를 마크와 매칭해
     TP(맞음)/FP(헛발화)로 분류하고, KWS 확률곡선까지 캐시(JSON)로 떨군다.
       python scripts/path2_kws_inspect.py --build            # 전 트립
       python scripts/path2_kws_inspect.py --build --trip 20260527_0654_등교

  2) VIEW (가벼움, tf 불필요): 캐시 + audio.wav 로드 → 파형 위에
       · 내가 찍은 모든 마크 (탑승역=회색점선, 매칭=초록, 놓침FN=주황점선)
       · 트리거 TP(파랑▲)/FP(빨강▲)
       · KWS 확률곡선 + 임계선(0.6)
     클릭=그 지점 재생, 행 클릭=이동, 선택 윈도우의 log-mel을 raw|CMN 나란히 표시
     (CMN이 고정 채널색을 빼는 걸 직접 비교). 휠 줌 / 스크롤 팬.
       python scripts/path2_kws_inspect.py                    # 파일 피커
       python scripts/path2_kws_inspect.py --trip 20260527_0654_등교

검출기·CMN·윈도우는 path2_dataset/path2_pipeline과 동일 상수를 쓴다(아래 mirror).
"""
import argparse
import atexit
import json
import os
import sys
import tempfile
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

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "data" / "raw" / "line1_live"
CACHE = ROOT / "reports" / "kws_inspect"

# --- feature/detector constants (MIRROR path2_dataset / path2_pipeline) --------
SR, N_FFT, HOP, N_MELS = 16000, 512, 256, 40
KWS_WIN = 1.0                       # KWS sliding window (s)
KWS_TRIG = 0.6                      # trigger threshold (== pipeline)
MATCH_S = 10.0                      # trigger<->mark match tolerance (== eval)
ENV_BUCKETS = 8000


# --------------------------------------------------------------------------- #
# shared feature helpers (no tensorflow — librosa only)
# --------------------------------------------------------------------------- #
def logmel(y_float: np.ndarray) -> np.ndarray:
    """power-dB mel (ref=1.0) — identical to P._full_logmel / D.to_logmel pre-CMN."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y_float, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=1.0).astype(np.float32)


def apply_cmn(lm: np.ndarray) -> np.ndarray:
    """per-mel-bin time-mean subtraction (== D.USE_CMN path)."""
    return lm - lm.mean(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# BUILD — train held-out LOO KWS, detect, classify, cache (heavy: tensorflow)
# --------------------------------------------------------------------------- #
def build_cache(only_trip: str | None = None):
    sys.path.insert(0, str(ROOT / "path2_fullpipe" / "scripts"))
    import path2_dataset as D
    import path2_poc as P
    import path2_pipeline as PL

    PL.LIVE_DIR = str(LIVE)
    PL.CLEAN_DIR = str(ROOT / "data" / "processed" / "wav")
    D.FEATURE_MODE = "logmel_cmn"
    CACHE.mkdir(parents=True, exist_ok=True)

    trips = PL.list_trips()
    clean = D.load_clean_sources(PL.CLEAN_DIR)
    loaded = {t: D.load_live_trip(os.path.join(str(LIVE), t)) for t in trips}
    targets = [only_trip] if only_trip else trips

    for held in targets:
        if held not in loaded:
            print(f"  skip {held}: not a trip"); continue
        test = loaded[held]
        tr = [loaded[t] for t in trips if t != held]
        Xk, Yk = D.build_kws(clean, tr, np.random.default_rng(0),
                             snr=(0.0, 25.0), spec_aug=False)
        model, norm, val = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)

        # --- KWS prob curve over the whole trip (== PL.kws_triggers internals) ---
        full = P._full_logmel(test.y)
        fps = SR / HOP
        kfr = int(round(KWS_WIN * fps)); fhop = max(1, int(round(0.25 * fps)))
        starts = list(range(0, full.shape[1] - kfr + 1, fhop))
        kp = P.predict_batch(model, [P._win_cmn(full, f, kfr, True) for f in starts], norm)[:, 1]
        ptimes = np.array(starts) / fps
        triggers = PL._debounce(kp, ptimes)             # debounced trigger times (s)

        # --- marks (ALL, from marks.json) + boarding flag --------------------
        info = json.loads((LIVE / held / "marks.json").read_text(encoding="utf-8"))
        seg = info["segments"][0]
        direction = seg.get("direction", "?")
        boarding = D.BOARDING.get(direction)
        marks = []
        for m in seg["marks"]:
            marks.append({"station": m["station"], "station_idx": m["station_idx"],
                          "sample_index": int(m["sample_index"]),
                          "t": int(m["sample_index"]) / SR,
                          "is_boarding": m["station"] == boarding, "matched": False})
        real = [m for m in marks if not m["is_boarding"]]

        # --- classify triggers vs real marks (mark-centric greedy == eval) ----
        used = [False] * len(triggers)
        for m in real:
            best = None
            for i, t in enumerate(triggers):
                if used[i] or abs(t - m["t"]) > MATCH_S:
                    continue
                if best is None or abs(t - m["t"]) < abs(triggers[best] - m["t"]):
                    best = i
            if best is not None:
                used[best] = True; m["matched"] = True
        # propagate matched flag back to the full marks list
        rk = {id(m): m for m in real}
        for m in marks:
            if id(m) in rk:
                m["matched"] = rk[id(m)]["matched"]

        def _prob_at(t):
            j = int(np.argmin(np.abs(ptimes - t)))
            return float(kp[j])

        trig_rows = [{"t": float(t), "prob": _prob_at(t),
                      "type": "TP" if used[i] else "FP"}
                     for i, t in enumerate(triggers)]
        n_tp = sum(used); n_fp = len(triggers) - n_tp
        n_hit = sum(1 for m in real if m["matched"])

        out = {
            "trip_id": held, "direction": direction, "sr": SR,
            "n_samples": int(len(test.y)), "kws_trig": KWS_TRIG, "kws_win": KWS_WIN,
            "detector": "LOO held-out baseline KWS (other 7 trips)",
            "val_acc": float(val),
            "summary": {"recall": n_hit, "n_marks": len(real),
                        "tp": n_tp, "fp": n_fp},
            "marks": marks, "triggers": trig_rows,
            "prob_times": [round(float(x), 3) for x in ptimes],
            "prob_vals": [round(float(x), 3) for x in kp],
        }
        (CACHE / f"{held}.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"  {held} ({direction}): recall {n_hit}/{len(real)}  "
              f"TP {n_tp}  FP {n_fp}  (val {val*100:.0f}%) -> cached")
    print(f"\ncache → {CACHE}")


# --------------------------------------------------------------------------- #
# VIEW — read-only inspector GUI (numpy + matplotlib + winsound, no tf)
# --------------------------------------------------------------------------- #
def load_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16), sr


def envelope(pcm: np.ndarray, n: int):
    if n <= 0 or len(pcm) == 0:
        return np.zeros(0), np.zeros(0)
    bucket = max(1, len(pcm) // n)
    trimmed = pcm[: bucket * n]
    env = np.abs(trimmed.reshape(-1, bucket)).max(axis=1).astype(np.float32)
    centers = (np.arange(len(env)) + 0.5) * bucket
    return centers, env


class Inspector:
    SPEC_WIN_S = KWS_WIN          # spectrogram shows the KWS window
    PLAY_S = 2.0

    def __init__(self, root, audio_path: Path, cache_path: Path):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            sys.exit("matplotlib not installed. run: pip install matplotlib")
        self._Figure, self._Canvas = Figure, FigureCanvasTkAgg

        self.pcm, self.sr = load_wav(audio_path)
        self.total_s = len(self.pcm) / self.sr
        self.data = json.loads(cache_path.read_text(encoding="utf-8"))
        self.marks = self.data["marks"]
        self.triggers = self.data["triggers"]
        self.ptimes = np.asarray(self.data["prob_times"], np.float32)
        self.pvals = np.asarray(self.data["prob_vals"], np.float32)

        self.sel_center = 0.0          # spectrogram/play window start (s)
        self.window_s = min(60.0, self.total_s)
        self.view0 = 0.0
        self.playing = False
        self._play_path = None
        atexit.register(self._cleanup)

        self.root = root
        root.title(f"KWS inspect — {self.data['trip_id']}")
        root.geometry("1500x900")
        self._build_ui()
        self._draw_wave()
        self._fill_table()
        self._draw_spec()

    # ---------- layout ----------
    def _build_ui(self):
        s = self.data["summary"]
        head = (f"{self.data['trip_id']}  [{self.data['direction']}]   "
                f"검출기: {self.data['detector']}   |   "
                f"recall {s['recall']}/{s['n_marks']}   TP {s['tp']}   FP {s['fp']}"
                f"   (KWS val {self.data['val_acc']*100:.0f}%)")
        top = tk.Frame(self.root); top.pack(fill="x", padx=8, pady=4)
        tk.Label(top, text=head, font=("TkDefaultFont", 10)).pack(side="left")
        legend = ("● mark-hit  ◌ mark-miss(FN)  · boarding(bogus)   "
                  "▲TP  ▲FP   — KWS prob / -- thr 0.6")
        tk.Label(top, text=legend, font=("TkDefaultFont", 9),
                 fg="#888").pack(side="right")

        self.fig = self._Figure(figsize=(15, 3.2), dpi=100, facecolor="#101418")
        self.ax = self.fig.add_subplot(111)
        self.axp = self.ax.twinx()             # prob curve axis (0..1)
        self.canvas = self._Canvas(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="x", padx=8, pady=(4, 0))
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self.scroll = ttk.Scrollbar(self.root, orient="horizontal",
                                    command=self._on_scrollbar)
        self.scroll.pack(fill="x", padx=8, pady=(0, 4))

        self.tlabel = tk.Label(self.root, text="", font=("Consolas", 10))
        self.tlabel.pack(anchor="w", padx=8)

        body = tk.Frame(self.root); body.pack(fill="both", expand=True, padx=8, pady=6)

        left = tk.Frame(body); left.pack(side="left", fill="y")
        cols = ("t", "kind", "label", "status", "prob")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=22)
        for c, w in zip(cols, (70, 70, 150, 80, 60)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor="w")
        self.tree.pack(side="left", fill="y")
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        sb.pack(side="left", fill="y"); self.tree.configure(yscrollcommand=sb.set)
        self.tree.tag_configure("TP", foreground="#3a7afe")
        self.tree.tag_configure("FP", foreground="#e0483a")
        self.tree.tag_configure("hit", foreground="#2faa55")
        self.tree.tag_configure("MISS", foreground="#e08a2a")
        self.tree.tag_configure("board", foreground="#888")
        self.tree.bind("<<TreeviewSelect>>", self._on_row)

        # spectrogram panel (raw | CMN) — what the KWS window looks like
        right = tk.Frame(body); right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        tk.Label(right, text="선택 윈도우 log-mel  (왼=raw / 오른=CMN)  — 클릭/행선택으로 갱신",
                 font=("TkDefaultFont", 9)).pack(anchor="w")
        self.sfig = self._Figure(figsize=(7, 3.4), dpi=100, facecolor="#101418")
        self.sax_raw = self.sfig.add_subplot(121)
        self.sax_cmn = self.sfig.add_subplot(122)
        self.scanvas = self._Canvas(self.sfig, master=right)
        self.scanvas.get_tk_widget().pack(fill="both", expand=True)

        ctrl = tk.Frame(right); ctrl.pack(anchor="w", pady=4)
        tk.Button(ctrl, text="▶ play window (Space)", command=self.toggle_play).pack(side="left")
        tk.Label(ctrl, text="  클릭=그 지점 재생+윈도우 선택").pack(side="left")
        self.root.bind("<space>", lambda e: self.toggle_play())

    # ---------- waveform ----------
    def _draw_wave(self):
        self.ax.clear(); self.axp.clear()
        self.ax.set_facecolor("#101418")
        cx, env = envelope(self.pcm, ENV_BUCKETS)
        ts = cx / self.sr
        self.ax.fill_between(ts, -env, env, color="#3a4452", linewidth=0)
        # marks
        for m in self.marks:
            if m["is_boarding"]:
                self.ax.axvline(m["t"], color="#888", ls=":", lw=1.0, alpha=0.7)
            elif m["matched"]:
                self.ax.axvline(m["t"], color="#2faa55", ls="-", lw=1.3, alpha=0.9)
            else:
                self.ax.axvline(m["t"], color="#e08a2a", ls="--", lw=1.3, alpha=0.9)
        # triggers (markers at top)
        ymax = float(np.abs(self.pcm).max() or 1)
        for tr in self.triggers:
            c = "#3a7afe" if tr["type"] == "TP" else "#e0483a"
            self.ax.plot([tr["t"]], [ymax * 0.92], marker="^", color=c, ms=7)
        # selection
        self.selspan = self.ax.axvspan(self.sel_center, self.sel_center + self.SPEC_WIN_S,
                                       color="#ffffff", alpha=0.12)
        self.ax.set_ylim(-ymax * 1.05, ymax * 1.05)
        self.ax.set_yticks([])
        # prob curve on twin axis
        self.axp.plot(self.ptimes, self.pvals, color="#d8c64a", lw=0.8, alpha=0.9)
        self.axp.axhline(KWS_TRIG, color="#d8c64a", ls="--", lw=0.7, alpha=0.6)
        self.axp.set_ylim(0, 1.02); self.axp.set_yticks([0, 0.6, 1.0])
        self.axp.tick_params(colors="#888", labelsize=7)
        self._apply_view()

    def _apply_view(self):
        x0 = max(0.0, min(self.view0, self.total_s - self.window_s))
        x1 = x0 + self.window_s
        self.ax.set_xlim(x0, x1)
        frac = self.window_s / self.total_s if self.total_s else 1.0
        self.scroll.set(x0 / self.total_s, min(1.0, x0 / self.total_s + frac))
        self.tlabel.config(
            text=f"view {x0:6.1f}–{x1:6.1f}s / {self.total_s:.1f}s   "
                 f"sel {self.sel_center:.2f}s   zoom {self.total_s/self.window_s:4.1f}x")
        self.canvas.draw_idle()

    # ---------- spectrogram ----------
    def _draw_spec(self):
        a = int(self.sel_center * self.sr)
        n = int(self.SPEC_WIN_S * self.sr)
        w = self.pcm[a:a + n].astype(np.float32) / 32768.0
        if len(w) < n:
            w = np.pad(w, (0, n - len(w)))
        lm = logmel(w)
        cm = apply_cmn(lm)
        for ax, M, ttl in ((self.sax_raw, lm, "raw log-mel"),
                           (self.sax_cmn, cm, "CMN (per-bin mean removed)")):
            ax.clear()
            ax.imshow(M, origin="lower", aspect="auto", cmap="magma")
            ax.set_title(ttl, color="#ccc", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        self.sfig.tight_layout()
        self.scanvas.draw_idle()

    # ---------- table ----------
    def _fill_table(self):
        rows = []
        for m in self.marks:
            st = "board" if m["is_boarding"] else ("hit" if m["matched"] else "MISS")
            rows.append((m["t"], "mark", m["station"], st, ""))
        for tr in self.triggers:
            rows.append((tr["t"], "trigger", "—", tr["type"], f"{tr['prob']:.2f}"))
        rows.sort(key=lambda r: r[0])
        self._rowmeta = []
        for t, kind, label, status, prob in rows:
            self.tree.insert("", "end",
                             values=(f"{t:7.2f}", kind, label, status, prob),
                             tags=(status,))
            self._rowmeta.append(t)

    def _on_row(self, _e):
        sel = self.tree.selection()
        if not sel:
            return
        i = self.tree.index(sel[0])
        t = self._rowmeta[i]
        self.sel_center = max(0.0, t)
        self.view0 = max(0.0, t - self.window_s / 2)
        self._refresh_selection()

    # ---------- interaction ----------
    def _on_click(self, e):
        if e.inaxes not in (self.ax, self.axp) or e.xdata is None:
            return
        self.sel_center = max(0.0, float(e.xdata))
        self._refresh_selection()
        self.play_from(self.sel_center)

    def _refresh_selection(self):
        try:
            self.selspan.remove()
        except Exception:
            pass
        self.selspan = self.ax.axvspan(self.sel_center, self.sel_center + self.SPEC_WIN_S,
                                       color="#ffffff", alpha=0.12)
        self._apply_view()
        self._draw_spec()

    def _on_scroll(self, e):
        if e.xdata is None:
            return
        factor = 0.8 if e.button == "up" else 1.25
        new_w = max(2.0, min(self.total_s, self.window_s * factor))
        # zoom around cursor
        self.view0 = e.xdata - (e.xdata - self.view0) * (new_w / self.window_s)
        self.window_s = new_w
        self._apply_view()

    def _on_scrollbar(self, *args):
        if args[0] == "moveto":
            self.view0 = float(args[1]) * self.total_s
        elif args[0] == "scroll":
            self.view0 += float(args[1]) * self.window_s * 0.2
        self._apply_view()

    # ---------- playback (winsound, raw audio) ----------
    def play_from(self, t):
        a = int(t * self.sr)
        n = int(self.PLAY_S * self.sr)
        self._play(self.pcm[a:a + n])

    def toggle_play(self):
        if self.playing:
            winsound.PlaySound(None, winsound.SND_PURGE)
            self.playing = False
        else:
            self.play_from(self.sel_center)

    def _play(self, chunk):
        self._cleanup()
        p = Path(tempfile.gettempdir()) / f"kwsi_{uuid.uuid4().hex}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.sr)
            wf.writeframes(chunk.astype(np.int16).tobytes())
        self._play_path = p
        winsound.PlaySound(str(p), winsound.SND_FILENAME | winsound.SND_ASYNC)
        self.playing = True

    def _cleanup(self):
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        if self._play_path and self._play_path.exists():
            try:
                self._play_path.unlink()
            except Exception:
                pass
        self._play_path = None


def pick_audio():
    initial = LIVE if LIVE.exists() else ROOT
    c = tk.Tk(); c.withdraw(); c.attributes("-topmost", True)
    picked = filedialog.askopenfilename(
        parent=c, title="Pick a trip audio.wav", initialdir=str(initial),
        filetypes=[("WAV", "*.wav"), ("All", "*.*")])
    c.destroy()
    return Path(picked) if picked else None


def _dialog_error(title, msg):
    """Show an error via messagebox so the .lnk launcher (pythonw, no console)
    doesn't fail silently."""
    try:
        r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
        messagebox.showerror(title, msg); r.destroy()
    except Exception:
        pass


def _run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="train LOO + cache (heavy)")
    ap.add_argument("--trip", help="trip_id (e.g. 20260527_0654_등교)")
    args = ap.parse_args()

    if args.build:
        build_cache(args.trip)
        return

    if args.trip:
        audio = LIVE / args.trip / "audio.wav"
        cache = CACHE / f"{args.trip}.json"
    else:
        audio = pick_audio()
        if not audio:                       # user cancelled the picker
            return
        cache = CACHE / f"{audio.parent.name}.json"

    if not audio.exists():
        _dialog_error("KWS inspect", f"audio.wav 없음:\n{audio}")
        return
    if not cache.exists():
        _dialog_error("KWS inspect — 캐시 없음",
                      f"이 트립의 검출 캐시가 없습니다:\n{cache}\n\n"
                      f"먼저 터미널에서 빌드하세요(학습, 몇 분):\n"
                      f"  python scripts/path2_kws_inspect.py --build --trip {cache.stem}\n\n"
                      f"또는 전체:  python scripts/path2_kws_inspect.py --build")
        return
    root = tk.Tk()
    Inspector(root, audio, cache)
    root.mainloop()


def main():
    # Surface unhandled errors via messagebox so the .lnk launcher (no console)
    # doesn't fail silently (== path2_recheck.main).
    try:
        _run()
    except SystemExit:
        raise
    except Exception:
        _dialog_error("path2_kws_inspect crashed", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
