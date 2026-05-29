"""Local CPU proof-of-concept: does the synthesis bridge the clean->cabin gap?

The clean Path-1 model fires on ZERO live announcements (measured).  Before
committing to a Colab notebook + the Monday live-inference plan, this script
answers the load-bearing question locally:

  Train KWS+CNN on {clean+train-noise synthesis (+/- real live)} from some
  trips, then run the FULL continuous pipeline over a HELD-OUT real trip and
  score per-station detection + classification.

Two conditions are compared so we learn *how much real live data matters*:
  A. synth-only      (clean announcements + real train noise, no real PA)
  B. synth + real    (also the held-in trips' real cabin-PA announcements)

If A already detects most held-out stations -> additive noise is the gap, and
3 Colab trips will be plenty.  If only B works -> PA EQ/reverb matters and we
lean on real data.  If neither -> the Monday live plan needs rethinking now.

Run:  python scripts/path2_poc.py
(float Keras models; INT8 quantisation done later in the notebook.)
"""
import collections
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
import tensorflow as tf
from tensorflow.keras import layers, models


def set_seeds(s: int = 0):
    """Pin every RNG so val accuracy stops swinging run-to-run (was 12-71%)."""
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(ROOT, "data", "processed", "wav")
LIVE_DIR = os.path.join(ROOT, "data", "raw", "line1_live")

# All marks hand-corrected (precise). Leave-one-out over the 3 trips.
TRIPS = ["20260527_0654_등교", "20260528_0642_등교", "20260527_1431_하교"]

KW = int(D.KWS_WIN * D.SR)
CW = int(D.CNN_WIN * D.SR)
SLIDE = int(0.25 * D.SR)
TRIG, CONF, MINRUN = 0.60, 0.50, 3
COOLDOWN_S = 20.0   # ignore the 2nd "이번역" of transfer/express stations; << ~119 s inter-station


def small_cnn(shape, n):
    return models.Sequential([
        layers.Input(shape=shape),
        layers.Conv2D(8, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(n, activation="softmax"),
    ])


def normalise(X):
    m, s = float(X.mean()), float(X.std()) + 1e-6
    return (X - m) / s, m, s


def train(X, Y, n_cls, epochs, seed=0):
    set_seeds(seed)
    Xn, m, s = normalise(X)
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    Xtr, Xva, Ytr, Yva = train_test_split(Xn, Y, test_size=0.15,
                                           stratify=Y, random_state=42)
    mdl = small_cnn(X.shape[1:], n_cls)
    mdl.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    classes = np.unique(Ytr)
    w = compute_class_weight("balanced", classes=classes, y=Ytr)
    cw = {int(c): float(wi) for c, wi in zip(classes, w)}
    mdl.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=epochs, batch_size=64,
            class_weight=cw, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=6, restore_best_weights=True)])
    _, acc = mdl.evaluate(Xva, Yva, verbose=0)
    return mdl, (m, s), acc


def predict_batch(mdl, mels, norm):
    m, s = norm
    x = ((np.asarray(mels, np.float32) - m) / s)[..., None]
    return mdl.predict(x, batch_size=256, verbose=0)


def _full_logmel(y):
    """power-dB mel of the whole signal (no CMN yet) + frames/sec."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=D.SR, n_fft=D.N_FFT, hop_length=D.MEL_HOP, n_mels=D.N_MELS)
    return librosa.power_to_db(mel, ref=1.0).astype(np.float32)


def _win_cmn(full, f0, nframes, cmn=True):
    w = full[:, f0:f0 + nframes]
    if w.shape[1] < nframes:
        w = np.pad(w, ((0, 0), (0, nframes - w.shape[1])))
    return w - w.mean(axis=1, keepdims=True) if cmn else w   # per-window CMN (= to_logmel)


def run_pipeline(y, kws, kn, cnn, cn, cmn=True):
    """Continuous KWS slide + CNN classify -> [(t, station, conf)] (= app_path1.c).

    The CNN window is taken AT the trigger onset (= "이번역은 [primary name]"),
    not +1.5 s (which used to land on "역입니다"/부역명). After a detection, a
    COOLDOWN_S cooldown swallows the 2nd "이번역" of transfer/express stations.
    """
    full = _full_logmel(y)
    fps = D.SR / D.MEL_HOP
    kfr = int(round(D.KWS_WIN * fps)); cfr = int(round(D.CNN_WIN * fps))
    fhop = max(1, int(round(0.25 * fps)))
    cooldown_fr = int(round(COOLDOWN_S * fps))
    starts = list(range(0, full.shape[1] - kfr + 1, fhop))
    kp = predict_batch(kws, [_win_cmn(full, f, kfr, cmn) for f in starts], kn)[:, 1]
    dets, run, rstart, j = [], 0, 0, 0
    while j < len(starts):
        if kp[j] > TRIG:
            if run == 0:
                rstart = starts[j]
            run += 1
            j += 1
        else:
            if run >= MINRUN:
                p = predict_batch(cnn, [_win_cmn(full, rstart, cfr, cmn)], cn)[0]
                k = int(p.argmax())
                dets.append((rstart / fps, D.TARGET13[k], float(p[k])))
                stop = rstart + cooldown_fr
                while j < len(starts) and starts[j] < stop:
                    j += 1
            else:
                j += 1
            run = 0
    return dets


def cnn_direct(test, cnn, cn, cmn=True):
    """Isolate CNN cross-trip accuracy: classify the [mark, mark+CNN_WIN] window
    at each mark (majority vote over small +-jitter, mirroring inference)."""
    full = _full_logmel(test.y)
    fps = D.SR / D.MEL_HOP
    cfr = int(round(D.CNN_WIN * fps))
    ok, preds = 0, []
    for station, idx in test.marks:
        f0 = int(round(idx / D.SR * fps))
        wins = [_win_cmn(full, f0 + int(round(d * fps)), cfr, cmn)
                for d in (-0.2, -0.1, 0.0, 0.1, 0.2)]
        pr = predict_batch(cnn, wins, cn)
        pred = collections.Counter(pr.argmax(1)).most_common(1)[0][0]
        preds.append(int(pred))
        ok += (D.TARGET13[pred] == station)
    return ok, len(test.marks), preds


def score(test, dets):
    """Match detections to ground-truth marks within +-12 s."""
    hit_det = hit_cls = 0
    used = [False] * len(dets)
    for station, idx in test.marks:
        t0 = idx / D.SR
        best = None
        for di, (dt, ds, dc) in enumerate(dets):
            if used[di] or abs(dt - t0) > 12.0:
                continue
            if best is None or abs(dt - t0) < abs(dets[best][0] - t0):
                best = di
        if best is not None:
            used[best] = True
            hit_det += 1
            hit_cls += (dets[best][1] == station)
    return hit_det, hit_cls, len(test.marks)


def main():
    try:                       # Windows console is cp949; keep Korean prints safe
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("loading sources ...")
    clean = D.load_clean_sources(CLEAN_DIR)
    print(f"  clean sources: {len(clean)} (target stations)\n")

    SNR, RVB = (0.0, 25.0), 0.0   # real noise only; reverb off (artificial-param mismatch)
    cd_ok_sum = cd_tot_sum = hc_sum = hd_sum = tot_sum = 0
    for held in TRIPS:
        train_ids = [t for t in TRIPS if t != held]
        trips = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in train_ids]
        test = D.load_live_trip(os.path.join(LIVE_DIR, held))
        rng = np.random.default_rng(0)
        t0 = time.time()
        Xk, Yk = D.build_kws(clean, trips, rng, snr=SNR, reverb_p=RVB)
        Xc, Yc, _ = D.build_cnn(clean, trips, rng, snr=SNR, reverb_p=RVB)
        kws, kn, kacc = train(Xk, Yk, 2, epochs=25)
        cnn, cn, cacc = train(Xc, Yc, len(D.TARGET13), epochs=35)
        cd_ok, cd_tot, preds = cnn_direct(test, cnn, cn)
        dets = run_pipeline(test.y, kws, kn, cnn, cn)
        hd, hc, tot = score(test, dets)
        gt = [D.LABEL_IDX[s] for s, _ in test.marks]
        print(f"===== held-out #{TRIPS.index(held)} ({held[9:13]}) - {tot} marks =====")
        print(f"  build {time.time()-t0:.0f}s | KWS X{Xk.shape} pos={int(Yk.sum())} "
              f"neg={int((Yk==0).sum())} | CNN X{Xc.shape}")
        print(f"  HELD-IN val:  KWS {kacc*100:.0f}%  CNN {cacc*100:.0f}%")
        print(f"  HELD-OUT cross-trip:  CNN-direct {cd_ok}/{cd_tot} ({cd_ok/cd_tot*100:.0f}%)"
              f"  |  pipeline {len(dets)} dets, matched {hd}/{tot}, correct {hc}/{tot}")
        print(f"    GT  idx: {gt}")
        print(f"    pred idx: {preds}")
        print(f"    pred hist: {dict(sorted(collections.Counter(preds).items()))}\n")
        cd_ok_sum += cd_ok; cd_tot_sum += cd_tot
        hc_sum += hc; hd_sum += hd; tot_sum += tot
    print(f"LOO totals:  CNN-direct {cd_ok_sum}/{cd_tot_sum} ({cd_ok_sum/cd_tot_sum*100:.0f}%)"
          f"  |  pipeline matched {hd_sum}/{tot_sum}, correct {hc_sum}/{tot_sum}")


if __name__ == "__main__":
    main()
