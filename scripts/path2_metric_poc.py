"""Local CPU proof-of-concept for the metric-learning station classifier.

The 13-class softmax CNN collapsed cross-trip (LOO held-out ~19% vs 8% chance):
it memorised the 13 boundaries and overfit each trip's channel/noise. Here we
instead train a small encoder so that the SAME announcement (different noise /
channel) maps to nearby embeddings, register one prototype per station, and
classify by nearest prototype. Loss = Prototypical Network (episodic), which is
exactly the inference rule (prototype distance), so train and test match.

Key lever for cross-trip: episodes draw each class's SUPPORT and QUERY from
DIFFERENT sources when available (synth vs a real trip, or two real trips), so
the encoder is pushed toward channel invariance, not just noise invariance.

Honest go/no-go: seeded leave-one-out over the 3 trips; does held-out nearest-
prototype clearly beat the 19% softmax baseline? Heavy training moves to Colab
only if this says yes.

Run:  python scripts/path2_metric_poc.py
"""
import collections
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D
import path2_poc as P                      # reuse set_seeds / _full_logmel / _win_cmn

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(ROOT, "data", "processed", "wav")
LIVE_DIR = os.path.join(ROOT, "data", "raw", "line1_live")
TRIPS = ["20260527_0654_등교", "20260528_0642_등교",
         "20260527_1431_하교", "20260528_2118_하교"]   # 2 등교 + 2 하교

EMB_DIM = 64
N_WAY = len(D.TARGET13)
N_SUPPORT, N_QUERY = 5, 5
EPISODES = int(os.environ.get("PROTO_EPISODES", "600"))   # lower for a quick smoke test
LR = 1e-3
REVERB_P = float(os.environ.get("PROTO_REVERB", "0.0"))   # synth channel-diversity aug


def build_encoder(shape, emb=EMB_DIM):
    return models.Sequential([
        layers.Input(shape=shape),
        layers.Conv2D(16, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(128, activation="relu"), layers.Dense(emb),
        layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=1)),
    ])


def group_by_class_source(Y, src):
    """label -> {source_group -> [sample indices]}."""
    by = {}
    for i, (y, s) in enumerate(zip(Y, src)):
        by.setdefault(int(y), {}).setdefault(s, []).append(i)
    return {lab: {g: np.asarray(ix) for g, ix in d.items()} for lab, d in by.items()}


def sample_episode(by, rng, n_s, n_q):
    """All classes; support/query from DIFFERENT source groups where possible."""
    labels = sorted(by.keys())
    sup, qry = [], []
    for lab in labels:
        groups = list(by[lab].keys())
        gi = rng.permutation(len(groups))
        sg = groups[gi[0]]
        qg = groups[gi[1]] if len(groups) > 1 else groups[gi[0]]
        sup.append(rng.choice(by[lab][sg], n_s, replace=len(by[lab][sg]) < n_s))
        qry.append(rng.choice(by[lab][qg], n_q, replace=len(by[lab][qg]) < n_q))
    return np.stack(sup), np.stack(qry), labels       # (n_way,n_s), (n_way,n_q)


def _sq_dists(q, protos):
    """(Nq,d),(Nw,d) -> (Nq,Nw) squared euclidean."""
    return (tf.reduce_sum(q ** 2, 1)[:, None]
            + tf.reduce_sum(protos ** 2, 1)[None, :]
            - 2.0 * tf.matmul(q, protos, transpose_b=True))


@tf.function
def proto_step(enc, opt, Xs, Xq, n_way, n_s, n_q):
    with tf.GradientTape() as tape:
        es = enc(Xs, training=True)
        eq = enc(Xq, training=True)
        protos = tf.reduce_mean(tf.reshape(es, (n_way, n_s, -1)), axis=1)
        logits = -_sq_dists(eq, protos)
        labels = tf.repeat(tf.range(n_way), n_q)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    grads = tape.gradient(loss, enc.trainable_variables)
    opt.apply_gradients(zip(grads, enc.trainable_variables))
    return loss


def train_encoder(X, Y, src, seed=0):
    P.set_seeds(seed)
    m, s = float(X.mean()), float(X.std()) + 1e-6
    Xn = ((X - m) / s).astype(np.float32)
    by = group_by_class_source(Y, src)
    enc = build_encoder(Xn.shape[1:])
    opt = tf.keras.optimizers.Adam(LR)
    opt.build(enc.trainable_variables)        # create slots now (not inside @tf.function)
    rng = np.random.default_rng(seed)
    for ep in range(EPISODES):
        sup, qry, _ = sample_episode(by, rng, N_SUPPORT, N_QUERY)
        Xs = tf.convert_to_tensor(Xn[sup.reshape(-1)])
        Xq = tf.convert_to_tensor(Xn[qry.reshape(-1)])
        proto_step(enc, opt, Xs, Xq, N_WAY, N_SUPPORT, N_QUERY)
    return enc, (m, s)


def embed(enc, X, m, s):
    return enc.predict(((X - m) / s).astype(np.float32), batch_size=256, verbose=0)


def register_protos(enc, X, Y, src, m, s, include_real):
    """Mean L2-normalised embedding per station. synth-only, or synth+train-real."""
    E = embed(enc, X, m, s)
    protos = np.zeros((N_WAY, EMB_DIM), np.float32)
    for lab in range(N_WAY):
        sel = [i for i in range(len(Y)) if Y[i] == lab and
               (include_real or src[i] == "synth")]
        protos[lab] = E[sel].mean(0)
    return protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9)


def proto_score(enc, protos, test, m, s):
    """Nearest-prototype over a small jitter, using the shared inference feature
    (D.window_feature) so train/test preprocessing match under any FEATURE_MODE."""
    ok, preds = 0, []
    for station, idx in test.marks:
        wins = [D.window_feature(test.y, idx, d) for d in (-0.1, 0.0, 0.1)]
        e = embed(enc, np.asarray(wins)[..., None], m, s)
        q = e.mean(0); q /= (np.linalg.norm(q) + 1e-9)
        pred = int(np.argmax(protos @ q))
        preds.append(pred); ok += (D.TARGET13[pred] == station)
    return ok, len(test.marks), preds


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    # PROTO_USE_CLEAN=0 -> live-only (drop clean+noise synth); prototypes from
    # real. PROTO_REAL_NOISE_AUG=K -> K extra real+more-real-noise copies/window.
    use_clean = os.environ.get("PROTO_USE_CLEAN", "1") == "1"
    rna = int(os.environ.get("PROTO_REAL_NOISE_AUG", "0"))
    clean = D.load_clean_sources(CLEAN_DIR)
    tags = (("synth", False), ("all", True)) if use_clean else (("real", True),)
    print(f"FEATURE_MODE = {D.FEATURE_MODE}   episodes = {EPISODES}   "
          f"use_clean = {use_clean}   real_noise_aug = {rna}")
    print(f"clean sources: {len(clean)}\n")
    tot = {tag: [0, 0] for tag, _ in tags}
    for held in TRIPS:
        trips = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in TRIPS if t != held]
        test = D.load_live_trip(os.path.join(LIVE_DIR, held))
        rng = np.random.default_rng(0)
        t0 = time.time()
        X, Y, src = D.build_metric_pool(clean, trips, rng, reverb_p=REVERB_P,
                                        use_clean=use_clean, real_noise_aug=rna)
        enc, (m, s) = train_encoder(X, Y, src)
        print(f"===== held-out #{TRIPS.index(held)} ({held[9:13]}) - {len(test.marks)} marks  "
              f"[build+train {time.time()-t0:.0f}s, pool {len(Y)}] =====")
        for tag, inc in tags:
            protos = register_protos(enc, X, Y, src, m, s, include_real=inc)
            ok, n, preds = proto_score(enc, protos, test, m, s)
            tot[tag][0] += ok; tot[tag][1] += n
            print(f"  proto={tag:5s}: held-out {ok}/{n} ({ok/n*100:3.0f}%)  "
                  f"hist={dict(sorted(collections.Counter(preds).items()))}")
        print(f"    GT idx: {[D.LABEL_IDX[st] for st,_ in test.marks]}\n")
    for tag, _ in tags:
        o, n = tot[tag]
        print(f"LOO proto={tag:5s}: {o}/{n} ({o/n*100:.0f}%)   "
              f"[clean-synth proto 35%, GRL 44%, chance 8%]")


if __name__ == "__main__":
    main()
