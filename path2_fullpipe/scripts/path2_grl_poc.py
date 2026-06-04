"""Lever 2 -- channel-adversarial training (DANN / gradient reversal) on top of
the metric-learning encoder, to push the embedding toward CHANNEL invariance.

Premise (CLAUDE.md): the cross-trip bottleneck is channel diversity (car PA /
cabin acoustics / mic placement), and we have only 4 real channels (trips). The
announcement signal itself is a fixed KORAIL recording, so any embedding axis
that separates the trips is pure channel nuisance. A gradient-reversal domain
head asks the encoder to make trips INDISTINGUISHABLE while ProtoNet still
separates stations -- removing the channel axis directly, with zero extra data
(trip labels already exist).

Design:
  * Encoder = the exact ProtoNet encoder (Flatten, 64-d L2 embedding) from
    path2_metric_poc -- unchanged, so the inference path / on-board model is
    identical. The domain head is TRAIN-ONLY; inference never runs it -> on-board
    cost 0.
  * GRL: forward identity, backward * -1. Total loss = proto_loss + lam*dom_ce.
    The reversal makes the encoder MAXIMISE domain confusion while the head
    MINIMISES domain CE. lam=0 reproduces the plain ProtoNet baseline (sanity).
  * Two variants for what counts as a domain (LOO trains on 3 real trips):
      - trips_only : 3-class (the 3 training trips); synth samples masked out.
      - with_synth : 4-class (synth + 3 trips); synth is its own domain.
  * Honest go/no-go: same seeded trip-LOO + nearest-prototype as the baseline.
    Judge on cross-trip held-out (proto=synth), vs logmel_cmn baseline 35%.

Run:  python scripts/path2_grl_poc.py
      PROTO_EPISODES=80 python scripts/path2_grl_poc.py   # quick smoke test
"""
import collections
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D
import path2_poc as P                  # set_seeds
import path2_metric_poc as M           # encoder + episode + prototype machinery

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras import layers, models

EMB_DIM = M.EMB_DIM
N_WAY = M.N_WAY
N_SUPPORT, N_QUERY = M.N_SUPPORT, M.N_QUERY
EPISODES = M.EPISODES
LR = M.LR
TRIPS = M.TRIPS

# (label, with_synth, lam). lam=0 is run once as a plumbing sanity check (must
# reproduce ~35%); both variants are identical at lam=0 (no domain gradient).
CONFIGS = [
    ("lam=0.0  (sanity)", True, 0.0),
    ("with_synth lam=0.05", True, 0.05),
    ("with_synth lam=0.1 ", True, 0.10),
    ("with_synth lam=0.3 ", True, 0.30),
    ("trips_only lam=0.05", False, 0.05),
    ("trips_only lam=0.1 ", False, 0.10),
    ("trips_only lam=0.3 ", False, 0.30),
]


@tf.custom_gradient
def grad_reverse(x):
    """Identity forward; negate gradient backward (DANN gradient-reversal)."""
    def grad(dy):
        return -dy
    return tf.identity(x), grad


def make_domain_head(n_dom, emb=EMB_DIM):
    return models.Sequential([
        layers.Input(shape=(emb,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_dom),
    ])


def build_domain_labels(src, with_synth):
    """Per-sample domain id + (#domains, mask_synth). real:<trip> are distinct
    domains; synth is domain 0 (with_synth) or -1=masked (trips_only)."""
    trips = sorted({s for s in src if s.startswith("real:")})
    if with_synth:
        groups = ["synth"] + trips
        idx = {g: i for i, g in enumerate(groups)}
        dom = np.array([idx[s] for s in src], np.int32)
        return dom, len(groups), False
    idx = {g: i for i, g in enumerate(trips)}
    dom = np.array([idx[s] if s.startswith("real:") else -1 for s in src], np.int32)
    return dom, len(trips), True


@tf.function
def proto_step_adv(enc, head, opt, Xs, Xq, ds, dq, n_way, n_s, n_q, lam, mask_synth):
    with tf.GradientTape() as tape:
        es = enc(Xs, training=True)
        eq = enc(Xq, training=True)
        protos = tf.reduce_mean(tf.reshape(es, (n_way, n_s, -1)), axis=1)
        logits = -M._sq_dists(eq, protos)
        labels = tf.repeat(tf.range(n_way), n_q)
        proto_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # domain CE on all episode embeddings, through GRL (encoder sees -grad)
        emb = tf.concat([es, eq], axis=0)
        dom = tf.concat([ds, dq], axis=0)
        dlogits = head(grad_reverse(emb), training=True)
        dce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.maximum(dom, 0), logits=dlogits)
        if mask_synth:
            w = tf.cast(dom >= 0, tf.float32)
            dom_loss = tf.reduce_sum(dce * w) / (tf.reduce_sum(w) + 1e-6)
        else:
            dom_loss = tf.reduce_mean(dce)
        loss = proto_loss + lam * dom_loss
    var = enc.trainable_variables + head.trainable_variables
    grads = tape.gradient(loss, var)
    opt.apply_gradients(zip(grads, var))
    return proto_loss, dom_loss


def train_grl(X, Y, src, dom, n_dom, lam, mask_synth, seed=0):
    P.set_seeds(seed)
    m, s = float(X.mean()), float(X.std()) + 1e-6
    Xn = ((X - m) / s).astype(np.float32)
    by = M.group_by_class_source(Y, src)
    enc = M.build_encoder(Xn.shape[1:])
    head = make_domain_head(n_dom)
    opt = tf.keras.optimizers.Adam(LR)
    opt.build(enc.trainable_variables + head.trainable_variables)
    rng = np.random.default_rng(seed)
    lam_t = tf.constant(lam, tf.float32)
    for _ in range(EPISODES):
        sup, qry, _ = M.sample_episode(by, rng, N_SUPPORT, N_QUERY)
        si, qi = sup.reshape(-1), qry.reshape(-1)
        proto_step_adv(enc, head, opt,
                       tf.convert_to_tensor(Xn[si]), tf.convert_to_tensor(Xn[qi]),
                       tf.convert_to_tensor(dom[si]), tf.convert_to_tensor(dom[qi]),
                       N_WAY, N_SUPPORT, N_QUERY, lam_t, mask_synth)
    return enc, (m, s)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    clean = D.load_clean_sources(M.CLEAN_DIR)
    print(f"FEATURE_MODE = {D.FEATURE_MODE}   episodes = {EPISODES}")
    print(f"clean sources: {len(clean)}   baseline (logmel_cmn, no GRL) = 35%\n")

    # Single-config mode (Lever 4 episode check etc): GRL_LAM set -> run only that
    # config; GRL_SYNTH=1 selects the with_synth variant. Default = full sweep.
    global CONFIGS
    sel_lam = os.environ.get("GRL_LAM")
    if sel_lam is not None:
        ws = os.environ.get("GRL_SYNTH", "0") == "1"
        tag = "with_synth" if ws else "trips_only"
        CONFIGS = [(f"{tag} lam={sel_lam}", ws, float(sel_lam))]

    # Pre-load trips once; rebuild pool per fold (depends on which trips train).
    summary = []
    for label, with_synth, lam in CONFIGS:
        tot_ok = tot_n = 0
        per_fold = []
        for held in TRIPS:
            trips = [D.load_live_trip(os.path.join(M.LIVE_DIR, t)) for t in TRIPS if t != held]
            test = D.load_live_trip(os.path.join(M.LIVE_DIR, held))
            rng = np.random.default_rng(0)
            X, Y, src = D.build_metric_pool(clean, trips, rng)
            dom, n_dom, mask_synth = build_domain_labels(src, with_synth)
            t0 = time.time()
            enc, (m, s) = train_grl(X, Y, src, dom, n_dom, lam, mask_synth)
            protos = M.register_protos(enc, X, Y, src, m, s, include_real=False)
            ok, n, preds = M.proto_score(enc, protos, test, m, s)
            tot_ok += ok; tot_n += n; per_fold.append(f"{ok}/{n}")
            print(f"  [{label}] held #{TRIPS.index(held)} ({held[9:13]}) "
                  f"{ok}/{n} ({ok/n*100:3.0f}%)  ndom={n_dom} "
                  f"[{time.time()-t0:.0f}s]  hist={dict(sorted(collections.Counter(preds).items()))}")
        acc = tot_ok / tot_n * 100
        summary.append((label, tot_ok, tot_n, acc, per_fold))
        print(f"  ==> {label}: LOO {tot_ok}/{tot_n} ({acc:.0f}%)   folds={per_fold}\n")

    print("=" * 64)
    print(f"{'config':22s}  LOO(proto=synth)   per-fold")
    print(f"{'baseline (no GRL)':22s}  17/48 (35%)        ['5/12','7/12','3/12','2/12']")
    for label, ok, n, acc, pf in summary:
        print(f"{label:22s}  {ok}/{n} ({acc:.0f}%)        {pf}")


if __name__ == "__main__":
    main()
