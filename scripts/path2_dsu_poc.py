"""Lever 5 go/no-go -- DSU / PatchDSU feature-statistics perturbation (and the
Flash-fit slim encoder) on the metric-learning station classifier.

Same honest protocol as path2_grl_poc.py: seeded trip leave-one-out, nearest-
prototype scoring on the held-out (unseen-channel) trip, identical default pool
(clean+noise synth) so numbers line up with the known baselines:

    baseline ProtoNet (logmel_cmn)      35%
    channel-adversarial GRL trips λ0.3  44%   (best model lever so far)
    chance                               8%

Each config swaps ONLY the encoder (M.build_encoder is monkey-patched to
path2_dsu.build_encoder); everything else -- episode sampler, ProtoNet loss,
prototype registration, scoring -- is reused from path2_metric_poc / path2_grl_poc
so this is apples-to-apples with the other levers.

Question being measured:
  * Does perturbing CONV-feature statistics (DSU/PatchDSU) beat the baseline,
    where input-space PCEN failed (CMN already removed static input EQ)?
  * Is it complementary to GRL (remove channel axis + manufacture channels)?
  * Does the slim head (needed for F411 Flash) hold accuracy, and does DSU
    recover any accuracy lost to slimming?

Run:  python scripts/path2_dsu_poc.py
      PROTO_EPISODES=60 python scripts/path2_dsu_poc.py          # quick smoke test
      DSU_CONFIGS=0,1,4 PROTO_EPISODES=600 python scripts/path2_dsu_poc.py
"""
import collections
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import path2_dataset as D
import path2_poc as P                 # set_seeds
import path2_metric_poc as M          # encoder / ProtoNet / prototypes / scoring
import path2_grl_poc as G             # channel-adversarial (GRL) training loop
import path2_dsu as DSU               # DSU / PatchDSU layers + encoder builder

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf

TRIPS = M.TRIPS

# Episodes (compute) -- the GRL sweep showed 600->2000 is wash, so 600 is the
# honest default; override with PROTO_EPISODES for a quick smoke test.
EPISODES = int(os.environ.get("PROTO_EPISODES", "600"))
M.EPISODES = G.EPISODES = EPISODES

# (label, encoder kwargs for path2_dsu.build_encoder, grl_lambda, with_synth).
# grl_lambda>0 -> train with the channel-adversarial head (trips_only domains).
CONFIGS = [
    ("baseline (no dsu/grl)",      dict(),                                              0.0, False),
    ("DSU p0.5 b1,b2",             dict(dsu="dsu", p=0.5),                              0.0, False),
    ("DSU p0.5 b1,b2,b3",          dict(dsu="dsu", p=0.5, places=("b1", "b2", "b3")),   0.0, False),
    ("PatchDSU 4x4 p0.5",          dict(dsu="patch", p=0.5, kh=4, kw=4),                0.0, False),
    ("PatchDSU 4x8 p0.5",          dict(dsu="patch", p=0.5, kh=4, kw=8),                0.0, False),
    ("slim (Flash-fit)",           dict(slim=True),                                     0.0, False),
    ("slim + DSU p0.5",            dict(slim=True, dsu="dsu", p=0.5),                   0.0, False),
    ("slim + PatchDSU 4x4",        dict(slim=True, dsu="patch", p=0.5, kh=4, kw=4),     0.0, False),
    ("DSU + GRL trips λ0.3",       dict(dsu="dsu", p=0.5),                              0.3, False),
    ("PatchDSU + GRL trips λ0.3",  dict(dsu="patch", p=0.5, kh=4, kw=4),                0.3, False),
]

_ORIG_BUILD = M.build_encoder


def patch_encoder(enc_kwargs):
    """Route M.build_encoder (used by BOTH M.train_encoder and G.train_grl) to
    the DSU encoder with these kwargs. emb stays M.EMB_DIM."""
    M.build_encoder = lambda shape, emb=M.EMB_DIM: DSU.build_encoder(shape, emb, **enc_kwargs)


def run_fold(held, enc_kwargs, lam, with_synth):
    trips = [D.load_live_trip(os.path.join(M.LIVE_DIR, t)) for t in TRIPS if t != held]
    test = D.load_live_trip(os.path.join(M.LIVE_DIR, held))
    rng = np.random.default_rng(0)
    X, Y, src = D.build_metric_pool(clean, trips, rng)         # default pool == GRL baseline
    if lam > 0:
        dom, n_dom, mask = G.build_domain_labels(src, with_synth)
        enc, (m, s) = G.train_grl(X, Y, src, dom, n_dom, lam, mask)
    else:
        enc, (m, s) = M.train_encoder(X, Y, src)
    protos = M.register_protos(enc, X, Y, src, m, s, include_real=False)   # proto=synth
    ok, n, preds = M.proto_score(enc, protos, test, m, s)
    return ok, n, preds, enc.count_params()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    global clean, CONFIGS
    clean = D.load_clean_sources(M.CLEAN_DIR)
    sel = os.environ.get("DSU_CONFIGS")
    if sel:
        idx = [int(i) for i in sel.split(",")]
        CONFIGS = [CONFIGS[i] for i in idx]
    print(f"FEATURE_MODE = {D.FEATURE_MODE}   episodes = {EPISODES}   "
          f"clean sources = {len(clean)}")
    print(f"baselines: ProtoNet 35% | GRL trips λ0.3 44% | chance 8%  "
          f"(Flash budget: F411 = 512 KB)\n")

    summary = []
    for label, enc_kwargs, lam, with_synth in CONFIGS:
        patch_encoder(enc_kwargs)
        tot_ok = tot_n = 0
        per_fold = []
        params = None
        for held in TRIPS:
            t0 = time.time()
            ok, n, preds, params = run_fold(held, enc_kwargs, lam, with_synth)
            tot_ok += ok; tot_n += n; per_fold.append(f"{ok}/{n}")
            print(f"  [{label}] held #{TRIPS.index(held)} ({held[9:13]}) "
                  f"{ok}/{n} ({ok/n*100:3.0f}%)  [{time.time()-t0:.0f}s]  "
                  f"hist={dict(sorted(collections.Counter(preds).items()))}")
        acc = tot_ok / tot_n * 100
        # int8 weights ~= 1 byte/param; leave headroom for KWS(16KB)+runtime under 512KB.
        flash = "OK " if params <= 460_000 else "BIG"
        summary.append((label, tot_ok, tot_n, acc, per_fold, params))
        print(f"  ==> {label}: LOO {tot_ok}/{tot_n} ({acc:.0f}%)  "
              f"params={params:,} [{flash} for Flash]  folds={per_fold}\n")
    M.build_encoder = _ORIG_BUILD

    print("=" * 78)
    print(f"{'config':28s}  LOO        params      folds")
    print(f"{'-- baseline ProtoNet':28s}  17/48(35%) 646,528     (reference)")
    print(f"{'-- GRL trips λ0.3':28s}  ~21/48(44%) 646,528     (best prior lever)")
    for label, ok, n, acc, pf, params in summary:
        print(f"{label:28s}  {ok}/{n}({acc:.0f}%)  {params:>9,}   {pf}")


if __name__ == "__main__":
    main()
