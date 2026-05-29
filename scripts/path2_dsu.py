"""Lever 5 -- feature-statistics perturbation (DSU / PatchDSU) for cross-trip
channel generalisation, plus a slimmed encoder that fits F411 Flash.

Two opinions motivated this file:

  (1) "the encoder is over-parameterised vs comparable studies." TRUE. The
      ProtoNet encoder's `Flatten(5x15x64=4800) -> Dense(128)` head alone is
      ~610k weights -> encoder.tflite 646 KB > F411's 512 KB Flash (the known
      deployment blocker). Comparable embedded KWS nets are far smaller:
      ResNet-15 (the PatchDSU paper's backbone) ~238k, ARM DS-CNN ~24-38k. The
      `slim=True` head adds one conv+pool block to shrink the flatten 4800->896,
      cutting the Dense head ~6x while KEEPING time resolution -- we must not use
      GlobalAveragePooling, which the project measured to collapse this task
      (time-averaging smears the short 본역명 token; cross-trip 19%).

  (2) "DSU / PatchDSU would help." PLAUSIBLE and on-theme. Our bottleneck is
      channel diversity (car PA / cabin / mic differs per trip; only 4 real
      channels). DSU (Li et al., ICLR 2022) models per-channel feature
      statistics (mean/std) as uncertain Gaussians and resamples them during
      training, synthesising new domains in FEATURE space. PatchDSU (arXiv
      2508.03190, a KWS-specific variant) does this per spectrogram patch, since
      a global statistic on a sparse spectrogram is skewed. Both are TRAINING-
      ONLY -> identity at inference, no params, on-board cost 0 (like the GRL
      domain head). Crucially this is NOT the PCEN failure: PCEN/CMN act on the
      INPUT (same space), so CMN neutered PCEN; DSU acts on learned CONV-feature
      statistics, a different space CMN does not touch. It is also complementary
      to GRL: GRL removes the channel axis, DSU manufactures more channels.

Honest go/no-go lives in path2_dsu_poc.py (seeded trip-LOO vs 35% baseline /
44% GRL). This file only defines the layers + encoder so the POC and the Colab
notebook share one implementation.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


# --------------------------------------------------------------------------- #
# DSU -- global per-channel feature-statistics uncertainty (ICLR 2022)
# --------------------------------------------------------------------------- #
class DSU(layers.Layer):
    """NHWC feature-statistics perturbation, training-only (identity at test).

    For each (sample, channel) it takes the mean/std over the spatial
    freq x time axes, estimates the uncertainty of those statistics as their
    variance ACROSS the batch (the only domain variation observable), resamples
    mean & std from N(stat, uncertainty), and re-normalises the feature map with
    the perturbed statistics. No trainable parameters.
    """

    def __init__(self, p: float = 0.5, factor: float = 1.0, eps: float = 1e-6, **kw):
        super().__init__(**kw)
        self.p, self.factor, self.eps = float(p), float(factor), float(eps)

    def _perturb(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)                 # (B,1,1,C)
        var = tf.reduce_mean(tf.square(x - mean), axis=[1, 2], keepdims=True)
        std = tf.sqrt(var + self.eps)
        m = tf.squeeze(mean, [1, 2])                                         # (B,C)
        s = tf.squeeze(std, [1, 2])
        mu_b = tf.reduce_mean(m, axis=0, keepdims=True)                      # (1,C)
        unc_m = tf.sqrt(tf.reduce_mean(tf.square(m - mu_b), 0, keepdims=True) + self.eps)
        s_b = tf.reduce_mean(s, axis=0, keepdims=True)
        unc_s = tf.sqrt(tf.reduce_mean(tf.square(s - s_b), 0, keepdims=True) + self.eps)
        beta = m + tf.random.normal(tf.shape(m)) * unc_m * self.factor       # perturbed mean
        gamma = s + tf.random.normal(tf.shape(s)) * unc_s * self.factor      # perturbed std
        x_norm = (x - mean) / std
        return x_norm * gamma[:, None, None, :] + beta[:, None, None, :]

    def call(self, x, training=None):
        if not training or self.p <= 0.0:
            return x
        return tf.cond(tf.random.uniform([]) < self.p,
                       lambda: self._perturb(x), lambda: x)

    def get_config(self):
        c = super().get_config()
        c.update(p=self.p, factor=self.factor, eps=self.eps)
        return c


# --------------------------------------------------------------------------- #
# PatchDSU -- per spectrogram-patch statistics uncertainty (arXiv 2508.03190)
# --------------------------------------------------------------------------- #
class PatchDSU(layers.Layer):
    """DSU applied independently to each of kh x kw freq x time patches.

    A spectrogram is sparse/non-stationary, so one global statistic per channel
    is skewed; tiling the feature map and perturbing each patch's statistics
    matches local time-frequency structure (the KWS-specific fix). Training-only,
    no trainable parameters. Static H,W (fixed input window) so the patch grid is
    resolved at build time.
    """

    def __init__(self, kh: int = 4, kw: int = 4, p: float = 0.5,
                 factor: float = 1.0, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.kh, self.kw = int(kh), int(kw)
        self.p, self.factor, self.eps = float(p), float(factor), float(eps)

    def build(self, shape):
        self.H, self.W, self.C = int(shape[1]), int(shape[2]), int(shape[3])
        self.Hk = -(-self.H // self.kh)             # ceil -> patch height
        self.Wk = -(-self.W // self.kw)             # ceil -> patch width
        self.Hp, self.Wp = self.Hk * self.kh, self.Wk * self.kw

    def _perturb(self, x):
        b = tf.shape(x)[0]
        xp = tf.pad(x, [[0, 0], [0, self.Hp - self.H], [0, self.Wp - self.W], [0, 0]],
                    mode="REFLECT")                                          # avoid zero-bias
        g = tf.reshape(xp, [b, self.kh, self.Hk, self.kw, self.Wk, self.C])  # tile into patches
        mean = tf.reduce_mean(g, axis=[2, 4], keepdims=True)                 # (b,kh,1,kw,1,C)
        var = tf.reduce_mean(tf.square(g - mean), axis=[2, 4], keepdims=True)
        std = tf.sqrt(var + self.eps)
        m = mean[:, :, 0, :, 0, :]                                           # (b,kh,kw,C)
        s = std[:, :, 0, :, 0, :]
        mu_b = tf.reduce_mean(m, axis=0, keepdims=True)
        unc_m = tf.sqrt(tf.reduce_mean(tf.square(m - mu_b), 0, keepdims=True) + self.eps)
        s_b = tf.reduce_mean(s, axis=0, keepdims=True)
        unc_s = tf.sqrt(tf.reduce_mean(tf.square(s - s_b), 0, keepdims=True) + self.eps)
        beta = m + tf.random.normal(tf.shape(m)) * unc_m * self.factor
        gamma = s + tf.random.normal(tf.shape(s)) * unc_s * self.factor
        g = (g - mean) / std * gamma[:, :, None, :, None, :] + beta[:, :, None, :, None, :]
        xp = tf.reshape(g, [b, self.Hp, self.Wp, self.C])
        return xp[:, :self.H, :self.W, :]

    def call(self, x, training=None):
        if not training or self.p <= 0.0:
            return x
        return tf.cond(tf.random.uniform([]) < self.p,
                       lambda: self._perturb(x), lambda: x)

    def get_config(self):
        c = super().get_config()
        c.update(kh=self.kh, kw=self.kw, p=self.p, factor=self.factor, eps=self.eps)
        return c


# --------------------------------------------------------------------------- #
# encoder builder (drop-in for path2_metric_poc.build_encoder)
# --------------------------------------------------------------------------- #
def _dsu(mode, p, factor, kh, kw, name):
    if mode == "dsu":
        return DSU(p=p, factor=factor, name=name)
    if mode == "patch":
        return PatchDSU(kh=kh, kw=kw, p=p, factor=factor, name=name)
    return None


def build_encoder(shape, emb: int = 64, dsu: str = "none", p: float = 0.5,
                  factor: float = 1.0, kh: int = 4, kw: int = 4,
                  places=("b1", "b2"), slim: bool = False, dense: int = 128):
    """ProtoNet encoder with optional DSU/PatchDSU and a Flash-fit slim head.

    dsu="none" reproduces path2_metric_poc.build_encoder EXACTLY (baseline). DSU
    modules are inserted after each block's conv+BN (before pooling) at `places`.
    slim=True adds a 4th conv+pool block, shrinking flatten 4800->896 (Dense head
    ~610k -> ~115k) while preserving time resolution. Inference path is identical
    to the baseline (DSU is identity at test), so the on-board model is unchanged.
    """
    inp = layers.Input(shape=shape)
    x = inp

    def block(x, filt, tag):
        x = layers.Conv2D(filt, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        d = _dsu(dsu, p, factor, kh, kw, f"dsu_{tag}") if tag in places else None
        if d is not None:
            x = d(x)
        return layers.MaxPooling2D(2)(x)

    x = block(x, 16, "b1")
    x = block(x, 32, "b2")
    x = block(x, 64, "b3")
    if slim:                                          # (5,15,64) -> (2,7,64)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(dense, activation="relu")(x)
    x = layers.Dense(emb)(x)
    x = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=1))(x)
    return models.Model(inp, x)


if __name__ == "__main__":
    # quick param-count comparison (the opinion-1 evidence)
    shape = (40, 126, 1)
    for name, kw in [("baseline", dict()),
                     ("DSU", dict(dsu="dsu")),
                     ("PatchDSU", dict(dsu="patch")),
                     ("slim", dict(slim=True)),
                     ("slim+DSU", dict(slim=True, dsu="dsu")),
                     ("slim dense64", dict(slim=True, dense=64))]:
        n = build_encoder(shape, **kw).count_params()
        print(f"{name:14s} params={n:>9,d}  (~{n/1024:.0f} KB int8 weights)")
