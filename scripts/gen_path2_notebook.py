"""Generate notebooks/path2_train.ipynb — Colab notebook for Path 2 (live cabin).

Path 2 station ID is a METRIC-LEARNING classifier (not 13-class softmax, which
collapsed cross-trip at ~19%). A small encoder maps a 2 s "이번역은 [primary
name]" window (+CMN) to a 64-d embedding; each station gets one prototype; we
classify by nearest prototype. Loss = Prototypical Network (episodic).

Strategy locked in by local seeded trip-LOO experiments (chance 8%):
  - clean+noise synth (old bulk)            = 35%
  - real-only, jitter only                  = 33%  (data-starved)
  - real-only + real-noise augmentation     = 38%  (> clean-synth, cleaner domain)
  - channel-adversarial GRL (trips λ=0.3)   = 44%  (best single lever)
  - episodes 600→2000                       = wash (compute is NOT the ceiling)
  - PCEN front-end                          = lost (CMN already removes static EQ)
  - real-RIR interp aug                     = infeasible (clean≠live recording)
This notebook combines the two winners: REAL-ONLY + heavy in-domain augmentation
+ GRL, trained large on GPU, and exports INT8 encoder + prototypes + meta.

One source of truth: the notebook git-clones the repo and imports
scripts/path2_dataset.py + path2_metric_poc.py + path2_grl_poc.py instead of
re-inlining logic. Data (clean wavs + 4 live trips, gitignored) comes from a
Drive zip. Re-run this script whenever the notebook content changes.

Run:  python scripts/gen_path2_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "path2_train.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Path 2 — 실차 안내방송 역 분류 (Colab, metric learning + GRL)

마이크로 녹음한 1호선 차내 안내방송으로 **현재 역**을 맞춥니다.
2-stage: **KWS 트리거**("이번역은" 검출) → **임베딩+prototype 분류**(어느 역).

### 전략 (로컬 트립단위 LOO로 확정, chance 8%)
| 접근 | cross-trip LOO |
|---|---|
| clean+노이즈 합성 (기존) | 35% |
| real-only (jitter만) | 33% (데이터 부족) |
| **real-only + 실노이즈 증강** | **38%** (도메인 깨끗, 모든 positive가 cross-trip) |
| **채널 적대 GRL (trips λ=0.3)** | **44%** (단일 최고 레버) |
| episode 600→2000 | wash (천장은 compute 아님) |
| PCEN front-end | 패배 (CMN이 정적 EQ를 이미 제거) |
| 실 RIR 보간 증강 | 불가 (clean ≠ live, 다른 녹음) |

핵심 진단: 안내방송은 코레일 단일 녹음의 결정론적 재생(고정 신호)이고, 변하는 건
(1) 가산 노이즈(합성으로 무한 생성), (2) 채널=차량 PA·객차·마이크(실 트립 수만큼만).
**병목은 데이터 양이 아니라 채널 다양성.** 그래서 이 노트북은 두 승자를 합칩니다:
**real-only + 강한 in-domain 증강 + GRL(채널 적대)**.

### 입력 규약 (로컬 스크립트와 동일, 한쪽만 바꾸면 mismatch)
- 분류 윈도우 = 1차 호명 **"이번역은 [본역명]"** 만 ([트리거 onset, +2.0s]).
- feature = 40-mel log-mel + **per-window CMN** (학습·추론·펌웨어 일관).
- 탑승역 마크(등교 구로 / 하교 성대)는 미녹음이라 drop.
- clean(서울교통공사)은 코레일 live와 **다른 녹음** → 기본 제외(`USE_CLEAN=False`).

**실행 전**: 런타임 → 런타임 유형 변경 → 하드웨어 가속기 **GPU**
"""),

    md("## 0. 환경 확인"),
    code("""
import tensorflow as tf
print("TensorFlow", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPU:", gpus if gpus else "없음 — 런타임 유형을 GPU로 바꾸세요")
!pip install -q librosa
import librosa; print("librosa", librosa.__version__)
"""),

    md("""
## 1. 코드 + 데이터 가져오기 (one source of truth)

- **코드**: 저장소를 clone 해 `scripts/`의 모듈을 그대로 import. (로컬 변경을 먼저 **git push** 하세요.)
- **데이터**(gitignore): 클린 wav + 4 트립을 zip으로 Drive에 올린 뒤 repo 경로로 풉니다.

zip 구조:
```
heynow_path2_data.zip
├── processed/wav/*.wav                              (서울교통공사 클린 — KWS/옵션용)
└── raw/line1_live/<trip_id>/audio.wav, marks.json   (4 트립 = 채널 4개)
```
"""),
    code("""
import os, sys, zipfile, glob

REPO = 'https://github.com/snup2e/hey-now.git'
!rm -rf /content/hey-now && git clone -q $REPO /content/hey-now
sys.path.insert(0, '/content/hey-now/scripts')

from google.colab import drive
drive.mount('/content/drive')
DATA_ZIP = '/content/drive/MyDrive/heynow/heynow_path2_data.zip'   # ← 본인 경로

with zipfile.ZipFile(DATA_ZIP) as z:
    z.extractall('/content/hey-now/data')

print("clean wav:", len(glob.glob('/content/hey-now/data/processed/wav/*.wav')))
print("trips:", [os.path.basename(d) for d in
                 glob.glob('/content/hey-now/data/raw/line1_live/*') if os.path.isdir(d)])
"""),
    code("""
import numpy as np, collections, time
import path2_dataset as D
import path2_poc as P            # set_seeds / small_cnn
import path2_metric_poc as MP    # encoder / ProtoNet / prototypes / scoring
import path2_grl_poc as G        # channel-adversarial (GRL) training loop

TRIPS = MP.TRIPS                 # all four one-way trips (2 등교 + 2 하교)
CLEAN_DIR = '/content/hey-now/data/processed/wav'
LIVE_DIR  = '/content/hey-now/data/raw/line1_live'
MP.CLEAN_DIR, MP.LIVE_DIR = CLEAN_DIR, LIVE_DIR
print("TARGET13:", D.TARGET13)
print("TRIPS:", TRIPS)
"""),

    md("""
## 2. 튜닝 노브 — real-only + 강한 증강 + GRL

증강은 학습 시점 전용이라 무거워도 OK(추론/펌웨어 불변). 모두 **in-domain**(실 안내방송
+ 실 객차잡음)이라 도메인을 벗어나지 않습니다.
"""),
    code("""
# ---- 데이터 전략 ----
USE_CLEAN      = False        # real-only. clean(서울교통공사)≠live(코레일) 다른 녹음 → 제외
REAL_NOISE_AUG = 16           # 윈도우당 실 객차잡음 추가 믹스 (강한 증강의 핵심: 33→38%)
REAL_JITTER    = 12           # 시간 jitter 샘플 수
JITTER_S       = 0.30         # jitter 폭 (±초)
SNR            = (-5.0, 25.0) # 더 어려운 저-SNR 포함 (강하게)
SPEC_AUG       = True         # SpecAugment 시간/주파수 마스킹
N_SYNTH        = 80           # USE_CLEAN=True일 때만 의미

# ---- 채널 적대(GRL) ----
LAMBDA_GRL     = 0.3          # 0=끄기. 로컬 best=trips_only λ=0.3 (44%). 단조상승이라 0.5도 시도해볼 만
DOM_WITH_SYNTH = False        # real-only면 항상 False (도메인=트립들)

# ---- 학습 규모 (GPU) ----
MP.EPISODES = G.EPISODES = 3000
MP.EMB_DIM  = 64

def build_pool(trips, rng, augment=True):
    \"\"\"augment=True: 학습용(강한 증강). False: prototype 등록용(깨끗·un-aug).\"\"\"
    return D.build_metric_pool(
        clean, trips, rng, use_clean=USE_CLEAN, n_synth=N_SYNTH, snr=SNR,
        real_noise_aug=(REAL_NOISE_AUG if augment else 0),
        real_jitter=(REAL_JITTER if augment else 3),
        jitter_s=(JITTER_S if augment else 0.1),
        spec_aug=(SPEC_AUG if augment else False))

def train_one(Xtr, Ytr, srctr):
    \"\"\"GRL if LAMBDA_GRL>0 else plain ProtoNet.\"\"\"
    if LAMBDA_GRL > 0:
        dom, n_dom, mask = G.build_domain_labels(srctr, with_synth=DOM_WITH_SYNTH)
        return G.train_grl(Xtr, Ytr, srctr, dom, n_dom, LAMBDA_GRL, mask)
    return MP.train_encoder(Xtr, Ytr, srctr)

clean = D.load_clean_sources(CLEAN_DIR)
INC_REAL = True   # real-only → prototypes from real (include_real). USE_CLEAN시에도 real 포함 가능
print(f"use_clean={USE_CLEAN} real_noise_aug={REAL_NOISE_AUG} jitter=±{JITTER_S}s x{REAL_JITTER} "
      f"snr={SNR} spec_aug={SPEC_AUG} | GRL λ={LAMBDA_GRL} | episodes={MP.EPISODES}")
"""),

    md("""
## 3. 정직한 LOO 평가 (go/no-go)

트립 단위 leave-one-out: held-out(미학습=새 채널)에서 최근접 prototype 정확도.
**held-out**이 판단 기준(held-in은 참고용). prototype은 깨끗한(un-aug) 풀에서 등록.
"""),
    code("""
tot = [0, 0]
for held in TRIPS:
    trips = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in TRIPS if t != held]
    test  = D.load_live_trip(os.path.join(LIVE_DIR, held))
    t0 = time.time()
    Xtr, Ytr, srctr = build_pool(trips, np.random.default_rng(0), augment=True)
    enc, (m, s)     = train_one(Xtr, Ytr, srctr)
    Xpr, Ypr, srcpr = build_pool(trips, np.random.default_rng(1), augment=False)  # clean protos
    protos = MP.register_protos(enc, Xpr, Ypr, srcpr, m, s, include_real=INC_REAL)
    ok, n, preds = MP.proto_score(enc, protos, test, m, s)
    tot[0] += ok; tot[1] += n
    print(f"held-out {held[9:13]}: {ok}/{n} ({ok/n*100:3.0f}%)  pool {len(Ytr)}  "
          f"[{time.time()-t0:.0f}s]  hist={dict(sorted(collections.Counter(preds).items()))}")
print(f"\\nLOO held-out: {tot[0]}/{tot[1]} ({tot[0]/tot[1]*100:.0f}%)   "
      f"[clean-synth 35%, real+aug 38%, GRL 44%, chance 8%]")
"""),

    md("""
## 4. 최종 모델 — 4 트립 전부로 학습 + prototype 등록

배포용은 모든 트립(채널 4개 전부)을 써서 학습합니다(평가 아님). 종착역(구로/성대)은
한 방향 트립에만 있으니 4 트립 모두 넣어야 prototype이 생깁니다.
"""),
    code("""
all_trips = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in TRIPS]
Xa, Ya, srca = build_pool(all_trips, np.random.default_rng(0), augment=True)
enc, (MEAN, STD) = train_one(Xa, Ya, srca)
# prototypes from a clean (un-augmented) pass for representativeness
Xp, Yp, srcp = build_pool(all_trips, np.random.default_rng(1), augment=False)
protos = MP.register_protos(enc, Xp, Yp, srcp, MEAN, STD, include_real=INC_REAL)
print("encoder params:", enc.count_params(), "| prototypes:", protos.shape,
      "| pool", len(Ya))
"""),

    md("""
## 5. 임계 보정 (abstain)

최근접 prototype의 cosine이 낮거나 top1−top2 margin이 작으면 "확인 중"으로 보류해
"틀린 역을 자신있게 표시"하는 최악을 막습니다. 깨끗한 풀(`Xp`)에서 곡선을 봅니다.
"""),
    code("""
Ep = MP.embed(enc, Xp, MEAN, STD)
sims = Ep @ protos.T                     # cosine (둘 다 L2-normed)
order = np.sort(sims, axis=1)
top1, top2 = order[:, -1], order[:, -2]
pred = sims.argmax(1); correct = (pred == Yp)
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    tau = np.quantile(top1[correct], 1 - q)
    keep = top1 >= tau
    prec = correct[keep].mean() if keep.any() else 0
    print(f"keep~{q:.0%}: tau={tau:.3f} -> kept {keep.mean():.0%}, precision {prec:.0%}")
TAU   = float(np.quantile(top1[correct], 0.2))            # ~80% coverage on this pool
DELTA = float(np.quantile((top1 - top2)[correct], 0.1))
print("TAU", round(TAU, 3), "DELTA", round(DELTA, 3))
"""),

    md("""
## 6. INT8 TFLite 인코더 + prototype 저장

STM32 배포: 인코더(.tflite) + 13×64 prototype + meta. prototype은 **양자화된 인코더
출력**으로 다시 등록해 추론과 일치시킵니다. (GRL 도메인 헤드는 학습 전용 → 미포함, 온보드 0)
"""),
    code("""
def to_int8_tflite(model, repr_X, path):
    def rep():
        idx = np.random.default_rng(0).choice(len(repr_X), min(200, len(repr_X)), replace=False)
        for i in idx:
            yield [repr_X[i:i+1].astype(np.float32)]
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    blob = c.convert(); open(path, 'wb').write(blob); return blob

Xa_n = ((Xa - MEAN) / STD).astype(np.float32)
blob = to_int8_tflite(enc, Xa_n, 'encoder.tflite')
print(f"encoder.tflite: {len(blob)/1024:.1f} KB")

interp = tf.lite.Interpreter(model_content=blob); interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
def q_embed(xn):
    si, zi = inp['quantization']; so, zo = out['quantization']
    e = np.zeros((len(xn), MP.EMB_DIM), np.float32)
    for k, x in enumerate(xn):
        q = np.clip(np.round(x/si + zi), -128, 127).astype(np.int8)[None]
        interp.set_tensor(inp['index'], q); interp.invoke()
        r = interp.get_tensor(out['index'])[0].astype(np.float32)
        e[k] = (r - zo) * so
    return e
# prototypes through the quantised encoder, from the clean (un-aug) pool
Xp_n = ((Xp - MEAN) / STD).astype(np.float32)
Eq = q_embed(Xp_n)
protos_q = np.zeros_like(protos)
for lab in range(len(D.TARGET13)):
    v = Eq[Yp == lab].mean(0); protos_q[lab] = v / (np.linalg.norm(v) + 1e-9)
print("quantised prototypes:", protos_q.shape)
"""),
    code("""
import json
np.save('prototypes.npy', protos_q)
meta = {
    'target13': D.TARGET13, 'emb_dim': MP.EMB_DIM,
    'norm_mean': MEAN, 'norm_std': STD,
    'feature': D.FEATURE_MODE, 'cmn': True,
    'sr': D.SR, 'n_mels': D.N_MELS, 'n_fft': D.N_FFT, 'mel_hop': D.MEL_HOP,
    'cnn_win_s': D.CNN_WIN, 'kws_win_s': D.KWS_WIN,
    'distance': 'cosine', 'tau': TAU, 'delta': DELTA, 'cooldown_s': 20.0,
    'train': {'use_clean': USE_CLEAN, 'real_noise_aug': REAL_NOISE_AUG,
              'jitter_s': JITTER_S, 'snr': list(SNR), 'spec_aug': SPEC_AUG,
              'grl_lambda': LAMBDA_GRL, 'episodes': MP.EPISODES},
    'prototypes': protos_q.tolist(),
}
with open('path2_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("saved path2_meta.json + prototypes.npy + encoder.tflite")
"""),

    md("""
## 7. KWS 트리거 (Stage 1) — INT8 TFLite (별도 트랙)

"이번역은" 1차 호명 검출. 분류 전에 언제 분류할지 정합니다. (분류기와 독립; KWS는
positive에 clean도 씁니다.)
"""),
    code("""
rng = np.random.default_rng(0)
Xk, Yk = D.build_kws(clean, all_trips, rng, snr=SNR)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
KMEAN, KSTD = float(Xk.mean()), float(Xk.std()) + 1e-6
Xk = (Xk - KMEAN) / KSTD
Xtr, Xva, Ytr, Yva = train_test_split(Xk, Yk, test_size=0.15, stratify=Yk, random_state=42)
P.set_seeds(0)
kws = P.small_cnn(Xk.shape[1:], 2)
# lr=5e-4 (not 1e-3) + more epochs/patience: the 2-class KWS collapses to a ~0.4
# constant output (val ~58%) on some folds at lr=1e-3. build_kws also keeps
# spec_aug=False (masking erases the short "이번역은" token -> label noise).
kws.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
w = compute_class_weight('balanced', classes=np.array([0, 1]), y=Ytr)
kws.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=60, batch_size=64,
        class_weight={0: w[0], 1: w[1]}, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)])
kb = to_int8_tflite(kws, Xtr, 'kws.tflite')
print(f"kws.tflite: {len(kb)/1024:.1f} KB | KMEAN {KMEAN:.2f} KSTD {KSTD:.2f}")
"""),

    md("## 8. 산출물 다운로드"),
    code("""
from google.colab import files
for f in ['encoder.tflite', 'kws.tflite', 'prototypes.npy', 'path2_meta.json']:
    files.download(f)
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU", "colab": {"provenance": []},
    },
    "nbformat": 4, "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUT}")
