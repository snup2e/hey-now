"""Generate notebooks/path1_train.ipynb — Colab notebook for Path 1.

Path 1 trains TWO INT8 TFLite models to embed in the STM32 firmware:
  - KWS (Stage 1): detects the "이번 역은" trigger
  - CNN (Stage 2): classifies the station name (성균관대 → 신도림, 14 stations)

Re-run this script whenever the notebook content changes.
Run:  python scripts/gen_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "path1_train.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Path 1 — KWS + CNN 모델 준비 (Colab)

마이크 없이, STM32 보드에 저장된 안내방송 음원을 모델에 직접 흘려보내는
**시뮬레이션** 방식입니다. 보드에 올릴 두 개의 INT8 TFLite 모델을 학습합니다.

| 모델 | 역할 | 입력 |
|---|---|---|
| **KWS (Stage 1)** | "이번 역은" 트리거 검출 | 1초 윈도우 |
| **CNN (Stage 2)** | 역 이름 분류 (성균관대→신도림 14역) | 2초 윈도우 |

데이터: 1호선 성균관대~신도림 안내방송 17개 음원
- 일반역: `"이번 역은"` 0~1.5s · `역 이름` 1.5~5.0s
- 환승역: 앞에 다른 안내가 붙어 `"이번 역은"`이 ~4s에 시작 → 자동 검출

**실행 전**: 런타임 → 런타임 유형 변경 → 하드웨어 가속기 **GPU**
"""),

    md("## 0. 환경 확인"),
    code("""
import tensorflow as tf
print("TensorFlow", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPU:", gpus if gpus else "없음 — 런타임 유형을 GPU로 바꾸세요")
"""),
    code("""
!pip install -q librosa
import librosa, numpy as np
print("librosa", librosa.__version__)
"""),

    md("""
## 1. 데이터 업로드

`data/processed/path1_target_wav.zip` (타겟 17개 16 kHz wav)을 Google
Drive에 올린 뒤 경로를 맞춰주세요.
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')

import zipfile, os
ZIP_PATH = '/content/drive/MyDrive/heynow/path1_target_wav.zip'   # ← 본인 경로로 수정

os.makedirs('/content/wav', exist_ok=True)
with zipfile.ZipFile(ZIP_PATH) as z:
    z.extractall('/content/wav')
print(len(os.listdir('/content/wav')), "개 wav 압축 해제 완료")
"""),

    md("""
## 2. 클립 분할 — "이번 역은" / 역 이름

일반역은 고정 타이밍, 환승역은 무음 검출로 `"이번 역은"` 시작점을 찾습니다.
"""),
    code("""
import glob
from pathlib import Path

SR = 16000
TARGET = ['성균관대','의왕','당정','군포','금정','명학','안양','관악',
          '석수','금천구청','독산','가산디지털단지','구로','신도림']

def load_wav(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)

def find_onset(y, t_start=3.0, t_end=6.5):
    frame, hop = 400, 160
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + np.arange(n)[:, None] * hop
    env = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-10)
    thr = env.max() * 0.08
    s, e = int(t_start * SR / hop), min(int(t_end * SR / hop), n)
    for i in range(max(s, 1), e):
        if env[i] > thr and env[i - 1] <= thr:
            return i * hop / SR
    return t_start

def slice_sec(y, t0, t1):
    a, b = int(t0 * SR), int(t1 * SR)
    seg = y[max(0, a):min(len(y), b)]
    if len(seg) < b - a:
        seg = np.pad(seg, (0, b - a - len(seg)))
    return seg

records = []
for path in sorted(glob.glob('/content/wav/*.wav')):
    stem = Path(path).stem
    station = stem.split('_')[0]
    if station not in TARGET:
        continue
    y = load_wav(path)
    if '환승' in stem:
        t0 = find_onset(y)
        kws_seg, name_seg = (t0, t0 + 1.5), (t0 + 1.5, t0 + 5.0)
    else:
        kws_seg, name_seg = (0.0, 1.5), (1.5, 5.0)
    records.append(dict(station=station, stem=stem, y=y,
                        kws=kws_seg, name=name_seg))

stations = sorted({r['station'] for r in records})
label_idx = {s: i for i, s in enumerate(stations)}
print(f"{len(records)}개 음원 | {len(stations)}개 역")
print("역:", stations)
"""),

    md("## 3. 공통 — log-mel spectrogram / 증강"),
    code("""
N_MELS, N_FFT, MEL_HOP = 40, 512, 256

def to_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=MEL_HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=1.0).astype(np.float32)

def make_windows(y, win_sec, hop_sec):
    w, h = int(win_sec * SR), int(hop_sec * SR)
    if len(y) < w:
        y = np.pad(y, (0, w - len(y)))
    return [y[i:i + w] for i in range(0, len(y) - w + 1, h)]

def augment_wave(y, rng):
    y = y * rng.uniform(0.7, 1.3)                       # 볼륨
    if rng.random() < 0.85:                             # 노이즈 (SNR 0~25dB)
        snr = rng.uniform(0, 25)
        sp = float(np.mean(y ** 2)) + 1e-9
        npow = sp / (10 ** (snr / 10))
        y = y + rng.normal(0, np.sqrt(npow), len(y)).astype(np.float32)
    shift = int(rng.integers(-int(0.15 * SR), int(0.15 * SR)))
    return np.roll(y, shift).astype(np.float32)

def spec_augment(mel, rng):
    mel = mel.copy()
    f, t = mel.shape
    floor = float(mel.min())
    for _ in range(int(rng.integers(0, 3))):
        wv = int(rng.integers(1, max(2, t // 8)))
        s = int(rng.integers(0, max(1, t - wv)))
        mel[:, s:s + wv] = floor
    for _ in range(int(rng.integers(0, 3))):
        wv = int(rng.integers(1, max(2, f // 6)))
        s = int(rng.integers(0, max(1, f - wv)))
        mel[s:s + wv, :] = floor
    return mel

from tensorflow.keras import layers, models

def small_cnn(input_shape, num_classes):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])

def to_int8_tflite(model, repr_X, path):
    def rep():
        idx = np.random.default_rng(0).choice(
            len(repr_X), size=min(200, len(repr_X)), replace=False)
        for i in idx:
            yield [repr_X[i:i + 1].astype(np.float32)]
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    blob = c.convert()
    with open(path, 'wb') as f:
        f.write(blob)
    return len(blob)
"""),

    md("""
## 4. KWS (Stage 1) — "이번 역은" 트리거

- positive: 각 음원의 `"이번 역은"` 1.5초 구간
- negative: 그 외 전부 (역 이름, "내리실 문은", 영어, 환승역 앞부분)
"""),
    code("""
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

KWS_WIN = 1.0
KWS_AUG_POS = 25      # positive는 데이터가 적어 많이 증강
KWS_AUG_NEG = 2

Xk, Yk = [], []
rng = np.random.default_rng(0)

for r in records:
    y = r['y']
    ks, ke = r['kws']
    # positive — "이번 역은"
    for w in make_windows(slice_sec(y, ks, ke), KWS_WIN, 0.25):
        Xk.append(to_logmel(w)); Yk.append(1)
        for _ in range(KWS_AUG_POS):
            Xk.append(spec_augment(to_logmel(augment_wave(w, rng)), rng)); Yk.append(1)
    # negative — "이번 역은" 이후 전체
    for w in make_windows(y[int(ke * SR):], KWS_WIN, 0.5):
        Xk.append(to_logmel(w)); Yk.append(0)
        for _ in range(KWS_AUG_NEG):
            Xk.append(spec_augment(to_logmel(augment_wave(w, rng)), rng)); Yk.append(0)
    # negative — 환승역의 "이번 역은" 앞부분
    if ks > 0.5:
        for w in make_windows(y[:int(ks * SR)], KWS_WIN, 0.5):
            Xk.append(to_logmel(w)); Yk.append(0)

Xk = np.array(Xk, dtype=np.float32)[..., None]
Yk = np.array(Yk, dtype=np.int32)
KMEAN, KSTD = float(Xk.mean()), float(Xk.std())
Xk = (Xk - KMEAN) / (KSTD + 1e-6)
print("KWS X", Xk.shape, "| positive", int(Yk.sum()), "| negative", int((Yk == 0).sum()))
"""),
    code("""
Xktr, Xkva, Yktr, Ykva = train_test_split(
    Xk, Yk, test_size=0.15, stratify=Yk, random_state=42)

cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=Yktr)
kws_model = small_cnn(Xk.shape[1:], 2)
kws_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

kws_hist = kws_model.fit(
    Xktr, Yktr, validation_data=(Xkva, Ykva),
    epochs=40, batch_size=64,
    class_weight={0: cw[0], 1: cw[1]},
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8, restore_best_weights=True)],
)
"""),
    code("""
_, acc = kws_model.evaluate(Xkva, Ykva, verbose=0)
pred = kws_model.predict(Xkva, verbose=0).argmax(1)
tp = int(((pred == 1) & (Ykva == 1)).sum())
fp = int(((pred == 1) & (Ykva == 0)).sum())
fn = int(((pred == 0) & (Ykva == 1)).sum())
print(f"KWS val accuracy: {acc*100:.1f}%")
print(f"trigger precision {tp/(tp+fp+1e-9)*100:.1f}% | recall {tp/(tp+fn+1e-9)*100:.1f}%")

kb = to_int8_tflite(kws_model, Xktr, 'kws.tflite')
print(f"kws.tflite: {kb/1024:.1f} KB")
"""),

    md("""
## 5. CNN (Stage 2) — 역 이름 분류

각 음원의 역 이름 구간(역명 2회 반복)을 2초 윈도우로 잘라 14역을 분류합니다.
"""),
    code("""
CNN_WIN = 2.0
CNN_AUG = 30

Xc, Yc = [], []
rng = np.random.default_rng(1)

for r in records:
    label = label_idx[r['station']]
    ns, ne = r['name']
    for w in make_windows(slice_sec(r['y'], ns, ne), CNN_WIN, 0.5):
        Xc.append(to_logmel(w)); Yc.append(label)
        for _ in range(CNN_AUG):
            Xc.append(spec_augment(to_logmel(augment_wave(w, rng)), rng)); Yc.append(label)

Xc = np.array(Xc, dtype=np.float32)[..., None]
Yc = np.array(Yc, dtype=np.int32)
CMEAN, CSTD = float(Xc.mean()), float(Xc.std())
Xc = (Xc - CMEAN) / (CSTD + 1e-6)
print("CNN X", Xc.shape, "| classes", len(stations))
"""),
    code("""
Xctr, Xcva, Yctr, Ycva = train_test_split(
    Xc, Yc, test_size=0.15, stratify=Yc, random_state=42)

cnn_model = small_cnn(Xc.shape[1:], len(stations))
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_hist = cnn_model.fit(
    Xctr, Yctr, validation_data=(Xcva, Ycva),
    epochs=60, batch_size=64,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=12, restore_best_weights=True)],
)
"""),
    code("""
import collections

_, acc = cnn_model.evaluate(Xcva, Ycva, verbose=0)
print(f"CNN val accuracy (윈도우): {acc*100:.1f}%")

# 음원 단위 — 역 이름 구간 윈도우 다수결
correct = 0
for r in records:
    ns, ne = r['name']
    wins = make_windows(slice_sec(r['y'], ns, ne), CNN_WIN, 0.5)
    mels = np.array([(to_logmel(w) - CMEAN) / (CSTD + 1e-6) for w in wins])[..., None]
    pred = collections.Counter(
        cnn_model.predict(mels, verbose=0).argmax(1)).most_common(1)[0][0]
    correct += (pred == label_idx[r['station']])
print(f"음원 단위 정확도: {correct}/{len(records)} = {correct/len(records)*100:.1f}%")

cb = to_int8_tflite(cnn_model, Xctr, 'cnn.tflite')
print(f"cnn.tflite: {cb/1024:.1f} KB")
"""),

    md("## 6. 결과물 저장"),
    code("""
import json
from google.colab import files

meta = {
    "stations": stations,
    "sample_rate": SR,
    "n_mels": N_MELS, "n_fft": N_FFT, "mel_hop": MEL_HOP,
    "kws_win_sec": KWS_WIN, "kws_norm_mean": KMEAN, "kws_norm_std": KSTD,
    "cnn_win_sec": CNN_WIN, "cnn_norm_mean": CMEAN, "cnn_norm_std": CSTD,
}
with open('path1_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

files.download('kws.tflite')
files.download('cnn.tflite')
files.download('path1_meta.json')
print("kws.tflite + cnn.tflite + path1_meta.json 다운로드 완료")
"""),

    md("""
---
### 다음 단계
- 받은 `kws.tflite` · `cnn.tflite` · `path1_meta.json`을 저장소 `models/` 폴더에
- 정확도가 낮으면: `KWS_AUG_POS` / `CNN_AUG` ↑, epoch ↑
- 모델이 준비되면 STM32CubeIDE 펌웨어로 통합 (보드 입수 후)
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"생성 완료: {OUT}  ({len(cells)} cells)")
