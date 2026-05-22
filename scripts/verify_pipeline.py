"""KWS + CNN 통합 파이프라인 로컬 검증.

학습된 INT8 tflite 2개를 음원에 흘려보내며
  트리거 검출 → 역 분류 → 이전역/현재역 갱신
을 시뮬레이션한다. 이 로직이 그대로 STM32 펌웨어 main.c가 된다.

Run:  python scripts/verify_pipeline.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import librosa
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
WAV_DIR = ROOT / "data" / "processed" / "wav"

meta = json.loads((MODELS / "path1_meta.json").read_text(encoding="utf-8"))
SR = meta["sample_rate"]
STATIONS = meta["stations"]
N_MELS, N_FFT, MEL_HOP = meta["n_mels"], meta["n_fft"], meta["mel_hop"]
KM, KS = meta["kws_norm_mean"], meta["kws_norm_std"]
CM, CS = meta["cnn_norm_mean"], meta["cnn_norm_std"]
KW = int(meta["kws_win_sec"] * SR)   # KWS window samples
CW = int(meta["cnn_win_sec"] * SR)   # CNN window samples

SLIDE_HOP = int(0.25 * SR)   # KWS sliding step
TRIG_THRESH = 0.60           # KWS trigger probability threshold
CONF_THRESH = 0.50           # min CNN confidence to accept
NAME_OFFSET = int(1.5 * SR)  # trigger start → station name
MIN_TRIG_RUN = 3             # real trigger = >=3 consecutive trigger windows
                             # (debounces brief KWS false positives)


def make_predictor(path):
    """Return predict(mel)->softmax for an INT8 tflite model."""
    itp = Interpreter(model_path=str(path))
    itp.allocate_tensors()
    ind, outd = itp.get_input_details()[0], itp.get_output_details()[0]
    i_scale, i_zp = ind["quantization"]
    o_scale, o_zp = outd["quantization"]

    def predict(mel):
        x = mel.astype(np.float32)[None, ..., None]
        xq = np.clip(np.round(x / i_scale + i_zp), -128, 127).astype(np.int8)
        itp.set_tensor(ind["index"], xq)
        itp.invoke()
        yq = itp.get_tensor(outd["index"])[0].astype(np.float32)
        return (yq - o_zp) * o_scale

    return predict


kws_predict = make_predictor(MODELS / "kws.tflite")
cnn_predict = make_predictor(MODELS / "cnn.tflite")


def to_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=MEL_HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=1.0).astype(np.float32)


def load_wav(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def _classify(y, trig_start):
    """Run CNN on the station-name span following a confirmed trigger."""
    seg = y[trig_start + NAME_OFFSET: trig_start + NAME_OFFSET + CW]
    if len(seg) < CW:
        seg = np.pad(seg, (0, CW - len(seg)))
    probs = cnn_predict((to_logmel(seg) - CM) / CS)
    k = int(probs.argmax())
    return dict(t=trig_start / SR, station=STATIONS[k], conf=float(probs[k]))


def run_pipeline(y):
    """Slide over audio, fire KWS, classify station. Same logic as main.c.

    A trigger is only confirmed after MIN_TRIG_RUN consecutive trigger
    windows, so brief KWS false positives are debounced away.
    """
    detections = []
    i = 0
    run = 0          # consecutive trigger-window count
    run_start = 0
    while i + KW <= len(y):
        mel = (to_logmel(y[i:i + KW]) - KM) / KS
        if kws_predict(mel)[1] > TRIG_THRESH:
            if run == 0:
                run_start = i
            run += 1
            i += SLIDE_HOP
        else:
            if run >= MIN_TRIG_RUN:
                detections.append(_classify(y, run_start))
                i = run_start + NAME_OFFSET + CW         # skip processed span
            else:
                i += SLIDE_HOP
            run = 0
    if run >= MIN_TRIG_RUN:                              # trailing trigger
        detections.append(_classify(y, run_start))
    return detections


def station_of(path):
    return Path(path).stem.split("_")[0]


def main():
    wavs = [w for w in sorted(glob.glob(str(WAV_DIR / "*.wav")))
            if station_of(w) in STATIONS]
    if not wavs:
        sys.exit("타겟 음원이 없음 — preprocess.py 먼저 실행")

    print(f"통합 파이프라인 검증 — 타겟 음원 {len(wavs)}개\n")
    print(f"{'음원':<22}{'트리거':>9}{'분류 역':>11}{'conf':>7}  결과")
    print("-" * 58)
    ok = 0
    for w in wavs:
        exp = station_of(w)
        dets = run_pipeline(load_wav(w))
        if dets:
            d = dets[0]
            hit = d["station"] == exp
            ok += hit
            mark = "O" if hit else f"X (정답 {exp})"
            print(f"{Path(w).stem:<22}{d['t']:>7.2f}s{d['station']:>11}"
                  f"{d['conf']*100:>6.0f}%  {mark}")
        else:
            print(f"{Path(w).stem:<22}{'트리거 없음':>20}  X")
    print("-" * 58)
    print(f"정확도: {ok}/{len(wavs)} = {ok / len(wavs) * 100:.1f}%\n")

    # 연속 재생 시나리오 — 이전역 / 현재역
    route = ["성균관대", "의왕", "당정", "군포"]
    by = {station_of(w): w for w in wavs}
    stream = np.concatenate([load_wav(by[s]) for s in route if s in by])
    print(f"연속 재생: {' → '.join(route)}")
    prev = cur = None
    for d in run_pipeline(stream):
        if d["conf"] < CONF_THRESH:
            continue
        prev, cur = cur, d["station"]
        print(f"  [{d['t']:6.1f}s] 지난 역: {prev or '-':<10} 현재 역: {cur}"
              f"  (conf {d['conf']*100:.0f}%)")


if __name__ == "__main__":
    main()
