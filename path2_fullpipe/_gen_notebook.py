# -*- coding: utf-8 -*-
"""Generator for path2_fullpipe.ipynb — MINIMAL: 위에서부터 전부 실행하면 검출-포함
융합 end-to-end 역정확도(최종 '쓸 수 있나')가 나오게. (open/close·HPSS비교·ceiling·
export 등 탐색용 셀은 제거 — 기록은 PATH2_RESULTS.md.)"""
import json, os

C = []
def md(*lines):  C.append({"cell_type": "markdown", "metadata": {}, "source": _src(lines)})
def code(*lines): C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(lines)})
def _src(lines):
    return [ln + ("\n" if i < len(lines) - 1 else "") for i, ln in enumerate(lines)]

md(
"# Path 2 — 검출-포함 융합 end-to-end (최종)",
"",
"실차 1호선 안내방송으로 **하차역**을 cross-trip(새 차량)으로 맞히는 최종 파이프라인.",
"",
"**KWS**(이번역은 검출) + **차임**(HPSS 톤분리, 출발) + **인코더**(역 이름) → 후보들 중 **노선 순서대로**",
"진짜 역을 고르는 **DP(흐름 전체 최적)**. 헛검출 자동 폐기, greedy 아니라 cascade 없음.",
"",
"위치는 **탑승역 앵커 + 노선 순서**가 골격, 인코더가 이름·복구, 차임이 게이트.",
"",
"**실행 전**: 런타임 → 런타임 유형 변경 → **GPU(T4)** + **고용량 RAM**. 그다음 셀1 경로 2줄만 맞추고",
"**위에서부터 전부 실행**하면 마지막에 최종 역정확도가 나옵니다. (trip4=차임 변종 제외 7트립 LOO)",
)

md("## 0. 환경")
code(
"import tensorflow as tf",
"print('TensorFlow', tf.__version__)",
"gpus = tf.config.list_physical_devices('GPU')",
"print('GPU:', gpus if gpus else '없음 — 런타임 유형을 GPU로')",
"!pip install -q librosa",
"import librosa; print('librosa', librosa.__version__)",
)

md(
"## 1. 코드 + 데이터",
"",
"### 업로드 — 무엇을 어디서 어디로",
"| 올릴 파일 | 로컬 위치 | Drive 업로드 위치 |",
"|---|---|---|",
"| **`path2_fullpipe/` 폴더 전체** | `E:\\imsisul\\path2_fullpipe\\` | `MyDrive/path2_fullpipe/` |",
"| (안에 `scripts/` + `door_events/` + `heynow_path2_data_8trip.zip`(457MB) 포함) | | |",
"",
"폴더째 드래그하면 1번에 끝. 셀 1의 `PKG`·`DATA_ZIP` 2줄만 맞추세요.",
)
code(
"import os, sys, glob, zipfile, shutil",
"",
"# ===== 고칠 경로 2줄 =====",
"PKG      = '/content/drive/MyDrive/path2_fullpipe'",
"DATA_ZIP = '/content/drive/MyDrive/path2_fullpipe/heynow_path2_data_8trip.zip'",
"# ========================",
"DATA = '/content/data'   # zip 푸는 로컬(빠름) — 그대로",
"",
"if not os.path.isdir(os.path.join(PKG, 'scripts')):",
"    PKG = os.getcwd()",
"assert os.path.isdir(os.path.join(PKG, 'scripts')), f'scripts/ not found under {PKG}'",
"sys.path.insert(0, os.path.join(PKG, 'scripts'))",
"",
"from google.colab import drive; drive.mount('/content/drive')",
"assert os.path.exists(DATA_ZIP), f'데이터 zip 없음: {DATA_ZIP}'",
"shutil.rmtree(DATA, ignore_errors=True)",
"with zipfile.ZipFile(DATA_ZIP) as z: z.extractall(DATA)",
"CLEAN_DIR = os.path.join(DATA, 'processed', 'wav')",
"LIVE_DIR  = os.path.join(DATA, 'raw', 'line1_live')",
"for d in glob.glob(os.path.join(LIVE_DIR, '*')):",
"    if os.path.isdir(d):",
"        de = os.path.join(d, 'door_events.json')",
"        src = os.path.join(PKG, 'door_events', os.path.basename(d) + '.door_events.json')",
"        if not os.path.exists(de) and os.path.exists(src): shutil.copy(src, de)",
"print('clean wav:', len(glob.glob(CLEAN_DIR + '/*.wav')))",
"print('trips:', sorted(os.path.basename(d) for d in glob.glob(LIVE_DIR+'/*') if os.path.isdir(d)))",
)

md("## 2. import + 노브")
code(
"import sys, numpy as np",
"# Drive의 최신 .py를 항상 다시 읽기 (모듈 캐시 stale 방지 — 재업로드 후 재시작 불필요)",
"for _m in [k for k in list(sys.modules) if k.startswith('path2_')]:",
"    del sys.modules[_m]",
"import path2_dataset as D, path2_poc as P",
"import path2_metric_poc as MP, path2_grl_poc as G",
"import path2_door_poc as DR, path2_pipeline as PL",
"",
"PL.CLEAN_DIR = MP.CLEAN_DIR = P.CLEAN_DIR = CLEAN_DIR",
"PL.LIVE_DIR  = MP.LIVE_DIR  = P.LIVE_DIR  = LIVE_DIR",
"DR.load_trip = lambda tid: PL.load_door(os.path.join(LIVE_DIR, tid))",
"",
"# ---- 노브 (융합: KWS+차임+인코더+시간간격 prior, 흐름 전체 DP) ----",
"MP.EPISODES = G.EPISODES = 1000   # 인코더 episode (2000 wash → 1000)",
"GRL_LAMBDA  = 0.3                 # 채널 적대",
"PL.EVENT_WIN['chime'] = 3.0       # 차임 윈도우(초)",
"PL.KWS_TRIG, PL.KWS_MINRUN = 0.6, 2",
"LAM_CHIME = 2.0                   # 차임 동반 보너스",
"LAM_TIME  = 0.015                 # 시간간격 prior 강도 (이웃 ~cycle 간격 선호; 클수록 타이밍 강제)",
"CYCLE     = 165.0                 # 역간 사이클(초)",
"",
"assert hasattr(PL, 'run_fusion_loo'), 'path2_pipeline.py가 옛 버전! 최신으로 덮어쓰고 런타임 재시작'",
"TRIPS = PL.list_trips()",
"print('TRIPS', len(TRIPS)); print('TARGET13', D.TARGET13)",
)

md(
"## 3. 융합 검출 파이프라인 (★ 최종 '쓸 수 있나')",
"",
"fold마다 KWS·차임·인코더 학습. KWS 후보(헛검출 많음) 중 **노선 순서대로 N개를 DP로** 고름:",
"",
"- **인코더 emission**: 그 위치 역 이름일 확률 (약하지만 후보 좁히면 도움)",
"- **차임 동반 보너스**: 진짜 정차였다는 증거",
"- **시간간격 prior**: 고른 이웃끼리 ~165s(한 역) 간격이도록 → 너무 붙은 가짜 후보 배제 (cascade 방지의 핵심)",
"- **단조**: 지나온 역 자동 배제, 노선 순서로 이름 확정",
"",
"카운팅 아님(간격 정합) → 헛검출 1개에 안 무너짐. 출력 `검출-포함 역정확도`가 최종 숫자.",
"`LAM_TIME` 올리면 타이밍 더 강제, `LAM_CHIME` 차임 의존도. (3모델×7fold, T4 ~40분)",
)
code(
"fz = PL.run_fusion_loo(grl_lambda=GRL_LAMBDA, lam_chime=LAM_CHIME, lam_time=LAM_TIME, cycle=CYCLE)",
)

md(
"## 4. (진단) 인코더 단독 — 7트립으로 개선됐나?",
"",
"노선prior·앵커 없이 **인코더만**의 per-mark cross-trip 정확도. 4트립 baseline(clean-synth 33%,",
"GRL 44%, chance 8%)과 비교 — 채널 4→7로 인코더가 나아졌는지 직접 확인.",
)
code(
"ep = PL.run_encoder_permark_loo(grl_lambda=GRL_LAMBDA)",
)

md(
"## 5. (진단) 각 모델 따로 학습 확인",
"",
"KWS·차임 검출기는 **별개 모델**(가중치 공유 X). 각자 val accuracy 학습곡선. (인코더는 episodic이라 §4로 확인)",
)
code(
"PL.plot_training_curves()",
"from IPython.display import Image; Image('model_training_curves.png')",
)

md(
"## 6. KWS false positive 억제 — hard-negative mining",
"",
"KWS false의 주범은 잡음이 아니라 *닮은 음성*('다음역은', 잡담) — denoising 안 통함(CMN이 이미",
"rumble 제거). 대신 **학습 트립에 KWS를 돌려 헛발화한 구간을 채굴 → negative로 추가 재학습**.",
"\"이 소리는 이번역은 아님\"을 직접 학습. cross-trip LOO로 mining 전/후 recall·false 비교.",
)
code(
"kf = PL.run_kws_hardneg_loo(rounds=1)   # rounds 늘리면 반복 채굴",
)

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "path2_fullpipe.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "|", len(C), "cells")
