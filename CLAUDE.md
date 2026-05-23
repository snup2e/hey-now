# CLAUDE.md

이 문서는 Claude Code가 이 저장소에서 작업할 때 참고하는 프로젝트 컨텍스트입니다.

## 프로젝트 개요

- **과목**: 임베디드시스템설계 (성균관대학교)
- **프로젝트명**: **Hey now!** — 지하철 안내방송 기반 하차역 알림 시스템 (Subway Announcement Location Notifier)
- **제품 모델명**: Hey now!
- **GitHub repo**: [snup2e/hey-now](https://github.com/snup2e/hey-now)
- **목표**: GPS·WiFi가 닿지 않는 지하 환경에서, 차내 안내방송("이번 역은 ~~역입니다")을 TinyML로 인식해 사용자에게 현재 위치/하차역을 알림
- **타겟 노선/구간**: 서울 지하철 **1호선 성균관대 → 신도림** (14역, 친구 통학 경로 기반)

### 2-Path 진행 전략

- **Path 1 (시뮬레이션)** — *현재 단계*. 마이크 없이, STM32 보드 Flash에 저장한
  안내방송 음원을 모델에 직접 흘려보내 "지난 역 / 현재 역"을 디스플레이에 표시.
  서울교통공사 공식 음원으로 학습하고 같은 음원으로 시연.
- **Path 2 (실차)** — 1호선 통학 구간에서 잡음 포함 음성을 자체 녹음해 학습,
  실제 차내 환경에서 시연 및 영상 제작.

## 페르소나 & 문제 정의

### 누가 쓰는가
- **청각장애인** (안내방송 자체를 듣지 못함)
- **상황적 청각차단 사용자** (에어팟·이어폰 착용, 음악·통화·집중으로 안내방송 놓침)

### 왜 필요한가
- 한국 지하철은 깊고, GPS·셀룰러·WiFi 측위가 부정확하거나 차단되는 구간이 많음
- 안내방송은 환경에 이미 존재하는 위치 신호 — 별도 인프라 없이 활용 가능
- 기존 청각장애 보조기기(Apple Sound Recognition, Neosensory Buzz, Lumename, SSIES)는 **사이렌·알람 등 위험음**에 집중하고 **지하철 안내방송 시나리오는 다루지 않음**

## 하드웨어 스택

| 부품 | 모델 | 비고 |
|---|---|---|
| MCU | STM32 F411RE Nucleo | Cortex-M4 @ 100MHz, 512KB Flash, 128KB SRAM, FPU 있음 |
| 마이크 | ICS43434 (I2S MEMS) | **Path 1에서는 미사용** — 보드 저장 음원을 직접 처리. Path 2에서 사용 |
| 출력 디스플레이 | 2.0" SPI LCD | "지난 역 / 현재 역" 표시. 패널 모델 확정 전까지 펌웨어는 UART(시리얼) 출력으로 대체 |
| 오디오 출력 | MAX98357A (I2S DAC + Class-D Amp) | 보유 중. 활용 여부 TBD (백업·비상 알람용) |
| 스피커 | 4Ω 3W 40mm 라운드 | MAX98357A 직결 |

구매처: 가치창조기술 (vctec.co.kr)

## 소프트웨어 스택

- **모델 학습**: Python + TensorFlow + Keras + librosa (Google Colab)
- **모델 변환**: TensorFlow Lite (INT8 양자화)
- **로컬 검증**: librosa + ai-edge-litert (노트북 PC, Python 3.13)
- **펌웨어 IDE**: STM32CubeIDE
- **AI 추론**: X-CUBE-AI 플러그인 (.tflite → C 코드 자동 생성)
- **신호처리**: CMSIS-DSP (log-mel spectrogram FFT)

## 데이터셋 전략

### 1. Seoul Metro 공개 음원 (Path 1)
- 서울교통공사 공개 안내방송 음원 — 1호선 역별 mp3
- 각 음원은 **한국어 안내 + 영어 안내** 순서로 구성 (역 이름이 한국어 구간에서 2회 반복)
- 한/영 경계를 무음만으로 정확히 가르기 어려워, **음원 전체를 해당 역 데이터로 사용**
- 일반역은 "이번 역은" 0~1.5s / 역 이름 1.5~5.0s, 환승역은 앞에 다른 안내가 붙어
  "이번 역은"이 ~4s에 시작 (무음 검출로 자동 분할)

### 2. 1호선 차내 자체 녹음 (Path 2)
- **녹음 구간: 성균관대 → 구로 (13역)** — Path 1의 14역(→신도림)에서 한 역 단축, 친구 통학 실제 구간 기준
- **녹음 장비: STM32 보드 + ICS43434 마이크 (폰 X)** — 추론 시점의 마이크 도메인과 일치시켜야 학습이 유효
- 보드가 USB-CDC로 raw PCM을 노트북에 스트리밍, 노트북 Python(`scripts/path2_capture_ui.py`)이 long.wav 저장 + 역 마킹
- 트립 끝나면 `scripts/path2_slice.py`가 마크 기준 ±윈도우로 잘라 클립화
- 통근/한산 시간대 각각 수집, 출입문·승객 잡담 자연 잡음 포함
- 친구가 3트립(왕복) 정도 측정 예정 — 역당 ~6 샘플 + 증강

### 3. 데이터 증강
- 볼륨 변화, 가우시안 노이즈 (SNR 0~25dB), 시간 시프트
- SpecAugment (시간/주파수 마스킹)
- 윈도우당 8~30배 증강 (역당 음원이 1~수 개뿐이라 필수)

### 4. 라이선스 & 커밋 정책
- 안내방송 원본(.mp3/.wav) 및 zip: **gitignore**, Google Drive 별도 저장
- 전처리 산출물: gitignore
- `data/metadata.csv` (파일→역 매핑)만 commit
- `.tflite` 모델은 commit OK (펌웨어 통합용)

## 디렉토리 구조

```
imsisul/
├── CLAUDE.md / README.md / .gitignore
├── docs/                  ← HTML 발표자료 (GitHub Pages: snup2e.github.io/hey-now)
│   ├── index.html
│   └── img/
├── data/
│   ├── raw/seoul_metro/   ← (gitignore) Path 1 안내방송 mp3
│   ├── raw/line1_live/    ← (gitignore) Path 2 트립별 audio.wav + marks.json
│   ├── processed/         ← (gitignore) 16kHz wav, 클립, Colab 업로드용 zip
│   ├── metadata.csv       ← Path 1 파일→역/변형 매핑
│   ├── path2_metadata.csv ← Path 2 클립→역/트립 매핑
│   └── sample/
├── notebooks/
│   ├── path1_train.ipynb          ← KWS+CNN 학습 (Colab)
│   └── path1_train_complete.ipynb ← 학습 실행 결과 보존본
├── models/
│   ├── kws.tflite                 ← "이번 역은" 트리거 (15.6KB)
│   ├── cnn.tflite                 ← 역 분류 14역 (16.4KB)
│   └── path1_meta.json            ← 라벨·정규화 상수·파라미터
├── firmware/              ← STM32 펌웨어 소스 + 통합 가이드
│   ├── melspec.c/h                ← log-mel 추출 (CMSIS-DSP)
│   ├── app_path1.c/h              ← KWS+CNN 파이프라인
│   ├── model_runner.c/h           ← X-CUBE-AI 래퍼
│   ├── app_demo.c/h               ← 데모 메인 루프
│   ├── display.c/h                ← 결과 출력 (UART, LCD 확정 후 교체)
│   ├── demo_audio.h / mel_filterbank.h / model_meta.h  ← 자산 헤더
│   └── INTEGRATION_GUIDE.md       ← 빌드·통합 단계별 가이드
└── scripts/               ← 전처리·학습·검증·수집 유틸리티
    ├── preprocess.py / build_metadata.py / split_clips.py
    ├── gen_notebook.py / gen_demo_audio.py / gen_firmware_assets.py
    ├── melspec_ref.py             ← log-mel 레퍼런스 (librosa 대조 검증)
    ├── verify_pipeline.py         ← KWS+CNN 통합 파이프라인 로컬 검증
    ├── path2_capture_ui.py        ← Path 2 USB-CDC 캡처 + 마킹 Tkinter UI
    ├── path2_slice.py             ← long.wav + marks → 역별 클립 + 메타데이터
    └── README_path2.md            ← 친구용 실차 녹음 단계별 가이드
```

## 진행 로드맵

| 단계 | 핵심 산출물 | 상태 |
|---|---|---|
| 중간발표 | 주제·아키텍처 확정, HTML 발표자료 | ✅ 완료 |
| Path 1 — 데이터 | Seoul Metro 음원 전처리, 클립 분할 | ✅ 완료 |
| Path 1 — 학습 | kws.tflite + cnn.tflite (INT8) | ✅ 완료 |
| Path 1 — 검증 | 통합 파이프라인 17/17 | ✅ 완료 |
| Path 1 — 펌웨어 | STM32 펌웨어 소스 + 통합 가이드 | ✅ 소스 완료, 보드 빌드·테스트 대기 |
| Path 1 — 시연 | 보드에서 데모 동작 확인 | ⬜ 친구 보드 빌드 후 |
| Path 2 — 수집 도구 | USB-CDC 캡처 UI + 마크 슬라이서 (mock 검증 완료) | ✅ 완료 |
| Path 2 — 수집 펌웨어 | ICS43434 I2S → USB-CDC 송신 펌웨어 | ⬜ Path 1 빌드 후 |
| Path 2 — 실차 녹음 | 친구 통학 시 3트립 (왕복) 수집 | ⬜ |
| Path 2 — 재학습 + 시연 | 라이브 데이터 합쳐 재학습, 실차 시연 | ⬜ |

## 현재 진행 상황

### 완료
- [x] 타겟 구간 확정 — 1호선 성균관대 → 신도림 (14역)
- [x] Seoul Metro 음원 확보 + 16kHz mono 전처리 + 메타데이터
- [x] KWS(트리거) + CNN(역 분류) 학습 → INT8 tflite
- [x] 통합 파이프라인 로컬 검증 (음원별 17/17, 연속 재생 시나리오 정상)
- [x] STM32 펌웨어 소스 + X-CUBE-AI 통합 가이드 작성

### 남은 일
- [ ] 친구 보드(NUCLEO-F411RE)에서 펌웨어 빌드·플래시·테스트
- [ ] 2.0" LCD 패널 모델 확정 → `display.c` LCD 드라이버 작성 (현재 UART 출력)
- [ ] Path 2 수집 펌웨어 — ICS43434 I2S DMA → USB-CDC 송신 (CubeMX에서 I2S+USB device 활성, Path 1과 별도 프로젝트)
- [ ] Path 2 실차 녹음 (친구 통학 ×3트립) → `path2_slice.py`로 클립화
- [ ] Path 2 재학습 (라이브 + Seoul Metro 합산, 13-class) → 실차 시연

## 핵심 기술 결정 사항

### 모델 아키텍처 — 2-Stage 구조

**Stage 1: KWS — "이번 역은" 트리거 검출**
- 입력: 16kHz, 1초 윈도우, log-mel (40 mel × 63 frame)
- 출력: trigger / non-trigger 2-class
- 모델: 작은 Conv2D CNN → `kws.tflite` 15.6KB
- 검증: val accuracy 95.7%, trigger precision 94.4% / recall 93.0%

**Stage 2: CNN — 역 이름 분류**
- 입력: 트리거 직후 2초 윈도우, log-mel (40 mel × 126 frame)
- 출력: 14역 분류 (성균관대~신도림)
- 모델: 작은 Conv2D CNN → `cnn.tflite` 16.4KB
- 검증: 윈도우 92.3%, 음원 단위 100% (17/17)

**파이프라인 (`verify_pipeline.py` = 펌웨어 `app_path1.c`)**
- KWS 1초 윈도우를 0.25초 간격 슬라이딩
- 트리거 임계 0.6, **연속 3윈도우 디바운스**로 오검출 차단
- 트리거 확정 시 시작점 +1.5초부터 2초를 CNN에 입력
- CNN confidence < 0.5 → 분류 보류

### 신호 처리
- 샘플링: 16kHz
- 특징: log-mel spectrogram (n_fft 512, hop 256, 40 mel) — librosa `melspectrogram` + `power_to_db`
- 펌웨어는 CMSIS-DSP FFT로 동일 알고리즘 재현 (`melspec_ref.py`로 librosa 대조 검증, 오차 ~1e-4 dB)

### 핀맵
- Path 1은 마이크를 쓰지 않으므로 I2S 입력 불필요
- LCD(SPI) 핀은 패널 확정 후 CubeMX에서 설정 — `firmware/INTEGRATION_GUIDE.md` 참조
- Path 2에서 ICS43434(I2S) 입력 핀맵 확정 예정

## 참고 자료

### 학술
- Lumename: Wearable Device for Hearing Impaired (arXiv 2508.01576) — 개인화 KWS + 햅틱
- SSIES: A TinyML device for risk identification for people with hearing loss (ScienceDirect, 2025) — 응급음 + DOA
- ARM ML-KWS-for-MCU: https://github.com/ARM-software/ML-KWS-for-MCU

### 도구·인프라
- TinyML 책 GitHub: https://github.com/yunho0130/tensorflow-lite
- X-CUBE-AI 가이드 (DigiKey): https://www.digikey.com/en/maker/projects/tinyml-getting-started-with-stm32-x-cube-ai/f94e1c8bfc1e4b6291d0f672d780d2c0
- SpecAugment 논문 (Google, INTERSPEECH 2019)

### 데이터
- 서울교통공사 공개 자료실 — 안내방송 음원 (라이선스 확인 필요)

## Claude Code 작업 시 주의사항

- **강의자료 PDF는 commit 금지** (`project_instruction/`은 .gitignore)
- **대용량 데이터셋(.wav, .mp3, .zip)은 commit 금지** — Google Drive 또는 별도 저장소 사용
- **자체 녹음 음원은 개인정보(승객 대화)** 포함 가능 — 학습 후 원본 보관 시 주의
- **.tflite 모델은 commit OK** (펌웨어 통합용)
- **STM32CubeIDE 빌드 산출물(Debug/, Release/) commit 금지**
- 한국어로 응답해도 OK, 코드 주석은 영어 권장
