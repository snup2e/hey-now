# CLAUDE.md

이 문서는 Claude Code가 이 저장소에서 작업할 때 참고하는 프로젝트 컨텍스트입니다.

## 프로젝트 개요

- **과목**: 임베디드시스템설계 (성균관대학교)
- **프로젝트명**: **Hey now!** — 지하철 안내방송 기반 하차역 알림 시스템 (Subway Announcement Location Notifier)
- **제품 모델명**: Hey now!
- **GitHub repo**: [snup2e/hey-now](https://github.com/snup2e/hey-now)
- **목표**: GPS·WiFi가 닿지 않는 지하 환경에서, 차내 안내방송("이번 역은 ~~역입니다")을 TinyML로 인식해 사용자에게 현재 위치/하차역을 알림
- **타겟 노선**: 서울 지하철 **1호선 일부 구간** (대상 역은 친구의 통근 경로 기반으로 추후 확정)
- **기말 발표 데모 시나리오**:
  1. 보드에 1호선 안내방송 음원을 재생 → 보드가 현재 역을 인식해 OLED/진동/LED로 표시
  2. 실제 1호선 차내에서 녹음한 잡음 포함 음성으로 강건성 시연
  3. 사용자가 등록한 "하차 예정역" 도착 직전 진동 알림

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
| MCU | STM32 F411RE Nucleo | Cortex-M4 @ 100MHz, 512KB Flash, 128KB SRAM, FPU 있음, **DAC 없음** |
| 마이크 | INMP441 (I2S MEMS) | I2S2 버스 연결 |
| 오디오 출력 | MAX98357A (I2S DAC + Class-D Amp) | 보유 중. 본 프로젝트에서 활용 여부 TBD (백업·비상 알람용) |
| 스피커 | 4Ω 3W 40mm 라운드 | MAX98357A 직결 |
| **출력 디바이스 (TBD)** | OLED 0.96" SSD1306 후보 | 역 이름 텍스트 표시용. 발주 필요 |
| **알림 디바이스 (TBD)** | 코인형 진동모터 후보 | 하차역 도착 알림용. 발주 필요 |

구매처: 가치창조기술 (vctec.co.kr)

> **출력 방식 결정 필요**: OLED(가독성) vs 진동(은밀성) vs 둘 다.

## 소프트웨어 스택

- **모델 학습**: Python + TensorFlow + Keras + librosa (Google Colab)
- **모델 변환**: TensorFlow Lite (int8 양자화)
- **펌웨어 IDE**: STM32CubeIDE
- **AI 추론**: X-CUBE-AI 플러그인 (.tflite → C 코드 자동 생성)
- **신호처리**: CMSIS-DSP (MFCC 추출, VAD)

## 데이터셋 전략

### 1. Seoul Metro 공개 음원 (Clean Reference)
- 서울교통공사 공개 안내방송 음원 활용
- 각 역의 표준 안내방송 (한국어/영어/중국어/일본어 4언어)
- **본 프로젝트는 한국어 안내 구간만 사용**
- 학습 데이터의 베이스라인 (clean signal)

### 2. 1호선 차내 자체 녹음 (Noisy Field Data)
- 팀원/협력자가 1호선 실제 탑승 중 스마트폰 녹음
- 통근 시간대(혼잡) + 한산한 시간대 분리 수집
- 신형(VVVF)/구형 차량 가능한 만큼 다양화
- 출입문 개폐, 승객 잡담, 휴대전화 소리 등 자연 잡음 포함

### 3. 데이터 증강
- Seoul Metro clean 음원 + 자체 녹음 잡음 트랙 합성 (다양한 SNR로 mix)
- SpecAugment (시간/주파수 마스킹)
- 가벼운 피치 시프트 / 시간 스트레치
- 목표: SNR -5 dB까지 동작

### 4. 라이선스 & 커밋 정책
- 자체 녹음 원본(.wav): **gitignore**, Google Drive 별도 저장
- 전처리 산출물(MFCC .npy): gitignore
- 샘플 1~2개만 `data/sample/` 에 commit (저장소 데모용)
- Seoul Metro 음원: 라이선스 확인 후 별도 저장소·링크로 관리

## 디렉토리 구조 (계획)

```
imsisul/
├── CLAUDE.md              ← 이 문서
├── README.md              ← 프로젝트 소개
├── .gitignore
├── docs/                  ← 발표자료, 보고서, 다이어그램
├── data/
│   ├── raw/               ← (gitignore) 원본 오디오
│   │   ├── seoul_metro/   ← (gitignore) 공식 안내방송 음원
│   │   └── field/         ← (gitignore) 1호선 자체 녹음
│   ├── processed/         ← (gitignore) MFCC, 스펙트로그램
│   ├── sample/            ← 소량 샘플 commit (저장소 데모)
│   └── README.md          ← 데이터셋 출처/라이선스/수집 방법
├── notebooks/             ← Colab 학습 노트북
│   ├── 01_eda.ipynb               ← 안내방송 음향 분석
│   ├── 02_preprocessing.ipynb     ← MFCC + SNR augmentation
│   ├── 03_training_stage1.ipynb   ← 트리거 검출 KWS
│   ├── 04_training_stage2.ipynb   ← 역 이름 분류
│   └── 05_quantization.ipynb      ← INT8 양자화 + 검증
├── models/                ← 학습된 모델
│   ├── stage1_trigger.tflite      ← "이번 역은" 트리거 (~5KB)
│   ├── stage2_station.tflite      ← 역 분류 (~50KB)
│   └── checkpoints/               ← (gitignore)
├── firmware/              ← STM32CubeIDE 프로젝트
│   ├── Core/
│   ├── Drivers/
│   ├── X-CUBE-AI/
│   └── Middlewares/
├── hardware/              ← 회로도, 핀맵
│   ├── pinout.md
│   └── schematic.png
└── scripts/               ← 유틸리티
    ├── record_subway.md           ← 자체 녹음 가이드
    ├── augment_snr.py             ← SNR 합성 증강
    └── tflite_to_carray.py        ← TFLite → C 헤더 변환
```

## 5주 로드맵

| 주차 | 단계 | 핵심 산출물 |
|---|---|---|
| Phase 0 (현재) | 사전 확정 | 교수님 컨펌, 부품 추가 발주, 타겟 역 확정, 역할 분담 |
| Phase 1 | 데이터셋 구축 | Seoul Metro 음원 수집 + 1호선 자체 녹음 첫 라운드 (10시간+) |
| Phase 2 | 모델 학습 | stage1.tflite + stage2.tflite (val_acc 90%+, FP < 1%) |
| Phase 3 | 하드웨어 통합 | INMP441 캡처 → MFCC 파이프라인 검증, 출력 디바이스 동작 확인 |
| Phase 4 | 펌웨어 통합 | 2-stage 풀스택 실시간 추론, 1호선 녹음 재생 테스트 |
| Phase 5 | 검증 & 시연 | 실차 시연 영상, SNR 강건성 평가, 발표 자료 |

## 현재 진행 상황 (Phase 0)

- [x] 프로젝트 주제 확정 (지하철 안내방송 기반 위치 알림)
- [x] 페르소나 정의 (청각장애 + 상황적 청각차단)
- [x] 타겟 노선 결정 (서울 1호선 일부 구간)
- [x] 데이터 소스 확정 (Seoul Metro 공식 음원 + 1호선 자체 녹음)
- [x] 기존 출튀 도우미 프로젝트 폐기
- [ ] 타겟 구간(역 목록) 확정 — 친구 통근 경로 기반
- [ ] 출력 디바이스 결정 (OLED / 진동 / 둘 다)
- [ ] 부품 추가 발주 (OLED·진동모터)
- [ ] 교수님께 주제 변경 컨펌
- [ ] STM32CubeIDE + X-CUBE-AI 설치
- [ ] Seoul Metro 음원 라이선스/이용약관 확인

## 핵심 기술 결정 사항

### 모델 아키텍처 — 2-Stage 구조

**Stage 1: 트리거 검출 (Always-on, lightweight)**
- 입력: 16kHz, 1초 윈도우, MFCC (40 mel × 40 frame)
- 출력: "이번 역은" / "다음 역은" / not-trigger 3-class
- 모델: 매우 작은 CNN (~5KB)
- 역할: stage2를 깨우는 게이트. 항상 동작하지만 전력 효율적

**Stage 2: 역 이름 분류 (Triggered)**
- 입력: 트리거 직후 2~3초 윈도우, MFCC
- 출력: N개 역 × 2방향 = ~20~30 class (1호선 일부 구간만)
- 모델: Conv2D 3층 + Dense (~50KB)
- Confidence thresholding (softmax max < 0.85 → 알림 보류)

### 신호 처리
- 샘플링: 16kHz (음성 안내방송 충분)
- 윈도우: 1초 (stage1), 2~3초 (stage2)
- MFCC: 40 계수, log-mel 기반
- VAD pre-gating으로 무음 구간 건너뛰기 (전력 절약)

### SNR 강건성 전략
1. Per-Channel Energy Normalization (PCEN) 적용
2. Multi-condition training (SNR 0~20dB 다양화)
3. SpecAugment (시간/주파수 마스킹)
4. 자체 녹음 잡음 트랙으로 augmentation
5. Confidence threshold로 불확실 케이스 거부

### 핀맵 (잠정)
```
INMP441 (I2S2 - 마이크 입력)
  VDD → 3.3V
  GND → GND
  SCK → PB10 (I2S2_CK)
  WS  → PB12 (I2S2_WS)
  SD  → PB15 (I2S2_SD)
  L/R → GND

OLED SSD1306 (I2C1 - 텍스트 출력, TBD)
  VCC → 3.3V
  GND → GND
  SCL → PB6 (I2C1_SCL)
  SDA → PB7 (I2C1_SDA)

진동모터 (GPIO PWM, TBD)
  IN  → PA8 (TIM1_CH1, PWM)

MAX98357A (I2S3 - 백업 알람음, 선택)
  BCLK → PC10 (I2S3_CK)
  LRC  → PA15 (I2S3_WS)
  DIN  → PC12 (I2S3_SD)
```
※ 핀맵은 STM32CubeMX에서 최종 검증 필요

## 참고 자료

### 학술
- Lumename: Wearable Device for Hearing Impaired (arXiv 2508.01576) — 개인화 KWS + 햅틱
- SSIES: A TinyML device for risk identification for people with hearing loss (ScienceDirect, 2025) — 응급음 + DOA
- ARM ML-KWS-for-MCU: https://github.com/ARM-software/ML-KWS-for-MCU
- TinySV: Speaker Verification in TinyML (ACM 2024)

### 도구·인프라
- TinyML 책 GitHub: https://github.com/yunho0130/tensorflow-lite
- X-CUBE-AI 가이드 (DigiKey): https://www.digikey.com/en/maker/projects/tinyml-getting-started-with-stm32-x-cube-ai/f94e1c8bfc1e4b6291d0f672d780d2c0
- SpecAugment 논문 (Google, INTERSPEECH 2019)
- PCEN 논문 (Wang et al., ICASSP 2017)

### 데이터
- 서울교통공사 공개 자료실 — 안내방송 음원 (라이선스 확인 필요)

## Claude Code 작업 시 주의사항

- **강의자료 PDF는 commit 금지** (`project_instruction/`은 .gitignore)
- **대용량 데이터셋(.wav, .npy)은 commit 금지** — Google Drive 또는 별도 저장소 사용
- **자체 녹음 음원은 개인정보(승객 대화)** 포함 가능 — 학습 후 원본 보관 시 주의
- **.tflite 모델은 commit OK** (펌웨어 통합용)
- **STM32CubeIDE 빌드 산출물(Debug/, Release/) commit 금지**
- **Seoul Metro 음원 직접 commit 금지** — 라이선스 미확정. 링크·다운로드 스크립트로 관리
- 한국어로 응답해도 OK, 코드 주석은 영어 권장
