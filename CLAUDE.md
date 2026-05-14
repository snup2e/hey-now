# CLAUDE.md

이 문서는 Claude Code가 이 저장소에서 작업할 때 참고하는 프로젝트 컨텍스트입니다.

## 프로젝트 개요

- **과목**: 임베디드시스템설계 (성균관대학교)
- **프로젝트명**: TinyML 출튀 도우미 (Smart Attendance Helper)
- **목표**: STM32 F411RE 보드 위에서 ESC-50 환경음 분류 + 자체 음성 키워드 인식을 동작시키고, 호명("김건형") 시 스피커로 "네!" 응답
- **기말 발표 데모 시나리오**:
  1. 강의실 환경음 5종 실시간 분류 (clapping / door_knock / coughing / keyboard_typing / silence)
  2. 호명 응답: 교수님이 "김건형" 호명 → 보드가 "네!" 음성 출력

## 하드웨어 스택

| 부품 | 모델 | 비고 |
|---|---|---|
| MCU | STM32 F411RE Nucleo | Cortex-M4 @ 100MHz, 512KB Flash, 128KB SRAM, FPU 있음, **DAC 없음** |
| 마이크 | INMP441 (I2S MEMS) | I2S2 버스 연결 |
| 오디오 출력 | MAX98357A (I2S DAC + Class-D Amp) | I2S3 버스 연결 |
| 스피커 | 4Ω 3W 40mm 라운드 (가치창조기술 P/N 14193) | MAX98357A 직결 |

구매처: 가치창조기술 (vctec.co.kr)

## 소프트웨어 스택

- **모델 학습**: Python + TensorFlow + Keras + librosa (Google Colab)
- **모델 변환**: TensorFlow Lite (int8 양자화)
- **펌웨어 IDE**: STM32CubeIDE
- **AI 추론**: X-CUBE-AI 플러그인 (.tflite → C 코드 자동 생성)
- **신호처리**: CMSIS-DSP (MFCC 추출)

## 데이터셋

- **ESC-50** (외부 다운로드): https://github.com/karolpiczak/ESC-50
  - 사용할 클래스: clapping, door_wood_knock, coughing, keyboard_typing, silence (자체 생성)
- **자체 녹음** (data/custom/):
  - "김건형" 호명 200샘플 (팀원 4명 × 50회)
  - Negative class (다른 이름 + 잡담) 100샘플
  - 강의실 배경음 100샘플
  - 16kHz, 모노, 1초 WAV

## 디렉토리 구조 (계획)

```
imsisul/
├── CLAUDE.md              ← 이 문서
├── README.md              ← 프로젝트 소개
├── .gitignore
├── docs/                  ← 발표자료, 보고서, 다이어그램
├── data/
│   ├── raw/               ← (gitignore) 원본 오디오
│   ├── processed/         ← (gitignore) MFCC 전처리 결과
│   ├── custom/            ← 자체 녹음 (소량 샘플만 commit)
│   └── README.md          ← 데이터셋 출처/라이선스
├── notebooks/             ← Colab 학습 노트북
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_quantization.ipynb
├── models/                ← 학습된 모델
│   ├── model.tflite       ← 최종 배포 모델 (commit)
│   └── checkpoints/       ← (gitignore) 학습 체크포인트
├── firmware/              ← STM32CubeIDE 프로젝트
│   ├── Core/
│   ├── Drivers/
│   ├── X-CUBE-AI/
│   └── Middlewares/
├── hardware/              ← 회로도, 핀맵
│   ├── pinout.md
│   └── schematic.png
└── scripts/               ← 유틸리티 스크립트
    ├── record_audio.py    ← 자체 녹음용
    └── wav_to_carray.py   ← "네" 음성 → C array
```

## 5주 로드맵

| 주차 | 단계 | 산출물 |
|---|---|---|
| Phase 0 (현재) | 사전 확정 | 교수님 컨펌, 부품 신청, 역할 분담 |
| Phase 1 | 데이터셋 구축 | 1,500+ 라벨 샘플 |
| Phase 2 | 모델 학습 | model.tflite (val_acc 80%+, < 80KB) |
| Phase 3 | 하드웨어 통합 | 마이크↔스피커 동작 확인 |
| Phase 4 | 펌웨어 통합 | 풀스택 실시간 추론 |
| Phase 5 | 검증 & 시연 | 발표 자료 + 시연 영상 |

## 현재 진행 상황 (Phase 0)

- [x] 프로젝트 주제 확정 ("출튀 도우미")
- [x] 시나리오 설계 (환경음 분류 + 호명 응답 하이브리드)
- [x] 하드웨어 BOM 확정 (INMP441 + MAX98357A + 4Ω 3W 스피커)
- [x] GitHub repo 초기 셋업
- [ ] 교수님께 시나리오 컨펌 메일 발송
- [ ] 부품 신청서 제출
- [ ] 팀 역할 분담 회의
- [ ] STM32CubeIDE + X-CUBE-AI 설치

## 핵심 기술 결정 사항

### 모델 아키텍처
- **입력**: 16kHz 1초 윈도우 → MFCC (40 mel bands × 40 frames)
- **모델**: Conv2D 3층 + Dense (depthwise separable 검토 중)
- **양자화**: INT8 (float32 대비 4배 압축)
- **목표 사이즈**: < 80KB Flash, < 50KB peak RAM

### 신호 처리
- **샘플링**: 16kHz (음성 + 환경음 모두 충분)
- **윈도우**: 1초 (FFT 1024, hop 512)
- **MFCC**: 40 계수, log-mel 기반

### 핀맵 (잠정)
```
INMP441 (I2S2 - 마이크 입력)
  VDD → 3.3V
  GND → GND
  SCK → PB10 (I2S2_CK)
  WS  → PB12 (I2S2_WS)
  SD  → PB15 (I2S2_SD)
  L/R → GND (Left channel)

MAX98357A (I2S3 - 오디오 출력)
  VIN  → 5V (또는 3.3V)
  GND  → GND
  BCLK → PC10 (I2S3_CK)
  LRC  → PA15 (I2S3_WS)
  DIN  → PC12 (I2S3_SD)
  GAIN → 플로팅 (기본 9dB)
  SD   → 3.3V (셧다운 비활성)
```
※ 핀맵은 STM32CubeMX에서 최종 검증 필요 (Nucleo 보드 LED/버튼 충돌 확인)

## 참고 자료

- ESC-50 데이터셋: https://github.com/karolpiczak/ESC-50
- TinyML 책 GitHub: https://github.com/yunho0130/tensorflow-lite
- ARM ML-KWS-for-MCU: https://github.com/ARM-software/ML-KWS-for-MCU
- jonnor ESC-CNN-microcontroller: https://github.com/jonnor/ESC-CNN-microcontroller
- X-CUBE-AI 가이드 (DigiKey): https://www.digikey.com/en/maker/projects/tinyml-getting-started-with-stm32-x-cube-ai/f94e1c8bfc1e4b6291d0f672d780d2c0
- STM32 acoustic scene classification (ST 공식): https://www.st.com/content/st_com/en/st-edge-ai-suite/case-studies/acoustic-scene-classification.html

## Claude Code 작업 시 주의사항

- **강의자료 PDF는 commit 금지** (`project_instruction/`은 .gitignore)
- **대용량 데이터셋(.wav, .npy)은 commit 금지** — Google Drive 또는 별도 저장소 사용
- **.tflite 모델은 commit OK** (펌웨어 통합용)
- **STM32CubeIDE 빌드 산출물(Debug/, Release/) commit 금지**
- 한국어로 응답해도 OK, 코드 주석은 영어 권장
