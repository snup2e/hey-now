# TinyML 출튀 도우미 (Smart Attendance Helper)

> **임베디드시스템설계 기말 프로젝트** — 성균관대학교

STM32 F411RE 보드 위에서 TinyML 모델을 실행해, 강의실 환경음을 분류하고 교수님의 호명("김건형")에 자동으로 "네!"라고 응답하는 시스템입니다.

## 데모 시나리오

1. **강의실 환경음 분류** — 박수 / 노크 / 기침 / 키보드 / 정적
2. **호명 응답** — "김건형" → 스피커로 "네!" 출력

## 시스템 구성

```
[INMP441 마이크] ──I2S──> [STM32 F411RE] ──I2S──> [MAX98357A] ──> [4Ω 스피커]
                              │
                              ├── X-CUBE-AI (TFLite Micro)
                              ├── MFCC (CMSIS-DSP)
                              └── 경량 CNN (INT8 양자화)
```

## 하드웨어 BOM

| 부품 | 모델 |
|---|---|
| MCU | STM32 F411RE Nucleo |
| 마이크 | INMP441 (I2S MEMS) |
| 오디오 IC | MAX98357A (I2S DAC + Amp) |
| 스피커 | 4Ω 3W 40mm 라운드 |

## 소프트웨어 스택

- **학습**: Python + TensorFlow + librosa (Google Colab)
- **변환**: TensorFlow Lite (INT8 양자화)
- **펌웨어**: STM32CubeIDE + X-CUBE-AI + CMSIS-DSP

## 데이터셋

- [ESC-50](https://github.com/karolpiczak/ESC-50) (환경음 클래스 선별 사용)
- 자체 녹음 (호명 키워드, 16kHz mono WAV)

## 진행 상황

자세한 진행 상황과 기술 결정은 [`CLAUDE.md`](./CLAUDE.md) 참조.

## 라이선스

학술 프로젝트 (비상업적 사용)
