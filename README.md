# 지하철 안내방송 기반 하차역 알림 시스템

> **임베디드시스템설계 기말 프로젝트** — 성균관대학교

GPS·WiFi가 닿지 않는 지하 환경에서, STM32 F411RE 보드가 차내 안내방송("이번 역은 ~~역입니다")을 TinyML로 인식해 현재 위치와 하차역을 알려주는 시스템입니다.

## 동기

- 한국 지하철은 깊고, GPS/셀룰러/WiFi 위치 측정이 부정확한 구간이 많음
- **청각장애인**은 안내방송을 들을 수 없고, **에어팟·이어폰을 낀 사용자**도 음악·통화·집중으로 안내를 놓침
- 기존 청각장애 보조기기(Apple Sound Recognition, Neosensory, Lumename, SSIES 등)는 사이렌·알람 같은 위험음에만 집중하고 **지하철 안내방송 시나리오는 빈 칸**
- 안내방송은 환경에 이미 존재하는 위치 신호 — 별도 인프라 없이 활용 가능

## 데모 시나리오

1. **공식 음원 재생 테스트** — Seoul Metro 공식 안내방송을 보드 마이크로 입력, OLED에 현재 역 표시
2. **실차 잡음 강건성** — 1호선 차내 녹음(혼잡 시간대 포함)으로 SNR 강건성 시연
3. **하차 알림** — 사용자가 등록한 하차역 도착 직전 진동 알림

## 시스템 구성

```
[INMP441 마이크] ──I2S──> [STM32 F411RE] ──> [OLED / 진동모터]
                              │
                              ├── CMSIS-DSP (MFCC, VAD)
                              ├── Stage 1: 트리거 KWS  (~5KB)
                              └── Stage 2: 역 분류 CNN (~50KB)
```

### 2-Stage 모델 구조

- **Stage 1 (Always-on)**: "이번 역은 / 다음 역은" 트리거 검출
- **Stage 2 (Triggered)**: 직후 2~3초 안에 등장하는 역 이름 분류
- Confidence threshold로 잘못된 알림(false positive) 억제

## 하드웨어 BOM

| 부품 | 모델 | 비고 |
|---|---|---|
| MCU | STM32 F411RE Nucleo | Cortex-M4 @ 100MHz |
| 마이크 | INMP441 (I2S MEMS) | I2S2 |
| 출력 (TBD) | OLED 0.96" SSD1306 후보 | 역 이름 텍스트 |
| 알림 (TBD) | 코인형 진동모터 후보 | 하차 도착 알림 |
| 백업 알람 | MAX98357A + 4Ω 스피커 | 선택 활용 |

## 소프트웨어 스택

- **학습**: Python + TensorFlow + Keras + librosa (Google Colab)
- **변환**: TensorFlow Lite (INT8 양자화)
- **펌웨어**: STM32CubeIDE + X-CUBE-AI + CMSIS-DSP

## 데이터셋

| 출처 | 용도 |
|---|---|
| 서울교통공사 공개 안내방송 음원 | Clean reference (학습 베이스라인) |
| 1호선 차내 자체 녹음 | 실제 잡음 환경 (강건성 학습) |
| SpecAugment + SNR 증강 | -5dB 까지 동작 목표 |

본 프로젝트는 **1호선 일부 구간**에 한정 (대상 역은 통근 경로 기반으로 추후 확정).

## 차별점

| 비교 대상 | 우리와의 차이 |
|---|---|
| Apple Sound Recognition / Neosensory Buzz | 위험음만 다룸. 안내방송 시나리오 미지원 |
| Lumename (arXiv 2025) | 일반 음성 명령 KWS, 지하철 도메인 X |
| SSIES (ScienceDirect 2025) | 4종 응급음 + DOA, 안내방송 X |
| 지도앱 위치 측위 | GPS·WiFi 의존 → 지하 음영지대 |

## 진행 상황

자세한 진행 상황·기술 결정·5주 로드맵은 [`CLAUDE.md`](./CLAUDE.md) 참조.

## 라이선스

학술 프로젝트 (비상업적 사용). 데이터셋·서울교통공사 음원의 라이선스 별도 준수.
