# Hey now!

> 지하철 안내방송 기반 하차역 알림 시스템 — **임베디드시스템설계 기말 프로젝트** (성균관대학교, 4조)

### [▶ 발표 슬라이드 열기](https://snup2e.github.io/hey-now/)

[![발표 슬라이드](https://img.shields.io/badge/%E2%96%B6%20%EB%B0%9C%ED%91%9C%20%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C-%EC%97%B4%EA%B8%B0-37B6A0?style=for-the-badge)](https://snup2e.github.io/hey-now/)

> 브라우저 단일 HTML 덱입니다. 키보드 `→`/`Space`로 다음, `F`로 전체화면, `?`로 단축키.
> 최종발표 덱·대본·캡처 PDF는 [`최종발표/`](./최종발표) 폴더에 있습니다.

GPS·WiFi가 닿지 않는 지하 환경에서, STM32 F411RE 보드가 차내 안내방송("이번 역은 ~~역입니다")과
출발 신호음을 TinyML로 인식해 현재 위치/하차역을 알려주는 시스템입니다. 제품 모델명은 **Hey now!** —
놓친 안내방송을 디바이스가 대신 잡아 사용자의 주의를 환기시킨다는 의미.

## 동기

- 한국 지하철은 깊고, GPS/셀룰러/WiFi 위치 측정이 부정확하거나 차단되는 구간이 많음
- **청각장애인**은 안내방송을 들을 수 없고, **에어팟·이어폰을 낀 사용자**도 음악·통화·집중으로 안내를 놓침
- 기존 청각장애 보조기기(Apple Sound Recognition, Neosensory Buzz, Lumename, SSIES 등)는 사이렌·알람 같은 위험음에만 집중하고 **지하철 안내방송 시나리오는 빈 칸**
- 안내방송은 환경에 이미 존재하는 위치 신호 — 별도 인프라 없이 활용 가능

## 2-Path 진행 전략

타겟 노선은 서울 지하철 **1호선**(친구 통학 경로 기반). 두 갈래로 진행했습니다.

| | **Path 1 — 시뮬레이션** | **Path 2 — 실차** |
|---|---|---|
| 입력 | 보드 Flash/스피커의 **공식 음원** | 1호선 차내 **실차 녹음** (혼잡·잡음 포함) |
| 구간 | 성균관대 → 신도림 (14역) | 성균관대 ↔ 구로 (13역) |
| 모델 | KWS(트리거) + CNN(역 분류) | 출발 신호음(차임) 검출 |
| 결과 | 깨끗한 음원에서 **역 분류 성공** | 역 이름 분류는 **실차의 벽**에 막힘 → **차임 검출로 피벗**, 온보드 동작 확인 |

> **핵심 서사**: 깨끗한 음원(Path 1)에서는 KWS+CNN으로 역 이름까지 정확히 분류된다. 그러나 실제 열차(Path 2)에서는
> 차량 PA·객차 음향·마이크 위치가 트립마다 달라 *역 이름 분류*가 일반화되지 않았다(cross-trip 천장 ~42%).
> 그래서 모든 열차에 공통인 **출발 신호음(닫힘 차임)** 검출로 방향을 틀어, 마이크 하나로 온디바이스 검출이 동작함을 확인했다.

## 시스템 구성

```
[ICS43434 마이크] ──I2S──> [STM32 F411RE] ──> [디스플레이: "하차벨이 울립니다"]
                              │
                              ├── 스트리밍 log-mel + CMN (자체 FFT)
                              ├── Path 1: KWS 트리거 → CNN 역 분류 (INT8)
                              └── Path 2: 출발 신호음 검출 CNN (INT8)
```

- **MCU**: STM32 F411RE Nucleo — Cortex-M4 @ 100MHz, 512KB Flash, 128KB SRAM, FPU
- **마이크**: ICS43434 I2S MEMS — NUCLEO 모르포에 F-F 직결(빵판은 실차 진동에 약해 폐기)
- **디스플레이**: 2.0" SPI LCD 예정이었으나 모듈 불량(흰 화면)으로 보류 → **PC 디스플레이 앱**으로 시연
- **오디오 백업**: MAX98357A I2S DAC + 4Ω 3W 스피커 (비상 알람용, 선택)

## 신호 처리 — 음성이 "이미지"가 되기까지

마이크가 받은 소리는 STM32 안에서 다음 단계를 거쳐 CNN이 읽는 한 장의 **log-mel spectrogram**이 됩니다.

| 단계 | 처리 | 왜 |
|---|---|---|
| **1. I2S 캡처** | MEMS → ΔΣ ADC → I2S 16-bit PCM, **16 kHz** | 음성 대역 대부분이 8 kHz 이하 → Nyquist로 16 kHz면 충분 |
| **2. STFT** | n_fft **512**, hop **256**, Hann window | 비정상(non-stationary) 음성을 짧은 구간에서 정상으로 근사해 주파수 분석 |
| **3. Mel filterbank** | 멜 스케일 **40개** 삼각 필터로 차원 축소 | 사람 귀의 비선형 주파수 인지 모방, 모델 경량화 |
| **4. Log** | 멜 에너지에 log 적용 → **log-mel** | 사람은 음량을 로그로 인지(Weber–Fechner) |
| **5. CMN** | per-window **채널 평균 정규화** | **결정적**. 마이크/채널 절대 레벨 mismatch 제거 — 없으면 라이브에서 검출 0 |

> 1초 윈도우 = 40 mel × **63 frame**, 2초 윈도우 = 40 mel × **126 frame**. CNN은 이 시간×주파수 패턴을
> 이미지처럼 보고 translation invariance로 약간의 시간 이동에 강건하게 분류합니다.
> 펌웨어는 CMSIS-DSP 없이 **자체 FFT**로 동일 알고리즘을 구현(`melspec_ref.py`로 librosa 대조 검증).

> ℹ️ 초기 계획의 PCEN은 **per-window CMN**으로 대체했습니다 — 실차 도메인 갭의 진짜 원인은 잡음이 아니라
> `power_to_db(ref=1.0)` **절대 레벨 mismatch**였고, CMN 적용으로 KWS가 라이브에서 회복했습니다.

## KWS란 — 왜 ASR이 아니라 KWS인가

KWS(Keyword Spotting)는 전체 음성 인식(ASR)의 축소판이 아니라 **별도의 경량 분야**입니다.

| | ASR (전체 음성 인식) | **KWS (Keyword Spotting)** |
|---|---|---|
| 목적 | 모든 발화를 텍스트로 | **정해진 단어/구간만 감지** |
| 모델 크기 | 수백 MB ~ GB | **수십 KB** |
| 대표 예 | Whisper, Google STT | "헤이 시리", "오케이 구글" |
| 우리 사용 | ❌ STM32 한계 초과 | ✅ 정형 안내방송에 최적 |

> 지하철 안내방송은 어휘가 한정되고 패턴이 정형이라 ASR을 풀로 돌릴 필요 없이 KWS만으로 풀린다 —
> 이것이 STM32 위에서 동작 가능한 결정적 이유.

## Path 1 — 공식 음원 2-Stage (시뮬레이션, 성공)

서울교통공사 공식 안내방송으로 학습·시연. **KWS 트리거 → CNN 역 분류**의 2-stage 구조.

| | **Stage 1 · KWS 트리거** | **Stage 2 · 역 분류 CNN** |
|---|---|---|
| 입력 | 1초 윈도우, log-mel 40×63 | 트리거 직후 2초, log-mel 40×126 |
| 출력 | "이번 역은" / 무관 2-class | 14역 분류 |
| 모델 | `kws.tflite` **15.6 KB** (INT8) | `cnn.tflite` **16.4 KB** (INT8) |
| 성능 | val **95.7%** · precision 94.4% / recall 93.0% | 윈도우 **92.3%** · 음원 단위 **100% (17/17)** |

**왜 2-stage인가**: ① Stage 1만 상시 동작해 평균 전력 최소화, ② Stage 2가 트리거 후에만 동작해
광고·잡담 오반응 방지, ③ 이진 트리거와 다중 분류를 각각 최적 구조로 설계.

**파이프라인** (`verify_pipeline.py` = 펌웨어 `app_path1.c`): KWS 1초 윈도우를 0.25초 간격 슬라이딩,
트리거 임계 0.6 + **연속 3윈도우 디바운스**로 오검출 차단, 트리거 확정 시 +1.5초부터 2초를 CNN에 입력,
CNN confidence < 0.5면 분류 보류.

## Path 2 — 실차의 벽과 피벗

실제 1호선 차내에서 **8 one-way 트립**(등교 4 + 하교 4)을 STM32 + ICS43434로 자체 녹음·정밀 마킹했습니다.

**부딪힌 벽 — cross-trip 일반화.** 안내방송은 코레일 단일 녹음의 반복 재생이라 *변하지 않는* 신호지만,
변하는 것은 **채널**(차량 PA·객차 음향·마이크 위치)이고 채널 수는 트립 수만큼뿐입니다. 트립을 4→8로 늘려도
역 *이름* 분류 정확도는 ~42%에서 천장이었습니다(13-class softmax 19% · ProtoNet 인코더 42% · 채널적대 GRL 44%).
KWS 트리거도 "음성 vs 음성"이라 false를 단독으로 억제하지 못했습니다. → 역 이름 인식 자체를 접었습니다.
(전체 실험·진단·재현 스크립트: [`PATH2_RESULTS.md`](./PATH2_RESULTS.md))

**피벗 — 출발 신호음(닫힘 차임) 검출.** 역 이름을 맞히는 대신, 모든 열차에 공통인 **출발 신호음**을
검출합니다. HPSS로 하모닉 톤을 분리해 검출하며, **7트립 LOO recall 94% / false 10.3per-trip**(confusion matrix
TP 74 / FN 5 / FP 72). 마이크 하나로 충분하고 별도 센서가 필요 없습니다.

**온디바이스 동작 확인 (2026-06-18).** `chime.tflite`(INT8, plain log-mel + CMN)를 X-CUBE-AI로 STM32에 올려
마이크 → 차임 검출 → "하차벨이 울립니다" 출력까지 **보드에서 동작 확인**. (입력 피딩 버그 해결: X-CUBE-AI가
입출력을 activation 버퍼 안에 할당하므로 네트워크 내부 입력 버퍼에 직접 기입하고 `weights=NULL`로 바인딩 유지.)
온보드 LCD는 ILI9341 모듈 불량으로 보류 → **PC 디스플레이 앱**(`scripts/path2_chime_display.py`)으로 시연.

## 학습 → 배포 파이프라인

```
수집 .wav (서울교통공사 + 1호선 자체 녹음)
   → librosa 전처리 (log-mel + CMN)
   → TensorFlow / Keras 학습
   → TFLite Converter + INT8 양자화  ← STM32에 욱여넣는 핵심
   → X-CUBE-AI 플러그인으로 C 코드 생성
   → STM32CubeIDE 펌웨어 통합 → Flash에 모델 embedded
```

**INT8 양자화**: Float32 가중치·활성화를 INT8로 변환 → 메모리 ~4× 압축 + Cortex-M4 SIMD 가속.
F411RE(512KB Flash·128KB SRAM)에는 Float32 모델이 못 올라가므로 필수.

## 하드웨어 BOM

| 부품 | 모델 | 비고 |
|---|---|---|
| MCU | STM32 F411RE Nucleo | Cortex-M4 @ 100MHz, 512KB Flash, 128KB SRAM, FPU |
| 마이크 | ICS43434 (I2S MEMS) | I2S2 circular DMA, 16kHz. 모르포 F-F 직결 |
| 디스플레이 | 2.0" SPI LCD (ILI9341) | 모듈 불량 보류 → PC 디스플레이 앱으로 시연 |
| 가속도계 | ADXL (키트) | Path 2 중간 설계에서 사용 검토 후 최종 구성에서 제외 |
| 백업 알람 | MAX98357A + 4Ω 3W 스피커 | 선택 활용 |

## 소프트웨어 스택

- **학습**: Python + TensorFlow + Keras + librosa (Google Colab)
- **변환**: TensorFlow Lite (INT8 양자화)
- **로컬 검증**: librosa + ai-edge-litert / tensorflow-cpu
- **펌웨어**: STM32CubeIDE + X-CUBE-AI + 자체 FFT log-mel

## 데이터셋

| 출처 | 용도 | 상태 |
|---|---|---|
| 서울교통공사 공개 안내방송 음원 | Path 1 clean reference (14역) | ✅ 전처리·학습·검증 완료 |
| 1호선 차내 자체 녹음 (8 one-way 트립) | Path 2 실차 잡음 환경 (13역) | ✅ 클린 수신·정밀 마킹 완료 |
| SpecAugment + SNR 증강 | 역당 음원이 적어 증강 필수 | ✅ |

원본 음원/대용량 데이터는 gitignore(개인정보 포함 가능) — `data/metadata.csv` 매핑과 `.tflite` 모델만 commit.

## 차별점

| 비교 대상 | 우리와의 차이 |
|---|---|
| Apple Sound Recognition / Neosensory Buzz | 위험음만 다룸. 안내방송 시나리오 미지원 |
| Lumename (arXiv 2025) | 일반 음성 명령 KWS, 지하철 도메인 X |
| SSIES (ScienceDirect 2025) | 4종 응급음 + DOA, 안내방송 X |
| 지도앱 위치 측위 | GPS·WiFi 의존 → 지하 음영지대 |

## 진행 상황 요약

- ✅ Path 1: KWS + CNN(INT8) 학습·검증(음원 17/17) + STM32 펌웨어 소스
- ✅ Path 2: 8트립 실차 녹음, 역 이름 분류 한계 진단(cross-trip 천장), **출발 신호음 검출로 피벗** (LOO recall 94%)
- ✅ 출발 신호음 검출 **온보드 동작 확인** + PC 디스플레이 시연
- ✅ 최종발표 덱 완성 ([`최종발표/`](./최종발표))

자세한 기술 결정·실험 로그는 [`CLAUDE.md`](./CLAUDE.md), [`PATH2_RESULTS.md`](./PATH2_RESULTS.md) 참조.

## 라이선스

학술 프로젝트 (비상업적 사용). 데이터셋·서울교통공사 음원의 라이선스 별도 준수.
