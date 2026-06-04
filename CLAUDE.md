# CLAUDE.md

이 문서는 Claude Code가 이 저장소에서 작업할 때 참고하는 프로젝트 컨텍스트입니다.

## ⚠️ 작업 규칙 (가장 중요 — 항상 먼저 적용)

1. **임의로 정의/결정하지 말 것.** 코드·실험에 들어가는 설계 결정(게이트 방식, 알고리즘 구조,
   파라미터·임계값·윈도우, 새 용어·개념 도입 등)은 Claude가 혼자 정하지 말고 **먼저 사용자에게
   물어보고 승인받은 뒤** 진행한다. 불가피하게 새 개념이 필요하면 코드에 박지 말고 "제안"으로
   분리해 질문으로 올린다.
2. **물리 가정(특히 타이밍)을 하드코딩 금지.** 정차시간·역간 간격·방송↔이벤트 시간차 같은 값은
   실차에서 급행·지연으로 **변동**한다(§11에서 cadence 기준은 이미 폐기). 절대/상대 시간 윈도우에
   의존하는 규칙은 기본적으로 쓰지 말고, 꼭 필요하면 사용자 확인을 받는다.
3. 기존에 검증된 용어·방법을 우선 사용한다. 모르면 추측 말고 묻는다.
4. **로컬 실험이 ~10분 넘게 걸릴 것 같으면 로컬에서 돌리지 말고 Colab GPU로 돌린다.**
   사용자는 **Colab Pro** 사용 중. 무거운 학습·LOO(특히 인코더·다fold)는 노트북/셀로 만들어
   Colab GPU에서 실행하게 하고, 로컬은 빠른 스모크/단위검증(≤10분)에만 쓴다.
5. **Colab 노트북을 만들거나 줄 때는 "무엇을 어디서 어디로 올리는지"를 항상 표로 명시한다.**
   열 = `올릴 파일 / 로컬 위치(절대경로) / Colab·Drive 업로드 위치 / 용도`. 노트북 README(또는
   노트북 첫 셀)에 이 표를 넣고, 셀에서 고쳐야 할 경로(예: `PKG`,`DATA_ZIP`)도 함께 적는다.
6. **모든 실험은 실차 실시간(스트리밍) 추론을 전제로 만든다.** 예측이 GT를 입력으로 쓰면 안 됨 —
   안내 위치/시각·역 개수(12개)·역 순서를 *미리 알고 깔아두는* 실험(예: 안내 GT 위치에서 분류→
   순서대로 박기)은 **금지**. 검출(KWS·차임)부터 스스로 하고, 과거 정보만으로 causal하게 위치를
   내야 한다. GT는 *채점에만* 쓴다(예측 입력 아님). "ceiling/상한" 같은 비실시간 수치는 보고하지
   않는다.

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
│   ├── path1_train_complete.ipynb ← 학습 실행 결과 보존본
│   └── path2_train.ipynb          ← Path 2 metric-learning 인코더+prototype+KWS (Colab, gen_path2_notebook.py로 생성)
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
    ├── path2_recheck.py           ← 트립 후 마크 후보정 GUI (matplotlib 파형, 클릭=그 지점부터 재생, 휠 줌/스크롤바, suspect 자동 줌, A_train 기본폴더). 재생커서는 별도라 클릭 위치 고정
    ├── path2_kws_inspect.py       ← KWS 점검 뷰어 (read-only). --build로 LOO held-out 검출 캐시(reports/kws_inspect/) → GUI에 마크·트리거 TP/FP·KWS 확률곡선·CMN before/after 스펙트로그램. 런처 Path2 KWS Inspect.lnk + kws_inspect.ico. 상세 PATH2_RESULTS §12-O
    ├── path2_dataset.py           ← 공유 데이터셋 빌더 (CMN + 실노이즈 합성 + 라이브 윈도우). build_kws / build_cnn(13-class) / build_metric_pool(metric). 1차호명 [onset,+2.0s] 윈도우, 탑승역 drop
    ├── path2_poc.py               ← 13-class softmax 로컬 LOO 검증 (시드 고정, class_weight, 1차호명 윈도우+20s 쿨다운)
    ├── path2_metric_poc.py        ← metric-learning 로컬 LOO 검증 (ProtoNet 인코더+64d 임베딩+prototype 최근접, cross-source 에피소드)
    ├── gen_path2_notebook.py      ← notebooks/path2_train.ipynb 생성기 (gen_notebook.py 컨벤션)
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
| Path 2 — 수집 도구 | 캡처 UI(등교/하교 picker · dBFS 레벨미터 · 왕복 segment) + 슬라이서 | ✅ 완료 |
| Path 2 — 수집 하드웨어 | ICS-43434 빵판 결선 + LCD F-F 직결 (LCD 모듈은 불량) | ✅ 결선 완료 |
| Path 2 — 수집 펌웨어 | I2S2 circular DMA → USART2 921600 raw 16-bit PCM 스트림 | ✅ 완료, 보드 플래시됨 |
| Path 2 — 실차 녹음 | 친구 통학 **4 one-way (2 등교 + 2 하교)** | ✅ 4트립 클린 수신·정밀마킹 완료 (0526하교 1개 클리핑 폐기). 0654등교·0642등교·1431하교·2118하교 |
| Path 2 — 분류기 실험 | 라이브 cross-trip 역 분류 go/no-go | 🟥 **softmax 13-class 19% / ProtoNet metric 35~42% — 둘 다 사용가능(90%)에 못 닿음** (천장=실 채널 다양성, 트립 4개). 아래 "분류기 실험 기록" 참조 |
| Path 2 — 재학습 + 시연 | metric-learning 인코더 Colab 학습 + INT8 tflite | 🟨 노트북(`path2_train.ipynb`) 완성. **데모 프레이밍 미정** (Claude AI와 상의 예정) |

## 현재 진행 상황

### 완료
- [x] 타겟 구간 확정 — Path 1 1호선 성균관대 → 신도림 (14역), Path 2 성균관대 → 구로 (13역)
- [x] Seoul Metro 음원 확보 + 16kHz mono 전처리 + 메타데이터
- [x] KWS(트리거) + CNN(역 분류) 학습 → INT8 tflite
- [x] 통합 파이프라인 로컬 검증 (음원별 17/17, 연속 재생 시나리오 정상)
- [x] Path 1 STM32 펌웨어 소스 + X-CUBE-AI 통합 가이드 작성
- [x] Path 2 하드웨어 결선 — ICS-43434는 **F-F로 NUCLEO 모르포 직결**(빵판 폐기, 빵판이 실차 진동에 약함이 확인됨), LCD F-F 직결(모듈 응답 없음, 보류)
- [x] Path 2 수집 펌웨어 (`E:\STM32CubeIDE\workspace\bringup`) — I2S2 circular DMA → USART2 921600 raw 16-bit mono PCM. 보드에 플래시 완료
- [x] Path 2 캡처 UI 개선 — 등교/하교 picker, dBFS 레벨미터(peak hold 포함), 스크롤 13역, ↻ 방향 전환(왕복 segment), 짝수 바이트 정렬 보장
- [x] Path 2 Trip #1 수신 — 2026-05-27 06:54 등교 (구로→성균관대, 35.8min). 오디오 품질 클린 (RMS 237, peak 4102/32768, 클리핑 0, LSB 256/256). 13/13 마크 모두 찍힘 — 단 금천구청·관악·안양 3개는 친구가 늦게 탭해서 sample_index가 실제 안내방송 시점과 어긋남 (LIS로 자동 검출됨)
- [x] `scripts/path2_recheck.py` — 트립 후 마크 후보정 GUI. matplotlib FigureCanvasTkAgg 파형(8000 bucket envelope) + 시작 시 파일 피커 + 좌클릭 즉시 재생 + 마우스 휠 줌(커서 중심) + 수평 스크롤바 + suspect로 점프 시 search window로 자동 줌. `p`로 2초 미리듣기, Space로 spinbox 윈도우 재생, Enter로 sample_index 갱신, .bak 자동 백업 후 저장. `Path2 Recheck.lnk` + `hey_now.ico`로 콘솔 없이 더블클릭 실행 가능 (pythonw + messagebox 크래시 다이얼로그).
- [x] Tk Treeview `<<TreeviewSelect>>` 가상 이벤트가 비동기 큐잉이라 `_suppress_select` sync 플래그로 막을 수 없는 무한 재귀 이슈 발견·수정 — `_refresh_table` 안의 `delete()+selection_set()`이 매 호출마다 이벤트 2개씩 큐에 쌓고, 그게 처리될 때 `_select → _refresh_table` 재진입으로 `update_idletasks()`가 영원히 큐를 비우지 못함. `_on_table_select`에서 `idx == self.selected_idx`면 일찍 빠지는 가드로 해결.
- [x] (2026-05-28) Path 2 4트립 수신·정리 — `A_train/`의 audio×4 중 **0526 하교는 클리핑(RMS 18134, peak 32767, clip 6.8만)로 폐기**, 나머지 3개를 `data/raw/line1_live/<trip_id>/`(audio.wav+marks.json)로 배치. bringup 테스트 잡음 폴더는 `_bringup_test/`로 비파괴 이동.
- [x] (2026-05-28) **로컬 검증으로 도메인 갭 진단** — 클린 14-class 모델은 라이브에서 트리거 **0개**(레벨정규화 후도 0). 원인은 목소리가 아니라 `power_to_db(ref=1.0)` 절대레벨 mismatch. **per-window CMN(채널 평균 정규화)** 적용 시 KWS val **97.7%**로 회복. → 학습·추론·펌웨어 melspec 모두에 CMN 반영 필요.
- [x] (2026-05-28) **자동 마크 위치추적은 불가 확정** — 클린 템플릿 NCC / live↔live NCC / MFCC+CMVN+subseq-DTW / 트립 전체 글로벌 DTW 스캔 4가지 모두 실패(비용 landscape 평평, 정답 역이 안 뜸). 열차 잡음(RMS가 안내방송과 비슷)이 특징을 덮어 알고리즘이 구별 못 함(사람 귀는 분리). → **정밀 마킹은 recheck GUI에서 사람이 청취 후 수행**해야 함.
- [x] (2026-05-28) `scripts/path2_dataset.py` (공유 데이터셋 빌더: CMN + 클린에 실 트레인노이즈 SNR 합성 + light reverb + 실 라이브 윈도우 + 트레인노이즈 negative), `scripts/path2_poc.py` (로컬 CPU 검증 — 합성/실데이터 조건 비교, held-out 트립 채점). tensorflow-cpu 2.21 로컬 설치.
- [x] (2026-05-28) **4트립 정밀 마킹 완료** — `path2_recheck.py`로 0654등교·1431하교·0642등교·2118하교 각 13역 안내방송 onset 청취 후 정확히 지정. A_train→`data/raw/line1_live/`로 동기화(오디오 바이트 동일 확인). 4번째 트립(2118하교) 수신 → **등교 2 + 하교 2** 방향 균형.
- [x] (2026-05-28) **탑승역 마크가 가짜임 발견** — 탑승역(등교 구로/하교 성대)은 이미 타고 있어 안내방송 미녹음인데, 캡처 UI에 13역이 다 있어 친구가 아무 데나 탭함. → 방향별로 탑승역 마크 drop(positive에서 제외, 노이즈로). clean 14-class가 라이브에서 구로로 붕괴했던 원인 중 하나(노이즈→구로 학습).
- [x] (2026-05-28) **1차 호명 윈도우 확정** — 클린 음원 측정: 트리거 onset 기준 "이번역은"~[0,0.7]s, 본역명~[0.9,1.8]s. 분류 윈도우 = **[onset, +2.0s] 고정**("이번역은 [본역명]"만). 코레일 차량은 부역명(마리오아울렛/안양예술공원/성결대/한세대/한국교통대)이 붙지만 서울교통공사 클린 음원엔 없음 → 1차 호명만 써야 신호 일치. 환승/급행역의 2번째 "이번역"은 **20s 쿨다운**으로 무시.
- [x] (2026-05-28) **path2_recheck.py 클릭 재생 버그 수정** — 클릭한 지점부터 재생(pre-roll 제거), 재생 커서를 편집 플레이헤드와 분리(클릭 위치 고정), 편집 중 마크엔 snap 안 함. 기본 폴더 A_train.
- [x] (2026-05-28) **분류기 방향 전환: 13-class softmax → metric-learning(ProtoNet 임베딩+prototype)** — 아래 "Path 2 분류기 실험 기록" 참조. `path2_metric_poc.py`, `gen_path2_notebook.py`, `notebooks/path2_train.ipynb` 작성.

### 남은 일 / 미결정
- [x] **데모 프레이밍 결정** — 최종 아키텍처 = **탑승역 앵커 + KWS 카운팅 + 분류기 교차검증 하이브리드**(상세 [PATH2_RESULTS.md](PATH2_RESULTS.md), 메모리 project-final-architecture). 분류기 단독은 cross-trip ~33% per-mark가 천장이라, 노선 단조성(시퀀스 prior)+탑승역 앵커로 75~100% 달성. 카운팅은 검출 완벽시 100%지만 검출오류 1개에 cascade(~48%)라 분류기가 안전장치. 데이터(채널) 계속 수집이 90%대의 길. (이전 "KWS 카운트 제외"는 번복 — 정확도론 가장 강력)
- [ ] `notebooks/path2_train.ipynb` Colab 실행 — episode 3000(미검증 레버) + INT8 `encoder.tflite`/`prototypes.npy`/`path2_meta.json` 산출. 데이터 zip(클린 wav + 4트립) Drive 업로드 필요. 로컬 변경 git push 선행.
- [x] KWS 트리거 회귀 복구 — 원인 2개: ① `build_kws`의 SpecAugment가 1초 윈도우의 짧은 "이번역은"을 마스킹해 positive를 라벨노이즈로 만듦(val 69%, 상수 0.4 출력, 0검출) → `spec_aug=False`(기본). ② `train` LR 1e-3가 불안정해 일부 fold 붕괴(val 58%) → `lr=5e-4`+epochs/patience↑. 복구 후 4 fold val 98~99%, 슬라이딩 검출 recall 48~79%(동작점별). **남은 한계=cross-trip 정밀도(false 과다)**: held-out 트립 노이즈에 오트리거 다수(8트립 고recall서 ~56/trip), 1431은 채널 약발. `scripts/path2_kws_recover.py`로 재현. **(2026-06-04) KWS false 단독억제 카드 3장 다 패배** — 매치드필터(PATH2_RESULTS §12-L)·hard-neg mining(§12-M: recall 75→58%)·고정구문 prototype 게이트(§12-N: real-FP AUC 0.45<chance)·front-end(CMN이 정적EQ 이미 제거). 공통원인=false가 "음성 vs 음성"+채널변이. → false는 **단독억제 아니라 교차게이트(KWS∧차임)·시퀀스로 흡수**가 본류. 잔여 카드=prototype 정렬탐색(§12-N, 대기). 근본레버=트립(채널) 수.
- [ ] `models/` Path 2 산출물 + `verify_pipeline.py`를 metric/CMN/13역으로 갱신, 펌웨어 `melspec.c`에 CMN 반영
- [ ] 하교 시연용 온보드 통합 펌웨어 — bringup I2S 마이크 DMA + 추론(+CMN) + ST7789V LCD. 디스플레이 입수 후 친구 빌드
- [ ] 친구 보드에서 Path 1 펌웨어 빌드·플래시·테스트 (별도 트랙)

## Path 2 데이터 수집 계획 (4 one-way trip 기준)

지하철 객차는 enclosed acoustic 환경이고 안내방송은 KORAIL 동일 녹음음원의 반복 재생이라 트립간 variation이 작음.

> ⚠️ **사후 정정**: "클래스당 effective ~75면 95%+"라는 초기 sample-complexity 가정은 **틀렸음**. 트립간 variation이 작다는 전제가 깨짐 — 차량 PA·객차 음향·마이크 위치가 트립마다 달라 **cross-trip 일반화가 진짜 병목**이고, 트립을 3→4로 늘려도(샘플 증가) 정확도가 안 올랐다(35~42% 천장). 즉 필요한 건 샘플 수가 아니라 **채널(트립) 다양성**. 아래 "Path 2 분류기 실험 기록" 참조.

**2일 통학 일정**

실제 수집·정밀마킹 완료 트립 (방향 2 등교 + 2 하교, 0526 1개는 클리핑 폐기):

| 트립 (trip_id) | 방향 | 오디오 품질 | 마크 | 상태 |
|---|---|---|---|---|
| 20260526_1942_하교 | 하교 | **RMS 18134, clip 6.8만** | — | ❌ 클리핑 폐기 |
| 20260527_0654_등교 | 등교 | 클린 RMS 237 | 13/13 정밀 | ✅ |
| 20260527_1431_하교 | 하교 | 클린 RMS 330 | 13/13 정밀 | ✅ |
| 20260528_0642_등교 | 등교 | 클린 RMS 228 | 13/13 정밀 | ✅ |
| 20260528_2118_하교 | 하교 | 클린 RMS 226 | 13/13 정밀 | ✅ (4번째) |

> 마크는 친구가 달리는 차에서 탭해 ±수초~수십초 부정확 → `path2_recheck.py`로 청취하며 정밀화 완료(자동화 4종 실패 확인). **각 트립의 탑승역 마크(등교 구로/하교 성대)는 가짜**(미녹음, 임의 탭) → 학습 시 drop. 종착역(등교 성대/하교 구로)은 실재. 트립이 곧 채널 1개라, **3→4트립으로 늘려도 cross-trip 분류 정확도는 안 올랐음**(아래 실험 기록).

**평가 방식 — 트립단위 LOO (정직)**
- 4트립 중 1개를 held-out(미학습=새 채널), 나머지 3개 + 클린으로 학습 → held-out 채점. 4 fold 평균.
- held-in val(같은 분포)은 높게 나오지만 **판단 기준은 held-out**(새 트립). 시드 고정으로 run간 요동 제거.
- 학습 데이터 = 클린 1차호명 + 실 트립 노이즈 합성 + 실 라이브 1차호명 윈도우. 증강은 노이즈/SNR(reverb는 wash).

**월요일(2026-06-01) 라이브 추론 계획 — 등교=노트북 모니터, 하교=온보드 LCD**

> ⚠️ 이 계획은 13-class 시절 작성. 분류기가 cross-trip ~40%라 **데모 프레이밍은 재검토 중**(Claude AI와 상의). 아래 하드웨어/펌웨어 분리 로직(등교=노트북 저위험, 하교=온보드)은 유효.

리스크를 등교 트립에서 0으로 두기 위해 **추론 위치를 트립별로 분리**:

- **등교 (오전, 저위험)**: 검증된 **bringup 펌웨어 그대로**(PCM만 USB-UART 스트리밍) → 노트북이 PCM 수신, **재학습 13-class 모델로 노트북에서 실시간 추론**(지난역/현재역 출력) + **raw PCM 동시 저장**(재학습용). 펌웨어 통합 리스크 0. "디스플레이 없이 노트북 모니터링"과 일치.
- **학교에서 디스플레이 인수** → 온보드 통합 펌웨어 빌드/플래시.
- **하교 (오후, 시연영상)**: **온보드 통합 펌웨어**(bringup I2S 마이크 DMA + app_path1 추론+CMN + ST7789V LCD)로 보드 단독 동작 촬영. 동시에 PCM 백업 저장.

> 등교에서 멀티플렉스(PCM+결정 동시 펌웨어 출력) 대신 **노트북측 추론**을 쓰는 이유: 같은 UX를 펌웨어 무변경으로 얻어 트립 날릴 위험 제거. offline post-edit(트립 audio를 `verify_pipeline.py`에 흘려 결정 추출→영상 오버레이)도 백업으로 항상 가능.

**트립 실패 정의 (재시도 필요)**
- 마크 누락 ≥ 50% (6역 이상)
- audio.wav LSB 분포 unique < 10 (결선 불안정)
- 전체 RMS > 5000 (사실상 클리핑)
- 늦은 탭(out-of-order) ≥ 6역 — `scripts/path2_recheck.py`로 후보정 가능한 수준이 5역 이하여야 train으로 사용

**탭 누락/늦은 탭 후보정 — Trip #1에서 발견된 패턴**
- 친구가 안내방송을 놓치고 다음 역 가서야 깨달은 뒤 스크롤 picker로 돌아가 늦게 탭하는 케이스. `marks.json`에는 station_idx와 sample_index가 어긋난 채 저장됨.
- 검출: `sample_index`로 정렬한 station_idx 순서가 monotone 아니면 LIS-complement로 out-of-order 자동 표시 (`scripts/path2_recheck.py`의 `flag_out_of_order`).
- 보정: 정상 마크 사이 시간 윈도우에서 안내방송 시작점을 청취 후 sample_index 패치.

매 트립 출발 전 사전 검증(실내 RMS 20~50) 통과 시 실패율 ~0.

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

**Path 2 (라이브) — Stage 2를 metric learning으로 전환** (2026-05-28)
- Stage 1 KWS는 동일(트리거 검출). **Stage 2는 13-class softmax 폐기 → 인코더+prototype**.
- **인코더**: 작은 Conv2D + **Flatten**(GAP 금지 — 짧은 본역명 토큰을 시간평균이 뭉갬) → 64-d L2-정규화 임베딩. log-mel(40)+**per-window CMN** 입력.
- **학습**: Prototypical Network(episodic). 같은 역(다른 노이즈/채널)=양성, 다른 역=음성. 에피소드의 support/query를 **다른 소스(synth↔real, real-tripA↔tripB)**에서 뽑아 채널 불변 유도.
- **추론**: 역당 prototype 1개(클린 임베딩 평균) 저장 → 입력 임베딩의 **최근접 prototype**. cosine < τ 또는 top1−top2 < δ 면 보류(abstain).
- **윈도우**: 분류 입력 = "이번역은 [본역명]" [트리거 onset, +2.0s]. 부역명/2차 안내 제외(코레일 부역명은 클린에 없음). 환승역 2번째 "이번역"은 20s 쿨다운으로 무시.
- **CMN은 학습·추론·펌웨어 melspec 모두 동일 적용 필수**(클린 모델이 라이브 0 트리거였던 절대레벨 mismatch 제거).

### Path 2 분류기 실험 기록 (cross-trip go/no-go)

문제: **학습 안 한 트립(=새 채널)에서 역을 맞히는가.** 시드 고정 + 트립단위 leave-one-out(LOO)으로 정직하게 채점. chance=8%(1/13), 사용가능 목표 ~90%.

| 접근 | cross-trip LOO | 비고 |
|---|---|---|
| 13-class softmax (GlobalAvgPool) | **19%** | 한 클래스로 붕괴. GAP가 본역명 토큰을 평균내 뭉갬(held-in val 11%) |
| 13-class softmax (Flatten) | ~19% | held-in val 76%로 회복했으나 cross-trip은 여전히 붕괴 |
| **ProtoNet metric (3트립)** | **42%** | 붕괴 안 함(예측 분산). 2-stage·CMN 유지 |
| ProtoNet metric (4트립) | 35% | **4번째 트립 추가해도 안 오름** → 병목=데이터 양 아님 |
| ProtoNet + reverb 채널증강 | 35% | wash. 합성 reverb는 실 채널차 못 흉내 |
| CMVN / EQ 증강 | 도움 없음 | 정적 EQ는 CMN이 이미 제거 |
| PCEN front-end | 10%→33% | ❌ 패배. CMN이 정적 EQ를 이미 제거해 PCEN 무의미(스케일 보정해도 baseline 미달) |
| **채널 적대 GRL (trips-only, λ=0.3)** | **44%** | ✅ 최고 모델 레버. 임베딩서 채널축 제거, 도메인 헤드는 학습 전용→온보드 비용 0 |
| 실 RIR 보간 증강 | 불가 | clean≠live(다른 녹음, envelope 상관 0.25)→deconvolution 입출력쌍 없음 |
| episode 600→2000 | 44→46% | wash. 천장은 compute 아님 |
| real-only + 실노이즈 증강 | 38% | clean 빼고 real만+증강이 clean-synth(35%)보다 나음. 모든 positive가 cross-trip |
| **+ 시퀀스 prior (노선 단조성)** | **75%** | ✅✅ 돌파. 3/4 트립 100%. 후처리(Viterbi/offset)라 온보드 비용 0 |

**진단/결론** (상세: [PATH2_RESULTS.md](PATH2_RESULTS.md)):
- **모델 레버 천장 ≈ 42~44%**(채널 4개 한계). PCEN 패배, 실 RIR 불가, episode wash 모두 확인 → 병목은 compute가 아니라 **실 채널(트립) 수**. GRL(채널 적대)이 최고 모델 레버(44%, 온보드 0), real-only+증강이 clean-synth보다 나음(38%>35%).
- **돌파구 = 시퀀스 prior**: 열차는 노선을 한 방향 단조 이동(방향 알려짐)이라 연속 안내방송=연속 역. per-mark cosine을 emission, 노선 위상을 transition으로 한 Viterbi/연속-런 디코딩이 **per-mark 33~42% → 75%(3/4 트립 100%)**. 데이터 누수 아닌 실제 물리 제약. 후처리라 **온보드 비용 0**. 90%는 트립 몇 개 더 모으면 도달.
- 핵심 통찰: 안내방송은 KORAIL 단일 녹음의 결정론적 재생(고정 신호). 변하는 건 노이즈(합성 가능) + **채널(실 트립 수만큼만)**. **CMN 필수**(없으면 라이브 0 트리거)이고 그 CMN이 정적 EQ를 지워 EQ/PCEN/RIR 증강을 무력화. **자동 마크 위치추적 불가**(NCC/DTW 4종 실패) → 정밀 마킹은 사람 청취.
- 기록 위치(시드 고정 로컬 재현): `path2_metric_poc.py`(metric/real-only LOO), `path2_grl_poc.py`(GRL), `path2_rir_feasibility.py`(RIR 불가 근거), `path2_seqprior_poc.py`(시퀀스 prior), `path2_export_clips.py`(자르기 청취용 wav).
- ⚠️ 배포 블로커: `encoder.tflite` 646KB(Flatten→Dense 61만 weight) > F411 Flash 512KB → 인코더 축소 필요. KWS(15.7KB)·시퀀스 디코드(후처리)는 문제 없음.

### 신호 처리
- 샘플링: 16kHz
- 특징: log-mel spectrogram (n_fft 512, hop 256, 40 mel) — librosa `melspectrogram` + `power_to_db`
- 펌웨어는 CMSIS-DSP FFT로 동일 알고리즘 재현 (`melspec_ref.py`로 librosa 대조 검증, 오차 ~1e-4 dB)

### 핀맵 (NUCLEO-F411RE)

**Path 2 수집 펌웨어 (확정, 보드 플래시됨)**

ICS-43434는 **F-F 점퍼로 모르포 직결** (빵판 사용 X — 친구 첫 트립에서 빵판 위 점퍼 진동으로 LSB가 0x00/0xFF만 들어오는 클리핑 데이터 생성된 사례 확인).

| 마이크 핀 | NUCLEO 핀 | 모르포 위치 |
|---|---|---|
| VCC | 3V3 | CN7-16 |
| GND | GND | CN10-9 |
| SEL | GND | CN10-20 (또는 마이크 위에서 GND↔SEL 짧은 점프) |
| BCLK | PB13 (I2S2_CK) | CN10-30 |
| LRCL (WS) | PB12 (I2S2_WS) | CN10-16 |
| DOUT (SD) | PB15 (I2S2_SD) | CN10-26 |

USART2 (ST-Link VCP, 921600 baud) — PA2/PA3는 보드 내부 라우팅, 외부 점퍼 불필요.

**Path 1 데모 LCD (배선만 완료, 모듈 불량으로 보류)**

| LCD 핀 | NUCLEO 핀 | 모르포 위치 |
|---|---|---|
| SCL | PA5 (SPI1_SCK) | CN10-11 |
| SDA | PA7 (SPI1_MOSI) | CN10-15 |
| CS | PB6 | CN10-17 |
| RS (DC) | PA9 | CN10-21 |
| RST | PA10 | CN10-33 |

- Path 1 모델 추론은 마이크 입력 없이 Flash 음원 사용 → I2S 비활성
- Path 2 수집 펌웨어는 LCD 사용 안 함 → SPI1 비활성 (배선만 살아 있음)

### Path 2 데이터 품질 검증 신호

매 트립 전 + WAV 사후 점검 시 봐야 할 패턴:

| 신호 | 정상 | 비정상 |
|---|---|---|
| 평상시 RMS | 20~50 | 1000+ (클리핑) |
| 평상시 dBFS | -50 이하 | -10 위 (노란/빨간 영역) |
| 박수 시 RMS | 1000+ 로 튐 | 변화 없음 |
| WAV LSB 분포 | 256가지 다 나옴 | 0x00, 0xFF 두 가지만 (= ΔΣ saturated, 결선 불안정 신호) |
| 클리핑 샘플 수 (전체) | 0 또는 극소 | 0.1% 이상 |

LSB 분포는 `python -c` 로 빠르게: `np.unique(arr & 0xFF).size`. 256이면 정상, 2면 결선 의심.

### 마이크 보호 원칙 (도메인 매칭)

학습 데이터는 추론 환경과 도메인이 일치해야 의미 있음. Path 2에서:

- ✓ **휴지/솜 윈드스크린** — 바람·핸들링 공기압만 차단, 음파는 통과
- ✗ **박스/통/두꺼운 천** — 공진 + 음향 도메인 변경 → 클린 데이터(Path 1)의 변주가 됨, 정보 가치 없음
- ✓ **열차 모터·HVAC·승객 잡담은 그대로 녹음** — 이것이 모델이 학습해야 할 노이즈 도메인
- ✓ 마이크 보드 + NUCLEO를 같은 받침에 고정 — 상대 진동 방지

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
