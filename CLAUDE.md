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
| 가속도계 | ADXL (키트 보유, 모델 확인 예정) | **Path 2 신규** — 열차 정차/주행 검출 = 주(主) 카운트. 학습 대상 아님(임계 기반 신호처리). I2C/SPI 결선 TBD. 과제로 사용 경험 있음 |
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
├── 최종발표/               ← 최종발표 HTML 덱(완성, 9장+Q&A). **상세 = 최종발표/README.md** (순서·배포·테마·문구정책)
│   ├── index.html / README.md / demo.mp4 / PretendardVariable.woff2
│   ├── Hey_now_최종발표.pdf       ← 캡처 PDF(12p, ?shot→Pillow). PPT 대용 배포본
│   ├── 발표대본.md               ← 7분 대본(슬라이드별·2인 분할·핵심 의사결정 ★8·9)
│   └── figures/ (architecture.png[사용자그림]·labeling.png[마킹GUI]·collect_00~09.jpg[카톡]·alert_preview.png)
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
│   └── path2_train.ipynb          ← (폐기) Path 2 metric-learning 인코더+KWS Colab 노트북. 현행=차임 검출 LOO + 가속도계 카운팅
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
    ├── path2_dataset.py           ← 공유 데이터셋 빌더 (CMN + 실노이즈 합성 + 라이브 윈도우). 1차호명 [onset,+2.0s] 윈도우, 탑승역 drop
    ├── path2_door_poc.py          ← **현행** 차임/문 검출 LOO (HPSS 하모닉 톤, win=3.0s, 트립단위). door-side 카운터 채점
    ├── path2_recheck.py / path2_event_mark.py ← 마크·이벤트(차임) 정밀 마킹 GUI
    ├── (폐기) path2_poc.py / path2_metric_poc.py / path2_grl_poc.py / path2_seqprior_poc.py / gen_path2_notebook.py ← 음향 분류기 실험(softmax·ProtoNet·GRL·시퀀스 prior). 기록은 PATH2_RESULTS §1~§12
    └── README_path2.md            ← 친구용 실차 녹음 단계별 가이드
    # 차임+가속도계 카운팅 상세·실험 로그: PATH2_RESULTS.md (§11 door 이벤트, §12 차임 HPSS/융합)
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
| Path 2 — 실차 녹음 | 친구 통학 **8 one-way (4 등교 + 4 하교)** | ✅ 8트립 클린 수신·정밀마킹 완료 (0526하교 1개 클리핑 폐기). trip4(2118하교)만 차임 +4세미톤 변종 |
| Path 2 — 음향 분류기(KWS·인코더) | 라이브 cross-trip 역 *이름* 분류 go/no-go | 🟥 **폐기.** softmax 13-class·ProtoNet 인코더·KWS 트리거 다 시도했으나 cross-trip 채널 천장으로 사용가능(90%) 미달. 상세 [PATH2_RESULTS.md](PATH2_RESULTS.md) → 차임+가속도계 카운팅으로 전환 |
| Path 2 — 차임-only 온디바이스 검출 + 디스플레이 | 마이크 → 차임 CNN → "하차벨이 울립니다" | ✅ **온보드 검출 동작 확인 (2026-06-18).** 입력피딩 버그 해결(내부버퍼 기입+weights=NULL+win2.0s). LCD는 ILI9341/흰화면으로 보류 → **PC 디스플레이 앱**으로 시연, 발표는 시연영상. 상세 [PATH2_RESULTS.md](PATH2_RESULTS.md) §13 |

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
- [x] **데모 아키텍처 2차 피벗 (2026-06-18)** — **가속도계 제거. 차임-only 온디바이스 검출 + 디스플레이.** 마이크 → 차임 CNN(plain logmel_cmn, HPSS 없음) → "하차벨이 울립니다" + 작게 예상역(저신뢰라 작게). 예상역 카운팅은 2단계. 상세 아래 "Path 2 온디바이스 차임 데모", [PATH2_RESULTS.md](PATH2_RESULTS.md) §13, 메모리 project-final-architecture.
- [x] **차임 검출 모델·펌웨어 작성 완료** — `models/chime.tflite`(INT8, plain logmel_cmn), `firmware/`(melspec_stream·app_chime·chime_runner·chime_meta.h), bringup main.c 배선, PC 디스플레이 앱+런처, 폰 테스트 클립.
- [x] ✅ **온보드 입력 피딩 버그 해결 (2026-06-18).** PC 0.99 vs 보드 50%로 *모델 vs 런타임*을 갈라 진단 → 원인은 외부 I/O가 아니라: ① **X-CUBE-AI가 입력/출력을 activation 버퍼 안에 할당**(allocate-inputs/outputs; `.ioc useInputAllocation=false`도 무시됨)이라 외부 `in[0].data`를 무시, ② `chime_runner.c`가 weights에 **테이블 주소**(`ai_chime_data_weights_get()`=`g_chime_weights_table`, blob 아님)를 넘겨 올바른 가중치 바인딩을 덮어써 출력이 상수 0.5. **해결 = ① 네트워크 내부 입력버퍼(`ai_chime_inputs_get()[0].data`)에 직접 기입+출력도 거기서 읽기, ② `ai_chime_create_and_init(net, act, NULL)`(weights=NULL → data_params_get의 올바른 바인딩 유지, Path1 방식).** win_s=2.0(126프레임) 재내보내 RAM 적합. selftest `const+2≠const−2` 확인, 차임 클립으로 **검출 동작 확인**.
- [x] **차임 온보드 검출 동작 확인** — 마이크 → 차임 검출 → UART "하차벨이 울립니다" → PC 디스플레이 앱. 발표는 시연영상으로 진행.
- [x] **LCD 보류 — 패널은 ILI9341 (ST7789 아님).** ILI9341 init 교체·CS 연속 트랜잭션(0x2C+픽셀 한 세션)·Mode0·390kHz·SWRESET까지 시도했으나 흰 화면 지속(모듈/배선 의심, SDO 없어 ID읽기 불가). **시연은 PC 디스플레이 앱으로 확정**, 온보드 LCD는 향후. bringup `app_config.h`의 `LCD_TEST_ONLY` 스위치=1로 색순환 테스트 가능(차임 AI 컴파일 제외).
- [x] **최종발표 덱 (2026-06-18, 거의 완성)** — `최종발표/index.html` (Terra 종설 템플릿 기반 단일파일 HTML 덱, 민트-틸 테마, Pretendard 임베드, ?shot=N 캡처). **상세는 `최종발표/README.md`** (슬라이드 순서·자료·테마·문구정책 표). 순서: 문제정의→시스템 아키텍처→**데이터 수집·라벨링·분석**(한 섹션, 분석=역별 안내방송 개수 EDA 막대그래프)→**접근①+벽 병합 한 슬라이드**(좌=KWS+CNN 깨끗한 음원 성공, 우=실제 열차 벽 표)→방향전환(출발 신호음, confusion matrix TP74/FN5/FP72=recall94%)→시연(demo.mp4)→결론. **온디바이스 검증 슬라이드는 폐기.** architecture.png(사용자 직접 그림)·demo.mp4·카톡 수집사진·라벨링 GUI 다 삽입됨.
  - **문구 정책(중요·재작업 금지):** 지적된 6개(디바운스·실차·cross-trip·결정론적 재생·트립·차임→출발 신호음)만 쉬운 말, **기본 기술용어(KWS·CNN·log-mel·CMN·INT8·ProtoNet·GRL·X-CUBE-AI·recall·LOO·도메인 갭)는 유지.** (한 번 과하게 풀었다가 원상복귀한 이력 — 다시 풀지 말 것.)
  - 표지 입력 완료: **4조 · 팀원 김건형·차현비 · 지도교수 정조운**. 결론 = 한계 2개(하드웨어 STM32 제약·주제 난도) + 성과(차임 검출), 한계2/3 해결과정 행 제거. 성능표는 confusion matrix(검출은 TN="정상 기각" 라벨, 분류는 정확도).
  - **배포 산출물 완료**: ① `최종발표/` 폴더째(상대경로 자산 7개 누락無), ② **캡처 PDF `최종발표/Hey_now_최종발표.pdf`**(12p, 2560×1440, `?shot=0..11` 헤드리스 크롬 PNG→Pillow 결합. 데모 10p는 PDF라 영상 대신 포스터→현장에서 demo.mp4 따로 재생. 덱 수정 시 재생성), ③ **`최종발표/발표대본.md`**(7분, 슬라이드별, 2인 분할 [A]1~7/[B]8~12, ★8·9가 핵심 의사결정). 상세 `최종발표/README.md`.
- [ ] 차임 false 동작점 튜닝 (선택) — `PL.run_chime_threshold_sweep()`로 false~2-3/trip 임계를 `chime_meta.h` 상수에만 반영.
- [ ] 친구 보드에서 Path 1 펌웨어 빌드·플래시·테스트 (별도 트랙)

### Path 2 온디바이스 차임 데모 (2026-06-18, 온보드 검출 동작 확인)

**데모 프로젝트 = `E:\STM32CubeIDE\workspace\bringup`** (Path 2 수집 펌웨어 베이스 — I2S2 24-bit 16kHz circular DMA 마이크 동작 중). 차임 데모로 개조함:
- 마이크 PCM(왼쪽채널 hi-16bit = 학습 도메인) → `app_chime_feed` → 검출 시 UART로 "하차벨이 울립니다".
- **펌웨어 파일**(`firmware/` 원본 → bringup `Core/{Inc,Src}`에 복사): `melspec_stream.c/h`(스트리밍 log-mel, **자체 FFT — CMSIS-DSP 불필요**, 188-col int16 링), `app_chime.c/h`(top_db→행CMN→/STD→추론→디바운스·쿨다운+진단 전역), `chime_runner.c/h`(X-CUBE-AI 10.x 래퍼, **in-activations 내부 IO 기입 + weights=NULL** — §13 해결), `chime_meta.h`(Colab 자동생성: 윈도우·정규화·임계), `mel_filterbank.h`(Path1 공유).
- **X-CUBE-AI**: 네트워크 이름 `chime`, float I/O, "Not selected" 앱, **allocate-inputs/outputs(IO가 activation 안)**. `ai_chime_create_and_init(net, act_addr[], NULL)` — activations만 우리 버퍼, **weights=NULL**(내부 바인딩 유지; `ai_chime_data_weights_get()`는 *테이블 주소*라 넘기면 가중치 깨짐). IO는 `ai_chime_inputs_get()[0].data`/`outputs_get()[0].data`(=arena 내부)에 직접 기입·판독. (롱패스 레지스트리 `LongPathsEnabled=1` 켜야 stedgeai 동작.)
- **진단**: main.c가 부팅 시 `[boot] ai_type/code`, `[selftest] const+2/−2`, 1초마다 `[dbg] half/eval/fill/p%/run/feat`. (selftest가 입력 피딩 정상 여부 판정 — 다르면 OK.)
- **검증된 사실**: INT8 tflite는 PC에서 차임 0.99/주행 0.0(`run_chime_loo` logmel_cmn 81%/13.9fp@3s). 보드 알고리즘 파이썬 미러도 0.99. 자체 FFT = numpy와 1e-14 일치. **입력피딩은 내부버퍼 기입+weights=NULL+win2.0s로 해결 → 보드에서도 PC와 동일하게 검출 동작 확인.**
- **PC 디스플레이**(시연/노트북 모니터, **온보드 LCD 대체**): `scripts/path2_chime_display.py`(**창모드 기본·F11 전체화면**, COM12 자동, P바 + **빨강 점멸 큰 알림**) + 더블클릭 `Hey now Display.lnk`(--fullscreen 제거됨). `[dbg] p=`·`하차벨이 울립니다` 줄만 파싱 → 펌웨어 변경 불필요. **시리얼 포트는 한 프로그램만** — 디버깅 땐 시리얼 모니터, 시연 땐 디스플레이 앱.
- **폰 테스트 클립**: `data/processed/chime_test/`(chime_A/B = 잘 잡히는 트립, noise = 안 잡혀야 정상) + README.
- 보드: **COM12**(ST-Link VCP), USART2 **921600**, 시리얼 인코딩 **UTF-8**.

## Path 2 데이터 수집 계획 (오디오 8 one-way trip 수집 완료)

지하철 객차는 enclosed acoustic 환경이고 안내방송은 KORAIL 동일 녹음음원의 반복 재생이라 트립간 variation이 작음.

> ⚠️ **사후 정정**: "클래스당 effective ~75면 95%+"라는 초기 sample-complexity 가정은 **틀렸음**. 차량 PA·객차 음향·마이크 위치가 트립마다 달라 **cross-trip 일반화가 진짜 병목**이고, 트립을 4→8로 늘려도(샘플 증가) 역 이름 분류 정확도가 안 올랐다(~42% 천장). 즉 음향 분류는 샘플 수가 아니라 **채널(트립) 다양성**이 병목 → 분류 폐기·카운팅 선회(위 "음향 분류기 실험" 참조).

> 📍 **가속도계 측정 트립은 1~2회만.** accel은 학습 대상이 아니라 임계 기반 신호처리라 트립 다수가 불필요 — stop-detection 임계·차임 시간정렬을 한두 번 트립으로 확인·튜닝하면 됨. (아래 오디오 8트립 표는 음향 분류기 실험 시절 수집분.)

**2일 통학 일정**

실제 수집·정밀마킹 완료 트립 (방향 2 등교 + 2 하교, 0526 1개는 클리핑 폐기):

| 트립 (trip_id) | 방향 | 오디오 품질 | 마크 | 상태 |
|---|---|---|---|---|
| 20260526_1942_하교 | 하교 | **RMS 18134, clip 6.8만** | — | ❌ 클리핑 폐기 |
| 20260527_0654_등교 | 등교 | 클린 RMS 237 | 13/13 정밀 | ✅ |
| 20260527_1431_하교 | 하교 | 클린 RMS 330 | 13/13 정밀 | ✅ |
| 20260528_0642_등교 | 등교 | 클린 RMS 228 | 13/13 정밀 | ✅ |
| 20260528_2118_하교 | 하교 | 클린 RMS 226 | 13/13 정밀 | ✅ **차임 +4세미톤 변종(trip4)** |
| 20260601_0654_등교 | 등교 | 클린 | 정밀 | ✅ |
| 20260601_1431_하교 | 하교 | 클린 | 정밀 | ✅ |
| 20260602_0653_등교 | 등교 | 클린 | 정밀 | ✅ |
| 20260602_1259_하교 | 하교 | 클린 | 정밀 | ✅ |

> 마크는 친구가 달리는 차에서 탭해 ±수초~수십초 부정확 → `path2_recheck.py`로 청취하며 정밀화 완료(자동화 4종 실패 확인). **각 트립의 탑승역 마크(등교 구로/하교 성대)는 가짜**(미녹음, 임의 탭) → drop. 종착역(등교 성대/하교 구로)은 실재. 트립이 곧 채널 1개라, **4→8트립으로 늘려도 cross-trip 역 이름 분류 정확도는 안 올랐음** → 분류 폐기, 차임+가속도계 카운팅으로 전환.

**평가 방식 — 트립단위 LOO (정직)**
- 8트립 중 1개를 held-out(미학습=새 채널), 나머지로 학습 → held-out 채점. fold 평균. 시드 고정으로 run간 요동 제거.
- held-in val(같은 분포)은 높게 나오지만 **판단 기준은 held-out**(새 트립). 차임 검출도 동일 LOO로 채점(94%/10.3fp = trip4 변종 제외 7트립).
- 차임 검출 학습 데이터 = 실 트립 닫힘차임 윈도우(HPSS 하모닉) + 노이즈 negative. CMN 적용.

**라이브 추론 계획 — 등교=노트북 모니터, 하교=온보드 LCD**

리스크를 등교 트립에서 0으로 두기 위해 **추론 위치를 트립별로 분리** (차임+가속도계 카운팅 기준):

- **등교 (오전, 저위험)**: **bringup 펌웨어 + ADXL 읽기 추가**(PCM + accel을 USB-UART 스트리밍) → 노트북이 수신, **노트북에서 실시간 카운팅**(가속도 정차 ∧ 차임 → 지난역/현재역) + **raw PCM·accel 동시 저장**(검증/튜닝용). 추론을 노트북에 두어 펌웨어 통합 리스크 최소화.
- **학교에서 디스플레이 인수** → 온보드 통합 펌웨어 빌드/플래시.
- **하교 (오후, 시연영상)**: **온보드 통합 펌웨어**(bringup I2S 마이크 DMA + 차임 검출(+CMN) + ADXL 정차 검출 + LCD)로 보드 단독 동작 촬영. 동시에 PCM·accel 백업 저장.

> 등교에서 **노트북측 추론**을 쓰는 이유: 같은 UX를 펌웨어 최소변경으로 얻어 트립 날릴 위험 제거. offline post-edit(트립 audio+accel을 카운팅 파이프라인에 흘려 결정 추출→영상 오버레이)도 백업으로 항상 가능.

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

**Path 2 (라이브) — 차임+가속도계 역 카운팅** (2026-06-04) — ⚠️ **2026-06-18 폐기됨.** 가속도계 빼고 **차임-only 검출 + 디스플레이**로 단순화(위 "Path 2 온디바이스 차임 데모" 참조). 아래는 가속도계 시절 기록(보존). 차임 검출·CMN·탑승역앵커는 그대로 재활용.

음향 분류기(KWS 트리거 + 13-class/ProtoNet 인코더)는 **폐기** — cross-trip 채널 천장(~42%)·KWS false(56/trip)을 못 넘음(시도 기록은 아래 "음향 분류기 실험" + [PATH2_RESULTS.md](PATH2_RESULTS.md)). 역 *이름*을 맞히는 대신 역을 **카운트**한다.

- **시작 앵커**: 탑승역 + 방향(캡처 UI 기존 기능). 시연 트립은 **완행**(모든 역 정차 → 정차수=역수 1:1), 노선은 한 방향 단조 이동.
- **가속도계(ADXL) = 주(主) 카운트**: 열차 모션(감속→정차 dwell→가속)으로 정차 이벤트 검출. moving↔stopped 상태머신, 정지 시 노이즈 플로어로 자동 캘리브(**dwell 시간 하드코딩 금지**, 작업규칙 #2). 모션은 물리라 **채널 불변** → 그동안의 채널 천장을 통째로 우회. **accel은 학습 대상 아님**(임계 기반 신호처리).
- **닫힘 차임(삐리리리) = 확인(보)**: HPSS 하모닉 톤 격리, win=3.0s, **7트립 LOO recall 94% / false 10.3/trip**(trip4 +4세미톤 변종 제외, 상세 PATH2_RESULTS §12-E). 차임은 **하드 AND 아님**(94%라도 AND는 under-count→cascade) — confidence를 올리고 신호대기 정차를 배제하는 용도. 차임 false 10.3은 mid-segment라 accel 정차 게이트가 흡수. 차임 없는 정차는 노선 prior(남은 역 수)로 보정. **trip4 변종(차임 검출 붕괴)은 accel이 음색 무관하게 정차를 잡아 구원.**
- **출력**: 카운트 + 노선 순서 → 지난역/현재역 (등교=노트북 모니터 / 하교=온보드 LCD).
- **CMN**: 차임 검출 melspec에도 동일 적용 필수(채널 절대레벨 mismatch 제거 — 클린 모델이 라이브 0 검출이던 원인).

### Path 2 음향 분류기 실험 — 폐기 (요약)

라이브 cross-trip 역 *이름* 분류를 여러 방법으로 시도했으나 **사용가능(90%)에 못 닿아 폐기**, 차임+가속도계 카운팅으로 전환. 전체 표·진단·재현 스크립트는 [PATH2_RESULTS.md](PATH2_RESULTS.md)(§1~§12)에 보존.

- 시도한 것(전부 천장): 13-class softmax(19%), ProtoNet 인코더(~42%), 채널적대 GRL(44%), 시퀀스 prior(75%, 단 *검출 완벽* 가정), KWS 트리거(false 56/trip), 그리고 false 억제용 PCEN·실 RIR·hard-neg mining·매치드필터·고정구문 prototype 게이트 — **모두 패배**.
- **근본 병목 = 실 채널(트립) 수.** 안내방송은 코레일 단일녹음의 결정론적 재생(고정신호)이라 변하는 건 노이즈(합성 가능)와 **채널**(차량 PA·객차 음향·마이크 위치 = 트립 수만큼만). 트립 4→8로 늘려도 모델 천장 안 올랐고, KWS false는 "음성 vs 음성"이라 단독 억제 불가. → 역 이름 인식 자체를 접고 **역 카운팅**으로 선회.
- **살아남아 재활용된 것**: ① **CMN**(없으면 라이브 0 검출), ② **차임 HPSS 검출**(94%/10.3fp, 카운팅의 확인 신호), ③ **탑승역 앵커 + 노선 단조성**(카운팅 골격), ④ **자동 마크 위치추적 불가** 교훈(정밀 마킹은 사람 청취 — NCC/DTW 4종 실패).
- 배포: 인코더 폐기로 `encoder.tflite` 646KB(>F411 512KB Flash) 블로커 소멸. 온디바이스는 차임 검출(가벼움) + accel 임계 로직뿐.

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
