# Path 2 — 실차 역분류 (현재: 실시간 KWS+차임+인코더 융합)

> 이전 **4트립 시대 실험(§1–10)은 `archive.md`로 분리.** 이 문서 = **8트립(§11) + 현재 풀파이프 작업(§12)**.
> 작업 규칙은 `CLAUDE.md` 상단(특히 규칙6: 모든 실험 **실차 실시간 전제**, GT-위치 깔기·ceiling 금지).

## 현재 상태 (2026-06-04)
- **데이터**: 8트립(4등교+4하교) `data/raw/line1_live/<trip_id>/{audio.wav,marks.json,door_events.json}`. trip4(20260528_2118_하교)=차임 **+4세미톤 변종** → 차임 실험서 제외(7트립).
- **딜리버러블**: `path2_fullpipe/` — Colab 노트북(7섹션) + `scripts/`(+ 신규 `path2_pipeline.py`) + `door_events/` + `heynow_path2_data_8trip.zip`(457MB) + `build_data_zip.py` + `_diag_*.py`.
- **확정 설계(실시간)**: KWS("이번역은") + 차임(HPSS 톤분리) + 인코더(GRL) 각자 검출 → **융합 DP**(`fuse_decode`/`run_fusion_loo`: 인코더 emission + 차임 보너스 + **시간간격 prior**(이웃 ~165s) + **단조**(지나온 역 배제))로 위치. 순수 카운팅은 cascade로 폐기.
- **측정 완료**: 차임 **HPSS 94%/10fp**(좌석 채널변이 해결, §12-E/F). 타이밍(§12-I): close→close ~165s, announce→close 역별 ±8s(자동방송=역별 고정), 급행=하교 당정구간 gap 폭발로 포착. **KWS hard-neg mining = 실패**(§12-M: recall 75→58%, false만 빠짐), **고정구문 prototype 게이트 = 실제 false엔 사망**(§12-N: real-FP AUC 0.45<chance, front-end 무효).
- **KWS 점검 툴**: `scripts/path2_kws_inspect.py`(§12-O) — LOO held-out 검출을 트립별 캐시(`reports/kws_inspect/`) + read-only GUI(마크·TP/FP·확률곡선·CMN before/after 스펙트로그램). 런처 `Path2 KWS Inspect.lnk`+`kws_inspect.ico`.
- **미측정(Colab GPU 대기)**: ① 노트북 §3 융합 검출-포함 역정확도(최종 "쓸 수 있나"), ② §4 인코더 7트립 per-mark(4트립 33%/44% 넘나).

## 다음 세션 시작점
1. **핵심 병목 = KWS false ~56/trip 여전.** 이번 세션에 단독 false-억제 카드 3장 다 소진: 매치드필터(§12-L)·hard-neg mining(§12-M)·고정구문 prototype 게이트(§12-N) **전부 패배**. 공통 원인 = false가 "음성 vs 음성"(잡담·"다음역은"·KTX)이라 신호레벨 분리 불가, cross-trip 채널 변이.
2. **남은 정직한 카드 = (a) prototype 정렬탐색**(§12-N, trigger ±2s 최적창 재점수 → 정렬 아티팩트 제거 후 사활 확정, 로컬 ≤5분), **(b) 교차게이트**(§11-C/§12-H, KWS∧차임 AND → 독립신호 곱으로 false 붕괴 — 이미 파이프라인에 `and_both`). false 단독억제가 아니라 **융합/시퀀스로 흡수**가 본류.
3. **Colab 측정**: §4(인코더 per-mark)→§5(학습곡선)→§3(융합, 무거움). 융합 노브(LAM_TIME/LAM_CHIME) 튜닝.
4. 근본 레버 = **트립(채널) 더 수집**. 배포(나중): 인코더 646KB>F411 512KB → 축소(Path-2 데모는 노트북 추론이라 비차단).

---

## 11. 8트립 재측정 — 이번역은 + 열림 + 닫힘 (2026-06-03 세션)

**맥락**: 데이터 4 → **8트립**(4 등교 + 4 하교) 도착. 사용자가 **타이밍(cadence) 기준은 폐기**(급행 정차로 운행시간 불안정) → **세 신호 = "이번역은" KWS + 문열림 + 문닫힘**으로 확정하고 8트립 학습 지시.

**8트립 목록** (`A_train/audio (N).wav` ↔ `marks (N).json` ↔ `audio (N).door_events.json`, N=1..8; 모두 13마크·페어링·이벤트 정상):

| N | trip_id | 방향 | dur | door open/close |
|---|---|---|---|---|
| 1 | 20260527_0654_등교 | 등교 | 2146s | 11/12 |
| 2 | 20260527_1431_하교 | 하교 | 2264s | 10/10 |
| 3 | 20260528_0642_등교 | 등교 | 2318s | 11/12 |
| 4 | 20260528_2118_하교 | 하교 | 1991s | 10/10 |
| 5 | 20260601_0654_등교 | 등교 | 2362s | 11/11 |
| 6 | 20260601_1431_하교 | 하교 | 2296s | 11/11 |
| 7 | 20260602_0653_등교 | 등교 | 2157s | 11/12 |
| 8 | 20260602_1259_하교 | 하교 | 2270s | 11/11 |

> **데이터 배치**: 새 4트립(5–8)을 `data/raw/line1_live/<trip_id>/`에 `audio.wav`+`marks.json`+`door_events.json`로 동기화 완료(`A_train`에서 복사, 오디오 길이 일치 검증). KWS는 이 live 디렉토리에서, door PoC는 `A_train/audio (N)`에서 읽음.
> **스크립트 변경**: `path2_kws_recover.py` `TRIPS`= live 디렉토리 glob(8트립 자동 인식). `path2_door_poc.py` `TRIPS=[1..8]`, 스테일한 `VARIANT_TRIP` 플래그 제거, false/trip 분모 = `len(TRIPS)`.

### 11-A. KWS "이번역은" — 8트립 LOO (`reports/path2_align/kws_8trip.log`)
fold별 val 95–97%. 슬라이딩 검출 임계 스윕(MATCH_S=10s, COOLDOWN=20s):

| MINRUN | TRIG | recall | false(8트립 합) | false/트립 |
|---|---|---|---|---|
| 1 | 0.5 | **83%** (80/96) | 447 | ~56 |
| 1 | 0.6 | 80% | 406 | ~51 |
| 2 | 0.6 | 71% | 287 | ~36 |
| 3 | 0.6 | 64% | 182 | ~23 |
| 3 | 0.7 | 65% (62/96) | 145 | ~18 |

- **recall 상승 확인**: 4트립 48–79% → 8트립 65–83%. 채널 다양성 효과가 정확히 여기 나타남.
- **false trigger 여전히 큼**(고recall서 ~56/트립). §7 한계(잡담·문소리·KTX 통과안내 오발화)가 8트립으로도 안 풀림 → KWS 단독은 "역마다 1번" 정밀 카운트 불가. 하이브리드에서 high-recall 후보 생성기 역할로는 OK.

### 11-B. 문 열림/닫힘 — 8트립 LOO (`reports/path2_align/door_8trip.log`, **win=1.0s**, pitch_aug=0)
3-class(주행/정차/출발) 검출. **열림 = 정차 정적상태**(open 후 dwell), **닫힘 = 출발 삐리리 차임**(close).

| 신호 | recall | false/트립 | 비고 |
|---|---|---|---|
| **닫힘(출발 차임)** | 81% (72/89) | **15.4** | false 적음. 단 trip4(2118하교) **2/10 붕괴** = 차임 음색 변종 차량(§10 관찰 8트립서도 지속) |
| **열림(정차 상태 런)** | **92%** (82/89) | 34.2 | recall 최고. 헛정차 많음(trip5 f58, trip6 f62) |
| 융합(OR, naive) | 92% | 35.0 | 상보성 못 살린 단순 OR |

per-trip 닫힘 recall: t1 10/12, t2 10/10, t3 11/12, **t4 2/10**, t5 10/11, t6 8/11, t7 11/12, t8 10/11.

- **닫힘차임 큰 폭 개선**: 4트립 1초윈도우 39% → 8트립 1초윈도우 **81%**. 채널 늘리니 1초 차임만으로도 잘 됨.
- ✅ **win=3.0s 재측정 완료** (`door_8trip_win3.log`, 2026-06-03): 닫힘 **83%/13.1fp**(1.0s 81%/15.4 대비 recall↑·false↓), 정차상태 92%/**26.4fp**(34.2→26.4). 3.0s가 전반적으로 살짝 낫다 → door 기본 win 후보. 단 trip4 변종은 3.0s서도 붕괴(출발 1/10), 정차상태도 trip4 3/10으로 1.0s(7/10)보다 나빠짐(변종은 결국 실 트립 필요).
- **trip4 변종 차임 문제 지속**: §10 결론("변종은 state로 회복, 실 트립이 학습에 있어야") 재확인.
- ⚠️ **open(열림)은 독립 이벤트로 미측정**: 3-class는 close(출발)만 이벤트, open은 정차상태 구간의 시작으로만 암묵 처리. 게이트 "역 성립 조건"에 open을 넣을지 정하려면 open-이벤트 검출 실험 필요(`_diag_openclose.py`).

### 11-C. 종합 진단 — "기준 추가"가 아니라 "교차 게이팅"
세 신호 단독 cross-trip recall/false: 이번역은 83%/56 · 닫힘 81%/15 · 열림 92%/34. **recall은 다 쓸만한데 false가 전부 과다** → 카운팅 cascade(검출 1개 어긋나면 위치 전체 밀림)로 단독 위치추정 불가.

**미검증 핵심 가설(다음 실험)**:
> 진짜 역 = **"이번역은" 안내 ∧ 문 닫힘 차임이 수 초 내 동시 발생.** KWS 오발화(잡담)와 헛차임은 **서로 무상관** → AND 게이트로 false가 곱으로 깎이고(56→소수), recall은 교집합(~0.8×0.8). 거기에 **시퀀스 prior**(노선 단조성, §3에서 검증된 +75% 레버) + **탑승역 앵커**로 빠진 것 보간 → §9 하이브리드를 데이터로 실증.

### 11-D. 다음 세션 — 미결정 (사용자에게 물어보던 중 중단)
다음 실험 옵션(택1):
1. **(추천) KWS∧닫힘 교차게이트 + 시퀀스** — 이번역은과 닫힘이 ±Δ 동시일 때만 '역' 카운트 → false 곱 제거. 거기에 Viterbi(노선 단조) 얹어 **cross-trip 역 정확도 LOO** 측정. "쓸 수 있나"를 가장 직접 답함. 새 융합 스크립트 + 8-fold(~40분 compute).
2. **열림(정차상태) false 튜닝 먼저** — recall 92%로 제일 높으니 MIN_STOP_S·merge 파라미터로 false(34/트립) 줄여 순수 카운터화. 재학습 불필요, 후처리만.
3. **분류기(역이름)까지 합친 풀 파이프라인** — KWS+door 게이트로 '역 발생' 잡고, 그 윈도우를 metric 분류기에 넣어 '어느 역'까지 end-to-end LOO. 가장 완전하지만 무거움(KWS+door+encoder 3개 학습).

> 추가로 빠르게 해볼 것: **`DOOR_WIN_S=3.0`으로 door_poc 재실행**(11-B 캐비엇), 8-fold라 수치 안정성↑.

**재현 (8트립, 시드 고정)**:
```bash
python scripts/path2_kws_recover.py        # 이번역은 8-fold LOO + 임계 스윕 → kws_8trip.log
python scripts/path2_door_poc.py           # 문 열림/닫힘 8-fold LOO → door_8trip.log
DOOR_WIN_S=3.0 python scripts/path2_door_poc.py   # 닫힘 차임 3초윈도우 재측정(미실행)
```

## 12. 풀 파이프라인 패키지 — 옵션3 결정 + 게이트 타이밍 실측 (2026-06-03 세션)

**결정**: §11-D에서 사용자가 **옵션 3(풀 파이프라인)**을 선택, 무거우니 **Colab 노트북**으로 패키징 요청. → 자립 폴더 `path2_fullpipe/` 생성:
- `path2_fullpipe.ipynb` — 8트립 LOO end-to-end (KWS·door·GRL인코더 3개 학습 → 교차게이트 → 분류 → anchored Viterbi → cross-trip 역정확도 + fusion-mode 비교표). 마지막에 encoder/kws/door INT8 tflite + prototype + meta(노선·앵커·게이트·Viterbi 파라미터) 내보내기.
- `scripts/path2_pipeline.py` (신규, 핵심: 교차게이트 + emission + anchored Viterbi + e2e 스코어링) + import 모듈 5개 사본.
- `door_events/`(8트립 json 백업), `build_data_zip.py`(8트립 zip 생성기 — **committed zip은 4트립·door없는 stale**), `_gen_notebook.py`/`_diag_*.py`(재현·게이트 튜닝), `README.md`.
- **데이터 정리**: 8트립 audio가 `A_train/audio (N)` ↔ `line1_live/<trip_id>` **md5 동일** 확인 → trips 1–4의 door_events를 live dir로 동기화(이제 8트립 모두 trip_id로 audio+marks+door_events 통일).

**게이트 타이밍 실측 (`_diag_timing`/`_diag_gate`, fold0, 18-epoch 약체 모델)** — §11-C 가설 **수정**:
- ⚠️ "이번역은 ∧ 닫힘이 **수 초 내 동시**"는 물리적으로 틀림. **방송은 정차 ~40–90s 전(주행 중) 송출** → 트리거→닫힘 간격 **median 66s(47–99s)**. 닫힘은 +20~115s 윈도우로 페어링해야 함(초기 MAXDWELL=50s 버그로 and_close=0매칭이었음).
- 진짜 역 판별의 핵심 = **방송 직후 열차가 실제로 정차**: 정차상태 구간 **start**가 [t−12,+45]s. 헛정차/과병합(100s+) 구간은 길이≤70s로 컷.

**교정 게이트 결과 (fold0, win=1.0, MOVING_THR=0.4·MIN_STOP_S=6.0)**:

| 게이트 | events | recall | false |
|---|---|---|---|
| kws (융합 없음) | 40 | 12/12 | 28 |
| and_close [+20,+115] | 28 | 11/12 | 17 |
| and_state (정차 start) | 24 | 11/12 | 13 |
| **and_both (둘 다)** | 22 | 11/12 | **11** |

→ recall 유지하며 **false 28→11(절반↓)**. §11-C 방향 입증(단, 약체 모델·1 fold). **anchored Viterbi에 stay(k=0) 전이 추가** → 게이트 false 이벤트가 경로를 밀지 않고 흡수(단위테스트: false중복→[1,1,2,3], skip→[1,3]). 

**남은 일 (Colab GPU)**: 노트북에서 인코더 2000 episode로 LOO 돌려 **실제 cross-trip 역정확도** 확정(로컬은 인코더 미학습이라 정확도 무의미, 게이트/디코드 로직만 검증됨).

### 12-A. 게이트 설계 전환 — 타이밍 윈도우 철회 → 이벤트-순서 상태기계 (사용자 지시)
**사용자 피드백(중요)**: 위 "타이밍 게이트"(트리거→닫힘 +20~115s, 정차 start [t−12,+45]s 등)는 **정차시간이 급행·지연으로 변동**하므로 **금지**. §11에서 폐기한 cadence 기준을 되살린 셈. + Claude가 임의로 용어/설계("페어링 윈도우","교정 게이트")를 정의한 것도 지적 → **CLAUDE.md 상단에 작업 규칙 추가**(①임의 결정 금지·먼저 질문 ②타이밍 하드코딩 금지 ③모르면 질문).

**새 게이트 = 이벤트 순서 상태기계 (시간 길이/간격 미사용, 순서만)**:
```
"이번역은 K역" ─ open ─ close ─→ (주행중, K→다음) ─ "이번역은 L역" ─ open ─ close ─→ …
   현재역=K        정차      close가 상태를 "K 정차"→"주행중"으로 전환
```
- 방송 = 현재 역 이름. open~close = 그 역 정차. **close = 출발 전환점.** 헛검출 방송은 "주행중" 구간에 떨어져 걸러짐.
- 누락(close/방송 빠짐) 처리 = **노선순서+탑승역 앵커 Viterbi가 보정**(§3, 사용자 확정).
- **미결(실험 대기)**: 역 성립에 open까지 필요한지 = open이 close만큼 검출되는지에 달림.
  - 사용자 음향 관찰: **open="탁" 한 번 → 1.0s 윈도우면 충분**, **close="삐리리 차임+탁" → 3.0s 필요**. CNN은 입력 길이 고정이라 한 모델로 둘을 못 함 → **이벤트별 독립 이진 검출기**(open@1.0s, close@3.0s). 이게 상태기계(방송→open→close)와 직접 맞물림.
  - `path2_pipeline.run_openclose_loo()`(= open@1s·close@3s 각각 8-fold LOO), 노트북 §3 셀. **8-fold×2라 로컬>10분 → Colab GPU에서 실행**(CLAUDE.md 규칙4 신설: 로컬 ~10분↑이면 Colab Pro로). 결과 보고 사용자가 'open+close' vs 'close 필수+open 선택', door 검출 구조(이벤트 검출기 2개만 vs 정차상태 백업 유지) 결정.
- ⚠️ `path2_pipeline.py`의 현재 게이트(timing 기반)는 **재작성 예정**(아직 미반영). 위 결정 후 상태기계로 교체.

### 12-B. door win 재측정 (§11-B 해소)
`door_8trip_win3.log`: 닫힘 **83%/13.1fp**, 정차상태 92%/**26.4fp** — win=3.0이 1.0보다 false↓로 살짝 나음(상세 §11-B). open 독립 검출은 12-A 실험에서 측정.

### 12-C. Colab 실측 결과 + 채널 변이 정면화 → chime/clunk 분리 (2026-06-03, Colab GPU)
**§3 open/close 독립 검출 (open@1s, close@3s, 8-fold LOO):**
- open: **71% / false 43.2/trip** → 열차소음과 유사, false 감당 불가 → **게이트 필수조건 불가**(open 제외 확정).
- close: **71% / 26.5/trip** 인데 **bimodal**: 등교 거의 완벽(0654 12/12, 0642 12/12, 0653 11/12…) / **하교 붕괴**(1431 3/10·2118 0/10·1431 4/11, val 38~60% = degenerate training). = 변종 차임 + 학습 불안정.

**§5 end-to-end LOO (옛 타이밍 게이트 placeholder, 인코더 2000ep):** 역정확도 per-mark 0~4/12, Viterbi 0~6/12 — 대체로 Viterbi가 per-mark를 **떨어뜨림**. 원인: 게이트 false가 많으면(36/trip) **앵커가 첫 false 이벤트에 꽂혀** 전체 디코드 붕괴. 단 false 적은 fold(0642, close 12/12)에선 Viterbi 3→**6**으로 상승 → **메커니즘은 살아있고 레버 = 게이트 false↓**(§11-C 재확인). ⚠️ 이 게이트는 폐기 대상이라 최종 성적 아님.

**사용자 핵심 진단**: 근본 병목 = **채널마다 close/open/KWS/역이름 인식 변이가 큼**. open=열차소음과 유사(false), close 차임도 차량별 음색 변종("사람 귀엔 같은데 모델엔 다름"). → 이게 프로젝트 내내의 cross-channel 병목(채널 수 적음).

**피벗 (사용자 아이디어, 정밀 검증 결정)**: 닫힘을 **차임(삐리리)** 과 **탁(문닫힘 충격음)** 으로 **분리 검출**, **순서상 연속(차임→탁)** 일 때만 '역바뀜'. 가설: **탁(broadband 충격)이 차임(톤)보다 채널-강건** → 변종 견딤. 시간 길이/간격 미사용(순서만).
- **구현**: `path2_event_mark.py`에 **'탁'(clunk, 키 `t`, 초록)** 마킹 추가. `build_data_zip.py`는 A_train→live door_events 자동 동기화(md5). `path2_pipeline.py`에 `load_event_marks`·`build_event_pool`(event∈open/close/chime/clunk)·`chime_clunk_sequence`(ordinal)·`run_chime_clunk_loo` 추가. 노트북 **§4** 셀.
- **대기**: 사용자가 8트립 '탁'을 GUI로 마킹 → `build_data_zip.py` 재실행 → zip 재업로드 → §4 실행. 결과(탁 단독 false↓? 연속 규칙 false↓?)로 door 카운터 구조 확정.
- **솔직한 천장**: 8채널로 완벽 cross-channel은 어려움; §9 하이브리드(앵커+노선순서+분류기 교차검증)가 불완전 검출 흡수 전제. 정확도는 채널 수와 함께 오름.

### 12-D. 탁 드롭 → 3종 확정 + 차임 HPSS 톤 분리 (2026-06-03)
**사용자 추가 관찰**: 탁(치이익 슬라이드음 포함)은 **좌석 위치 의존** — 문 근처 좌석엔 차임+치이익, 가운데엔 차임+탁(치이익 없음). **탁은 있는 트립/없는 트립** → 신뢰 카운터 불가. **최종 결론(사용자) = 차임 + 이번역(KWS) + 인코더 3종.** (탁·open 드롭, §12-A/C의 clunk 실험 코드는 보존하되 비주력.)

**남은 문제 = 차임 오염의 채널 변이**: 같은 차임도 좌석따라 치이익이 섞이고 안 섞임 → 검출기가 채널 변이로 인식. 해법: **차임 톤만 분리.** 차임=지속 톤(harmonic), 치이익=광대역, 탁=충격(percussive).
- **기법 = HPSS(harmonic-percussive separation) + margin.** `librosa.decompose.hpss`를 mel power에 적용, harmonic만 사용, margin>1로 광대역 치이익을 residual로 버림. → `path2_dataset.FEATURE_MODE='logmel_cmn_harmonic'`(HPSS_MARGIN=3, kernel=(31,17)). 학습·추론 둘 다 `D.to_logmel` 한 함수 → mismatch 0.
- **추론 비용 검토(사용자 우려)**: mel-domain median 2개 = µs. Path-2는 **노트북 추론**(CLAUDE.md)이라 비용 무시. 한 정거장 이동시간(60~119s) 대비 헤드룸 압도적. F411 이식 시만 time-median lookahead 버퍼 필요(비차단).
- **구현/검증**: `path2_pipeline.run_chime_loo(mode)`·`run_chime_compare()`(베이스라인 vs HPSS, per-window `to_logmel` 슬라이드로 추론도 HPSS 동일 적용). 노트북 **§4** = `PL.run_chime_compare()`. HPSS 피처 shape=베이스라인 동일(모델 불변) 확인, 2트립 경로 스모크 통과. **8-fold HPSS vs 베이스라인은 Colab GPU에서 측정 예정**(하교 붕괴 회복 여부가 판정).
- 사용자가 차임 마크 정밀화 중. 마킹 후 `build_data_zip.py`(A_train→live 자동 동기화) 재실행 → 재업로드 → §4.

**trip4 차임 변종 분석 (`_diag_chime_variant.py`, `chime_variant.png`)**: 마킹 완료 후 사용자가 trip4(2118 하교)만 차임이 다르다(고주파) 보고 → HPSS로 톤 격리해 트립별 차임 톤 정량:
- 6/8 트립 차임 기본음 **~531Hz** 공유(N7 `[531,734,984,1625]` 등). **trip4만 531 없고 664/812/953/1172Hz = 기본음 531→664 (+4세미톤 ↑)**. 사용자 청취("고주파")와 일치. 유일 outlier(N2 하교도 531 보유).
- **결정**: 차임 검출 실험에서 **trip4 제외(7트립)**, KWS·인코더엔 유지(변종은 차임뿐). 노트북 §4 `CHIME_TRIPS`에서 `2118_하교` 필터. 단일 +4st 변종이라 그냥 넣으면 두 피치 학습으로 흔들림(§10 pitch-aug 패배 전례).
- **통합 옵션(미결, 보류)**: +4st가 깨끗한 시프트라 **타깃 pitch-aug**(나머지 트립 차임을 +3~+5st 올려 학습)로 재포함 시도 가능하나 §10 리스크(false 폭증) → 7트립 baseline 먼저 보고 결정.
- **차임 윈도우 1.5→3.0s** (사용자 청취: 차임 3~4s 유지): 검출기가 전체 멜로디를 봐야 더 특징적·false↓. `EVENT_WIN['chime']=3.0`(§2 노브). HPSS가 늘어난 윈도우의 치이익/탁을 어차피 제거.

### 12-E. ✅ HPSS 차임 검출 — Colab 7트립 LOO 결과 (win=3.0s, trip4 제외)
| feature | recall | false/trip | 학습 안정성 |
|---|---|---|---|
| logmel_cmn (베이스라인) | 58/79 **73%** | 16.6 | **붕괴**: val 45·57·64%, 0642 **0/12**, 1431 4/11 |
| **logmel_cmn_harmonic (HPSS)** | 74/79 **94%** | **10.3** | **전부 안정** val 98~100% |

- **클린 승리**: recall +21점, false −38%, **학습 붕괴 제거**(0642 0→12/12, 1431 4→11/11). 좌석-의존 치이익(광대역)이 학습 불안정·채널변이의 주범이었고 **HPSS 톤 격리가 그걸 제거** → 사용자 가설(좌석 오염=채널변이) 데이터 입증. 추론 비용 µs(노트북). 
- **결론**: 차임이 신뢰할 cross-trip 카운터(94%/10.3fp, 7트립). 3종(차임+이번역+인코더) 중 door-side 카운터 확정.
- **남은 false 10.3/trip**: 카운터엔 과검출이라 → 이벤트-순서 상태기계 + 노선순서 prior + 탑승역 앵커가 흡수해야(다음). trip4(+4st 변종)는 HPSS로도 피치차는 안 메워지니 별도(pitch-aug 옵션).

### 12-F. 이벤트-순서 상태기계 end-to-end 구현 (사용자 결정)
3종(차임+이번역+인코더) 합친 최종 파이프라인 구현(`path2_pipeline.run_statemachine_loo`):
- **확정 안내** = 다음 KWS 안내 전에 차임(출발)이 따라오는 KWS 트리거(`confirm_announcements`, **순서만·시간 미사용**) → 주행 중 잡담 오발화 제거.
- 확정 안내들의 인코더 emission → **anchored Viterbi**(노선 단조 + 탑승역 앵커)로 역 이름.
- fold 안에서 **신호별 FEATURE_MODE 전환**: KWS·인코더 = logmel_cmn, 차임 = logmel_cmn_harmonic(HPSS). (인코더 emission이 FEATURE_MODE 의존이라 정확히 분리 필요.)
- (이 블록의 GT-위치 enc-Viterbi 실험은 §12-G에서 폐기됨 — 실시간 전제 위반.)

**검출-포함(실시간) 디코더로 일원화**: 카운팅 단독은 cascade로 죽음(false 차임 1개에 뒤 다 밀림). → **융합 DP**(KWS후보 중 노선순서대로 DP 선택, 인코더 emission+차임 보너스+시간간격 prior+단조)로 실시간 위치추정. (§12-J)

### 12-G. (삭제) "안내 GT 위치 깔기" 비실시간 실험 폐기 — 사용자 지시
GT 안내 위치를 깔고 분류→노선순서로 박던 실험(소위 96% 같은 ceiling)은 **실차 실시간 추론과 무관**하여 **기록에서 폐기**(CLAUDE.md 규칙6). 그 수치는 인코더가 아니라 "12개 연속 순서 prior"가 만든 것이라 무의미. **모든 실험은 검출(KWS·차임)부터 스스로 하는 실시간 전제로만 진행.** `run_statemachine_loo`(GT 위치 사용분)·`chime_count_route`는 코드에서 제거.

### 12-H. 융합 디코더 구현 (사용자 설계 — 흐름 전체 최적)
사용자 설계: "이번 역만 보지 말고 history 전체를 보고 결정." → `fuse_decode` + `run_fusion_loo`(노트북 §6):
- **DP(Viterbi)**: KWS 후보(시간순, 헛검출 포함 ~68) 중 **노선 순서대로 N개 선택**. 후보 i를 여행순서 p에 놓는 점수 = `log p(역_p | 후보 i)`(인코더 emission, 탑승역 앵커+노선으로 역_p 결정) + `λ·(후보 i 뒤 차임 동반?)`. 안 고른 후보 = KWS 헛검출 자동 폐기. **greedy 아닌 전체최적 → 한 번 틀려도 cascade 없음**.
- 4경우(차임±·KWS±)를 하드 if-else 대신 **소프트 증거로 전체 점수화**(사용자 우려대로 greedy는 cascade). 차임=전진 보너스, 인코더=이름 증거, 노선순서+앵커=골격.
- 단위검증: 헛검출 2개 섞인 5후보에서 진짜 3개 골라 routes [1,2,3] ✓. 
- **이게 검출-포함(실시간) '쓸 수 있나' 숫자.** Colab GPU에서 측정(노트북 §3, 3모델×7fold).
- **융합 DP(run_fusion_loo) 결과 18%** (`fz`): KWS 후보(헛검출 ~56) 중 진짜 12개를 인코더 emission으로 못 고름(per-mark 33% 약함) → 4/12만 매칭. **검출-선택이 진짜 병목** 확인.

### 12-I. 인코더 폐기 → 타이밍-정렬 필터 (사용자 결정, KWS+차임만)
사용자: "인코더 버려도 KWS·차임은 못 포기(음향 TinyML 핵심). false만 거르면 됨. 타이밍 추가." → **타이밍을 측정해 false 필터로.**

**(1) announce→open 변동 원인 규명** (`_diag_announce_open.py`): 전체 std 16s지만 **역내(같은 역 반복) std=6s, 역간 std=14s** → 변동은 랜덤 아니라 **역별 상수**. = 관제 자동방송이 **역마다 고정 위치에서 트리거**(거리/접근 다름). 방향도 영향(가산 등교99/하교51). announce→open vs 직전주행 r=+0.45(속도 2차효과). → **역별 타이밍은 일관(±6s), 모델링 가치 있음.**

**(2) 실전 타이밍 테이블** (`_diag_timing_table.py`, `timing_table.json`, open 실전불가라 announce+close만):
- **Δac(announce→close)**: 역내 std 8s(전체18), med 74 range[46,124] — 역별 일관. KWS↔차임 페어링 윈도우.
- **Δca(close→다음announce)**: 정상 std 1~15s(med 93). ⚠️ **하교에서 급행 skip 데이터로 포착**: 당정→명학 699s(2역 skip), 당정→금정 561s(군포 skip) — gap 폭발+도착역 2~3 앞. → gap÷사이클로 skip 역산 가능.
- 사이클(close→close) ~165s.

**(3) 타이밍-정렬 필터** (`timing_decode`/`run_timing_loo`, 노트북 §3, **인코더 X·KWS+차임만·가벼움**):
1. **announce-backed close만 채택**(직전 [45~130s]에 KWS) → 잡담 KWS·헛차임 동시 제거.
2. **close grid 카운트 + gap÷165로 급행 skip 보정**.
3. 위치 = 탑승역 앵커 + d×(센 수). 노선순서가 이름. 
- 단위검증: 정상[1,2,3]·급행skip[1,4]·헛차임거부[1,2] 통과. Colab §3에서 검출-포함 역정확도 측정 예정.
- **타이밍 사용 옵션 A(global)/B(역별테이블)/C(순서만)/D(페이스보정) 중 B+D 채택**(역별 일관성이 데이터로 정당화 + 페이스배율로 급행/지연 흡수). 현재 구현은 global 윈도우(견고), 역별 테이블은 JSON에 있어 추후 적용.

### 12-J. 순수 카운팅 사망 확인 → 융합 DP + 시간간격 prior (사용자 아이디어 종합)
- **`run_timing_loo`(인코더 X 카운팅) 5/12** — skip 꺼도 동일. 원인: **KWS false 46개가 빽빽해 announce-backed 필터가 무력**(차임마다 [45~130s]전에 KWS 우연히 있음→전부 confirm→over-count→cascade). **순수 카운팅 dead 확정**(KWS false 56 벽).
- **사용자 핵심 아이디어 2개**: ① 지나온 역은 확률에서 제외 ② 인코더는 약하니 후보를 좁혀라. → anchored Viterbi의 **단조(뒤로 못감)=①** 이미 내장. ②는 **타이밍이 후보를 좁히면 약한 인코더(33%@13지선다)도 2~3지선다선 쓸만**.
- **융합 DP에 시간간격 prior 추가**(`fuse_decode` lam_time): 고른 이웃 후보끼리 ~cycle(165s) 간격이도록 transition 페널티 → 너무 붙은 가짜 후보 배제. **카운트가 아니라 *간격 정합*이라 헛검출 1개에 cascade 없음.** + 차임 보너스 + 인코더 emission + 단조. 단위검증: 가산 근처 가짜(120s) 배제하고 ~165 간격 진짜 [1,2,3] 선택 ✓.
- **= KWS+차임+인코더+타이밍+지나온역배제 전부 결합** (`run_fusion_loo`, 노트북 §3). 5/12(카운팅)·18%(인코더선택만)보다 나을 것으로 기대. 단 8채널 한계는 잔존(인코더 약함·검출 시끄러움) → 천장은 트립 수가 올림. Colab GPU 측정 예정.

### 12-K. 실시간 전제 확정 + 비실시간 실험 폐기 + KWS hard-neg + 진단 (사용자 지시)
- **CLAUDE.md 규칙6 신설**: 모든 실험은 실차 실시간(스트리밍) 전제. GT를 예측 입력으로 쓰는 "안내 GT위치 깔기"류 금지, "ceiling" 비실시간 수치 보고 안 함. → §12-G의 96%(GT위치 enc-Viterbi) 폐기, 코드에서 `run_statemachine_loo`·`chime_count_route` 제거.
- **KWS denoising 질문**: HPSS는 차임(톤 vs 광대역)에 통했지만 KWS false는 *음성 vs 음성*("다음역은"·잡담)이라 denoising으로 안 잡힘. rumble은 CMN이 이미 제거 중. → KWS false엔 **hard-negative mining**이 정답.
- **KWS hard-negative mining 구현**(`run_kws_hardneg_loo`/`train_kws_hardneg`/`_mine_kws_fp`, 노트북 §6): 학습 트립에 KWS 돌려 헛발화 구간 채굴 → negative 추가 재학습(rounds회). cross-trip LOO로 mining 전/후 recall·false 비교. (실시간 충실: 학습데이터+GT로 채굴, held-out은 채점만.)
- **진단 추가**: `run_encoder_permark_loo`(노트북 §4, 인코더 단독 per-mark 7트립 — 4트립 33%/44% 대비 개선?), `plot_training_curves`(§5, KWS·차임 학습곡선 따로 = 3모델 독립 확인).
- 노트북 7섹션: §3 융합(최종) + §4~6 진단. 측정 권장순서 §4(가벼움)→§5→§6→§3.

### 12-L. 매치드 필터 "이번역은" 검출 — ❌ KWS에 패배 (2026-06-04, 로컬)
**동기**: "이번역은"은 코레일 단일녹음의 결정론적 재생(고정신호 ×12마크×8트립)이라, 학습된 CNN-KWS 대신 **노이즈 속 알려진 고정신호 검출 = 매치드 필터**(고정신호의 최적 선형 검출기)로 false를 줄일 수 있나 검증. 가설: 템플릿 상관은 "이 *특정* 구절"과의 일치를 직접 재므로 잡담·"다음역은"은 spectro-temporal 패턴이 달라 상관이 낮을 것. (과거 NCC/DTW 4종 실패는 **클린 서울교통공사 템플릿**(다른 녹음, corr 0.25)으로 **변하는 역이름**을 찾은 것 → 여기선 **live 도메인의 고정 prefix**라 다른 문제.)

**구현** `scripts/path2_matchfilter_poc.py` (학습 없음 — 템플릿=마크 정렬평균, 전체 8-fold LOO가 mel 34s+연산 수초. KWS는 폴드마다 CNN 학습). 채점 = `path2_kws_recover.{debounce,score}`(MATCH_S=10·COOLDOWN=20) 그대로 import → **완전 동급 비교**. 진단이 가리킨 개선 다 포함: **노이즈 화이트닝**(1/std per-mel-bin, rumble 지배 제거=GCC-PHAT 아이디어) + **트립별 템플릿 max**(평균 블러 대신 채널 다양성). 템플릿 0.7/1.0/1.4s 스윕. CMN log-mel 코사인(=정규화 NCC), 슬라이딩 0.25s, MINRUN=1(peak가 뾰족해 연속디바운스는 recall 붕괴).

**결과 (최고 변종 maxtrip+whiten, 8트립 LOO)**:
| 동작점 | 매치드 필터 | KWS (kws_8trip.log) |
|---|---|---|
| 고recall | 81%/56fp (1.0s) · 85~88%/60 (0.7s) | **83%/56fp** |
| 저false(~17/trip) | **33%/11** (cos.50) ~34%/17 | **65%/18 · 71%/36** |

**판정 = 패배.** 고recall 잡음점에선 KWS와 동률(~80%/56)이나, **정작 중요한 false 억제 영역에선 KWS 압승**(같은 ~17fp서 65% vs 34%). 핵심 진단: **진짜 마크 peak-cos 0.44~0.47 ≈ 노이즈 p99 0.42~0.45** — 모든 템플릿 길이·변종에서 **진짜/잡음 분포가 겹쳐 분리 gap 0**. 임계 하나로 recall↑·false↓ 동시 불가. 짧은 템플릿은 recall·false 같이 올라 gap 그대로.

**원인 = 또 그 cross-trip 채널 변이.** 7채널 평균 템플릿이 held-out 채널과 ~0.45밖에 안 맞는데, "다음역은"·잡담(같은 목소리, "역은" 공유)이 정확히 같은 0.45로 올라옴. 학습된 CNN이 노이즈증강으로 채널 불변을 *약간 더* 배워 저false서 오히려 나음. → **§12-K 데이터 확정**: KWS false는 "음성 vs 음성"이라 상관/denoising으로 안 잡힘. 매치드필터도 같은 벽.

**결론**: 매치드 필터는 레버 아님(주력 제외, 스크립트는 음성기록으로 보존). false 억제의 옳은 길 = **KWS hard-neg mining**(§12-K). 미검증 잔여 카드 = 오디오 핑거프린팅(정확-녹음 랜드마크 해싱, 노트북측)이나 채널 안정성 의존이라 같은 한계 가능성. 근본 레버는 변함없이 **트립(채널) 수**.

### 12-M. KWS hard-neg mining — ❌ 실패 (2026-06-04, 로컬, `run_kws_hardneg_loo` rounds=1)
§12-K에서 구현한 mining을 8트립 LOO로 측정. 학습 트립서 헛발화 구간(마크 >10s 밖 트리거) 채굴 → negative 추가 재학습 → held-out 채점.

| | recall | false/trip |
|---|---|---|
| baseline | 72/96 **75%** | 38 |
| +hardneg(~270/트립) | 56/96 **58%** | 10 |

**판정 = 사용자 기준("false↓ AND recall 유지") 미통과.** false는 크게 줄지만(38→10) **recall이 같이 무너짐(-17pt).** 트립별로 **이득/손해 양극화**: Pareto 개선 2트립(0654#1 10→12·f36→4, 0653 5→6·f16→3) vs recall 붕괴(0642 8→**3**, 1431#1 9→5, 2118 9→5, 0654#2 12→8). 최다 채굴(348) 트립이 최악 붕괴 → **무제한 채굴이 positive 희석**.
- **방법론 결함**: 단일 operating-point 비교라 "분리도 향상"과 "임계값 보수화"를 못 가름 — neg 추가는 PR곡선을 *아래로 미끄러뜨릴* 뿐일 수 있고(대부분 트립이 그 모양), 정직한 판정은 matched-recall에서 false 비교 필요.
- **원인** = §12-L/N과 동일: KWS가 "이번역은"을 음소로 못 배우고 채널단서에 기댐 → 학습 트립 FP가 진짜 트리거와 특징공간서 겹쳐, 그걸 neg로 박으면 경계가 positive 쪽으로 밀림. cross-trip 비전이.
- 재현: `path2_fullpipe/scripts/path2_pipeline.run_kws_hardneg_loo()`.

### 12-N. 고정구문 prototype 게이트 probe — ❌ 실제 false엔 사망 + front-end 무효 (2026-06-04, 로컬)
**동기**(사용자): "이번 역은"은 코레일 단일녹음의 *고정 파형*(역이름과 달리 불변)이라 "그 고정구문과 닮았나?"(prototype 거리)로 false를 직접 칠 수 있나. + 차임 HPSS처럼 음성용 front-end(밴드패스/프리엠퍼시스)로 KWS를 더 잘 듣게? — **학습 없이 상한(raw logmel_cmn 임베딩 코사인)만** 측정해 사활 판정. `scripts/path2_kws_probe.py`.

**측정 (8-트립 LOO, prototype = 나머지 7트립 "이번역은"[onset,+0.7s] 임베딩 평균, ROC-AUC):**
| negative | meanAUC | 해석 |
|---|---|---|
| 랜덤 pure-noise | **0.731** | chance(0.5) 위 → 고정구문 시그니처는 채널 넘어 *존재* |
| **실제 CNN false-trigger** | **0.451** | **chance 아래** — prototype이 진짜보다 false를 더 높게 점수 |
| front-end(HP@250/preemph) | 0.70/0.71 | **하락**(neg 코사인 0.04→0.14↑) → 무효 |

- **raw 템플릿/레퍼런스 게이트 = 죽음.** 랜덤잡음(0.73)은 낙관적이었고, 실제 false는 유성음(다른 방송·잡담)이라 고정구문과 닮음. recall 지키는 게이트(keep-all-TP)는 false를 35.9→32.0(겨우 11%)밖에 못 깎음. §12-L 매치드필터 패배와 동일 벽.
- **front-end 무효 확정**: CMN이 정적 EQ를 이미 제거 → HP는 positive·negative를 같이 올려 간극 축소. 음성 enhancement는 recall 레버일 뿐 false 못 줄임(혼동원도 음성).
- ⚠️ **잔여 아티팩트**: TP를 trigger 시각(구문 onset 아님)서 점수 → cos TP가 onset정렬 0.17 대비 0.01~0.17로 낮음 → 0.451이 *과소평가*일 수 있음. **사용자 결정 = 정렬탐색**(trigger ±2s 최적 0.7s창 재점수)으로 아티팩트 제거 후 최종 사활 — **대기 중**. 정렬 후도 ~0.5면 진짜 사망, 0.7대 회복이면 학습임베딩(metric)+정렬 게이트 시도(Colab).

### 12-O. KWS 점검 뷰어 `path2_kws_inspect.py` (2026-06-04)
KWS가 한 트립서 *무엇을* 트리거했는지 눈/귀로 검사하는 read-only 툴(recheck/doormark UX). 마크 미수정(뷰어).
- **build**(무거움, tf): held-out LOO — 보는 트립 빼고 7트립 학습 → 그 트립 슬라이딩 검출 → 트리거를 마크에 매칭(TP/FP) + 확률곡선 캐시(`reports/kws_inspect/<trip>.json`). 8트립 캐시됨: recall 10/7/9/8/11/11/5/7(=68/96), FP 43/30/27/38/44/35/17/53(real-FP probe와 일치).
- **view**(가벼움): 파형 위 **모든 마크**(매칭=초록●/놓침FN=주황◌/탑승역=회색·) + **트리거 TP파랑▲·FP빨강▲** + **KWS 확률곡선+임계0.6** + 좌측 시간순 표(`kind=mark`의 hit/MISS/board ↔ `kind=trigger`의 TP/FP는 *같은 매칭의 양면*, hit수==TP수==recall) + 우측 선택윈도우 **log-mel raw|CMN 나란히**(CMN이 고정 채널색 제거하는 걸 직접 비교). 클릭=재생, 휠=줌.
- 실행: `python scripts/path2_kws_inspect.py [--trip <id>]`, 또는 `Path2 KWS Inspect.lnk` 더블클릭(pythonw, 크래시·캐시없음 messagebox). 새 트립은 `--build --trip <id>` 선행.
