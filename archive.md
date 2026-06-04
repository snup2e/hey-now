# Path 2 — 아카이브 (§1–10, 4트립 시대 / 2026-05~06-03)

> 현재 작업은 `PATH2_RESULTS.md`(8트립 §11 + 풀파이프 §12). 이건 그 이전 기록.
> 현재 문서가 참조하는 §3(시퀀스 prior)·§7(KWS복구)·§8(카운팅 cascade)·§9(하이브리드)·§10(도어 state)이 여기 있음.

---

# Path 2 — 실차 역 분류 실험 종합 (중간평가)

> **한 줄 요약**: 안내방송으로 하차역을 맞추는 cross-trip(새 차량) 분류에서, per-window 분류는
> 채널(차량) 4개 한계로 **~33–42%가 천장**. 모델 레버(PCEN·GRL·증강)로는 ~44%까지가 한계지만,
> **노선의 물리적 순서(시퀀스 prior)** 를 후처리로 얹으면 **75%(트립 4개 중 3개 100%)** 로 점프.
> 시퀀스 디코딩은 **모델 비용 0**이라 온보드에 그대로 적용 가능.

## 1. 문제 정의
- **태스크**: 차내 안내방송("이번역은 ○○역") 2초 윈도우 → 13역 중 현재 역.
- **핵심 난점 = cross-trip 일반화**: 학습 안 한 트립(=새 차량 PA·객차 음향·마이크 위치)에서 맞히기.
- 안내방송은 코레일 단일 녹음의 결정론적 재생(고정 신호)이라, 변하는 건 (1) 가산 노이즈(합성으로 무한 생성), (2) **채널 = 실 트립 수만큼만 존재(4개)**.
- **평가 프로토콜**: 트립 단위 leave-one-out(LOO). 4트립 중 1개 held-out(새 채널), 나머지 3 학습 → held-out 채점, 4 fold 평균. **시드 고정**. chance = 8%(1/13).

## 2. 시도한 레버와 결과 (cross-trip LOO)

| 레버 | LOO | 판정 | 핵심 이유 |
|---|---|---|---|
| ProtoNet metric (clean+노이즈 합성, baseline) | 35% | 기준 | — |
| **PCEN front-end** | 10%→33% | ❌ 패배 | per-window CMN이 정적 채널 EQ를 이미 제거 → PCEN 무의미 (스케일 보정해도 baseline 미달) |
| **채널 적대 GRL (trips-only, λ=0.3)** | **44%** | ✅ 최고 모델 레버 | 임베딩에서 트립(채널) 축을 직접 제거. **도메인 헤드는 학습 전용 → 온보드 비용 0** |
| **실 RIR 보간 증강** | — | ❌ 불가 | clean(서울교통공사)≠live(코레일) **다른 녹음**(envelope 상관 0.25, CMN log-mel 상관 −0.09) → deconvolution 입출력쌍 없음. + CMN이 EQ 제거 |
| **episode 600→2000** | 44%→46% | ➖ wash | 천장은 compute 아님(한 fold는 오히려 붕괴) |
| **real-only (clean 제거, jitter만)** | 33% | ➖ | clean 빼니 데이터 부족 |
| **real-only + 실노이즈 증강** | **38%** | ✅ baseline 상회 | clean+노이즈 합성보다 나음. 도메인 깨끗 + 모든 positive pair가 cross-trip |
| real-only + 강증강 + GRL (Colab) | ~42% | — | 데이터 전략 × 최고 레버. 천장(~42–44%) 재확인 |

**모델 레버의 천장 ≈ 42–44%.** 채널 4개로는 물리적으로 여기까지. 추가 정확도는 compute/증강이 아니라 **트립(채널) 수**가 필요.

## 3. 돌파구 — 시퀀스 prior (노선 단조성)
열차는 알려진 1차원 노선을 **한 방향으로 단조 이동**(등교→성균관대 / 하교→구로, 방향은 캡처 UI가 앎). 따라서 연속된 안내방송 = 연속된 역. 이는 **데이터 누수가 아니라 실제 물리 제약**.

인코더의 per-mark cosine을 **emission**, 노선 위상을 **transition**으로 한 Viterbi / 연속-런 디코딩:

| 디코더 | LOO | 0654 / 0642 / 1431 / 2118 |
|---|---|---|
| per-mark argmax (prior 없음) | 33% | 42 / 67 / 25 / 0 |
| Viterbi (단조 transition, α=10) | 71% | 100 / 100 / 83 / 0 |
| **연속-런 (시작 오프셋 탐색)** | **75%** | **100 / 100 / 100 / 0** |

- **트립 4개 중 3개가 100%.** 유일 실패(2118)는 랜덤이 아니라 **"한 칸 밀림"**(종착역 anchor 약화로 전체 런이 1역 시프트) — 트립 더 모으면 사라질 fragility.
- **90%는 트립 몇 개만 더(붕괴 fold 감소) 모으면 도달 가능.**
- 시퀀스 디코딩 = 결정 스트림 후처리 → **추론/Flash 비용 0**, 온보드에 그대로 적용.

## 4. 핵심 발견 (재사용 가능한 교훈)
1. **clean ≠ live**: 서울교통공사 클린과 코레일 실차는 다른 녹음. clean+노이즈 합성은 "녹음 차이"를 억지로 메우게 만들어 저가치. real-only + 실노이즈 증강이 더 나음(38%>35%).
2. **CMN이 정적 EQ를 제거**: 채널의 주파수 응답(EQ)은 CMN이 기계정밀도로 지움 → EQ 기반 증강/PCEN/RIR은 원리상 wash. 살아남는 건 reverb뿐인데 추정 불가.
3. **GRL이 채널 불변을 직접 학습**: 트립을 혼동시키는 적대 학습이 가장 효과적인 모델 레버(+9점), 온보드 비용 0.
4. **시퀀스 prior가 진짜 usability 레버**: 모델 천장(42%)을 후처리로 75%까지. 임베디드에 공짜로 얹힘.

## 5. 배포 현황 & 남은 과제
| 항목 | 상태 |
|---|---|
| KWS 트리거 (kws.tflite 15.7KB) | ✅ 정상 (val ~92–94%) |
| 역 분류 인코더 (encoder.tflite) | 🟥 **646KB — F411 Flash 512KB 초과**. `Flatten→Dense(128)`=61만 weight. 풀링 추가/Dense 축소로 ~100KB대 필요(축소 후 정확도 재측정) |
| 시퀀스 디코드 온보드 통합 | ⬜ 후처리(Viterbi/offset)만 결정 루프에 추가. 모델 변경 없음 |
| abstain 보정 | ⬜ held-out 기반으로 재작성 필요(현재 in-sample 누수로 무효). 단 시퀀스 prior가 들어가면 역할 축소 |

## 6. 재현 (시드 고정, 로컬 CPU)
```bash
python scripts/path2_metric_poc.py                 # metric LOO (clean-synth 35%)
PATH2_USE_CLEAN=0 PROTO_REAL_NOISE_AUG=4 \
  python scripts/path2_metric_poc.py               # real-only + aug 38%  (env: PROTO_USE_CLEAN)
python scripts/path2_grl_poc.py                    # GRL λ 스윕 (trips_only 0.3 = 44%)
python scripts/path2_rir_feasibility.py            # RIR 불가 근거 (CMN-EQ, clean≠live)
python scripts/path2_seqprior_poc.py               # 시퀀스 prior 33→75% (offset/Viterbi)
python scripts/path2_count_poc.py                  # KWS 카운팅 vs 분류기 비교
python scripts/path2_kws_recover.py                # KWS 검출 복구 LOO + 임계 스윕
python scripts/path2_eval_report.py                # 평가 리포트(confusion·확률 히트맵) -> reports/path2_eval/
python scripts/path2_export_clips.py               # 모든 자르기를 청취용 wav로 출력
```
Colab 학습: `notebooks/path2_train.ipynb` (real-only + 강증강 + GRL → INT8 encoder + prototypes + meta).
공유 데이터셋 빌더: `scripts/path2_dataset.py` (`build_metric_pool`에 `use_clean`/`real_noise_aug`/`spec_aug`/`jitter_s`).

## 7. KWS 트리거 (Stage 1) 복구
검출이 0이면 카운팅·분류기 둘 다 무용 → 복구가 선결. 원인 2개:
- **SpecAugment가 positive 파괴**: 1초 윈도우 마스킹이 짧은 "이번역은"을 지워 라벨노이즈화 → 모델 붕괴(val 69%, 상수 0.4 출력, 0검출). → `build_kws(spec_aug=False)`.
- **학습 불안정(LR 1e-3)**: 일부 fold 붕괴(val 58%). → `train(lr=5e-4, epochs↑, patience↑)`.

복구 후 4 fold val 98~99%, 슬라이딩 검출 recall 48~79%(동작점별). 단 **잘 되는 채널(0654·2118)은 11~12/12** 발화(=예전 수준), 평균은 약체 채널(1431 1/12)이 끌어내림. 남은 한계 = **cross-trip 정밀도**(held-out 노이즈 오트리거, 채널 다양성 병목).

## 8. KWS 카운팅 vs 분류기 (방식 비교, LOO 48마크)
| 방식 | LOO |
|---|---|
| KWS 카운팅 — 완벽 검출 | **100%** |
| KWS 카운팅 — 트리거 1개 누락(평균) | 46% (cascade) |
| 메트릭 분류기 — per-mark | 33% |
| 메트릭 분류기 + 시퀀스 prior | 75% |
| **메트릭 분류기 + 시퀀스 + 탑승역 앵커** | **100%** |
| 메트릭 분류기 + 시퀀스, 트리거 1개 누락(평균) | 55% |

검출이 완벽하면 카운팅이 최강(100%)이나 검출오류 1개에 cascade(~46%). 분류기는 이름을 읽어 누락에 강건(55%>46%).

## 9. 최종 아키텍처 & 다음 세션 계획
**최종 = 탑승역 앵커 + KWS 카운팅(위치 추적) + 분류기(이름 교차검증·드리프트 감지) 하이브리드.**
- 깨끗한 검출 + 앵커면 100%, 검출오류는 분류기가 안전장치로 흡수.
- 분류기 보정력은 자기 정확도만큼 → cross-trip 정확도(채널 다양성)가 핵심 레버.

**남은 배포 과제**: ① `encoder.tflite` 646KB > F411 512KB Flash → 인코더 축소, ② 시퀀스 디코드 온보드 통합(후처리, 비용 0), ③ KWS 오트리거 정밀도(채널 더 필요).

**다음 세션 (데이터 10트립 시, 2026-06-05 예정)**:
- 4트립 → **10트립**으로 LOO 재실행 (`path2_seqprior_poc.py`, `path2_kws_recover.py`, `path2_eval_report.py`). 채널 2.5배 → 분류기 per-mark·KWS 정밀도·약체 fold 감소 기대(단 3→4는 wash였으니 상승폭은 측정해봐야 함). 10-fold라 수치도 더 안정적.
- 데이터 zip 재생성(`heynow_path2_data.zip`에 신규 트립 추가) → Colab 재학습.
- 그 위에서 인코더 축소 + 시퀀스 디코드 온보드 통합 진행.

## 10. 도어 이벤트 기반 state 카운팅 (2026-05-30 세션)

**맥락 재정의**: "맨 TinyML이 cross-trip reliable"은 실현가능 데이터 규모에서 안 됨(KWS recall 54~69% + 오트리거 23/trip, 분류기 per-mark 33%, 학습곡선 k=1·2·3 평평). 그래서 TinyML을 *할 수 있는 일*(거친 음향 이벤트)로 재배치 → **여러 cue로 주행/정차 STATE를 정하고 정차 수를 카운트.** 최종(사용자 결정) = **10트립 + 문 여닫힘 + 이번역 KWS 이중 카운팅**, CNN은 10트립에서 의미 있으면 추가, 탑승역 앵커로 시작 고정.

**도구**: `scripts/path2_event_mark.py`(+`event_mark.ico`,`Path2 Door Mark.lnk`) — 문열림/문닫힘 마킹 GUI(연속재생·이번역 참조선·A_train 기본). `scripts/path2_door_poc.py` — 3-class(주행/정차/출발) 검출 + 카운팅 비교 LOO.

**도어 음향(청취)**: 출발 "삐리리리" 차임 = 닫힘에만, 가장 특징적 → 카운트 마커. 단 **차임 피치가 차량모델마다 다름**(트립1-3 동일, 트립4 변종). 열림은 약함(특히 가운데 좌석). 좌석위치(문근처 트립1·2 / 가운데 3·4)가 새 채널축.

**측정 (trip-LOO, 출발 차임 검출 / 정차-상태 런 / 융합)**:

| 레버 | LOO recall | false/trip | 판정 |
|---|---|---|---|
| 1초 윈도우(차임만) | 39% | 12.8 | ❌ 약함(트립2·3·4 붕괴) |
| **3초 윈도우(차임+닫힘 둘 다)** | **64%** | 8.2 | ✅ 둘 다 봐야 함(트립2 10→100%) |
| 3초 + pitch-aug(변종 가짜생성) | 50% | 18 | ❌ 패배(false폭증, 변종 40→10%) |
| **정차-상태 런 카운팅** | **82%** | 20.2 | ✅ 변종·좌석 robust(트립3 33→92%, 트립4 40→90%) — but false 높음 |
| 융합(출발 OR 정차) | 82% | 20.8 | naive 구현, 상보성 못 살림 |

**결론**: ① "차임+닫힘 둘 다" 한 윈도우가 정답. ② **정차-상태 카운팅이 변종/좌석 fold를 직접 회복**(state는 ~20s dwell이라 차임 피치·좌석 무관) → 방향 검증. ③ 증강(pitch-aug 등)은 또 패배 — 변종은 *실제 트립*이 학습에 있어야. ④ state vs 차임-이벤트는 **트립마다 상보적**(트립2는 이벤트가 강함).

**남은 일 (10트립에서)**: state의 **false 억제** = 시간표/운행주기 **cadence prior**(실 정차 ≈ 2분마다 ~20s, off-cadence 헛검출 제거) + dual-cue(정차 런 + 차임 + 이번역 KWS) + threshold 튜닝. 4트립 튜닝은 전이 안 되니 10트립에서 함께. state 머신·cadence·카운팅은 전부 후처리 → **온보드 비용 0**.
