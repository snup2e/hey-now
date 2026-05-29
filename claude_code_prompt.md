# Claude Code 작업 지시 — Path 2 cross-trip 정확도 끌어올리기

## 배경 (먼저 읽어)

`CLAUDE.md`를 먼저 정독해. 특히 "Path 2 분류기 실험 기록" 섹션. 요약하면:

- 우리는 지하철 안내방송으로 하차역을 알리는 TinyML 프로젝트. STM32 F411 **온보드 단독 추론**이 최종 타겟(128KB SRAM, CMSIS-DSP).
- Stage 1 KWS("이번 역은" 트리거)는 cross-trip 11~13/13으로 잘 됨. **Stage 2 역 분류가 병목.**
- Stage 2는 13-class softmax(19%) → ProtoNet metric learning(35~42%)으로 전환됨. 여전히 사용가능(90%)에 못 닿음.
- **진단: 병목은 데이터 양이 아니라 채널 다양성.** 안내방송은 KORAIL 단일 녹음의 결정론적 재생(고정 신호)이고, 트립마다 변하는 건 (1) 가산 노이즈(합성으로 학습됨, 해결), (2) 채널=차량 PA·객차 음향·마이크 위치(실 트립 수만큼만 존재=4개). 3→4트립으로 늘려도 안 올랐음.
- **현실적 목표: 42% → 55~65%.** 90%는 4채널로 물리적으로 불가. 거기에 더해 abstain으로 "틀린 역을 자신있게 표시"하는 최악을 방지.

## 절대 규칙

1. **한 번에 레버 하나만.** 각 레버를 적용한 뒤 반드시 LOO로 채점하고, baseline 대비 cross-trip 정확도 변화를 표로 보고한 뒤 멈춰서 내 확인을 받아. 여러 레버를 동시에 적용하지 마.
2. **평가 프로토콜 고정 (정직하게):** 트립단위 leave-one-out. 4트립 중 1개 held-out(미학습=새 채널), 나머지 3 + 클린으로 학습 → held-out 채점. 4 fold 평균. **시드 고정**으로 run간 요동 제거. held-in val은 참고용일 뿐 **판단 기준은 항상 cross-trip held-out.**
3. **온보드 제약 위반 금지:** 추가하는 전처리/모델은 CMSIS-DSP로 STM32 F411(128KB SRAM, FPU 있음)에 이식 가능해야 함. 무거운 연산(대형 transformer, RIR 실시간 합성 등 추론 경로에 들어가는 것)은 거부하고 이유를 설명해. 단, **증강은 학습 시점에만 쓰이므로 무거워도 OK.**
4. **전처리 일관성:** melspec/정규화를 바꾸면 **학습·로컬추론(verify_pipeline)·펌웨어(melspec.c)** 세 곳 모두에 동일 적용해야 함. 한쪽만 바꾸면 과거 "CMN 없어서 라이브 0트리거" mismatch가 재발. 펌웨어 반영은 코드만 작성하고 "보드 검증 대기"로 표시.
5. **재현성:** 모든 실험은 `python scripts/...`로 로컬 CPU 재현 가능해야 함(시드 고정). 결과는 기존 `path2_metric_poc.py` 출력 포맷과 일치시켜.
6. KWS Stage 1은 건드리지 마(별도 트랙). 이번 작업은 Stage 2 분류기만.

## 시작 전 진단

먼저 다음을 보고해 (코드 수정 전):
- `scripts/path2_dataset.py`, `scripts/path2_metric_poc.py`의 현재 melspec/정규화 함수 시그니처와 CMN 적용 위치.
- 현재 baseline LOO 정확도를 **다시 한 번 재현 실행**해서 숫자 확정(시드 고정). 이게 모든 비교의 기준점.
- 4트립 각각의 클립 수, held-out fold별 클래스 분포.

## 레버 1 — PCEN으로 채널 정규화 강화 (가장 먼저)

현재 `power_to_db + per-window CMN`을 **PCEN(Per-Channel Energy Normalization)**으로 교체.
- 근거: PCEN은 채널별 게인 + 시간상수까지 정규화해 차량별 PA 음색/레벨 차이를 CMN보다 강하게 억제. FS-KWS에서 PCEN만 추가해도 cross-domain 정확도가 올랐다는 보고(EdgeSpot, Interspeech 류).
- 구현: 학습은 `librosa.pcen` 또는 직접 구현(파라미터 `gain, bias, power, time_constant`). **학습·추론 동일 파라미터.**
- PCEN 파라미터는 우선 문헌 기본값으로 고정 → LOO 채점 → 그 다음 1~2개 값만 가볍게 스윕.
- 펌웨어: PCEN의 IIR smoother는 CMSIS-DSP로 이식 가능. `melspec.c`에 적용할 의사코드/C 스니펫까지 작성하되 "보드 검증 대기"로 표시.
- **채점 후 멈춤.** baseline vs PCEN cross-trip LOO 표 보고.

## 레버 2 — 채널 적대 학습 (GRL)

ProtoNet 인코더 위에 작은 도메인 분류 헤드(4-class: 어느 트립이냐) + **gradient reversal layer**를 붙여 채널 불변 임베딩 유도.
- 근거: channel adversarial training(GRL)은 채널이 적을 때 채널 정보를 임베딩에서 직접 제거하는 직접적 방법. 트립 라벨은 이미 있으므로 데이터 추가 0.
- 구현: 인코더는 그대로, forward에 GRL→Dense→4-class 헤드 추가. 손실 = ProtoNet loss + λ·(도메인 분류 손실, GRL 통과). λ는 0(=baseline)부터 점진 증가(0.05, 0.1, 0.3) 스윕.
- 주의: 클린 음원도 하나의 도메인으로 라벨링할지(5-class) 트립만(4-class)로 할지 두 가지 다 시도하고 비교.
- 인코더 구조/임베딩 차원/Flatten은 유지(GAP 금지 — 본역명 토큰 뭉개짐).
- **추론 경로에는 도메인 헤드 미포함**(학습 전용) → 온보드 비용 0. 명시해.
- **채점 후 멈춤.** baseline vs (레버1 적용본) vs +GRL, λ별 cross-trip LOO 표 보고.

## 레버 3 — 실 RIR 보간 증강 (레버 1·2가 효과 났을 때만)

4트립에서 추출한 실제 채널 특성을 보간해 가상 채널 생성(랜덤 reverb 아님 — 그건 이미 wash로 확인됨).
- 접근: (a) 각 트립의 클린-대비 채널 응답을 추정(예: 같은 안내방송 구간의 클린 대비 스펙트럴 차이/추정 RIR), (b) 트립 쌍 사이를 보간/혼합해 가상 채널 응답 생성, (c) 클린 1차호명에 합성 적용 → 학습 데이터에만 추가.
- 증강은 학습 시점 전용이라 무거워도 OK. 추론/펌웨어 불변.
- 구현 전에 먼저 "추정 가능한가" 타당성을 1트립으로 빠르게 점검하고 보고(불가하면 솔직히 말하고 스킵).
- **채점 후 멈춤.**

## 레버 4 — episode 수 (확인용, 마지막)

로컬 episode 600 → 더 큰 값(예: 2000)으로 늘려 천장이 compute인지 확인.
- CLAUDE.md 진단상 wash 예상. 레버 1~3 적용본 위에서 한 번만 확인하고 결과 보고.

## 마무리 — abstain 캘리브레이션 (레버 끝난 뒤)

최종 인코더로 abstain 임계(τ: cosine 최소, δ: top1−top2 최소)를 LOO에서 캘리브레이션.
- 목표: "확신할 때만 표시, 아니면 '확인 중'". 표시한 것 중 정확도(precision)와 표시 비율(coverage)의 트레이드오프 곡선을 보고.
- 이게 데모에서 "틀린 역 자신있게 표시" 최악을 막는 안전장치.

## 산출물

- 각 레버: 코드 변경(diff) + LOO 결과표(baseline 대비) + 온보드 영향 한 줄.
- 최종: `models/` Path 2 산출물 갱신 계획, 펌웨어 `melspec.c` 반영분(보드 검증 대기 표시), abstain 곡선.
- 전부 시드 고정 로컬 재현 가능. git push는 내가 별도로 지시.

지금 "시작 전 진단"부터 실행하고 baseline 숫자 확정한 뒤 멈춰.
