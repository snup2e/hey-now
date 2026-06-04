# path2_fullpipe — Path 2 풀 파이프라인 (Colab)

§11-D 옵션 3. 실차 1호선 안내방송으로 **하차역**을 cross-trip(새 차량)으로 맞히는
**end-to-end 파이프라인**을 8트립 leave-one-out으로 검증한다.

```
KWS "이번역은"  ─┐
출발차임(close) ─┼─ 교차게이트(AND) ── 역 이벤트 ── metric 인코더 분류 ── anchored Viterbi ── 역 시퀀스
정차상태(stop)  ─┘                                   (cosine→emission)   (노선 단조 + 탑승역 앵커)
```

## 왜 융합인가
8트립 단독 cross-trip(§11): 이번역은 KWS 83%/56fp · 닫힘 81%/15fp · 열림 92%/34fp.
recall은 쓸만하나 **false가 전부 과다** → 단독 카운팅은 cascade로 위치 추정 불가.
KWS 오발화(잡담·KTX) ⟂ 헛차임/헛정차 → **독립 신호가 겹치는 곳만 '역'으로 인정(AND)**
하면 false가 곱으로 깎인다(§11-C 가설). 게이트가 한 역을 놓쳐도 anchored Viterbi가
skip을 허용해 위치를 보존한다. 게이트·디코드는 후처리 → **온보드(F411) 비용 0**.

## 폴더 구성
```
path2_fullpipe/
├── path2_fullpipe.ipynb     ← Colab 노트북 (메인)
├── scripts/                 ← import 모듈 (one source of truth, 로컬 repo 사본)
│   ├── path2_dataset.py        공유 데이터셋 빌더 (KWS/CNN/metric 풀)
│   ├── path2_poc.py            train / small_cnn / 슬라이딩 추론 헬퍼
│   ├── path2_metric_poc.py     ProtoNet 인코더 + prototype
│   ├── path2_grl_poc.py        채널 적대(GRL) 학습 루프
│   ├── path2_door_poc.py       door 3-class 검출 + 카운팅 (윈도우/검출 재사용)
│   └── path2_pipeline.py       ★ 교차게이트 + emission + anchored Viterbi + e2e 스코어링
├── door_events/             ← 8트립 door_events.json (데이터 zip에 없을 때 백업)
├── build_data_zip.py        ← 8트립 데이터 zip 생성기 (로컬 1회 실행)
├── _gen_notebook.py         ← 노트북 생성 스크립트 (재현용)
└── README.md
```

## 실행 순서
### 1) 데이터 zip 만들기 (로컬, 1회)
committed `heynow_path2_data.zip`은 **stale**(4트립·door_events 없음). 8트립용을 새로 만든다.
```
cd E:\imsisul
python path2_fullpipe/build_data_zip.py
# → path2_fullpipe/heynow_path2_data_8trip.zip  (clean wav + 8트립 audio/marks/door_events)
```

### 2) 업로드 — 무엇을 어디서 어디로

| 올릴 파일 | 로컬 위치 (절대경로) | Drive 업로드 위치 | 용도 |
|---|---|---|---|
| **`path2_fullpipe/` 폴더 전체** | `E:\imsisul\path2_fullpipe\` | `MyDrive/path2_fullpipe/` | 노트북 + `scripts/` 모듈 + `door_events/` |
| **`heynow_path2_data_8trip.zip`** (457MB) | `E:\imsisul\path2_fullpipe\heynow_path2_data_8trip.zip` | (폴더에 이미 포함 → 같이 올라감) | 8트립 audio+marks+door_events + clean wav |

> 폴더째 드래그하면 zip도 함께 올라가니 업로드는 **한 번**이면 끝. zip만 따로 다른 곳에 두면
> 셀 1의 `DATA_ZIP` 경로를 거기로 맞추면 된다.

**셀 1에서 고칠 경로 2줄** (Drive에 폴더째 올린 경우):
```python
PKG      = '/content/drive/MyDrive/path2_fullpipe'
DATA_ZIP = '/content/drive/MyDrive/path2_fullpipe/heynow_path2_data_8trip.zip'
```

### 3) 노트북 실행
런타임 → **GPU(T4 추천)** + **고용량 RAM**(Colab Pro). 모델이 작아 T4면 충분하고 compute unit을
아낀다 — **A100은 과잉**(병목은 librosa 전처리=CPU라 체감 이득 적고 units 3~4배 소모). 셀 1에서
고칠 건 **`PKG`·`DATA_ZIP` 2줄뿐**. `DATA='/content/data'`는 zip 푸는 로컬 작업폴더라 **그대로 둔다**
(Drive로 바꾸면 느려짐). 무거운 실험은 로컬 대신 여기서 돈다(로컬 ≤10분 스모크만 — CLAUDE.md 규칙4).
- **§3 open/close 검출**: 4-class로 open·close 독립 검출 LOO. 게이트 '역 성립 조건'(open+close
  vs close-only)을 이 결과로 결정. (8-fold)
- **§4 end-to-end LOO**: KWS·door·인코더 3개 학습 → 게이트 → 분류 → Viterbi → cross-trip 역정확도.
  ⚠️ 현재 게이트는 **타이밍 윈도우 구버전 → 이벤트-순서 상태기계로 교체 예정**(§3 결과 후). 빠른
  점검은 셀 2에서 `MP.EPISODES = 400`.
- **§6–7**: 8트립 전부로 최종 학습 → encoder/kws/door INT8 tflite + prototype + meta 내보내기.

## 결과 읽기 (셀 3 출력)
fusion-mode 비교표:

| mode | 뜻 |
|---|---|
| `kws` | 융합 없음 — KWS 트리거마다 분류 (cascade 베이스라인) |
| `and_close` | KWS ∧ 출발차임 (§11-C 핵심 가설) |
| `and_state` | KWS ∧ 정차상태 구간 |
| `and_both` | KWS ∧ 정차 ∧ 그 안의 close (가장 엄격) |

- **gate recall** = 게이트 이벤트가 실제 역에 매칭된 비율 = 정확도의 천장.
- AND가 `kws` 대비 **false/trip을 크게 줄이며** recall을 지키면 가설 입증.
- **Viterbi acc > per-mark acc** = 시퀀스 prior(앵커+단조)가 분류 천장을 끌어올림.

## 입력 규약 (로컬 스크립트와 동일 — 한쪽만 바꾸면 mismatch)
- 분류 윈도우 = 1차 호명 **"이번역은 [본역명]"** ([트리거 onset, +2.0s]).
- feature = 40-mel log-mel + **per-window CMN** (학습·추론·펌웨어 일관).
- door 윈도우 = `DR.WIN_S`(기본 1.0s; 3.0s = 차임+닫힘 둘 다).
- clean(서울교통공사)은 코레일 live와 **다른 녹음** → 인코더는 real-only(`use_clean=False`),
  KWS positive에만 clean 사용.
- door_events.json의 `sample_index`는 audio.wav와 샘플 단위로 정렬됨(md5 검증).

## 노트북에서 바꿀 만한 노브 (셀 2)
| 노브 | 기본 | 의미 |
|---|---|---|
| `DR.WIN_S` | 1.0 | door 윈도우(초). 3.0이면 차임+닫힘 둘 다 보게 됨 |
| `DR.MOVING_THR` | 0.4 | P(주행)<thr=정차. 낮추면 헛정차↓ |
| `DR.MIN_STOP_S` | 6.0 | 최소 정차 길이(초). 짧은 오분류 제거 |
| `MP.EPISODES` | 2000 | GRL 인코더 episode 수 (GPU). 점검은 400~600 |
| `GRL_LAMBDA` | 0.3 | 채널 적대 강도 (로컬 best trips_only 0.3) |
| `PL.KWS_TRIG` | 0.6 | KWS 트리거 임계 (고recall) |
| `PL.CLOSE_MIN_S`/`CLOSE_MAX_S` | 20/115 | 방송→닫힘차임 간격 윈도우(초, 실측 median 66s) |
| `PL.STOP_LEAD_S`/`STOP_LAG_S` | 12/45 | 방송 직후 정차구간 START 윈도우(초) |

전체 맥락은 repo 루트 `PATH2_RESULTS.md`(특히 §3 시퀀스 prior, §9 하이브리드, §11 8트립) 참조.
