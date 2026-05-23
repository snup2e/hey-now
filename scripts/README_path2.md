# Path 2 — 1호선 실차 녹음 가이드

성균관대 → 구로 구간(13역)을 STM32 보드로 직접 녹음해서 데이터셋을 만드는 과정입니다.
폰 마이크는 안 씁니다 (실제 추론할 때의 마이크 도메인과 달라지면 학습이 의미 없음).

## 한 줄 요약

1. 보드(ICS43434 마이크 결선됨) USB로 노트북에 연결
2. `python scripts/path2_capture_ui.py --port <COMx> --direction north`
3. 트립 동안 안내방송이 시작될 때마다 **스페이스바**
4. 도착해서 종료 → `python scripts/path2_slice.py` → 클립·메타데이터 자동 생성

## 0. 준비물

- STM32 NUCLEO-F411RE + ICS43434 I2S 마이크 (결선 + 펌웨어는 `firmware/path2_recorder/` 참조 — 별도)
- 노트북 (USB-A 또는 USB-C 어댑터) + 마이크로 USB 케이블 (NUCLEO ST-LINK 포트용)
- Python 3.10+ : `pip install pyserial`

## 1. 녹음 시작 전

보드를 노트북에 USB 연결하면 가상 COM 포트가 생깁니다.

- **Windows**: 장치관리자 → 포트 (COM & LPT) → `STMicroelectronics Virtual COM Port (COMx)` 확인. 보통 COM3~COM9.
- **macOS/Linux**: `ls /dev/tty.usbmodem*` 또는 `ls /dev/ttyACM*`

```powershell
# 상행 (성균관대 → 구로) 시작 예시
python scripts/path2_capture_ui.py --port COM5 --direction north

# 하행 (구로 → 성균관대)
python scripts/path2_capture_ui.py --port COM5 --direction south
```

창이 뜨면 즉시 녹음 시작. **첫 안내방송 전에 띄워두기.**

## 2. 트립 중

- 안내방송 "이번 역은 ___" 시작 들리면 **스페이스바**
- 노란색 ▶ 표시가 다음 역으로 자동 이동
- 헷갈리거나 놓쳤으면 → 해당 역 버튼을 마우스로 클릭 (재마크도 동일하게 덮어쓰기)
- 마크가 살짝 늦거나 빠르게 눌려도 OK — 슬라이서가 ±4~12초 마진으로 자름

**팁**: 환승역(금정·금천구청·구로 등)은 안내방송이 길어서 "이번 역은"이 ~4초쯤 늦게 시작합니다. 들릴 때 누르면 됨.

## 3. 종료

- 마지막 역 도착 → 우측 하단 **종료 & 저장** (또는 창 X)
- 파일 두 개가 저장됨:
  - `data/raw/line1_live/<trip_id>/audio.wav` — 트립 통째 녹음
  - `data/raw/line1_live/<trip_id>/marks.json` — 역별 마크 타임스탬프
- `<trip_id>` = `YYYYMMDD_HHMM_<direction>` (예: `20260524_0742_north`)

## 4. 클립 + 메타데이터 생성

녹음 끝나고 노트북에서:

```bash
python scripts/path2_slice.py            # 새로 추가된 트립만 처리
python scripts/path2_slice.py --trip 20260524_0742_north   # 특정 트립만
python scripts/path2_slice.py --pre 5 --post 14            # 윈도우 조정
```

생성물:
- `data/processed/line1_clips/<trip_id>/<역>_<seq>.wav` — 16초 클립 (마크 -4s ~ +12s)
- `data/path2_metadata.csv` — clip_file / station / trip_id / direction / mark_offset_sec / clip_sec / sample_rate

같은 트립을 다시 슬라이스하면 메타데이터 행이 덮어쓰기됩니다 (clip_file 키 기준).

## 5. 노트북 없이 보드 검증 (mock 모드)

보드/펌웨어가 아직 없어도 UI 동작은 확인 가능:

```bash
python scripts/path2_capture_ui.py \
    --mock-wav data/processed/wav/성균관대.wav \
    --direction north
```

mock 모드는 입력 wav를 무한 반복 재생하면서 실시간처럼 스트림합니다. UI 인터랙션, 저장 포맷, 슬라이서까지 전부 보드 없이 검증 가능.

## 트러블슈팅

| 증상 | 원인/대응 |
|---|---|
| 창은 뜨는데 elapsed 시계가 0:00에서 안 움직임 | USB 포트 오인식. 다른 COM 포트로 재시도, 보드 펌웨어가 USB-CDC enable 됐는지 확인 |
| 녹음이 너무 빠르게/느리게 들림 | `--sr` 값 mismatch. 펌웨어가 보내는 샘플레이트와 일치시켜야 함 |
| 마크를 다 놓쳤음 | 트립 폐기 말고 일단 audio.wav 보관 → 나중에 Audacity로 수동 마킹 후 marks.json 직접 작성 가능 (스키마는 기존 파일 참조) |
| 파일이 손상되거나 wav 헤더가 깨짐 | Recorder가 정상 stop을 못 부른 경우. 데이터 자체는 raw PCM이라 살릴 수 있음 — 별도 복구 도구 필요 |

## 데이터 정책 복습

- `data/raw/line1_live/`, `data/processed/line1_clips/` — **gitignore** (대용량 + 잡음에 승객 대화 잠재 포함)
- `data/path2_metadata.csv` — **commit OK** (파일명만 들어감)
- 원본 audio.wav는 Google Drive 별도 백업 권장
