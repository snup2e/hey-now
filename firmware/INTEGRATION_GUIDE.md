# Path 1 펌웨어 통합 가이드

STM32 NUCLEO-F411RE 보드에서 **KWS + CNN으로 지하철 역을 인식**하는
Path 1 데모 펌웨어를 빌드·실행하는 단계별 안내입니다.

마이크는 쓰지 않습니다 — 보드 Flash에 저장된 안내방송 음원(`demo_audio.h`)을
모델에 흘려보내 "지난 역 / 현재 역"을 출력하는 **시뮬레이션**입니다.

---

## 0. 준비물

- **보드**: STM32 NUCLEO-F411RE
- **STM32CubeIDE** (설치되어 있음)
- **X-CUBE-AI** 패키지 (3단계에서 설치 — 처음이어도 괜찮음)
- 이 저장소의 `firmware/` 폴더와 `models/kws.tflite`, `models/cnn.tflite`

### 펌웨어 파일 구성 (`firmware/`)

| 파일 | 역할 |
|---|---|
| `model_meta.h` | 역 라벨·정규화 상수·파이프라인 임계값 |
| `mel_filterbank.h` | librosa mel 필터뱅크 (log-mel용) |
| `melspec.c/.h` | log-mel spectrogram 추출 (CMSIS-DSP) |
| `app_path1.c/.h` | KWS+CNN 파이프라인 (`verify_pipeline.py`와 동일) |
| `model_runner.c/.h` | X-CUBE-AI 네트워크 호출 래퍼 |
| `demo_audio.h` | 데모 음원 (성균관대·의왕, int16 PCM) |
| `display.c/.h` | 결과 출력 (현재 UART, LCD 확정 후 교체) |
| `app_demo.c/.h` | 데모 메인 루프 |

---

## 1. CubeMX 프로젝트 생성

1. STM32CubeIDE → **File → New → STM32 Project**
2. **Board Selector** 탭에서 `NUCLEO-F411RE` 선택 → Next
3. 프로젝트 이름 `heynow_path1` → Finish
4. "Initialize all peripherals with their default mode?" → **Yes**
   (USART2가 ST-LINK 가상 COM 포트로 자동 설정됨)

`.ioc` 파일이 열리면:

- **Clock Configuration** 탭: HCLK가 **100 MHz**인지 확인
- USART2: 기본값 그대로 (115200 baud, ST-LINK VCP 연결)

---

## 2. CMSIS-DSP 활성화

`melspec.c`가 FFT(`arm_rfft_fast_f32`)에 CMSIS-DSP를 씁니다.

1. `.ioc` → **Software Packs → Select Components**
2. **CMSIS → CMSIS DSP Library** 체크 → OK
3. 프로젝트 속성 → C/C++ Build → Settings → MCU GCC Compiler →
   Preprocessor에 `ARM_MATH_CM4` 정의 추가 (이미 있으면 생략)

> CMSIS-DSP가 Software Packs에 없으면, ST의 CMSIS-DSP 소스를 직접
> 프로젝트에 추가하고 `arm_rfft_fast_f32` 관련 파일을 포함하세요.

---

## 3. X-CUBE-AI 설치

1. **Help → Manage Embedded Software Packages**
2. **STMicroelectronics** 탭 → `X-CUBE-AI` 최신 버전 체크 → Install
3. 설치 후 `.ioc` → **Software Packs → Select Components**
4. **X-CUBE-AI → Core** 체크, Application은 `Application Template` 선택 → OK

---

## 4. 모델 추가 (X-CUBE-AI)

`.ioc` 좌측 메뉴에 생긴 **X-CUBE-AI** 항목을 클릭:

1. **Add network** → 네트워크 이름을 정확히 **`kws`** 로 입력
   - Model type: `TFLite`
   - Browse → `models/kws.tflite` 선택
2. **Add network** 한 번 더 → 이름 **`cnn`**, `models/cnn.tflite` 선택
3. 각 네트워크의 **Inputs/Outputs 데이터 타입을 `float`** 로 설정
   (모델은 INT8이지만 X-CUBE-AI가 양자화를 내부 처리 → `model_runner.c`가
   float 입출력을 그대로 넘길 수 있음)
4. 각 네트워크 **Analyze** 클릭 → Flash/RAM 사용량 확인
   - 두 모델 합쳐 Flash ~수십 KB, RAM(activations) ~수십 KB 예상
5. **Project → Generate Code** (또는 저장 시 자동 생성)

생성되면 프로젝트에 `kws.h`, `kws_data.h`, `cnn.h`, `cnn_data.h` 등이
추가됩니다. `model_runner.c`가 이 헤더를 include 합니다.

> 네트워크 이름이 `kws`/`cnn`이 아니면 `model_runner.c`의 `ai_*` 함수명이
> 안 맞습니다. 이름을 정확히 맞추거나, `model_runner.c`의 함수명을 생성된
> 것에 맞춰 수정하세요.

---

## 5. firmware/ 소스 복사

`firmware/`의 파일들을 CubeMX 프로젝트로 복사합니다:

- `.c` 파일 → 프로젝트 `Core/Src/`
- `.h` 파일 → 프로젝트 `Core/Inc/`
- `models/kws.tflite`, `models/cnn.tflite`는 4단계에서 X-CUBE-AI가
  이미 가져갔으므로 따로 복사 불필요

복사 후 프로젝트 탐색기에서 새 파일들이 보이는지 확인하세요.

---

## 6. main.c 연결

CubeMX가 생성한 `Core/Src/main.c`를 두 군데 수정합니다.

**(a) app_demo 헤더 포함** — `/* USER CODE BEGIN Includes */` 안에:

```c
/* USER CODE BEGIN Includes */
#include "app_demo.h"
/* USER CODE END Includes */
```

**(b) 데모 실행** — `main()`의 `while (1)` 직전 `/* USER CODE BEGIN 2 */` 안에:

```c
/* USER CODE BEGIN 2 */
app_demo_run();        // 자체 무한 루프 — 아래 while(1)에는 도달하지 않음
/* USER CODE END 2 */
```

**(c) printf → UART 리타겟** — `/* USER CODE BEGIN 4 */` 안에:

```c
/* USER CODE BEGIN 4 */
int __io_putchar(int ch) {
    HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}
/* USER CODE END 4 */
```

> `huart2`는 CubeMX가 USART2로 생성한 핸들입니다. 이름이 다르면 맞추세요.

---

## 7. 빌드 & 플래시

1. **Project → Build Project** (망치 아이콘)
2. 빌드 성공 시 보드를 USB로 연결 → **Run** (▶)
3. PC에서 시리얼 터미널 열기 (STM32CubeIDE 내장 터미널 또는 PuTTY 등)
   - 포트: ST-Link Virtual COM Port
   - 속도: **115200**, 8-N-1, 인코딩 **UTF-8**

### 예상 출력

```
[Path1] Hey now! | Path 1 demo
[Path1] 지난 역: -  |  현재 역: 성균관대  (conf 100%)
[Path1] 지난 역: 성균관대  |  현재 역: 의왕  (conf 100%)
[Path1] 지난 역: -  |  현재 역: 성균관대  (conf 100%)
...
```

데모 음원 2곡(성균관대 → 의왕)이 반복 재생되며, 트리거가 잡힐 때마다
"지난 역 / 현재 역"이 갱신됩니다.

---

## 8. 트러블슈팅

| 증상 | 해결 |
|---|---|
| **Flash 초과** (region `FLASH` overflowed) | `demo_audio.h` 클립이 312 KB로 큼. `scripts/gen_demo_audio.py`의 `CLIP_SEC`를 4.0으로 줄이거나 `DEMO`를 1곡으로 줄여 다시 생성 |
| **`kws.h` 없음** | 4단계 X-CUBE-AI 코드 생성을 안 했거나 네트워크 이름이 다름 |
| **`arm_rfft_fast_f32` 미정의** | 2단계 CMSIS-DSP 미활성화 |
| **시리얼에 한글 깨짐** | 터미널 인코딩을 UTF-8로 |
| **출력이 안 나옴** | 6-(c) printf 리타겟 확인, baud 115200 확인 |
| **RAM 부족** | X-CUBE-AI activation 버퍼가 큼 — Analyze에서 크기 확인 |

---

## 9. 다음 단계 — LCD

현재 결과는 UART로 출력됩니다. 2.0" SPI LCD 패널이 확정되면:

1. CubeMX에서 SPI + GPIO(CS/DC/RST) 핀 설정
2. `display.c`의 `lcd_init` / `lcd_show_route` / `lcd_show_message`
   본문을 LCD 드라이버(ILI9341 등) 호출로 교체
3. 인터페이스(`display.h`)는 그대로 — `app_demo.c`는 수정 불필요

한글 표시는 역 이름(14개)에 쓰인 글자만 담은 비트맵 폰트가 필요합니다.
