// display.c — Path 1 result output.
//
// 임시 구현: 2.0" LCD 패널이 확정되기 전까지 결과를 UART(시리얼)로
// 출력합니다. NUCLEO-F411RE의 USART2가 ST-LINK 가상 COM 포트에
// 연결되므로 PC 시리얼 모니터(115200 baud, UTF-8)로 확인할 수 있습니다.
//
// LCD 패널 확정 후, 이 파일의 함수 본문을 LCD 드라이버 호출로
// 교체하면 됩니다 (인터페이스 display.h는 그대로 유지).
#include "display.h"
#include <stdio.h>

void lcd_init(void)
{
    // USART2 + printf retarget은 CubeMX/main.c에서 설정 — INTEGRATION_GUIDE 참조.
}

void lcd_show_route(const Path1State *st)
{
    const char *prev = (st->previous_station >= 0)
                           ? STATION_NAMES[st->previous_station] : "-";
    const char *cur  = (st->current_station >= 0)
                           ? STATION_NAMES[st->current_station] : "-";
    printf("[Path1] 지난 역: %s  |  현재 역: %s  (conf %.0f%%)\r\n",
           prev, cur, (double)(st->current_conf * 100.0f));
}

void lcd_show_message(const char *line1, const char *line2)
{
    printf("[Path1] %s | %s\r\n", line1, line2);
}
