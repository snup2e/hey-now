// Path 1 demo loop — replays Flash-resident announcement clips through
// the KWS+CNN pipeline and shows the route on the LCD.
//
// This is the "음원 재생 시뮬레이션": no microphone — demo_clips[] stand
// in for live audio. Identical pipeline to scripts/verify_pipeline.py.
#include "app_demo.h"
#include "app_path1.h"
#include "demo_audio.h"
#include "display.h"

#include "stm32f4xx_hal.h"   // HAL_Delay — provided by the CubeMX project

void app_demo_run(void)
{
    path1_init();
    lcd_init();
    lcd_show_message("Hey now!", "Path 1 demo");
    HAL_Delay(1500);

    Path1State state;
    path1_state_reset(&state);

    for (;;) {
        for (int c = 0; c < NUM_DEMO_CLIPS; c++) {
            // feed one announcement clip through KWS + CNN
            path1_process(demo_clips[c].pcm, demo_clips[c].n_samples, &state);
            lcd_show_route(&state);
            HAL_Delay(3000);     // hold the result before the next clip
        }
    }
}
