// Path 1 application pipeline — KWS trigger + CNN station classification.
//
// Mirrors scripts/verify_pipeline.py (verified 17/17 on the host):
//   slide 1 s window -> KWS -> on a debounced "이번 역은" trigger,
//   feed the following 2 s to the CNN -> update previous/current station.
#ifndef APP_PATH1_H
#define APP_PATH1_H

#include <stdint.h>
#include "model_meta.h"

// ---- model inference interface ----
// Implemented in model_runner.c once X-CUBE-AI has generated the network
// code for kws.tflite / cnn.tflite.
//   kws_infer : in  = normalized log-mel, N_MELS*KWS_MEL_FRAMES floats
//               out = 2-class softmax { non-trigger, trigger }
//   cnn_infer : in  = normalized log-mel, N_MELS*CNN_MEL_FRAMES floats
//               out = NUM_STATIONS softmax
void kws_infer(const float *mel_norm, float *out_prob);
void cnn_infer(const float *mel_norm, float *out_prob);

// ---- pipeline state ----
typedef struct {
    int   previous_station;   // STATION_NAMES index, -1 = none
    int   current_station;    // STATION_NAMES index, -1 = none
    float current_conf;       // CNN confidence of current_station
} Path1State;

void path1_init(void);
void path1_state_reset(Path1State *st);

// Run KWS+CNN over one announcement clip (int16 PCM, e.g. from demo_audio.h).
// *st is updated each time a station is confirmed (previous <- current).
void path1_process(const int16_t *audio, int n_samples, Path1State *st);

#endif  // APP_PATH1_H
