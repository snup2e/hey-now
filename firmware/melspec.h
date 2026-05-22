// log-mel spectrogram extraction for Path 1 (STM32 / CMSIS-DSP).
//
// Reproduces librosa.power_to_db(melspectrogram(y), ref=1.0) so the
// on-device features match training. Verified against librosa via
// scripts/melspec_ref.py (max error ~1e-4 dB).
#ifndef MELSPEC_H
#define MELSPEC_H

#include <stdint.h>
#include "model_meta.h"

// Call once at startup (initializes the rFFT instance + Hann window).
void melspec_init(void);

// Compute raw log-mel (dB) for one window of an int16 PCM clip.
//   audio         : full clip (read directly from Flash, never copied)
//   total_samples : length of audio[]
//   win_start     : window start index into audio[]
//   win_len       : KWS_WIN_SAMPLES or CNN_WIN_SAMPLES (fixes frame count)
//   out_mel       : caller buffer, size >= N_MELS * (1 + win_len / MEL_HOP)
//                   mel-major row-major: out_mel[m * (*n_frames) + t]
//   n_frames      : receives the number of time frames
// Samples outside [win_start, win_start+win_len) or outside the clip are
// treated as 0 — identical to librosa center-padding a sliced window.
// The caller applies per-model normalization: (mel - MEAN) / STD.
void melspec_compute(const int16_t *audio, int total_samples,
                     int win_start, int win_len,
                     float *out_mel, int *n_frames);

#endif  // MELSPEC_H
