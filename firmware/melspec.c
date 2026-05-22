// log-mel spectrogram extraction for Path 1 — see melspec.h.
//
// Pipeline (identical to scripts/melspec_ref.py, verified vs librosa):
//   1. center-pad the window by N_FFT/2 zeros each side
//   2. per frame: apply periodic Hann window, real FFT
//   3. power spectrum |X|^2
//   4. mel filterbank matmul (mel_filterbank.h)
//   5. 10*log10, then clip to (max - 80 dB)   [librosa top_db=80]
#include "melspec.h"
#include "mel_filterbank.h"

#include "arm_math.h"
#include <math.h>

#define N_BINS (N_FFT / 2 + 1)
#define PAD    (N_FFT / 2)
#define TOP_DB 80.0f

static arm_rfft_fast_instance_f32 s_fft;
static float s_hann[N_FFT];

// scratch buffers (kept static to avoid large stack frames)
static float s_frame[N_FFT];     // windowed frame / FFT input (destroyed by FFT)
static float s_fftout[N_FFT];    // packed real-FFT output
static float s_power[N_BINS];    // power spectrum

void melspec_init(void)
{
    arm_rfft_fast_init_f32(&s_fft, N_FFT);
    // periodic Hann window: 0.5 - 0.5*cos(2*pi*n/N)  (librosa fftbins=True)
    for (int n = 0; n < N_FFT; n++) {
        s_hann[n] = 0.5f - 0.5f * cosf(2.0f * PI * (float)n / (float)N_FFT);
    }
}

// int16 PCM -> float in [-1, 1)  (matches librosa.load scaling)
#define PCM_SCALE (1.0f / 32768.0f)

void melspec_compute(const int16_t *audio, int total_samples,
                     int win_start, int win_len,
                     float *out_mel, int *n_frames)
{
    const int nf = 1 + win_len / MEL_HOP;     // center-padded frame count
    *n_frames = nf;

    float db_max = -1.0e30f;

    for (int t = 0; t < nf; t++) {
        // build one center-padded, Hann-windowed frame (int16 -> float here)
        const int base = win_start + t * MEL_HOP - PAD;
        for (int n = 0; n < N_FFT; n++) {
            const int idx = base + n;
            const int rel = idx - win_start;       // position within window
            // zero outside the window OR outside the clip (matches librosa
            // center-padding of a sliced window)
            const float s = (rel >= 0 && rel < win_len &&
                             idx >= 0 && idx < total_samples)
                                ? (float)audio[idx] * PCM_SCALE
                                : 0.0f;
            s_frame[n] = s * s_hann[n];
        }

        // real FFT  (s_frame is consumed; s_fftout is packed)
        arm_rfft_fast_f32(&s_fft, s_frame, s_fftout, 0);

        // power spectrum from CMSIS packed layout:
        //   [0] = DC real, [1] = Nyquist real, [2k]/[2k+1] = Re/Im of bin k
        s_power[0]        = s_fftout[0] * s_fftout[0];
        s_power[N_FFT / 2] = s_fftout[1] * s_fftout[1];
        for (int k = 1; k < N_FFT / 2; k++) {
            const float re = s_fftout[2 * k];
            const float im = s_fftout[2 * k + 1];
            s_power[k] = re * re + im * im;
        }

        // mel filterbank + 10*log10
        for (int m = 0; m < N_MELS; m++) {
            float e = 0.0f;
            for (int k = 0; k < N_BINS; k++) {
                e += mel_filterbank[m][k] * s_power[k];
            }
            if (e < 1.0e-10f) e = 1.0e-10f;
            const float db = 10.0f * log10f(e);
            out_mel[m * nf + t] = db;
            if (db > db_max) db_max = db;
        }
    }

    // librosa top_db: clip everything below (max - 80 dB)
    const float floor = db_max - TOP_DB;
    const int total = N_MELS * nf;
    for (int i = 0; i < total; i++) {
        if (out_mel[i] < floor) out_mel[i] = floor;
    }
}
