// Path 1 application pipeline — see app_path1.h.
#include "app_path1.h"
#include "melspec.h"
#include "model_runner.h"

// log-mel scratch buffers (static: ~10 KB + ~20 KB, avoids huge stack)
static float s_kws_mel[N_MELS * KWS_MEL_FRAMES];
static float s_cnn_mel[N_MELS * CNN_MEL_FRAMES];

void path1_init(void)
{
    melspec_init();
    models_init();
}

void path1_state_reset(Path1State *st)
{
    st->previous_station = -1;
    st->current_station  = -1;
    st->current_conf     = 0.0f;
}

static int argmax(const float *v, int n)
{
    int best = 0;
    for (int k = 1; k < n; k++) {
        if (v[k] > v[best]) best = k;
    }
    return best;
}

// Classify the station-name span that follows a confirmed trigger.
static void classify(const int16_t *audio, int total_samples,
                     int trig_start, Path1State *st)
{
    const int name_start = trig_start + NAME_OFFSET_SAMPLES;
    int nf;
    melspec_compute(audio, total_samples, name_start, CNN_WIN_SAMPLES,
                    s_cnn_mel, &nf);

    const int count = N_MELS * nf;
    for (int j = 0; j < count; j++) {
        s_cnn_mel[j] = (s_cnn_mel[j] - CNN_NORM_MEAN) / CNN_NORM_STD;
    }

    float prob[NUM_STATIONS];
    cnn_infer(s_cnn_mel, prob);

    const int best = argmax(prob, NUM_STATIONS);
    if (prob[best] >= CONF_THRESH) {
        st->previous_station = st->current_station;
        st->current_station  = best;
        st->current_conf     = prob[best];
    }
}

void path1_process(const int16_t *audio, int n_samples, Path1State *st)
{
    int i = 0;
    int run = 0;          // consecutive trigger-window count (debounce)
    int run_start = 0;

    while (i + KWS_WIN_SAMPLES <= n_samples) {
        int nf;
        melspec_compute(audio, n_samples, i, KWS_WIN_SAMPLES,
                        s_kws_mel, &nf);

        const int count = N_MELS * nf;
        for (int j = 0; j < count; j++) {
            s_kws_mel[j] = (s_kws_mel[j] - KWS_NORM_MEAN) / KWS_NORM_STD;
        }

        float p[2];
        kws_infer(s_kws_mel, p);

        if (p[1] > TRIG_THRESH) {                 // "이번 역은" trigger window
            if (run == 0) run_start = i;
            run++;
            i += SLIDE_HOP_SAMPLES;
        } else {
            if (run >= MIN_TRIG_RUN) {            // debounced trigger confirmed
                classify(audio, n_samples, run_start, st);
                i = run_start + NAME_OFFSET_SAMPLES + CNN_WIN_SAMPLES;
            } else {
                i += SLIDE_HOP_SAMPLES;
            }
            run = 0;
        }
    }
    if (run >= MIN_TRIG_RUN) {                     // trailing trigger
        classify(audio, n_samples, run_start, st);
    }
}
