// model_runner.c — connects app_path1's kws_infer()/cnn_infer() to the
// X-CUBE-AI generated networks.
//
// Assumes X-CUBE-AI code generation with:
//   * kws.tflite  -> network name "kws"
//   * cnn.tflite  -> network name "cnn"
//   * Inputs/Outputs data type: float  (X-CUBE-AI then handles the
//     INT8 quantize/dequantize internally, so app_path1.c can pass
//     normalized float log-mel and read float softmax directly)
//
// If X-CUBE-AI is new to you, follow INTEGRATION_GUIDE.md steps 4-6
// first — that is where these networks are generated.
#include "app_path1.h"
#include "model_runner.h"

// --- X-CUBE-AI generated headers (created in the CubeMX project) ---
#include "kws.h"
#include "kws_data.h"
#include "cnn.h"
#include "cnn_data.h"

static ai_handle s_kws = AI_HANDLE_NULL;
static ai_handle s_cnn = AI_HANDLE_NULL;

// activation buffers (sizes come from the generated *_data.h)
AI_ALIGNED(32) static ai_u8 s_kws_act[AI_KWS_DATA_ACTIVATIONS_SIZE];
AI_ALIGNED(32) static ai_u8 s_cnn_act[AI_CNN_DATA_ACTIVATIONS_SIZE];

void models_init(void)
{
    ai_kws_create_and_init(&s_kws, s_kws_act, NULL);
    ai_cnn_create_and_init(&s_cnn, s_cnn_act, NULL);
}

void kws_infer(const float *mel_norm, float *out_prob)
{
    ai_buffer *in  = ai_kws_inputs_get(s_kws, NULL);
    ai_buffer *out = ai_kws_outputs_get(s_kws, NULL);
    in[0].data  = AI_HANDLE_PTR(mel_norm);   // N_MELS * KWS_MEL_FRAMES floats
    out[0].data = AI_HANDLE_PTR(out_prob);   // 2-class softmax
    ai_kws_run(s_kws, in, out);
}

void cnn_infer(const float *mel_norm, float *out_prob)
{
    ai_buffer *in  = ai_cnn_inputs_get(s_cnn, NULL);
    ai_buffer *out = ai_cnn_outputs_get(s_cnn, NULL);
    in[0].data  = AI_HANDLE_PTR(mel_norm);   // N_MELS * CNN_MEL_FRAMES floats
    out[0].data = AI_HANDLE_PTR(out_prob);   // NUM_STATIONS softmax
    ai_cnn_run(s_cnn, in, out);
}
