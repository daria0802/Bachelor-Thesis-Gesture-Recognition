/**
 * @file    tflite_inference.cpp
 * @brief   TensorFlow Lite Micro inference implementation.
 * @author  Daria Kinash
 * @date    2026
 */

#include "tflite_inference.h"
#include "app_config.h"

#include "fsl_debug_console.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <new>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>

#include "../model/model_data.h"
#include "normalization_params.h"

/* TFLM log hooks – required symbols called by the library internally. */
extern "C" void DebugLog(const char *s)                                            { PRINTF("%s", s); }
extern "C" int  DebugVsnprintf(char *buf, size_t n, const char *fmt, va_list args) { return vsnprintf(buf, n, fmt, args); }

namespace {

constexpr int kTensorArenaSize = 128 * 1024; /* Tensor arena size in bytes */

/* Tensor arena in OCRAM (outside DTCM) */
__attribute__((section(".bss.$SRAM_OC.tensor_arena")))
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];

/* Storage for the interpreter object, sized at compile time */
alignas(alignof(tflite::MicroInterpreter))
uint8_t g_interpreter_storage[sizeof(tflite::MicroInterpreter)];

static tflite::MicroInterpreter *g_interpreter = nullptr; /* null until first call */

} /* namespace */

/**
 * @brief  Return the index of the largest element in logits[0..n-1].
 * @param  logits  Pointer to float array.
 * @param  n       Number of elements.
 * @return Index of the maximum value; 0 if n <= 0.
 */
static int argmax_n(const float *logits, int n)
{
    if (n <= 0) return 0;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best])
            best = i;
    return best;
}

/**
 * @brief  Apply z-score normalisation to one sample using per-channel params.
 * @param  val  Raw sensor value.
 * @param  ch   Channel index (0-5).
 * @return Normalised value; uses std = 1 if norm_std[ch] is zero.
 */
static inline float zscore(float val, int ch)
{
    float s = (norm_std[ch] != 0.f) ? norm_std[ch] : 1.f;
    return (val - norm_mean[ch]) / s;
}

/**
 * @brief  Return the raw sensor value for channel c at time step t.
 * @param  ax,ay,az  Accelerometer arrays.
 * @param  gx,gy,gz  Gyroscope arrays.
 * @param  t         Time step index.
 * @param  c         Channel index (0=ax,1=ay,2=az,3=gx,4=gy,5=gz).
 * @return Raw float value.
 */
static inline float channel_val(
    const float *ax, const float *ay, const float *az,
    const float *gx, const float *gy, const float *gz,
    int t, int c)
{
    switch (c) {
        case 0: return ax[t];
        case 1: return ay[t];
        case 2: return az[t];
        case 3: return gx[t];
        case 4: return gy[t];
        case 5: return gz[t];
        default: return 0.f;
    }
}

/** @brief Destroy and reset the interpreter so it can be rebuilt on the next call. */
static void interpreter_destroy(void)
{
    if (g_interpreter != nullptr) {
        g_interpreter->~MicroInterpreter();
        g_interpreter = nullptr;
    }
}

extern "C" {

/* Returned on error; value is outside the valid range [0, NUM_CLASSES) */
#define PREDICT_CLASS_ERROR  ((uint8_t)NUM_CLASSES)

uint8_t predict_class(
    const float *ax, const float *ay, const float *az,
    const float *gx, const float *gy, const float *gz,
    size_t window_len_actual)
{
    if (window_len_actual == 0)
        return PREDICT_CLASS_ERROR;

    const tflite::Model *model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        PRINTF("Inference: model schema mismatch (%u vs %d)\r\n",
               model->version(), TFLITE_SCHEMA_VERSION);
        return PREDICT_CLASS_ERROR;
    }

    static bool resolver_initialized = false;
    static tflite::MicroMutableOpResolver<24> resolver;
    if (!resolver_initialized) {
        resolver.AddFullyConnected();
        resolver.AddSoftmax();
        resolver.AddReshape();
        resolver.AddQuantize();
        resolver.AddDequantize();
        resolver.AddRelu();
        resolver.AddMul();
        resolver.AddAdd();
        resolver.AddExpandDims();
        resolver.AddShape();
        resolver.AddStridedSlice();
        resolver.AddPack();
        resolver.AddTranspose();
        resolver.AddMean();
        resolver.AddSqueeze();
        resolver.AddConv2D();
        resolver.AddRelu6();
        resolver.AddAveragePool2D();
        resolver.AddMaxPool2D();
        resolver_initialized = true;
    }

    if (g_interpreter == nullptr) {
        tflite::InitializeTarget();
        g_interpreter = new (g_interpreter_storage) tflite::MicroInterpreter(
            model, resolver, g_tensor_arena, kTensorArenaSize);
        if (g_interpreter->AllocateTensors() != kTfLiteOk) {
            static unsigned alloc_fail_count;
            if (alloc_fail_count++ < 3)
                PRINTF("Inference: AllocateTensors failed\r\n");
            interpreter_destroy();
            return PREDICT_CLASS_ERROR;
        }
    }

    TfLiteTensor *input = g_interpreter->input(0);
    if (input == nullptr) {
        PRINTF("Inference: null input tensor\r\n");
        return PREDICT_CLASS_ERROR;
    }

#if MODEL_USE_INT8
    if (input->type != kTfLiteInt8) {
        PRINTF("Inference: expected int8 input, got type %d\r\n", input->type);
        return PREDICT_CLASS_ERROR;
    }
    int8_t *input_data = input->data.int8;
    float   in_scale   = input->params.scale;
    int32_t in_zp      = input->params.zero_point;
    if (in_scale == 0.f) {
        PRINTF("Inference: int8 input scale is zero\r\n");
        return PREDICT_CLASS_ERROR;
    }
#else
    if (input->type != kTfLiteFloat32) {
        PRINTF("Inference: expected float32 input, got type %d\r\n", input->type);
        return PREDICT_CLASS_ERROR;
    }
    float *input_data = input->data.f;
#endif

    /* Detect input layout: (1,T,C) time-first or (1,C,T) channel-first */
    int ndims        = input->dims->size;
    int input_time   = 0;
    int input_ch     = 0;
    bool channel_major = false;

    if (ndims >= 3) {
        int d1 = input->dims->data[1], d2 = input->dims->data[2];
        if (d1 == N_CHANNELS && d2 != N_CHANNELS) {
            channel_major = true;
            input_ch   = d1;
            input_time = d2;
        } else {
            input_time = d1;
            input_ch   = d2;
        }
    } else if (ndims == 2) {
        input_time = input->dims->data[0];
        input_ch   = input->dims->data[1];
    }

    if (input_time <= 0 || input_ch != N_CHANNELS ||
        (size_t)input_time > window_len_actual) {
        PRINTF("Inference: bad input shape [%d,%d,%d]\r\n",
               ndims > 0 ? input->dims->data[0] : 0,
               ndims > 1 ? input->dims->data[1] : 0,
               ndims > 2 ? input->dims->data[2] : 0);
        return PREDICT_CLASS_ERROR;
    }

    if (channel_major) {
        for (int t = 0; t < input_time; t++) {
            for (int c = 0; c < N_CHANNELS; c++) {
                float norm = zscore(channel_val(ax, ay, az, gx, gy, gz, t, c), c);
#if MODEL_USE_INT8
                int32_t q = (int32_t)roundf(norm / in_scale) + in_zp;
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                input_data[c * input_time + t] = (int8_t)q;
#else
                input_data[c * input_time + t] = norm;
#endif
            }
        }
    } else {
        for (int t = 0; t < input_time; t++) {
            for (int c = 0; c < N_CHANNELS; c++) {
                float norm = zscore(channel_val(ax, ay, az, gx, gy, gz, t, c), c);
#if MODEL_USE_INT8
                int32_t q = (int32_t)roundf(norm / in_scale) + in_zp;
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                input_data[t * N_CHANNELS + c] = (int8_t)q;
#else
                input_data[t * N_CHANNELS + c] = norm;
#endif
            }
        }
    }

    if (g_interpreter->Invoke() != kTfLiteOk) {
        static unsigned invoke_fail_count;
        if (invoke_fail_count++ < 3)
            PRINTF("Inference: Invoke failed\r\n");
        return PREDICT_CLASS_ERROR;
    }

    TfLiteTensor *output = g_interpreter->output(0);
    if (output == nullptr) {
        PRINTF("Inference: null output tensor\r\n");
        return PREDICT_CLASS_ERROR;
    }

    int out_elems = 1;
    for (int i = 0; i < output->dims->size; i++)
        out_elems *= output->dims->data[i];
    if (out_elems <= 0)
        return PREDICT_CLASS_ERROR;

    int n_classes = (out_elems < NUM_CLASSES) ? out_elems : NUM_CLASSES;

#if MODEL_USE_INT8
    if (output->type != kTfLiteInt8) {
        PRINTF("Inference: expected int8 output, got type %d\r\n", output->type);
        return PREDICT_CLASS_ERROR;
    }
    float   out_scale = output->params.scale;
    int32_t out_zp    = output->params.zero_point;
    float   logits[NUM_CLASSES];
    for (int i = 0; i < n_classes; i++)
        logits[i] = (output->data.int8[i] - out_zp) * out_scale;
    const float *logit_ptr = logits;
#else
    if (output->type != kTfLiteFloat32) {
        PRINTF("Inference: expected float32 output, got type %d\r\n", output->type);
        return PREDICT_CLASS_ERROR;
    }
    const float *logit_ptr = output->data.f;
#endif

    int idx = argmax_n(logit_ptr, n_classes);
    return (uint8_t)idx;
}

} /* extern "C" */
