/**
 * @file    tflite_inference.h
 * @brief   TensorFlow Lite Micro inference interface.
 * @author  Daria Kinash
 * @date    2026
 */

#ifndef TFLITE_INFERENCE_H
#define TFLITE_INFERENCE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Run 1D-CNN inference on one sliding window of IMU data.
 *
 * The function normalises each channel with pre-computed z-score parameters,
 * fills the model input tensor, invokes the interpreter, and returns argmax
 * over the softmax output.
 *
 * Quantisation mode is selected at build time via MODEL_USE_INT8 in app_config.h:
 *  - 0: float32 input/output
 *  - 1: int8 quantised input/output
 *
 * @param  ax               Accelerometer X array.
 * @param  ay               Accelerometer Y array.
 * @param  az               Accelerometer Z array.
 * @param  gx               Gyroscope X array.
 * @param  gy               Gyroscope Y array.
 * @param  gz               Gyroscope Z array.
 * @param  window_len_actual Number of valid samples in each array.
 * @return Predicted class index (0-based), or NUM_CLASSES on any error.
 */
uint8_t predict_class(
    const float *ax, const float *ay, const float *az,
    const float *gx, const float *gy, const float *gz,
    size_t window_len_actual
);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_INFERENCE_H */
