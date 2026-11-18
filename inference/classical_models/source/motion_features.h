/**
 * @file    motion_features.h
 * @brief   Feature extraction for IMU gesture recognition. Computes 5 features per channel.
 * @author  Daria Kinash
 * @date    2026
 */

#ifndef MOTION_FEATURES_H
#define MOTION_FEATURES_H

#include <stdint.h>
#include <stddef.h>

#define MOTION_N_FEATS     30
#define MOTION_WINDOW_SIZE 128

/**
 * @brief  Compute the feature vector for one sliding window.
 * @param  ax,ay,az        Accelerometer arrays (g).
 * @param  gx,gy,gz        Gyroscope arrays (°/s).
 * @param  window_size     Number of valid samples in each array.
 * @param  output_features Output buffer.
 */
void motion_compute_features(
    const float *ax, const float *ay, const float *az,
    const float *gx, const float *gy, const float *gz,
    size_t window_size,
    float *output_features
);

#endif /* MOTION_FEATURES_H */
