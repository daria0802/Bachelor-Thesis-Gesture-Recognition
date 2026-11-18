/**
 * @file    motion_features.c
 * @brief   IMU feature extraction: mean, MAD, min, max, and
 *          mean-centred zero-crossing rate for each sensor channel.
 * @author  Daria Kinash
 * @date    2026
 */

#include "motion_features.h"
#include <math.h>
#include <float.h>

/**
 * @brief  Arithmetic mean of an array.
 * @param  x     Input array.
 * @param  size  Number of elements.
 * @return Mean value; 0 if size == 0.
 */
static float smean(const float *x, size_t size)
{
    if (size == 0) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++)
        sum += x[i];
    return sum / (float)size;
}

/**
 * @brief  Mean absolute deviation from a pre-computed mean.
 * @param  x     Input array.
 * @param  size  Number of elements.
 * @param  mean  Pre-computed mean of x.
 * @return MAD value; 0 if size == 0.
 */
static float smad(const float *x, size_t size, float mean)
{
    if (size == 0) return 0.0f;
    float acc = 0.0f;
    for (size_t i = 0; i < size; i++)
        acc += fabsf(x[i] - mean);
    return acc / (float)size;
}

/**
 * @brief  Minimum value in an array.
 * @param  x     Input array.
 * @param  size  Number of elements.
 * @return Minimum; 0 if size == 0.
 */
static float smin(const float *x, size_t size)
{
    if (size == 0) return 0.0f;
    float v = FLT_MAX;
    for (size_t i = 0; i < size; i++)
        if (x[i] < v) v = x[i];
    return v;
}

/**
 * @brief  Maximum value in an array.
 * @param  x     Input array.
 * @param  size  Number of elements.
 * @return Maximum; 0 if size == 0.
 */
static float smax(const float *x, size_t size)
{
    if (size == 0) return 0.0f;
    float v = x[0];
    for (size_t i = 1; i < size; i++)
        if (x[i] > v) v = x[i];
    return v;
}

/**
 * @brief  Count sign changes of the mean-centred signal (ZCR about the mean).
 * @param  x     Input array.
 * @param  size  Number of elements.
 * @param  mean  Pre-computed mean of x.
 * @return Number of zero crossings; 0 if size < 2.
 */
static int szcr_mean_centered(const float *x, size_t size, float mean)
{
    if (size < 2) return 0;
    int count = 0;
    for (size_t i = 0; i < size - 1; i++)
    {
        float a = x[i]     - mean;
        float b = x[i + 1] - mean;
        if ((a >= 0.f && b < 0.f) || (a < 0.f && b >= 0.f))
            count++;
    }
    return count;
}

void motion_compute_features(
    const float *ax, const float *ay, const float *az,
    const float *gx, const float *gy, const float *gz,
    size_t window_size,
    float *output_features)
{
    const float *signals[6] = {ax, ay, az, gx, gy, gz};
    float means[6];

    for (int i = 0; i < 6; i++)
        means[i] = smean(signals[i], window_size);

    int idx = 0;
    for (int i = 0; i < 6; i++)
    {
        const float *s = signals[i];
        float        m = means[i];
        output_features[idx++] = m;
        output_features[idx++] = smad(s, window_size, m);
        output_features[idx++] = smin(s, window_size);
        output_features[idx++] = smax(s, window_size);
        output_features[idx++] = (float)szcr_mean_centered(s, window_size, m);
    }
}
