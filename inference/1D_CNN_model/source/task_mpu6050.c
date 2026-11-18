/**
 * @file    task_mpu6050.c
 * @brief   MPU6050 acquisition and 1D-CNN inference task.
 *          Reads the sensor at 50 Hz, fills a sliding window of 100 samples
 *          (50% overlap), runs TFLite Micro inference, and prints the result.
 * @author  Daria Kinash
 * @date    2026
 */

#include "task_mpu6050.h"
#include "task_button.h"
#include "app_config.h"
#include "mpu6050.h"
#include "tflite_inference.h"
#include "logger.h"

#include "fsl_debug_console.h"
#include "FreeRTOS.h"
#include "task.h"

#include <inttypes.h>
#include <string.h>
#include <math.h>

/* DWT cycle counter registers (Cortex-M, ARMv7-M) */
#define DWT_CONTROL    (*((volatile uint32_t *)0xE0001000))
#define DWT_CYCCNTENA  (1UL << 0)
#define DWT_CYCCNT     (*((volatile uint32_t *)0xE0001004))
#define DWT_CPU_CLOCK  600000000UL

/** @brief Enable the DWT cycle counter. */
static inline void enable_cycle_counting(void)
{
    (*((volatile uint32_t *)0xE000EDFC)) |= (1UL << 24); /* set TRCENA in DEMCR */
    DWT_CONTROL |= DWT_CYCCNTENA;
    DWT_CYCCNT = 0;
}

/** @brief Return the current DWT cycle count. */
static inline uint32_t get_cycle_count(void)
{
    return DWT_CYCCNT;
}

/**
 * @brief  Convert a DWT cycle delta to microseconds.
 * @param  cycles  Number of CPU cycles.
 * @return Elapsed time in microseconds.
 */
static inline uint32_t cycles_to_us(uint32_t cycles)
{
    return (uint32_t)(((uint64_t)cycles * 1000000ULL) / (uint64_t)DWT_CPU_CLOCK);
}

/**
 * @brief  Print an error message and halt the system permanently.
 *         Called when the MPU6050 returns an I2C error — the sensor data
 *         stream is no longer valid and a power-cycle is required to recover.
 * @param  reason  Short description of the failure printed over UART.
 */
__attribute__((noreturn))
static void fatal_halt(const char *reason)
{
    PRINTF("FATAL: %s - halting\r\n", reason);
    taskDISABLE_INTERRUPTS();
    for (;;) {}
}

void MPU6050Task(void *arg)
{
    MPU6050Context *ctx = (MPU6050Context *)arg;

    (void)MPU6050_SetActive(false);

    const TickType_t period      = pdMS_TO_TICKS(1000 / MPU6050_ODR_HZ);
    TickType_t       nextWakeTime = xTaskGetTickCount();
    bool             lastStreaming = false;

    /* Per-channel buffers for the sliding window */
    static float ax_window[WINDOW];
    static float ay_window[WINDOW];
    static float az_window[WINDOW];
    static float gx_window[WINDOW];
    static float gy_window[WINDOW];
    static float gz_window[WINDOW];

    static size_t   window_idx         = 0;
    static uint32_t last_sample_cycles = 0; /* DWT count when the last sample entered the window */

    static uint32_t total_inferences     = 0;
    static uint32_t total_inference_time = 0;
    static uint32_t max_inference_time   = 0;
    static uint32_t total_latency        = 0;
    static uint32_t max_latency          = 0;

    memset(ax_window, 0, sizeof(ax_window));
    memset(ay_window, 0, sizeof(ay_window));
    memset(az_window, 0, sizeof(az_window));
    memset(gx_window, 0, sizeof(gx_window));
    memset(gy_window, 0, sizeof(gy_window));
    memset(gz_window, 0, sizeof(gz_window));

    enable_cycle_counting();

    PRINTF("WIN=%d STRIDE=%d ODR=%d Hz CLASSES=%d INT8=%d\r\n",
           WINDOW, STRIDE, MPU6050_ODR_HZ, NUM_CLASSES, MODEL_USE_INT8);

    for (;;)
    {
        const bool streaming = stream_is_enabled();
        if (streaming != lastStreaming)
        {
            (void)MPU6050_SetActive(streaming);
            lastStreaming = streaming;
            window_idx          = 0;
            last_predicted_class = 255;
        }

        if (streaming)
        {
            uint8_t  status = 0;
            status_t st     = MPU6050_ReadReg(&ctx->sensorHandle,
                                               MPU6050_REG_INT_STATUS,
                                               &status, 1);
            if (st != kStatus_Success)
                fatal_halt("MPU6050 INT_STATUS I2C read failed");

            if ((status & 0x01U) != 0U)
            {
                mpu6050_data_t d;
                if (MPU6050_ReadSensorData(&ctx->sensorHandle, &d) != kStatus_Success)
                    fatal_halt("MPU6050 sensor data I2C read failed");

                ax_window[window_idx] = MPU6050_AccelRawToG(d.accelX, ctx->accelFSR);
                ay_window[window_idx] = MPU6050_AccelRawToG(d.accelY, ctx->accelFSR);
                az_window[window_idx] = MPU6050_AccelRawToG(d.accelZ, ctx->accelFSR);
                gx_window[window_idx] = MPU6050_GyroRawToDps(d.gyroX, ctx->gyroFSR);
                gy_window[window_idx] = MPU6050_GyroRawToDps(d.gyroY, ctx->gyroFSR);
                gz_window[window_idx] = MPU6050_GyroRawToDps(d.gyroZ, ctx->gyroFSR);

                if (window_idx == WINDOW - 1)
                    last_sample_cycles = get_cycle_count();

                window_idx++;

                if (window_idx >= WINDOW)
                {
                    uint32_t t0 = get_cycle_count();

                    uint8_t predicted_class = predict_class(
                        ax_window, ay_window, az_window,
                        gx_window, gy_window, gz_window,
                        WINDOW);

                    uint32_t t1             = get_cycle_count();
                    uint32_t inference_time = cycles_to_us(t1 - t0);
                    uint32_t latency        = cycles_to_us(t1 - last_sample_cycles);

                    total_inferences++;
                    total_inference_time += inference_time;
                    if (inference_time > max_inference_time) max_inference_time = inference_time;
                    total_latency += latency;
                    if (latency > max_latency) max_latency = latency;

                    if (predicted_class != last_predicted_class)
                    {
                        const char *name = (predicted_class < NUM_CLASSES) ? class_names[predicted_class] : "?";
                        PRINTF("Gesture: %s | Inf: %u us | Lat: %u us\r\n", name, inference_time, latency);
                        last_predicted_class = predicted_class;
                    }

                    // (void)logger_write((uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS),
                    //                    predicted_class);

                    if (total_inferences % 10U == 0U)
                    {
                        uint32_t avg_inf = total_inference_time / total_inferences;
                        uint32_t avg_lat = total_latency        / total_inferences;
                        PRINTF("Stats: n=%u AvgInf=%u us MaxInf=%u us AvgLat=%u us MaxLat=%u us\r\n",
                               total_inferences, avg_inf, max_inference_time, avg_lat, max_latency);
                    }

                    if (STRIDE < WINDOW)
                    {
                        size_t overlap = WINDOW - STRIDE;
                        memmove(ax_window, &ax_window[STRIDE], overlap * sizeof(float));
                        memmove(ay_window, &ay_window[STRIDE], overlap * sizeof(float));
                        memmove(az_window, &az_window[STRIDE], overlap * sizeof(float));
                        memmove(gx_window, &gx_window[STRIDE], overlap * sizeof(float));
                        memmove(gy_window, &gy_window[STRIDE], overlap * sizeof(float));
                        memmove(gz_window, &gz_window[STRIDE], overlap * sizeof(float));
                        window_idx = overlap;
                    }
                    else
                    {
                        window_idx = 0;
                    }
                }
            }
        }

        vTaskDelayUntil(&nextWakeTime, period);
    }
}
