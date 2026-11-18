/**
 * @file    task_mpu6050.h
 * @brief   MPU6050 acquisition task – collects IMU samples, fills the
 *          sliding window, and triggers gesture inference.
 * @author  Daria Kinash
 * @date    2026
 */

#ifndef TASK_MPU6050_H
#define TASK_MPU6050_H

#include <stdint.h>
#include <mpu6050.h>

/** @brief Task argument passed from main.c to MPU6050Task(). */
typedef struct {
    mpu6050_handle_t sensorHandle; /* Initialised MPU6050 driver handle */
    uint8_t          accelFSR;     /* Accelerometer full-scale range */
    uint8_t          gyroFSR;      /* Gyroscope full-scale range */
} MPU6050Context;

#define MPU6050_ODR_HZ 50  /* Sensor output data rate used by the task (Hz) */

/** @brief FreeRTOS task: reads IMU at 50 Hz, runs inference. */
void MPU6050Task(void *arg);

#endif /* TASK_MPU6050_H */
