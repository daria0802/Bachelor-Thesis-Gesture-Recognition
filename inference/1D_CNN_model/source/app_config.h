/**
 * @file    app_config.h
 * @brief   Application constants.
 * @author  Daria Kinash
 * @date    2026
 */

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <stdint.h>

#define MPU6050_ADDR    (0x68)    /* MPU6050 7-bit I2C address (AD0 = GND). */
#define I2C_BAUDRATE_HZ (400000U) /* LPI2C1 target baud rate in Hz.         */

#define BUTTON_GPIO  GPIO5 /* GPIO port of the user push-button. */
#define BUTTON_PIN   0U    /* Pin index of the user push-button.  */

#define N_CHANNELS  6   /* IMU channels: ax, ay, az, gx, gy, gz.       */
#define WINDOW      100 /* Sliding window length in samples.             */
#define STRIDE      50  /* Hop size between consecutive windows.         */
#define NUM_CLASSES 5   /* Number of output gesture classes.             */

/* Model input/output type: 0 = float32, 1 = int8 (quantised).
 * Must match the model in model/model_data.h. */
#define MODEL_USE_INT8  1

/* Class names in the same order as model output indices */
static const char *class_names[NUM_CLASSES] = {
    "idle", "accept", "left", "right", "cancel"
};

/* Stores the last printed class; 255 means no prediction yet */
static uint8_t last_predicted_class = 255;

#endif /* APP_CONFIG_H */
