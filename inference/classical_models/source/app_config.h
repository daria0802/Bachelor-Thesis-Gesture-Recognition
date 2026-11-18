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

#endif /* APP_CONFIG_H */
