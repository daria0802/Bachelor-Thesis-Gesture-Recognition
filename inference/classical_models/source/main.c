/**
 * @file    main.c
 * @brief   System entry point for the classical ML gesture recognition firmware.
 *          Initialises board clocks, I2C, and the MPU6050, then starts
 *          three FreeRTOS tasks: StartupTask (logger), ButtonTask, MPU6050Task.
 * @author  Daria Kinash
 * @date    2026
 */

#include <stdio.h>
#include <stdbool.h>

#include "fsl_device_registers.h"
#include "board.h"
#include "pin_mux.h"
#include "clock_config.h"
#include "fsl_debug_console.h"

#include "FreeRTOS.h"
#include "task.h"

#include "app_config.h"
#include "i2c.h"
#include "mpu6050.h"
#include "logger.h"
#include "task_button.h"
#include "task_mpu6050.h"

/**
 * @brief  One-shot task that initialises the SD-card logger after the FreeRTOS scheduler starts.
 */
static void StartupTask(void *arg)
{
    (void)arg;
    if (logger_init() != kStatus_Success)
        PRINTF("Logger inactive (no SD card or mount failure)\r\n");
    vTaskDelete(NULL);
}

int main(void)
{
    BOARD_InitBootPins();
    BOARD_InitBootClocks();
    BOARD_InitDebugConsole();

    I2C_Init();

    mpu6050_config_t mpuCfg = {0};
    mpuCfg.I2C_SendFunc    = MPU6050_I2C_Send;
    mpuCfg.I2C_ReceiveFunc = MPU6050_I2C_Receive;
    mpuCfg.slaveAddress    = MPU6050_ADDR;
    mpuCfg.gyroFSR         = MPU6050_GYRO_FS_500;
    mpuCfg.accelFSR        = MPU6050_ACCEL_FS_8;
    mpuCfg.dlpfBW          = MPU6050_DLPF_BW_42;

    mpu6050_handle_t mpuHandle = {0};
    mpuHandle.I2C_SendFunc    = MPU6050_I2C_Send;
    mpuHandle.I2C_ReceiveFunc = MPU6050_I2C_Receive;
    mpuHandle.slaveAddress    = MPU6050_ADDR;

    if (MPU6050_Init(&mpuHandle, &mpuCfg) != kStatus_Success)
        PRINTF("MPU6050 init failed\r\n");

    static MPU6050Context mpuCtx;
    mpuCtx.sensorHandle = mpuHandle;
    mpuCtx.accelFSR     = MPU6050_ACCEL_FS_8;
    mpuCtx.gyroFSR      = MPU6050_GYRO_FS_500;

    PRINTF("System ready. Press button to start/stop gesture recognition.\r\n");

    BaseType_t rc;
    rc = xTaskCreate(StartupTask, "Startup", 1024,     NULL,    tskIDLE_PRIORITY + 3, NULL);
    if (rc != pdPASS) PRINTF("xTaskCreate(Startup) failed: %ld\r\n", (long)rc);
    rc = xTaskCreate(ButtonTask,  "Button",  256,      NULL,    tskIDLE_PRIORITY + 2, NULL);
    if (rc != pdPASS) PRINTF("xTaskCreate(Button)  failed: %ld\r\n", (long)rc);
    rc = xTaskCreate(MPU6050Task, "MPU6050", 4096, &mpuCtx, tskIDLE_PRIORITY + 1, NULL);
    if (rc != pdPASS) PRINTF("xTaskCreate(MPU6050) failed: %ld\r\n", (long)rc);

    vTaskStartScheduler();

    for (;;) {}
}
