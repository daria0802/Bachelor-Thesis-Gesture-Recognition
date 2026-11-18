/**
 * @file    mpu6050.c
 * @brief   MPU6050 driver implementation.
 * @author  Daria Kinash
 * @date    2026
 */

#include <mpu6050.h>
#include "i2c.h"

status_t MPU6050_Init(mpu6050_handle_t *handle, mpu6050_config_t *config)
{
    assert(handle);
    assert(config);
    assert(config->I2C_SendFunc);
    assert(config->I2C_ReceiveFunc);

    uint8_t tmp[1] = {0};

    handle->I2C_SendFunc    = config->I2C_SendFunc;
    handle->I2C_ReceiveFunc = config->I2C_ReceiveFunc;
    handle->slaveAddress = config->slaveAddress;

    /* Check WHO_AM_I register */
    if (MPU6050_ReadReg(handle, MPU6050_REG_WHO_AM_I, tmp, 1) != kStatus_Success)
    {
        return kStatus_Fail;
    }

    if (tmp[0] != MPU6050_WHO_AM_I_VALUE)
    {
        return kStatus_Fail;
    }

    /* Wake up the sensor */
    if (MPU6050_WriteReg(handle, MPU6050_REG_PWR_MGMT_1, MPU6050_CLOCK_PLL_XGYRO) != kStatus_Success)
    {
        return kStatus_Fail;
    }

    /* Wait for sensor to stabilize */
    volatile uint32_t delay = 100000;
    while (delay--)
    {
        __asm("nop");
    }

    /* Configure gyroscope full scale range */
    if (MPU6050_WriteReg(handle, MPU6050_REG_GYRO_CONFIG, config->gyroFSR << 3) != kStatus_Success)
    {
        return kStatus_Fail;
    }

    /* Configure accelerometer full scale range */
    if (MPU6050_WriteReg(handle, MPU6050_REG_ACCEL_CONFIG, config->accelFSR << 3) != kStatus_Success)
    {
        return kStatus_Fail;
    }

    /* Configure digital low pass filter */
    if (MPU6050_WriteReg(handle, MPU6050_REG_CONFIG, config->dlpfBW) != kStatus_Success)
    {
        return kStatus_Fail;
    }

    if (MPU6050_WriteReg(handle, MPU6050_REG_SMPLRT_DIV, 19) != kStatus_Success)
	{
		return kStatus_Fail;
	}

    /* Disable FIFO */
    if (MPU6050_WriteReg(handle, MPU6050_REG_FIFO_EN, 0x00) != kStatus_Success)
    {
        return kStatus_Fail;
    }

    if (MPU6050_WriteReg(handle, MPU6050_REG_INT_ENABLE, 0x01) != kStatus_Success)
    {
        return kStatus_Fail;
    }

    return kStatus_Success;
}

status_t MPU6050_ReadSensorData(mpu6050_handle_t *handle, mpu6050_data_t *sensorData)
{
    assert(handle);
    assert(sensorData);

    uint8_t buffer[14];

    if (MPU6050_ReadReg(handle, MPU6050_REG_ACCEL_XOUT_H, buffer, 14) != kStatus_Success)
    {
        return kStatus_Fail;
    }

    sensorData->accelX = (int16_t)((buffer[0] << 8) | buffer[1]);
    sensorData->accelY = (int16_t)((buffer[2] << 8) | buffer[3]);
    sensorData->accelZ = (int16_t)((buffer[4] << 8) | buffer[5]);

    sensorData->temp = (int16_t)((buffer[6] << 8) | buffer[7]);

    sensorData->gyroX = (int16_t)((buffer[8] << 8) | buffer[9]);
    sensorData->gyroY = (int16_t)((buffer[10] << 8) | buffer[11]);
    sensorData->gyroZ = (int16_t)((buffer[12] << 8) | buffer[13]);

    return kStatus_Success;
}

status_t MPU6050_WriteReg(mpu6050_handle_t *handle, uint8_t reg, uint8_t val)
{
    assert(handle);
    assert(handle->I2C_SendFunc);

    return handle->I2C_SendFunc(handle->slaveAddress, reg, 1U, val);
}

status_t MPU6050_ReadReg(mpu6050_handle_t *handle, uint8_t reg, uint8_t *val, uint8_t bytesNumber)
{
    assert(handle);
    assert(handle->I2C_ReceiveFunc);
    assert(val);

    return handle->I2C_ReceiveFunc(handle->slaveAddress, reg, 1U, val, bytesNumber);
}

status_t MPU6050_SetSleep(mpu6050_handle_t *handle, bool sleep)
{
    assert(handle);

    uint8_t reg = 0;

    if (MPU6050_ReadReg(handle, MPU6050_REG_PWR_MGMT_1, &reg, 1) != kStatus_Success)
    {
        return kStatus_Fail;
    }

    if (sleep)
    {
        reg |= MPU6050_PWR1_SLEEP_BIT;
    }
    else
    {
        reg &= ~MPU6050_PWR1_SLEEP_BIT;
        reg &= ~MPU6050_PWR1_CLKSEL_BIT;
        reg |= MPU6050_CLOCK_PLL_XGYRO;
    }

    return MPU6050_WriteReg(handle, MPU6050_REG_PWR_MGMT_1, reg);
}

status_t MPU6050_I2C_Send(uint8_t deviceAddr, uint32_t reg, uint8_t regSize, uint32_t value)
{
    uint8_t data = (uint8_t)value;
    return I2C_Send(deviceAddr, reg, regSize, &data, 1U);
}

status_t MPU6050_I2C_Receive(uint8_t deviceAddr, uint32_t reg, uint8_t regSize, uint8_t *rxData, uint8_t length)
{
    return I2C_Receive(deviceAddr, reg, regSize, rxData, length);
}

status_t MPU6050_SetActive(bool enable)
{
    uint8_t reg = 0;
    status_t st = I2C_ReadReg(0x68, 0x6B, &reg);
    if (st != kStatus_Success) return st;
    if (enable) reg &= ~0x40U; else reg |= 0x40U;
    return I2C_WriteReg(0x68, 0x6B, reg);
}
