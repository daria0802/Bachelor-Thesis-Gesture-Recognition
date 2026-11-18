/**
 * @file    mpu6050.h
 * @brief   MPU6050 6-axis IMU driver: register map, data types,
 *          configuration constants, and public API.
 * @author  Daria Kinash
 * @date    2026
 */

#ifndef _FSL_MPU6050_H_
#define _FSL_MPU6050_H_

#include "fsl_common.h"

/* I2C addresses */
#define MPU6050_I2C_ADDRESS_AD0_LOW   0x68U  /* AD0 pin = GND */
#define MPU6050_I2C_ADDRESS_AD0_HIGH  0x69U  /* AD0 pin = VCC */

/* Register map */
#define MPU6050_REG_SELF_TEST_X         0x0D
#define MPU6050_REG_SELF_TEST_Y         0x0E
#define MPU6050_REG_SELF_TEST_Z         0x0F
#define MPU6050_REG_SELF_TEST_A         0x10
#define MPU6050_REG_SMPLRT_DIV          0x19
#define MPU6050_REG_CONFIG              0x1A
#define MPU6050_REG_GYRO_CONFIG         0x1B
#define MPU6050_REG_ACCEL_CONFIG        0x1C
#define MPU6050_REG_FF_THR              0x1D
#define MPU6050_REG_FF_DUR              0x1E
#define MPU6050_REG_MOT_THR             0x1F
#define MPU6050_REG_MOT_DUR             0x20
#define MPU6050_REG_ZRMOT_THR           0x21
#define MPU6050_REG_ZRMOT_DUR           0x22
#define MPU6050_REG_FIFO_EN             0x23
#define MPU6050_REG_I2C_MST_CTRL        0x24
#define MPU6050_REG_I2C_SLV0_ADDR       0x25
#define MPU6050_REG_I2C_SLV0_REG        0x26
#define MPU6050_REG_I2C_SLV0_CTRL       0x27
#define MPU6050_REG_I2C_SLV1_ADDR       0x28
#define MPU6050_REG_I2C_SLV1_REG        0x29
#define MPU6050_REG_I2C_SLV1_CTRL       0x2A
#define MPU6050_REG_I2C_SLV2_ADDR       0x2B
#define MPU6050_REG_I2C_SLV2_REG        0x2C
#define MPU6050_REG_I2C_SLV2_CTRL       0x2D
#define MPU6050_REG_I2C_SLV3_ADDR       0x2E
#define MPU6050_REG_I2C_SLV3_REG        0x2F
#define MPU6050_REG_I2C_SLV3_CTRL       0x30
#define MPU6050_REG_I2C_SLV4_ADDR       0x31
#define MPU6050_REG_I2C_SLV4_REG        0x32
#define MPU6050_REG_I2C_SLV4_DO         0x33
#define MPU6050_REG_I2C_SLV4_CTRL       0x34
#define MPU6050_REG_I2C_SLV4_DI         0x35
#define MPU6050_REG_I2C_MST_STATUS      0x36
#define MPU6050_REG_INT_PIN_CFG         0x37
#define MPU6050_REG_INT_ENABLE          0x38
#define MPU6050_REG_INT_STATUS          0x3A
#define MPU6050_REG_ACCEL_XOUT_H        0x3B
#define MPU6050_REG_ACCEL_XOUT_L        0x3C
#define MPU6050_REG_ACCEL_YOUT_H        0x3D
#define MPU6050_REG_ACCEL_YOUT_L        0x3E
#define MPU6050_REG_ACCEL_ZOUT_H        0x3F
#define MPU6050_REG_ACCEL_ZOUT_L        0x40
#define MPU6050_REG_TEMP_OUT_H          0x41
#define MPU6050_REG_TEMP_OUT_L          0x42
#define MPU6050_REG_GYRO_XOUT_H         0x43
#define MPU6050_REG_GYRO_XOUT_L         0x44
#define MPU6050_REG_GYRO_YOUT_H         0x45
#define MPU6050_REG_GYRO_YOUT_L         0x46
#define MPU6050_REG_GYRO_ZOUT_H         0x47
#define MPU6050_REG_GYRO_ZOUT_L         0x48
#define MPU6050_REG_EXT_SENS_DATA_00    0x49
#define MPU6050_REG_EXT_SENS_DATA_01    0x4A
#define MPU6050_REG_EXT_SENS_DATA_02    0x4B
#define MPU6050_REG_EXT_SENS_DATA_03    0x4C
#define MPU6050_REG_EXT_SENS_DATA_04    0x4D
#define MPU6050_REG_EXT_SENS_DATA_05    0x4E
#define MPU6050_REG_EXT_SENS_DATA_06    0x4F
#define MPU6050_REG_EXT_SENS_DATA_07    0x50
#define MPU6050_REG_EXT_SENS_DATA_08    0x51
#define MPU6050_REG_EXT_SENS_DATA_09    0x52
#define MPU6050_REG_EXT_SENS_DATA_10    0x53
#define MPU6050_REG_EXT_SENS_DATA_11    0x54
#define MPU6050_REG_EXT_SENS_DATA_12    0x55
#define MPU6050_REG_EXT_SENS_DATA_13    0x56
#define MPU6050_REG_EXT_SENS_DATA_14    0x57
#define MPU6050_REG_EXT_SENS_DATA_15    0x58
#define MPU6050_REG_EXT_SENS_DATA_16    0x59
#define MPU6050_REG_EXT_SENS_DATA_17    0x5A
#define MPU6050_REG_EXT_SENS_DATA_18    0x5B
#define MPU6050_REG_EXT_SENS_DATA_19    0x5C
#define MPU6050_REG_EXT_SENS_DATA_20    0x5D
#define MPU6050_REG_EXT_SENS_DATA_21    0x5E
#define MPU6050_REG_EXT_SENS_DATA_22    0x5F
#define MPU6050_REG_EXT_SENS_DATA_23    0x60
#define MPU6050_REG_I2C_SLV0_DO         0x63
#define MPU6050_REG_I2C_SLV1_DO         0x64
#define MPU6050_REG_I2C_SLV2_DO         0x65
#define MPU6050_REG_I2C_SLV3_DO         0x66
#define MPU6050_REG_I2C_MST_DELAY_CTRL  0x67
#define MPU6050_REG_SIGNAL_PATH_RESET   0x68
#define MPU6050_REG_MOT_DETECT_CTRL     0x69
#define MPU6050_REG_USER_CTRL           0x6A
#define MPU6050_REG_PWR_MGMT_1          0x6B
#define MPU6050_REG_PWR_MGMT_2          0x6C
#define MPU6050_REG_FIFO_COUNTH         0x72
#define MPU6050_REG_FIFO_COUNTL         0x73
#define MPU6050_REG_FIFO_R_W            0x74
#define MPU6050_REG_WHO_AM_I            0x75

#define MPU6050_WHO_AM_I_VALUE  0x68U  /* Expected response from WHO_AM_I register */

/* Clock source options for PWR_MGMT_1 CLKSEL field */
#define MPU6050_CLOCK_INTERNAL    0x00  /* Internal 8 MHz oscillator */
#define MPU6050_CLOCK_PLL_XGYRO   0x01  /* PLL with X gyro reference */
#define MPU6050_CLOCK_PLL_YGYRO   0x02
#define MPU6050_CLOCK_PLL_ZGYRO   0x03
#define MPU6050_CLOCK_PLL_EXT32K  0x04
#define MPU6050_CLOCK_PLL_EXT19M  0x05
#define MPU6050_CLOCK_KEEP_RESET  0x07

/* Gyroscope full-scale range (GYRO_CONFIG) */
#define MPU6050_GYRO_FS_250   0x00  /* ±250 °/s  */
#define MPU6050_GYRO_FS_500   0x01  /* ±500 °/s  */
#define MPU6050_GYRO_FS_1000  0x02  /* ±1000 °/s */
#define MPU6050_GYRO_FS_2000  0x03  /* ±2000 °/s */

/* Accelerometer full-scale range (ACCEL_CONFIG) */
#define MPU6050_ACCEL_FS_2   0x00  /* ±2 g  */
#define MPU6050_ACCEL_FS_4   0x01  /* ±4 g  */
#define MPU6050_ACCEL_FS_8   0x02  /* ±8 g  */
#define MPU6050_ACCEL_FS_16  0x03  /* ±16 g */

/* Digital Low-Pass Filter bandwidth (CONFIG) */
#define MPU6050_DLPF_BW_256  0x00  /* 256 Hz */
#define MPU6050_DLPF_BW_188  0x01  /* 188 Hz */
#define MPU6050_DLPF_BW_98   0x02  /* 98 Hz  */
#define MPU6050_DLPF_BW_42   0x03  /* 42 Hz  */
#define MPU6050_DLPF_BW_20   0x04  /* 20 Hz  */
#define MPU6050_DLPF_BW_10   0x05  /* 10 Hz  */
#define MPU6050_DLPF_BW_5    0x06  /* 5 Hz   */

/* PWR_MGMT_1 bit masks */
#define MPU6050_PWR1_DEVICE_RESET_BIT  0x80
#define MPU6050_PWR1_SLEEP_BIT         0x40
#define MPU6050_PWR1_CYCLE_BIT         0x20
#define MPU6050_PWR1_TEMP_DIS_BIT      0x08
#define MPU6050_PWR1_CLKSEL_BIT        0x07

/* Raw sensor output from one 14-byte burst read */
typedef struct _mpu6050_data
{
    int16_t accelX; /* Raw accelerometer X */
    int16_t accelY; /* Raw accelerometer Y */
    int16_t accelZ; /* Raw accelerometer Z */
    int16_t temp;   /* Raw temperature */
    int16_t gyroX;  /* Raw gyroscope X */
    int16_t gyroY;  /* Raw gyroscope Y */
    int16_t gyroZ;  /* Raw gyroscope Z */
} mpu6050_data_t;

/* I2C function pointer types used by the driver */
typedef status_t (*MPU6050_I2C_SendFunc_t)(uint8_t deviceAddress,
    uint32_t subAddress, uint8_t subaddressSize, uint32_t txBuff);
typedef status_t (*MPU6050_I2C_ReceiveFunc_t)(uint8_t deviceAddress,
    uint32_t subAddress, uint8_t subaddressSize, uint8_t *rxBuff, uint8_t rxBuffSize);

/* Runtime driver handle — created once and passed to all API calls */
typedef struct _mpu6050_handle
{
    MPU6050_I2C_SendFunc_t    I2C_SendFunc;
    MPU6050_I2C_ReceiveFunc_t I2C_ReceiveFunc;
    uint8_t                   slaveAddress;  /* 7-bit I2C address */
} mpu6050_handle_t;

/* Init parameters passed to MPU6050_Init() */
typedef struct _mpu6050_config
{
    MPU6050_I2C_SendFunc_t    I2C_SendFunc;
    MPU6050_I2C_ReceiveFunc_t I2C_ReceiveFunc;
    uint8_t                   slaveAddress;  /* 7-bit I2C address */
    uint8_t                   gyroFSR;       /* Gyro full-scale range */
    uint8_t                   accelFSR;      /* Accel full-scale range */
    uint8_t                   dlpfBW;        /* DLPF bandwidth setting */
} mpu6050_config_t;

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief  Init the MPU6050.
 * @param  handle  Driver handle to fill.
 * @param  config  Sensor configuration.
 * @return kStatus_Success or kStatus_Fail.
 */
status_t MPU6050_Init(mpu6050_handle_t *handle, mpu6050_config_t *config);

/**
 * @brief  Read all six axes in one burst.
 * @param  handle      Initialised driver handle.
 * @param  sensorData  Output buffer.
 * @return kStatus_Success or kStatus_Fail.
 */
status_t MPU6050_ReadSensorData(mpu6050_handle_t *handle, mpu6050_data_t *sensorData);

/**
 * @brief  Write one byte to a sensor register.
 * @param  handle  Initialised driver handle.
 * @param  reg     Register address.
 * @param  val     Byte to write.
 * @return kStatus_Success or kStatus_Fail.
 */
status_t MPU6050_WriteReg(mpu6050_handle_t *handle, uint8_t reg, uint8_t val);

/**
 * @brief  Read one or more bytes from a sensor register.
 * @param  handle       Initialised driver handle.
 * @param  reg          Start register address.
 * @param  val          Output buffer.
 * @param  bytesNumber  Number of bytes to read.
 * @return kStatus_Success or kStatus_Fail.
 */
status_t MPU6050_ReadReg(mpu6050_handle_t *handle, uint8_t reg, uint8_t *val, uint8_t bytesNumber);

/**
 * @brief  Set or clear the SLEEP bit in PWR_MGMT_1.
 * @param  handle  Initialised driver handle.
 * @param  sleep   true to sleep, false to wake.
 * @return kStatus_Success or kStatus_Fail.
 */
status_t MPU6050_SetSleep(mpu6050_handle_t *handle, bool sleep);

/**
 * @brief  Return accelerometer sensitivity in LSB/g for the given FSR.
 * @param  fsr  One of MPU6050_ACCEL_FS_* constants.
 * @return Sensitivity in LSB/g.
 */
static inline float MPU6050_GetAccelSensitivity(uint8_t fsr)
{
    switch (fsr)
    {
        case MPU6050_ACCEL_FS_2:  return 16384.0f;
        case MPU6050_ACCEL_FS_4:  return 8192.0f;
        case MPU6050_ACCEL_FS_8:  return 4096.0f;
        case MPU6050_ACCEL_FS_16: return 2048.0f;
        default:                  return 16384.0f;
    }
}

/**
 * @brief  Return gyroscope sensitivity in LSB/(°/s) for the given FSR.
 * @param  fsr  One of MPU6050_GYRO_FS_* constants.
 * @return Sensitivity in LSB/(°/s).
 */
static inline float MPU6050_GetGyroSensitivity(uint8_t fsr)
{
    switch (fsr)
    {
        case MPU6050_GYRO_FS_250:  return 131.0f;
        case MPU6050_GYRO_FS_500:  return 65.5f;
        case MPU6050_GYRO_FS_1000: return 32.8f;
        case MPU6050_GYRO_FS_2000: return 16.4f;
        default:                   return 131.0f;
    }
}

/**
 * @brief  Convert raw accelerometer value to g.
 * @param  raw  Raw 16-bit ADC value.
 * @param  fsr  Configured full-scale range.
 * @return Acceleration in g.
 */
static inline float MPU6050_AccelRawToG(int16_t raw, uint8_t fsr)
{
    return (float)raw / MPU6050_GetAccelSensitivity(fsr);
}

/**
 * @brief  Convert raw gyroscope value to degrees per second.
 * @param  raw  Raw 16-bit ADC value.
 * @param  fsr  Configured full-scale range.
 * @return Angular velocity in °/s.
 */
static inline float MPU6050_GyroRawToDps(int16_t raw, uint8_t fsr)
{
    return (float)raw / MPU6050_GetGyroSensitivity(fsr);
}

/**
 * @brief  Convert raw temperature value to degrees Celsius.
 * @param  raw  Raw 16-bit ADC value.
 * @return Temperature in °C.
 */
static inline float MPU6050_TempRawToCelsius(int16_t raw)
{
    return ((float)raw / 340.0f) + 36.53f;
}

/** @brief I2C write adapter for the MPU6050_I2C_SendFunc_t type. */
status_t MPU6050_I2C_Send(uint8_t deviceAddr, uint32_t reg,
                          uint8_t regSize, uint32_t value);

/** @brief I2C read adapter for the MPU6050_I2C_ReceiveFunc_t type. */
status_t MPU6050_I2C_Receive(uint8_t deviceAddr, uint32_t reg,
                             uint8_t regSize, uint8_t *rxData, uint8_t length);

/**
 * @brief  Wake or sleep the sensor via the generic I2C layer.
 * @param  enable  true to wake, false to sleep.
 * @return kStatus_Success or kStatus_Fail.
 */
status_t MPU6050_SetActive(bool enable);

#if defined(__cplusplus)
}
#endif

#endif /* _FSL_MPU6050_H_ */
