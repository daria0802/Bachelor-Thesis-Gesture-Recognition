/**
 * @file    i2c.h
 * @brief   LPI2C1 master driver interface for the MIMXRT1052.
 * @author  Daria Kinash
 * @date    2026
 */

#ifndef I2C_H
#define I2C_H

#include <stdbool.h>
#include <stdint.h>
#include "fsl_common.h"
#include "fsl_lpi2c.h"

#define I2C_MASTER_BASEADDR LPI2C1  /* LPI2C1 peripheral */

/** @brief Initialise LPI2C1 pin mux, clock, and master configuration. */
void I2C_Init(void);

/**
 * @brief  Write bytes to a slave device register.
 * @param  slaveAddr    7-bit I2C slave address.
 * @param  regAddr      Target register address.
 * @param  regAddrSize  Register address width in bytes (usually 1).
 * @param  data         Data to write.
 * @param  dataSize     Number of bytes to write.
 * @return kStatus_Success or an SDK error code.
 */
status_t I2C_Send(uint8_t slaveAddr, uint32_t regAddr, uint8_t regAddrSize,
                  const uint8_t *data, uint32_t dataSize);

/**
 * @brief  Read bytes from a slave device register.
 * @param  slaveAddr    7-bit I2C slave address.
 * @param  regAddr      Source register address.
 * @param  regAddrSize  Register address width in bytes (usually 1).
 * @param  rxBuff       Receive buffer.
 * @param  rxSize       Number of bytes to read.
 * @return kStatus_Success or an SDK error code.
 */
status_t I2C_Receive(uint8_t slaveAddr, uint32_t regAddr, uint8_t regAddrSize,
                     uint8_t *rxBuff, uint32_t rxSize);

/**
 * @brief  Write a single byte to one register.
 * @param  slaveAddr  7-bit I2C slave address.
 * @param  regAddr    Register address.
 * @param  data       Byte value to write.
 * @return kStatus_Success or an SDK error code.
 */
status_t I2C_WriteReg(uint8_t slaveAddr, uint32_t regAddr, uint8_t data);

/**
 * @brief  Read a single byte from one register.
 * @param  slaveAddr  7-bit I2C slave address.
 * @param  regAddr    Register address.
 * @param  data       Pointer to receive the byte.
 * @return kStatus_Success or an SDK error code.
 */
status_t I2C_ReadReg(uint8_t slaveAddr, uint32_t regAddr, uint8_t *data);

/**
 * @brief  Check whether a transfer is currently in progress.
 * @return true if the bus is busy.
 */
bool I2C_IsBusy(void);

/**
 * @brief  Return the LPI2C1 hardware peripheral pointer.
 * @return Pointer to LPI2C_Type.
 */
LPI2C_Type *I2C_GetBase(void);

#endif /* I2C_H */
