/**
 * @file    i2c.c
 * @brief   LPI2C1 master driver for the MIMXRT1052.
 * @author  Daria Kinash
 * @date    2026
 */

#include "i2c.h"
#include "fsl_device_registers.h"
#include "fsl_iomuxc.h"
#include "fsl_clock.h"
#include "app_config.h"

static bool s_i2cInitialized = false; /* Prevents calling init more than once */

/** @brief Configure GPIO_AD_B1_00/01 for LPI2C1 SCL/SDA with pull-ups. */
static void I2C_InitPins(void)
{
    IOMUXC_SetPinMux(IOMUXC_GPIO_AD_B1_00_LPI2C1_SCL, 1U);
    IOMUXC_SetPinMux(IOMUXC_GPIO_AD_B1_01_LPI2C1_SDA, 1U);

    uint32_t pad = IOMUXC_SW_PAD_CTL_PAD_DSE(6U) |
                   IOMUXC_SW_PAD_CTL_PAD_PKE_MASK |
                   IOMUXC_SW_PAD_CTL_PAD_PUE_MASK |
                   IOMUXC_SW_PAD_CTL_PAD_PUS(3U);
    IOMUXC_SetPinConfig(IOMUXC_GPIO_AD_B1_00_LPI2C1_SCL, pad);
    IOMUXC_SetPinConfig(IOMUXC_GPIO_AD_B1_01_LPI2C1_SDA, pad);
}

/**
 * @brief  Set LPI2C clock mux to USB1_PLL_PFD0, apply divider, and return
 *         the resulting source frequency.
 * @return LPI2C source clock in Hz.
 */
static uint32_t I2C_InitClock(void)
{
    CLOCK_SetMux(kCLOCK_Lpi2cMux, 0U);
    CLOCK_SetDiv(kCLOCK_Lpi2cDiv, 5U);
    CLOCK_EnableClock(kCLOCK_Lpi2c1);

    uint32_t freq = CLOCK_GetFreq(kCLOCK_Usb1PllPfd0Clk);
    uint32_t div  = CLOCK_GetDiv(kCLOCK_Lpi2cDiv) + 1U;
    return freq / (div ? div : 1U);
}

void I2C_Init(void)
{
    if (s_i2cInitialized)
        return;

    I2C_InitPins();

    uint32_t i2cClockHz = I2C_InitClock();

    lpi2c_master_config_t masterConfig;
    LPI2C_MasterGetDefaultConfig(&masterConfig);
    masterConfig.baudRate_Hz = I2C_BAUDRATE_HZ;
    masterConfig.enableDoze  = false;
    LPI2C_MasterInit(I2C_MASTER_BASEADDR, &masterConfig, i2cClockHz);

    s_i2cInitialized = true;
}

status_t I2C_Send(uint8_t slaveAddr, uint32_t regAddr, uint8_t regAddrSize,
                  const uint8_t *data, uint32_t dataSize)
{
    if (!s_i2cInitialized || !data || dataSize == 0)
        return kStatus_InvalidArgument;

    lpi2c_master_transfer_t xfer = {0};
    xfer.slaveAddress   = slaveAddr;
    xfer.direction      = kLPI2C_Write;
    xfer.subaddress     = regAddr;
    xfer.subaddressSize = regAddrSize;
    xfer.data           = (uint8_t *)data;
    xfer.dataSize       = dataSize;
    xfer.flags          = kLPI2C_TransferDefaultFlag;
    return LPI2C_MasterTransferBlocking(I2C_MASTER_BASEADDR, &xfer);
}

status_t I2C_Receive(uint8_t slaveAddr, uint32_t regAddr, uint8_t regAddrSize,
                     uint8_t *rxBuff, uint32_t rxSize)
{
    if (!s_i2cInitialized || !rxBuff || rxSize == 0)
        return kStatus_InvalidArgument;

    lpi2c_master_transfer_t xfer = {0};
    xfer.slaveAddress   = slaveAddr;
    xfer.direction      = kLPI2C_Read;
    xfer.subaddress     = regAddr;
    xfer.subaddressSize = regAddrSize;
    xfer.data           = rxBuff;
    xfer.dataSize       = rxSize;
    xfer.flags          = kLPI2C_TransferDefaultFlag;
    return LPI2C_MasterTransferBlocking(I2C_MASTER_BASEADDR, &xfer);
}

status_t I2C_WriteReg(uint8_t slaveAddr, uint32_t regAddr, uint8_t data)
{
    return I2C_Send(slaveAddr, regAddr, 1U, &data, 1U);
}

status_t I2C_ReadReg(uint8_t slaveAddr, uint32_t regAddr, uint8_t *data)
{
    return I2C_Receive(slaveAddr, regAddr, 1U, data, 1U);
}

bool I2C_IsBusy(void)
{
    if (!s_i2cInitialized)
        return false;
    return (LPI2C_MasterGetStatusFlags(I2C_MASTER_BASEADDR) & kLPI2C_MasterBusyFlag) != 0;
}

LPI2C_Type *I2C_GetBase(void)
{
    return I2C_MASTER_BASEADDR;
}
