/**
 * @file    logger.h
 * @brief   SD-card logger – records (timestamp_ms, predicted_class) rows to file using FATFS.
 * @author  Daria Kinash
 * @date    2026
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <stdint.h>
#include "fsl_common.h"
#include "sdmmc_config.h"
#include <stdio.h>

/**
 * @brief  Mount the SD card and open (or create) file.
 * @return kStatus_Success on success, kStatus_Fail on any FATFS error.
 */
status_t logger_init(void);

/**
 * @brief  Append one prediction record to the open CSV file.
 * @param  tMs             Timestamp in milliseconds (from xTaskGetTickCount).
 * @param  predicted_class Predicted gesture class index.
 * @return kStatus_Success on success, kStatus_Fail on write error.
 */
status_t logger_write(uint32_t tMs, uint8_t predicted_class);

/** @brief Flush and close the log file. */
void logger_close(void);

#endif /* LOGGER_H */
