/**
 * @file    logger.c
 * @brief   SD-card logger implementation using FATFS.
 * @author  Daria Kinash
 * @date    2026
 */

#include "logger.h"
#include "fsl_debug_console.h"
#include "ff.h"
#include "diskio.h"
#include "fsl_sd_disk.h"
#include "fsl_sd.h"
#include "FreeRTOS.h"
#include "semphr.h"

#include <string.h>

extern sd_card_t g_sd;

static FATFS s_fileSystem;
static FIL   s_fileObject;
static bool  s_isMounted = false; /* true after f_mount succeeds */
static bool  s_isOpen    = false; /* true after f_open succeeds  */

BYTE work[FF_MAX_SS]; /* Work buffer for f_mkfs */

/**
 * @brief  Check SD card is inserted, init the host, and power-cycle the card.
 * @return kStatus_Success if card is ready, kStatus_Fail otherwise.
 */
static status_t sdcardWaitCardInsert(void)
{
    BOARD_SD_Config(&g_sd, NULL, BOARD_SDMMC_SD_HOST_IRQ_PRIORITY, NULL);

    if (SD_HostInit(&g_sd) != kStatus_Success)
    {
        PRINTF("\r\nSD host init fail\r\n");
        return kStatus_Fail;
    }

    if (SD_PollingCardInsert(&g_sd, kSD_Inserted) == kStatus_Success)
    {
        SD_SetCardPower(&g_sd, false);
        SD_SetCardPower(&g_sd, true);
    }
    else
    {
        PRINTF("\r\nCard detect fail.\r\n");
        return kStatus_Fail;
    }

    return kStatus_Success;
}

status_t logger_init(void)
{
    if (s_isOpen)
        return kStatus_Success;

    if (sdcardWaitCardInsert() != kStatus_Success)
    {
        PRINTF("SD card not ready.\r\n");
        return kStatus_Fail;
    }

    const TCHAR driverNumberBuffer[3U] = { SDDISK + '0', ':', '/' };
    FRESULT error;

    error = f_mount(&s_fileSystem, driverNumberBuffer, 0U);
    if (error)
    {
        PRINTF("Mount volume failed: %d\r\n", error);
        return kStatus_Fail;
    }
    s_isMounted = true;

#if (FF_FS_RPATH >= 2U)
    error = f_chdrive((char const *)&driverNumberBuffer[0U]);
    if (error)
    {
        PRINTF("Change drive failed.\r\n");
        return kStatus_Fail;
    }
#endif

#if FF_USE_MKFS
    if (f_mkfs(driverNumberBuffer, 0, work, sizeof work))
    {
        PRINTF("Make file system failed.\r\n");
        return kStatus_Fail;
    }
#endif

    error = f_mkdir(_T("/data"));
    if (error != FR_OK && error != FR_EXIST)
    {
        PRINTF("Create dir failed: %d\r\n", error);
        return kStatus_Fail;
    }

    error = f_open(&s_fileObject, _T("/data/preds.csv"),
                   FA_OPEN_APPEND | FA_WRITE);
    if (error)
    {
        PRINTF("Open file failed: %d\r\n", error);
        return kStatus_Fail;
    }

    s_isOpen = true;

    if (f_size(&s_fileObject) == 0)
    {
        const char *hdr = "timestamp_ms,predicted_class\r\n";
        UINT bw;
        f_write(&s_fileObject, hdr, strlen(hdr), &bw);
        f_sync(&s_fileObject);
    }

    f_lseek(&s_fileObject, f_size(&s_fileObject));
    return kStatus_Success;
}

status_t logger_write(uint32_t tMs, uint8_t predicted_class)
{
    if (!s_isMounted || !s_isOpen)
        return kStatus_Fail;

    char line[32];
    int len = snprintf(line, sizeof(line), "%lu,%u\r\n",
                       (unsigned long)tMs, (unsigned)predicted_class);
    if (len <= 0 || len >= (int)sizeof(line))
        return kStatus_Fail;

    UINT bw;
    FRESULT res = f_write(&s_fileObject, line, len, &bw);
    if (res != FR_OK || bw != (UINT)len)
    {
        PRINTF("LOGGER: Write error (%d)\r\n", res);
        return kStatus_Fail;
    }

    f_sync(&s_fileObject);
    return kStatus_Success;
}

void logger_close(void)
{
    if (s_isOpen)
    {
        f_sync(&s_fileObject);
        f_close(&s_fileObject);
        s_isOpen = false;
    }

    if (s_isMounted)
    {
        f_mount(NULL, "0:", 1);
        s_isMounted = false;
    }

    PRINTF("Logger deinitialized.\r\n");
}
