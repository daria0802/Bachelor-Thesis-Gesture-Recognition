/**
 * @file    task_button.c
 * @brief   User button task – reads the button press, turns streaming on or off,
 *          and saves the log file when streaming stops.
 * @author  Daria Kinash
 * @date    2026
 */

#include "task_button.h"
#include "app_config.h"
#include "logger.h"

#include "fsl_gpio.h"
#include "fsl_clock.h"
#include "fsl_debug_console.h"

#include "FreeRTOS.h"
#include "task.h"

static volatile bool s_streaming = false; /* Shared with the MPU task */

bool stream_is_enabled(void)
{
    return s_streaming;
}

void ButtonTask(void *arg)
{
    (void)arg;
    CLOCK_EnableClock(kCLOCK_Gpio5);

    const gpio_pin_config_t inCfg = {kGPIO_DigitalInput, 0, kGPIO_NoIntmode};
    GPIO_PinInit(BUTTON_GPIO, BUTTON_PIN, &inCfg);

    bool wasReleased = true;

    for (;;)
    {
        const bool isReleased = (GPIO_PinRead(BUTTON_GPIO, BUTTON_PIN) != 0U);
        if (wasReleased && !isReleased)
        {
            s_streaming = !s_streaming;
            PRINTF("Gesture recognition: %s\r\n", s_streaming ? "STARTED" : "STOPPED");

            if (!s_streaming)
                logger_close();
        }
        wasReleased = isReleased;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}
