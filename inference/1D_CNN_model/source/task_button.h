/**
 * @file    task_button.h
 * @brief   User button FreeRTOS task – controls gesture streaming state.
 * @author  Daria Kinash
 * @date    2026
 */

#ifndef TASK_BUTTON_H
#define TASK_BUTTON_H

#include <stdbool.h>

/** @brief FreeRTOS task that polls the button and toggles the streaming flag. */
void ButtonTask(void *arg);

/**
 * @brief  Return the current streaming state.
 * @return true if gesture acquisition is enabled.
 */
bool stream_is_enabled(void);

#endif /* TASK_BUTTON_H */
