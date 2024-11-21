/**
 * @file  cuda_timing.h
 * @author Erik Fabrizzi
 * @date 19.11.2024
 * @brief Defines macros to handle timing of cuda kernels.
 */ 

#pragma once

#ifdef CUDA_TIMING

typedef char time_event;

/**
 * @brief Defines a cuda compatible timing event.
 *   
 * @param event_name Variable name of the event.
 * 
 * @note This macro defines 3 variables:
 *      cudaEvent_t start_<event_name> , stop_<event_name>;
 *      time_event(alias for char) <event_name>.
 *      The ladder is needed for suggestion by intellisense.
 */
#define TIME_EVENT_DEFINE(event_name)\
    static_assert(sizeof(#event_name) > 1, "TIME_EVENT_CREATE requires an argument."); \
    volatile time_event event_name;                                                             \
    cudaEvent_t start_##event_name, stop_##event_name;                                 \

/**
 * @brief Initializes a timing event.
 *   
 * @param event_name Variable name of the event.
 * 
 * @note This macro calls cudaEventCreate on the events defined 
 *       by TIME_EVENT_DEFINE.
 */
#define TIME_EVENT_CREATE(event_name)\
    cudaEventCreate(&start_##event_name);                                              \
    cudaEventCreate(&stop_##event_name);

/**
 * @brief Starts event recording.
 *   
 * @param event_name Variable name of the event.
 * 
*/
#define TIME_START(event_name) \
    cudaEventRecord(start_##event_name, 0);

/**
 * @brief Stops event recording and saves it to a float.
 *   
 * @param event_name Variable name of the event.
 * @param timing_var Variable that hosts elapsed time, needs to be a float.
 * 
*/
#define TIME_STOP_SAVE(event_name, timing_var) \
    cudaEventRecord(stop_##event_name, 0);     \
    cudaEventSynchronize(stop_##event_name);   \
    cudaEventElapsedTime(&timing_var, start_##event_name, stop_##event_name);

/**
 * @brief Destroyes timing event.
 *   
 * @param event_name Variable name of the event.
 * @param timing_var Variable that hosts elapsed time, needs to be a float.
 * 
*/
#define TIME_EVENT_DESTROY(event_name) cudaEventDestroy(start_##event_name); cudaEventDestroy(stop_##event_name);

#else

#define TIME_EVENT_DEFINE(event_name)
#define TIME_EVENT_CREATE(event_name)
#define TIME_START(event_name)
#define TIME_STOP_SAVE(event_name, timing_var)
#define TIME_EVENT_DESTROY(event_name)

#endif
