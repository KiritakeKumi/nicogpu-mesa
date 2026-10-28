/* SPDX-License-Identifier: MIT */
/* ngv winsys backend "hal": forwards to the nicogpu_winsys library of the
 * functional model (the functional model, on the HAL "sim" backend)..
 *
 * Compiled only with -DNGV_WITH_HAL. The ICD links the model libraries when
 * halprebuilt/ supplies them; see vulkan/meson.build. Their binaries come
 * from the model's CMake build. The vtable entries are the
 * library functions themselves: this file adds no logic, which is the whole
 * point -- the model's winsys is the reference for the semantics the other
 * backends must match.
 */
#ifdef NGV_WITH_HAL
#include "ngv_winsys.h"

const struct ngv_winsys_ops ngv_winsys_hal_ops = {
   .name = "hal",
   .enumerate = nicogpu_ws_enumerate,
   .device_create = nicogpu_ws_device_create,
   .device_destroy = nicogpu_ws_device_destroy,
   .device_get_info = nicogpu_ws_device_get_info,
   .device_read_timestamp = nicogpu_ws_device_read_timestamp,
   .device_read_timestamp_ex = nicogpu_ws_device_read_timestamp_ex,
   .device_is_lost = nicogpu_ws_device_is_lost,
   .device_get_lost_info = nicogpu_ws_device_get_lost_info,
   .set_log_fn = nicogpu_ws_set_log_fn,
   .bo_alloc = nicogpu_ws_bo_alloc,
   .bo_map = nicogpu_ws_bo_map,
   .bo_unmap = nicogpu_ws_bo_unmap,
   .bo_free = nicogpu_ws_bo_free,
   .queue_create = nicogpu_ws_queue_create,
   .queue_destroy = nicogpu_ws_queue_destroy,
   .queue_get_status = nicogpu_ws_queue_get_status,
   .queue_get_fault = nicogpu_ws_queue_get_fault,
   .syncobj_create = nicogpu_ws_syncobj_create,
   .syncobj_destroy = nicogpu_ws_syncobj_destroy,
   .syncobj_reset = nicogpu_ws_syncobj_reset,
   .syncobj_signal = nicogpu_ws_syncobj_signal,
   .syncobj_wait = nicogpu_ws_syncobj_wait,
   .submit = nicogpu_ws_submit,
};
#endif /* NGV_WITH_HAL */
