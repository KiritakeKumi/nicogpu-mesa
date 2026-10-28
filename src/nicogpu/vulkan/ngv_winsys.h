/* SPDX-License-Identifier: MIT */
/*
 * ngv_winsys.h -- the winsys interface of the Nicogpu Vulkan driver "ngv".
 *
 * The driver (ngv_device.c, ngv_memory.c, ngv_sync.c, ngv_cmd_buffer.c) talks to
 * the device ONLY through `struct ngv_winsys_ops`.  The vtable is the C function
 * table of nicogpu_winsys.h (HAL v1): same call names, same
 * argument and error-code conventions (0 or a negative NICOGPU_ERR_* == -errno),
 * same types.  Two real backends implement it:
 *
 *   hal   forwards 1:1 to the nicogpu_winsys library of the functional model
 * (the functional model): the "sim" HAL backend today, the HAL "drm" backend if
 *         that is ever written.  Used by the x86 CTS farm.  ngv_winsys_hal.c
 *   drm   native C implementation on the nicogpu.ko render node (nicogpu_uapi.h,
 *         ABI v6, see NICOGPU_ABI_VERSION): GEM/VM_BIND/queue/syncobj/submit ioctls plus the winsys
 *         policy the kernel does not provide (GPU VA pool, handle tables,
 *         device-lost latch).  The production backend.  ngv_winsys_drm.c
 *
 * plus `stub` (ngv_winsys_stub.c): an in-process fake with no device behind it;
 * it exists so that the driver can be built, loaded and smoke-tested on a
 * machine that has neither the model nor the hardware.  It is only selected
 * explicitly (NGV_WINSYS=stub); a stub device never executes anything.
 *
 * Backend selection (ngv_winsys.c): $NGV_WINSYS = stub | hal | drm | auto
 * (default auto = drm, then hal; the stub is never auto-selected).
 *
 * All objects are u32 handles scoped to one device (BO / queue / syncobj), the
 * device pointer is opaque and backend private.
 *
 * Thread contract (batch 3): ANY op may be called concurrently from any
 * number of threads on one device (application threads, the Vulkan runtime's
 * submit thread); every backend serializes internally, the caller adds no
 * lock.  The only exception is device_destroy: no other call may be in
 * flight or start.  Locks are small and per backend (hal: one device mutex
 * around bookkeeping + HAL calls; drm: ioctls unlocked, small locks for the
 * latch / queue table / BO table; stub: one mutex), and none is held while
 * a call sleeps: syncobj_wait polls under the lock and sleeps outside it
 * (hal) or blocks in the kernel / on a condvar (drm / stub), so a waiter
 * never keeps a submit or a host signal from running.  The log hook is
 * called with a backend lock held in hal, so it must not call back into the
 * winsys.  ngv_device.mem_lock (ngv_memory.c) still orders the driver's own
 * BO bookkeeping; the winsys does not rely on it.
 */
#ifndef NGV_WINSYS_H
#define NGV_WINSYS_H

#include "nicogpu_winsys.h" /* synced from the functional model/include; includes nicogpu_hal.h */

#ifdef __cplusplus
extern "C" {
#endif

struct ngv_winsys_ops {
   const char *name; /* "stub", "hal", "drm" */

   /* Device enumeration; same contract as nicogpu_ws_enumerate (count = total). */
   int (*enumerate)(struct nicogpu_ws_device_desc *descs, uint32_t max_descs, uint32_t *count);

   int (*device_create)(const char *uri, const struct nicogpu_ws_device_create_args *args,
                        struct nicogpu_ws_device **out);
   void (*device_destroy)(struct nicogpu_ws_device *dev);
   int (*device_get_info)(struct nicogpu_ws_device *dev, struct nicogpu_device_info *info);
   int (*device_read_timestamp)(struct nicogpu_ws_device *dev, uint64_t *ts);
 /* winsys v2 : calibrated timestamp, see nicogpu_hal.h struct nicogpu_timestamp_info */
   int (*device_read_timestamp_ex)(struct nicogpu_ws_device *dev, struct nicogpu_timestamp_info *info);
   int (*device_is_lost)(struct nicogpu_ws_device *dev);
   int (*device_get_lost_info)(struct nicogpu_ws_device *dev, struct nicogpu_ws_lost_info *out);
   int (*set_log_fn)(struct nicogpu_ws_device *dev, nicogpu_ws_log_fn fn, void *user);

   int (*bo_alloc)(struct nicogpu_ws_device *dev, const struct nicogpu_ws_bo_args *args,
                   struct nicogpu_ws_bo_info *out);
   int (*bo_map)(struct nicogpu_ws_device *dev, uint32_t handle, void **ptr);
   int (*bo_unmap)(struct nicogpu_ws_device *dev, uint32_t handle);
   int (*bo_free)(struct nicogpu_ws_device *dev, uint32_t handle);

   int (*queue_create)(struct nicogpu_ws_device *dev, uint32_t *handle);
   int (*queue_destroy)(struct nicogpu_ws_device *dev, uint32_t handle);
   int (*queue_get_status)(struct nicogpu_ws_device *dev, uint32_t handle, uint32_t *status);
   int (*queue_get_fault)(struct nicogpu_ws_device *dev, uint32_t handle,
                          struct nicogpu_queue_fault_info *info);

   int (*syncobj_create)(struct nicogpu_ws_device *dev, uint32_t flags, uint32_t *handle);
   int (*syncobj_destroy)(struct nicogpu_ws_device *dev, uint32_t handle);
   int (*syncobj_reset)(struct nicogpu_ws_device *dev, const uint32_t *handles, uint32_t count);
 /* winsys v2 : host signal */
   int (*syncobj_signal)(struct nicogpu_ws_device *dev, const uint32_t *handles, uint32_t count);
   int (*syncobj_wait)(struct nicogpu_ws_device *dev, const uint32_t *handles, uint32_t count,
                       int64_t timeout_ns, uint32_t flags, uint32_t *first_signaled);

   int (*submit)(struct nicogpu_ws_device *dev, const struct nicogpu_ws_submit_args *args);

 /* : the DRM file descriptor of the device (owned by the winsys, valid until device_destroy), or -1 when the
    * backend has none (stub, hal) or the kernel has no DRM core syncobjs (ABI < 6).  Optional (NULL = -1).  The Mesa
    * runtime's vk_drm_syncobj creates and waits on syncobjs of this file; submit entries name them with
    * NICOGPU_WS_SYNC_F_DRM. */
   int (*drm_fd)(struct nicogpu_ws_device *dev);
};

/* A device opened through some backend. */
struct ngv_winsys {
   const struct ngv_winsys_ops *ops;
   struct nicogpu_ws_device *dev;
};

/* Backend tables (ngv_winsys_*.c). hal is NULL unless built with NGV_WITH_HAL. */
extern const struct ngv_winsys_ops ngv_winsys_stub_ops;
extern const struct ngv_winsys_ops ngv_winsys_drm_ops;
extern const struct ngv_winsys_ops ngv_winsys_hal_ops;

/* Backend chosen by $NGV_WINSYS / auto; NULL if the name is unknown or no
 * backend is compiled in for it. */
const struct ngv_winsys_ops *ngv_winsys_select(void);

/* Maps a (negative) NICOGPU_ERR_* to the Vulkan result for the operation class. */
#include <vulkan/vulkan_core.h>
VkResult ngv_winsys_result(int err);

#ifdef __cplusplus
}
#endif
#endif /* NGV_WINSYS_H */
