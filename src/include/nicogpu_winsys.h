/* SPDX-License-Identifier: MIT */
/*
 * nicogpu_winsys.h -- Nicogpu winsys, v0.
 *
 * The device-side library the future Mesa driver ("ngv") links
 * against.  It is a policy layer over the frozen HAL v0 (nicogpu_hal.h) and adds
 * exactly what a Vulkan driver needs beyond raw HAL calls:
 *
 *   - device enumeration (drmGetDevices2-shaped) and one-step device setup:
 *     open + query info + create the default VM;
 *   - a GPU-VA allocator and an "allocate == bind" BO lifecycle, one call per
 *     VkDeviceMemory (free = UNMAP + GEM close + VA release);
 *   - winsys-scoped u32 handles for BOs/queues/syncobjs plus ordered teardown
 *     of anything a client forgot to destroy;
 *   - submit with syncobj handle translation and post-submit fault polling
 *     (sticky device-lost, the source of VK_ERROR_DEVICE_LOST);
 *   - an optional per-device log hook (NICOGPU_WINSYS_LOG=1|2 logs to stderr).
 *
 * Style follows nicogpu_hal.h: C ABI, every call returns 0 or a negative
 * NICOGPU_ERR_*, thread-safe per device (see "Thread contract" below).
 * Error codes, struct nicogpu_device_info and the BO/VM constants are shared
 * with the HAL on purpose: nicogpu_hal.h stays the single device vocabulary and
 * the winsys never invents its own device facts.  Moving from the model to
 * real hardware swaps the HAL backend ("sim" -> "drm"); this header, the
 * winsys and the Mesa driver stay unchanged.
 *
 * PROVISIONAL decisions are tracked in the winsys design notes.
 *
 * Thread contract (batch 3). Every call except nicogpu_ws_device_destroy
 * may be made concurrently from any number of threads on one device; the
 * library serializes internally (one device mutex around the handle tables,
 * the VA pool, the device-lost latch and the HAL call), the caller adds no
 * lock.  Calls are atomic with respect to each other.  Nothing sleeps with
 * the lock held: nicogpu_ws_syncobj_wait polls the HAL with timeout 0 under the
 * lock and sleeps (growing backoff, capped at 1 ms, never past the deadline)
 * outside it, so a blocked waiter never stalls a submit or a signal on
 * another thread (wait-before-signal across threads works).  The log hook
 * runs with the device lock held and must not call back into the winsys.
 * nicogpu_ws_device_destroy needs exclusive access: no other call may be in
 * flight or start afterwards.  Handles of one device must not be used on
 * another.  Objects destroyed while another thread still uses them yield
 * NICOGPU_ERR_NOENT, never undefined behavior (a wait on a syncobj that is
 * destroyed meanwhile ends with NICOGPU_ERR_NOENT). */
#ifndef NICOGPU_WINSYS_H
#define NICOGPU_WINSYS_H

#include <stdint.h>

/* nicogpu_hal.h supplies the error codes (NICOGPU_OK / NICOGPU_ERR_*), struct
 * nicogpu_device_info, the NICOGPU_BO_DOMAIN_ and NICOGPU_BO_HOST_VISIBLE constants,
 * NICOGPU_VM_*, NICOGPU_QUEUE_*, NICOGPU_SYNCOBJ_*, NICOGPU_TIMEOUT_INFINITE and struct
 * nicogpu_cmdbuf used below. */
#include "nicogpu_hal.h"

#define NICOGPU_WS_API_VERSION 3 /* winsys v3, tracks HAL v3 (v3: version only, DRM core syncobjs live in the drm
                                 backend; v1 = D8: structured faults, WAIT_FOR_SUBMIT, device status;
 v2: syncobj_signal, WAIT_AVAILABLE, calibrated timestamp, lost reason/time) */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- logging --------------------------------------------------------------- */
/* Levels are ordered: lower = more severe; a hook sees every message and
 * filters itself.  Messages: lifecycle at INFO, BO/submit traffic at DEBUG,
 * faults at ERROR, teardown leftovers at WARN. */
#define NICOGPU_WS_LOG_ERROR 0
#define NICOGPU_WS_LOG_WARN 1
#define NICOGPU_WS_LOG_INFO 2
#define NICOGPU_WS_LOG_DEBUG 3

typedef void (*nicogpu_ws_log_fn)(void* user, int level, const char* msg);

/* ---- device enumeration ------------------------------------------------------ */
struct nicogpu_ws_device_desc {
  uint32_t struct_size; /* in: bytes available; smaller prefixes are filled */
  char uri[64];         /* pass to nicogpu_ws_device_create */
  char backend[16];     /* "sim", "drm" */
  char name[64];        /* human readable */
  uint32_t vendor_id;
  uint32_t device_id;
};

/* Lists the devices this process can open.  *count is always set to the total
 * number of devices; at most max_descs entries are written.  max_descs > 0
 * with descs == NULL is NICOGPU_ERR_INVAL.  v0 lists exactly the in-process
 * functional model ("sim"). */
int nicogpu_ws_enumerate(struct nicogpu_ws_device_desc* descs, uint32_t max_descs, uint32_t* count);

/* ---- device ------------------------------------------------------------------ */
typedef struct nicogpu_ws_device nicogpu_ws_device;

struct nicogpu_ws_device_create_args {
  uint32_t struct_size; /* in: >= sizeof(this struct as of v0) */
  uint32_t flags;       /* must be 0 */
  char client[32];      /* optional label used in log lines, e.g. "mesa-ngv" */
};

/* Opens `uri` (NULL = nicogpu_ws_enumerate[0].uri / $NICOGPU_DEVICE), queries the
 * info and creates the device's single VM.  args may be NULL (defaults).
 * On success the caller owns the device and must nicogpu_ws_device_destroy it.
 * Enabling $NICOGPU_WINSYS_LOG (1 = info, 2 = debug) mirrors log output to
 * stderr in addition to any hook installed later. */
int nicogpu_ws_device_create(const char* uri, const struct nicogpu_ws_device_create_args* args,
                          nicogpu_ws_device** out);

/* Destroys the device and everything still alive on it, in the order the HAL
 * requires (queues, syncobjs, UNMAP every BO, free BOs, destroy the VM,
 * close).  Exclusive: no other call on `dev` may be in flight, none may follow. */
void nicogpu_ws_device_destroy(nicogpu_ws_device* dev);

/* Same prefix rules as nicogpu_device_query_info; returns the cached open-time
 * query without touching the device. */
int nicogpu_ws_device_get_info(nicogpu_ws_device* dev, struct nicogpu_device_info* info);

int nicogpu_ws_device_read_timestamp(nicogpu_ws_device* dev, uint64_t* ts);
/* v2: device timestamp + two host clock samples around the read (calibrated timestamps); see nicogpu_hal.h. */
int nicogpu_ws_device_read_timestamp_ex(nicogpu_ws_device* dev, struct nicogpu_timestamp_info* info);

/* Returns 1 once any queue of this device has been found GUILTY of a GPU
 * fault, or the HAL reports the whole device lost (nicogpu_device_get_status,
 * HAL v1; sticky for the device lifetime either way), else 0, or a negative
 * error.  The Vulkan layer maps 1 to VK_ERROR_DEVICE_LOST.  Only the guilty
 * queue gets NICOGPU_ERR_CANCELED from submit; other queues keep working until
 * the client decides to honor the loss. */
int nicogpu_ws_device_is_lost(nicogpu_ws_device* dev);

/* Why the device is lost (vkGetDeviceFaultInfo / VK_ERROR_DEVICE_LOST
 * reports).  Meaningful once nicogpu_ws_device_is_lost() returned 1. */
#define NICOGPU_WS_LOST_NONE 0u
#define NICOGPU_WS_LOST_QUEUE_FAULT 1u /* first guilty queue; `fault` is its structured record */
#define NICOGPU_WS_LOST_DEVICE 2u      /* HAL device status says the whole device is gone */
struct nicogpu_ws_lost_info {
  uint32_t struct_size; /* in: bytes available; smaller prefixes are filled */
  uint32_t reason;      /* NICOGPU_WS_LOST_* */
  uint32_t queue;       /* winsys handle of the first guilty queue, 0 if none */
  uint32_t pad;
  struct nicogpu_queue_fault_info fault; /* valid for NICOGPU_WS_LOST_QUEUE_FAULT */
  struct nicogpu_device_status status;   /* HAL device status at query time */
};
int nicogpu_ws_device_get_lost_info(nicogpu_ws_device* dev, struct nicogpu_ws_lost_info* out);

/* Installs (fn == NULL removes) the log hook; `user` is passed back opaque. */
int nicogpu_ws_set_log_fn(nicogpu_ws_device* dev, nicogpu_ws_log_fn fn, void* user);

/* ---- buffer objects: allocate == VM_BIND --------------------------------------- */
/* Every winsys BO is mapped for its whole lifetime in the device's VM at the
 * returned VA, so a VkDeviceMemory is one call.  Free is the exact inverse
 * (synchronous UNMAP, GEM close, VA release).  Sparse/rebind support is a
 * PROVISIONAL extension (DESIGN.md). */
struct nicogpu_ws_bo_args {
  uint32_t struct_size; /* in: >= sizeof(this struct as of v0) */
  uint64_t size;        /* bytes; rounded up to 4 KiB; must be nonzero */
  uint64_t alignment;   /* 0 = auto (4 KiB / 64 KiB / 2 MiB by size class);
                           else power of two >= 4096, used for placement and VA */
  uint32_t domain;      /* NICOGPU_BO_DOMAIN_VRAM or NICOGPU_BO_DOMAIN_GTT */
  uint32_t flags;       /* NICOGPU_BO_HOST_VISIBLE (VRAM only) */
  uint32_t vm_flags;    /* NICOGPU_VM_* permissions for the implicit bind */
  uint64_t va;          /* 0 = allocate from the winsys VA pool (first fit,
                           alignment rules above); else exact VA, must satisfy
                           the same alignment and lie inside the user window */
};
struct nicogpu_ws_bo_info {
  uint32_t struct_size; /* in: >= sizeof(this struct as of v0) */
  uint32_t handle;      /* winsys BO handle */
  uint64_t va;          /* GPU VA of [0, size) for the device lifetime */
  uint64_t size;        /* page-rounded size in bytes */
};

int nicogpu_ws_bo_alloc(nicogpu_ws_device* dev, const struct nicogpu_ws_bo_args* args,
                     struct nicogpu_ws_bo_info* out);
int nicogpu_ws_bo_map(nicogpu_ws_device* dev, uint32_t handle, void** ptr); /* stable until free */
int nicogpu_ws_bo_unmap(nicogpu_ws_device* dev, uint32_t handle);
int nicogpu_ws_bo_free(nicogpu_ws_device* dev, uint32_t handle);

/* ---- queues -------------------------------------------------------------------- */
/* One VM per device in v0, so queues need no VM argument (PROVISIONAL). */
int nicogpu_ws_queue_create(nicogpu_ws_device* dev, uint32_t* handle);
int nicogpu_ws_queue_destroy(nicogpu_ws_device* dev, uint32_t handle);
/* NICOGPU_QUEUE_STATUS_OK / NICOGPU_QUEUE_STATUS_GUILTY; seeing GUILTY also latches
 * the device-lost flag. */
int nicogpu_ws_queue_get_status(nicogpu_ws_device* dev, uint32_t handle, uint32_t* status);
/* Structured fault record of a queue (HAL v1; cause, CP error, MMU fault VA /
 * access / level / VMID).  Same prefix rules as the HAL call; a non-guilty
 * queue returns OK with cause NONE.  Replaces scraping the sim's stderr. */
int nicogpu_ws_queue_get_fault(nicogpu_ws_device* dev, uint32_t handle, struct nicogpu_queue_fault_info* info);

/* ---- sync objects (binary; timeline is emulated above this layer) ---------------- */
int nicogpu_ws_syncobj_create(nicogpu_ws_device* dev, uint32_t flags, uint32_t* handle);
int nicogpu_ws_syncobj_destroy(nicogpu_ws_device* dev, uint32_t handle);
int nicogpu_ws_syncobj_reset(nicogpu_ws_device* dev, const uint32_t* handles, uint32_t count);
/* v2: host signal (HAL nicogpu_syncobj_signal): every handle must exist (NOENT, nothing changes); the syncobjs then
 * count as signaled and available. */
int nicogpu_ws_syncobj_signal(nicogpu_ws_device* dev, const uint32_t* handles, uint32_t count);
/* timeout_ns is relative; 0 = poll; NICOGPU_TIMEOUT_INFINITE waits forever.
 * flags: NICOGPU_SYNCOBJ_WAIT_ALL (else wait for any) | NICOGPU_SYNCOBJ_WAIT_FOR_SUBMIT | (v2) NICOGPU_SYNCOBJ_WAIT_AVAILABLE
 * (return as soon as a fence is attached, not when it signals; fence-less is never INVAL then).
 * *first_signaled (may be NULL) receives the index of the first signaled
 * handle, like HAL.
 *
 * Fence-less syncobjs (HAL v1 / DRM semantics, replaces the v0 EINVAL->ETIME
 * translation): with WAIT_FOR_SUBMIT a syncobj that has no fence yet is "not
 * signaled"; the wait lasts up to the timeout for a submit / host signal from
 * another thread (since the thread contract the sim-backed winsys behaves
 * like the drm backend here; a single-threaded caller that waits forever on a
 * fence-less syncobj hangs, as it would on DRM); without the flag it is
 * NICOGPU_ERR_INVAL.
 * Vulkan host waits (vkWaitForFences, binary vkWaitSemaphores) pass the flag. */
int nicogpu_ws_syncobj_wait(nicogpu_ws_device* dev, const uint32_t* handles, uint32_t count,
                         int64_t timeout_ns, uint32_t flags, uint32_t* first_signaled);

/* ---- submission ------------------------------------------------------------------ */
#define NICOGPU_WS_SYNC_F_DRM (1u << 0)  /* drm backend, NICOGPU_FEATURE_DRM_SYNCOBJ: `handle` is a DRM core syncobj of the device's
                                         DRM file (the one nicogpu_ws_drm_fd() returns); `point` is the timeline point */
struct nicogpu_ws_sync {   /* references winsys syncobj handles (not HAL's) */
  uint32_t handle;
  uint32_t flags;       /* 0, or NICOGPU_WS_SYNC_F_DRM (drm backend with NICOGPU_FEATURE_DRM_SYNCOBJ) */
  uint64_t point;       /* must be 0 unless NICOGPU_WS_SYNC_F_DRM on a timeline syncobj (NICOGPU_FEATURE_SYNCOBJ_TIMELINE) */
};
struct nicogpu_ws_submit_args {
  uint32_t struct_size; /* in: >= sizeof(this struct as of v0) */
  uint32_t queue;       /* winsys queue handle */
  uint32_t num_cmdbufs;
  const struct nicogpu_cmdbuf* cmdbufs; /* HAL type: {va, size_dw, flags=0} */
  uint32_t num_waits;
  const struct nicogpu_ws_sync* waits;   /* must already have a fence */
  uint32_t num_signals;
  const struct nicogpu_ws_sync* signals; /* receive the job's fence */
};

/* Submits like nicogpu_submit, translating sync handles.  The sim executes
 * synchronously, so on NICOGPU_OK the job has completed; the winsys still polls
 * the queue status afterwards and latches device-lost if the job faulted
 * (a faulted job completes with an error-signaled fence, HAL v0 section 4). */
int nicogpu_ws_submit(nicogpu_ws_device* dev, const struct nicogpu_ws_submit_args* args);

#ifdef __cplusplus
}
#endif
#endif /* NICOGPU_WINSYS_H */
