/* SPDX-License-Identifier: MIT */
/*
 * nicogpu_hal.h -- Nicogpu device HAL (user-space side), v0.
 *
 * One C API, several backends:
 *   "sim"   the C++ functional model in-process (this repository, P0)
 *   "drm"   the Linux nicogpu.ko render node (future; same header)
 *
 * The API deliberately mirrors the planned kernel UAPI (the UAPI spec draft)
 * one-to-one, so the Mesa winsys written against it runs unchanged on the model
 * (x86 CTS farm) and on hardware.  Rules:
 *   - every call returns 0 or a negative NICOGPU_ERR_* (Linux errno values);
 *   - objects are u32 handles scoped to one nicogpu_device (like GEM handles);
 *   - structs carry no host pointers except where the kernel UAPI will carry a
 *     __u64 user pointer (arrays of ops / syncs);
 *   - GPU virtual addresses are chosen by user space (VM_BIND model).
 * Not thread-safe per device in P0: callers serialize (Mesa does per-device).
 * The P0 v0 contract was approved on 2026-09-24. Breaking changes require a
 * new API version; Linux ioctl numbers are assigned separately.
 */
#ifndef NICOGPU_HAL_H
#define NICOGPU_HAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NICOGPU_HAL_API_VERSION 3 /* v0 (P0, frozen 2026-09-24) + additive v1 (D8, 2026-10-10): queue fault record,
 syncobj WAIT_FOR_SUBMIT, device status + additive v2 (kernel ABI v5):
                                  syncobj_signal, WAIT_AVAILABLE, calibrated timestamp, lost reason/time in the device
 status, device_info features/gtt_bo_max + v3 (kernel ABI v6): comment-level only, the drm backend maps
                                  syncobjs onto DRM core syncobjs (see "sync objects" below), feature bits 8/10/11;
                                  every v0/v1/v2 call keeps its behaviour */

/* ---- errors (negative Linux errno values, fixed so every OS agrees) ------ */
#define NICOGPU_OK 0
#define NICOGPU_ERR_NOENT (-2)      /* bad handle / unknown object */
#define NICOGPU_ERR_IO (-5)         /* backend I/O failure */
#define NICOGPU_ERR_NOMEM (-12)     /* host or device memory exhausted */
#define NICOGPU_ERR_BUSY (-16)
#define NICOGPU_ERR_NODEV (-19)     /* no such backend / device */
#define NICOGPU_ERR_INVAL (-22)     /* invalid argument */
#define NICOGPU_ERR_NOSPC (-28)     /* VA space / heap exhausted */
#define NICOGPU_ERR_TIME (-62)      /* wait timed out */
#define NICOGPU_ERR_NOTSUP (-95)
#define NICOGPU_ERR_CANCELED (-125) /* queue is guilty of a GPU fault: device lost */

typedef struct nicogpu_device nicogpu_device;

/* ---- device --------------------------------------------------------------- */
/* uri: "sim", "sim:cores=2", "drm:/dev/dri/renderD128".
 * NULL or "" -> $NICOGPU_DEVICE, else "sim". */
int nicogpu_device_open(const char *uri, nicogpu_device **out);
void nicogpu_device_close(nicogpu_device *dev);

#define NICOGPU_PAGE_4K (1u << 0)
#define NICOGPU_PAGE_64K (1u << 1)
#define NICOGPU_PAGE_2M (1u << 2)

struct nicogpu_device_info {
   uint32_t struct_size;      /* in: bytes available; smaller prefixes are accepted */
   uint32_t api_version;      /* out: NICOGPU_HAL_API_VERSION of the backend */
   char backend[16];          /* "sim", "drm" */
   uint32_t vendor_id, device_id, revision;
   char isa_name[32];         /* generated ISA table name, e.g. "nicogpu_isa_sample" */
   char isa_version[32];
   uint32_t num_cores;
   uint32_t sp_per_core;
   uint32_t warp_size;        /* == subgroupSize */
   uint32_t warp_slots_per_sp;
   uint32_t max_vregs;        /* per thread */
   uint32_t num_uregs;        /* per warp */
   uint32_t max_workgroup_threads;
   uint32_t max_shared_bytes; /* per workgroup */
   uint32_t max_user_uregs;   /* preloaded uniform registers per dispatch */
   uint32_t va_bits;
   uint64_t va_user_start;    /* [start, end) is available to user space for VM_BIND */
   uint64_t va_user_end;
   uint32_t page_sizes;       /* NICOGPU_PAGE_* */
   uint32_t pad0;
   uint64_t vram_size;
   uint64_t vram_visible_size; /* CPU-mappable VRAM (BAR window) */
   uint64_t gtt_size;          /* host memory the GPU may map */
   uint64_t timestamp_freq_hz;
 /* ---- v2: appended, older callers ignore them (struct_size prefix rule) ---- */
   uint64_t features;         /* NICOGPU_FEATURE_* bits of the backend */
   uint64_t gtt_bo_max;       /* largest single GTT BO, 0 = only limited by gtt_size */
   uint32_t info_flags;       /* NICOGPU_INFO_F_*: provenance of the numbers (drm backend: kernel GET_GPU_INFO) */
 uint32_t sp_units; /* NICOGPU_SP_UNIT_* the SP was built with; valid only if info_flags has NICOGPU_INFO_F_SP_UNITS */
};
/* features: same bit positions as the kernel's NICOGPU_FEAT_* (ABI v5) for the ones that exist on both sides */
#define NICOGPU_FEATURE_TIMESTAMP (1ull << 1)         /* nicogpu_device_read_timestamp / _ex work */
#define NICOGPU_FEATURE_SYNCOBJ_WAIT_AVAILABLE (1ull << 2)
#define NICOGPU_FEATURE_SYNCOBJ_SIGNAL (1ull << 3)    /* nicogpu_syncobj_signal */
#define NICOGPU_FEATURE_DEVICE_STATUS (1ull << 4)     /* nicogpu_device_get_status carries lost reason / time */
#define NICOGPU_FEATURE_QUEUE_FAULT (1ull << 5)
#define NICOGPU_FEATURE_SYNCOBJ_TIMELINE (1ull << 8)  /* v3: `point` of nicogpu_sync is honoured (drm backend, kernel ABI v6) */
#define NICOGPU_FEATURE_DRM_SYNCOBJ (1ull << 10)     /* v3: syncobj handles are DRM core syncobjs (exportable) */
#define NICOGPU_FEATURE_SYNC_FILE (1ull << 11)       /* v3: sync_file import / export of a syncobj is available */
/* info_flags (drm backend; the sim reports 0 = "model-defined"): a set bit means the value was read from hardware */
#define NICOGPU_INFO_F_RM_CAPS (1u << 0)     /* num_cores / sp_per_core / warp_size */
#define NICOGPU_INFO_F_RM_REV (1u << 1)      /* revision */
#define NICOGPU_INFO_F_TS_RM (1u << 2)       /* timestamp_freq_hz from the RM register */
#define NICOGPU_INFO_F_TS_SHELL (1u << 3)    /* timestamp_freq_hz from the shell clock meter */
#define NICOGPU_INFO_F_GTT_UNCACHED (1u << 4) /* drm backend, non-coherent host: GTT is CPU-uncached/WC, not HOST_CACHED */
#define NICOGPU_INFO_F_SP_UNITS (1u << 6)    /* sp_units was reported by the RM (CAPS[26:24] with CAPS[27] UNITS_VALID); clear = unknown */
/* sp_units: optional SP execution units (same bits as RM CAPS[26:24] and the kernel's NICOGPU_SP_UNIT_*).  valu/salu/lsu/... are
 * always there.  A clear NICOGPU_INFO_F_SP_UNITS means "unknown" (an RM older than the unit bits, or no RM regmap), NOT "none". */
#define NICOGPU_SP_UNIT_SFU (1u << 0)        /* rcp / rsq / sqrt / exp2 / log2 / sin / cos, integer divide */
#define NICOGPU_SP_UNIT_XLANE (1u << 1)      /* readlane / writelane / ballot / shuffle ... */
#define NICOGPU_SP_UNIT_FP64 (1u << 2)
#define NICOGPU_SP_UNIT_F16CVT (1u << 3) /* (ruled 2026-10-24): VALU fcvt.f32.f16 / fcvt.f16.f32 / pkcvt.f16.f32 (RM CAPS[29]) */
int nicogpu_device_query_info(nicogpu_device *dev, struct nicogpu_device_info *info);
/* Current GPU timestamp (for calibrated timestamps). */
int nicogpu_device_read_timestamp(nicogpu_device *dev, uint64_t *ts);

/* v2: the device timestamp together with two host clock samples taken around the read (calibrated timestamps,
 * VK_EXT_calibrated_timestamps).  Mirrors kernel DRM_IOCTL_NICOGPU_GET_TIMESTAMP.  clock_id: 0 or 1 = CLOCK_MONOTONIC,
 * 4 = CLOCK_MONOTONIC_RAW (Linux clock ids on every OS); anything else NICOGPU_ERR_INVAL. */
struct nicogpu_timestamp_info {
   uint32_t struct_size; /* in: bytes available; smaller prefixes are filled */
   uint32_t clock_id;    /* in */
   uint64_t ts;          /* out: device ticks */
   uint64_t freq_hz;     /* out: ticks per second (== nicogpu_device_info.timestamp_freq_hz) */
   uint64_t cpu_ns_before; /* out: host clock before the read, ns */
   uint64_t cpu_ns_after;  /* out: host clock after it */
};
int nicogpu_device_read_timestamp_ex(nicogpu_device *dev, struct nicogpu_timestamp_info *info);

/* ---- buffer objects --------------------------------------------------------- */
#define NICOGPU_BO_DOMAIN_VRAM 1u /* device-local */
#define NICOGPU_BO_DOMAIN_GTT 2u  /* host memory, snooped (HOST_COHERENT|HOST_CACHED); uncached/WC if NICOGPU_INFO_F_GTT_UNCACHED */

#define NICOGPU_BO_HOST_VISIBLE (1u << 0) /* VRAM only: place in the CPU-visible window */

struct nicogpu_bo_create_args {
   uint64_t size;      /* bytes, rounded up to 4 KiB */
   uint64_t alignment; /* 0 = backend default; power of two >= 4096 */
   uint32_t domain;    /* NICOGPU_BO_DOMAIN_* */
   uint32_t flags;     /* NICOGPU_BO_* */
};
/* BO size / placement granule. */
#define NICOGPU_PAGE_SIZE 4096ull
/* Default VA / PA alignment of a BO whose `alignment` is 0, from its page-rounded size: the size classes let the
 * page-table writer use 2M / 64K pages.  Shared by the sim HAL, the winsys and the Mesa drm winsys so VA and PA
 * alignment agree. */
static inline uint64_t nicogpu_default_va_align(uint64_t size)
{
   return size >= (2ull << 20) ? (2ull << 20) : size >= (64ull << 10) ? (64ull << 10) : NICOGPU_PAGE_SIZE;
}
/* New BOs are always zero-filled. */
int nicogpu_bo_create(nicogpu_device *dev, const struct nicogpu_bo_create_args *args, uint32_t *handle);
/* Drops the handle; storage lives on while VM mappings reference it. */
int nicogpu_bo_free(nicogpu_device *dev, uint32_t handle);
/* CPU mapping (write-combined for VRAM, cached for GTT).  Stable until free. */
int nicogpu_bo_map(nicogpu_device *dev, uint32_t handle, void **ptr);
int nicogpu_bo_unmap(nicogpu_device *dev, uint32_t handle);

/* ---- GPU virtual memory ----------------------------------------------------- */
int nicogpu_vm_create(nicogpu_device *dev, uint32_t *vm_id);
int nicogpu_vm_destroy(nicogpu_device *dev, uint32_t vm_id);

#define NICOGPU_VM_BIND_MAP 0u             /* map bo[bo_offset, +range) at va */
#define NICOGPU_VM_BIND_UNMAP 1u           /* unmap [va, +range) */
#define NICOGPU_VM_BIND_MAP_SPARSE_NULL 2u /* reads 0 + "non-resident", writes dropped */

#define NICOGPU_VM_READ (1u << 0)
#define NICOGPU_VM_WRITE (1u << 1)
#define NICOGPU_VM_EXEC (1u << 2)
#define NICOGPU_VM_UNCACHED (1u << 3)

struct nicogpu_vm_bind_op {
   uint32_t op;        /* NICOGPU_VM_BIND_* */
   uint32_t flags;     /* NICOGPU_VM_* (MAP only) */
   uint32_t bo_handle; /* MAP only */
   uint32_t pad;
   uint64_t bo_offset; /* 4 KiB aligned */
   uint64_t va;        /* 4 KiB aligned */
   uint64_t range;     /* 4 KiB multiple */
};
/* Ops apply in order with "replace" semantics: a MAP over an existing
 * mapping (e.g. a sparse-null reservation) replaces that part of it.
 * P0: synchronous; the TLB is invalidated before return. */
int nicogpu_vm_bind(nicogpu_device *dev, uint32_t vm_id, const struct nicogpu_vm_bind_op *ops, uint32_t num_ops);

/* ---- queues ------------------------------------------------------------------ */
#define NICOGPU_QUEUE_PRIORITY_NORMAL 0u
int nicogpu_queue_create(nicogpu_device *dev, uint32_t vm_id, uint32_t priority, uint32_t *queue_id);
int nicogpu_queue_destroy(nicogpu_device *dev, uint32_t queue_id);

#define NICOGPU_QUEUE_STATUS_OK 0u
#define NICOGPU_QUEUE_STATUS_GUILTY 1u /* a job of this queue faulted: VK_ERROR_DEVICE_LOST */
int nicogpu_queue_get_status(nicogpu_device *dev, uint32_t queue_id, uint32_t *status);

/* ---- sync objects (binary in v0; `point` reserved for timelines) ------------ */
/* v3: the sim backend keeps these as plain binary syncobjs. The drm backend may map them onto the DRM core
 * syncobjs of the kernel (NICOGPU_FEATURE_DRM_SYNCOBJ: DRM_IOCTL_SYNCOBJ_*, submit entries flagged NICOGPU_SYNC_F_DRM;
 * the WAIT_* flags below are the same bits as DRM_SYNCOBJ_WAIT_FLAGS_*, but DRM timeouts are absolute).  Nothing here
 * changes for callers; `nicogpu_sync.point` becomes meaningful only where NICOGPU_FEATURE_SYNCOBJ_TIMELINE is set. */
#define NICOGPU_SYNCOBJ_CREATE_SIGNALED (1u << 0)
#define NICOGPU_SYNCOBJ_WAIT_ALL (1u << 0)
#define NICOGPU_SYNCOBJ_WAIT_FOR_SUBMIT (1u << 1) /* v1: a syncobj without a fence is "not signaled yet", not INVAL */
#define NICOGPU_SYNCOBJ_WAIT_AVAILABLE (1u << 2)  /* v2: wait until a fence is ATTACHED (submitted or host-signaled),
                                                  not until it signals; implies WAIT_FOR_SUBMIT.  Differs from
                                                  WAIT_FOR_SUBMIT, which still waits for completion.  The DRM
                                                  DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE semantics (VK_SYNC_WAIT_PENDING). */
#define NICOGPU_TIMEOUT_INFINITE INT64_MAX

int nicogpu_syncobj_create(nicogpu_device *dev, uint32_t flags, uint32_t *handle);
int nicogpu_syncobj_destroy(nicogpu_device *dev, uint32_t handle);
int nicogpu_syncobj_reset(nicogpu_device *dev, const uint32_t *handles, uint32_t count);
/* v2: host signal (kernel DRM_IOCTL_NICOGPU_SYNCOBJ_SIGNAL, present since ABI v2): attach an already-signaled fence.
 * Every handle must exist, otherwise nothing changes (NICOGPU_ERR_NOENT); count 0 is a no-op (like reset). */
int nicogpu_syncobj_signal(nicogpu_device *dev, const uint32_t *handles, uint32_t count);
/* timeout_ns is relative; 0 = poll.  Returns NICOGPU_ERR_TIME on timeout.
 * Waiting on a syncobj that has no fence yet is NICOGPU_ERR_INVAL unless
 * NICOGPU_SYNCOBJ_WAIT_FOR_SUBMIT is set (v1): then it counts as not signaled
 * (the wait keeps waiting for a fence to be attached; the sim, which runs
 * jobs synchronously inside nicogpu_submit, can never see one appear during a
 * wait and reports NICOGPU_ERR_TIME right away, as the timeout-expiry result).
 * *first_signaled may be NULL. */
int nicogpu_syncobj_wait(nicogpu_device *dev, const uint32_t *handles, uint32_t count, int64_t timeout_ns,
                      uint32_t flags, uint32_t *first_signaled);

/* ---- v1: structured fault record and device status (D8, 2026-10-10) --------------
 * Values and field meanings mirror the kernel UAPI (nicogpu_uapi.h
 * struct nicogpu_queue_fault, ABI v3+): the drm backend is a field-by-field copy
 * of DRM_IOCTL_NICOGPU_QUEUE_FAULT, the sim backend fills the same record from
 * the CP/MMU registers it already reads. */
#define NICOGPU_FAULT_CAUSE_NONE 0u      /* the queue is not guilty */
#define NICOGPU_FAULT_CAUSE_TIMEOUT 1u   /* a job did not finish in time (hang) */
#define NICOGPU_FAULT_CAUSE_CP_ERROR 2u /* CP halted on one of its jobs: cp_error says why (command-set spec section 6) */
#define NICOGPU_FAULT_CAUSE_MMU_FAULT 3u /* ... together with an MMU fault of the queue's VMID: mmu_* valid */
#define NICOGPU_FAULT_CAUSE_FW_PANIC 4u  /* CP firmware died (drm backend only) */
#define NICOGPU_FAULT_CAUSE_DEV_LOST 5u  /* innocent: the context died with a GPU-zone reset (drm backend only) */

#define NICOGPU_FAULT_F_MMU_VALID (1u << 0)    /* mmu_* describe a latched MMU fault */
#define NICOGPU_FAULT_F_MMU_MULTI (1u << 1)    /* the MMU dropped later faults while this one was latched */
#define NICOGPU_FAULT_F_DEVICE_LOST (1u << 2)  /* every reset level failed: the whole device is lost */

#define NICOGPU_FAULT_ACCESS_READ 0u
#define NICOGPU_FAULT_ACCESS_WRITE 1u
#define NICOGPU_FAULT_ACCESS_EXEC 2u

struct nicogpu_queue_fault_info {
   uint32_t struct_size; /* in: bytes available; smaller prefixes are filled */
   uint32_t status;      /* out: NICOGPU_QUEUE_STATUS_* */
   uint32_t cause;       /* out: NICOGPU_FAULT_CAUSE_* */
 uint32_t cp_error; /* out: CP_ERROR code of the CP halt (command-set spec section 6), 0 if none */
   uint32_t flags;       /* out: NICOGPU_FAULT_F_* */
   int32_t err;          /* out: error of the failing job (negative errno, -EIO), 0 if not guilty */
   uint64_t fence_seq;   /* out: sequence number of the failing job (0: none) */
   uint32_t mmu_code;    /* out: MMU.md 6.1 fault code (1 NOT_PRESENT, 2 NON_CANONICAL, 3 PERMISSION,
                            4 BAD_PTE, 5 BAD_PHYS_ADDR, 6 MISALIGNED) */
   uint32_t mmu_access;  /* out: NICOGPU_FAULT_ACCESS_* */
   uint32_t mmu_level;   /* out: page-table level of the deciding entry */
   uint32_t mmu_vmid;    /* out */
   uint64_t mmu_va;      /* out: faulting byte VA */
   uint64_t mmu_pte;     /* out: the entry read at mmu_level */
   uint32_t resets;      /* out: CP/firmware/GPU-zone resets since the CP came up (device wide) */
   uint32_t gpu_resets;  /* out: ... of which GPU-zone resets */
};
/* The record is latched when the queue turns GUILTY and stays until the queue
 * is destroyed.  A queue that is not guilty returns NICOGPU_OK with cause NONE
 * and every other field zero (status OK). */
int nicogpu_queue_get_fault(nicogpu_device *dev, uint32_t queue_id, struct nicogpu_queue_fault_info *info);

#define NICOGPU_DEVICE_STATUS_LOST (1u << 0) /* the whole device is lost (reset ladder exhausted / device gone) */
#define NICOGPU_DEVICE_STATUS_PR_WEDGED (1u << 1) /* v2: partial reconfiguration failed and could not roll back (implies LOST) */
#define NICOGPU_DEVICE_STATUS_IO_ERROR (1u << 2)  /* v2: ring / doorbell access failed, recovery pending (informational) */
/* v2 lost_reason (kernel NICOGPU_LOST_*) */
#define NICOGPU_DEVICE_LOST_NONE 0u
#define NICOGPU_DEVICE_LOST_RESET_FAILED 1u /* the reset ladder (CP, firmware, GPU zone) was exhausted */
#define NICOGPU_DEVICE_LOST_PR_WEDGED 2u    /* PR failed twice and the rollback failed too */
struct nicogpu_device_status {
   uint32_t struct_size; /* in: bytes available; smaller prefixes are filled */
   uint32_t flags;       /* out: NICOGPU_DEVICE_STATUS_* */
   uint32_t resets;      /* out: CP/firmware/GPU-zone resets so far (device wide) */
   uint32_t gpu_resets;  /* out: ... of which GPU-zone resets */
 /* ---- v2 (kernel GET_DEV_STATUS): describe the MOST RECENT loss, kept after a recovery ---- */
   uint32_t lost_reason; /* out: NICOGPU_DEVICE_LOST_*, 0 if never lost */
   int32_t lost_err;     /* out: negative errno that ended the failing flow, 0 if none */
   uint64_t lost_time_ns; /* out: CLOCK_MONOTONIC at that moment, 0 = never */
   uint32_t lost_count;  /* out: how many times the device was declared lost */
   uint32_t pad;
   uint64_t now_ns;      /* out: CLOCK_MONOTONIC now (lost age = now_ns - lost_time_ns) */
};
/* Device-wide loss, independent of any queue (ECC, GPU-zone reset failure,
 * PCIe surprise removal): VK_ERROR_DEVICE_LOST even for a client whose queues
 * never faulted.  A queue-only fault does NOT set NICOGPU_DEVICE_STATUS_LOST (the
 * winsys keeps its own per-queue sticky latch).  The sim never loses the whole
 * device unless a test hook says so. */
int nicogpu_device_get_status(nicogpu_device *dev, struct nicogpu_device_status *st);

/* ---- submission --------------------------------------------------------------- */
struct nicogpu_cmdbuf {
   uint64_t va;      /* GPU VA of a packet stream in the queue's VM (see REGMAP-draft §3) */
   uint32_t size_dw;
   uint32_t flags;   /* must be 0 */
};
struct nicogpu_sync {
   uint32_t handle;
   uint32_t flags;   /* must be 0 */
   uint64_t point;   /* must be 0 unless NICOGPU_FEATURE_SYNCOBJ_TIMELINE (v3, drm backend only) */
};
struct nicogpu_submit_args {
   uint32_t queue_id;
   uint32_t num_cmdbufs;
   const struct nicogpu_cmdbuf *cmdbufs;
   uint32_t num_waits;
   uint32_t num_signals;
   const struct nicogpu_sync *waits;   /* all must signal before the job runs */
   const struct nicogpu_sync *signals; /* get the job's fence */
};
int nicogpu_submit(nicogpu_device *dev, const struct nicogpu_submit_args *args);

#ifdef __cplusplus
}
#endif
#endif /* NICOGPU_HAL_H */
