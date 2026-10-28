/* SPDX-License-Identifier: MIT */
/* ngv_drm_ioctl.h -- thin typed wrappers over the nicogpu.ko render-node ioctls.
 *
 * Kept free of both nicogpu_hal.h and nicogpu_uapi.h so that the policy layer
 * (ngv_winsys_drm.c, which speaks nicogpu_winsys.h types) and the kernel-ABI layer
 * (ngv_drm_ioctl.c, which speaks nicogpu_uapi.h) can live in separate translation
 * units: the two headers define the same names (struct nicogpu_cmdbuf,
 * NICOGPU_TIMEOUT_INFINITE, ...) with HAL-vs-kernel spellings.  Every function
 * returns 0 or a negative errno (== NICOGPU_ERR_* values).  Struct layouts below
 * are the kernel's; ngv_drm_ioctl.c static-asserts every one of them.
 * PROVISIONAL: tracks ABI v5 (nicogpu_uapi.h); a v4 kernel is still accepted, without the
 * v5 ioctls (GET_GPU_INFO / GET_TIMESTAMP / GET_DEV_STATUS / WAIT_AVAILABLE): the caller checks the ABI it got.
 */
#ifndef NGV_DRM_IOCTL_H
#define NGV_DRM_IOCTL_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NGV_DRM_ABI_VERSION 6 /* the ABI this layer was written against (v6: additive) */
#define NGV_DRM_ABI_MIN 4     /* oldest kernel ABI it still works with (v5 only adds ioctls and a wait flag) */

struct ngv_drm_info {
   uint32_t abi_version;
   uint32_t msi_vectors;
   uint32_t shell_id, shell_version;
   uint64_t bar_len[3];
};

/* The fields of struct nicogpu_gpu_info (ABI v5 GET_GPU_INFO) the winsys turns into struct nicogpu_device_info. */
struct ngv_drm_gpu_info {
   uint32_t abi_version, info_flags;
 uint32_t sp_units; /* NICOGPU_SP_UNIT_*, valid iff info_flags has NICOGPU_GPU_INFO_F_SP_UNITS */
   uint64_t features;
   uint32_t vendor_id, device_id, revision;
   uint32_t num_cores, sp_per_core, warp_size, warp_slots_per_sp;
   uint32_t max_vregs, num_uregs, max_workgroup_threads, max_shared_bytes, max_user_uregs;
   uint32_t va_bits, page_sizes, max_vms, max_submit_cmdbufs, max_submit_syncs, max_bind_ops;
   uint64_t va_user_start, va_user_end;
   uint64_t vram_size, vram_visible_size, gtt_size, gtt_bo_max, vram_used, gtt_used;
   uint64_t timestamp_freq_hz;
};

/* struct nicogpu_dev_status (ABI v5 GET_DEV_STATUS) */
struct ngv_drm_dev_status {
   uint32_t status, lost_reason;
   int32_t lost_err;
   uint64_t lost_time_ns;
   uint32_t lost_count, resets, gpu_resets;
   uint64_t now_ns;
};

#define NGV_DRM_DEVSTAT_LOST (1u << 0)      /* == NICOGPU_DEVSTAT_LOST == HAL NICOGPU_DEVICE_STATUS_LOST */
#define NGV_DRM_DEVSTAT_PR_WEDGED (1u << 1) /* == NICOGPU_DEVSTAT_PR_WEDGED == HAL NICOGPU_DEVICE_STATUS_PR_WEDGED */
#define NGV_DRM_DEVSTAT_IO_ERROR (1u << 2)  /* == NICOGPU_DEVSTAT_IO_ERROR == HAL NICOGPU_DEVICE_STATUS_IO_ERROR */

/* struct nicogpu_timestamp (ABI v5 GET_TIMESTAMP) */
struct ngv_drm_timestamp {
   uint32_t clock_id; /* in: 0/1 CLOCK_MONOTONIC, 4 CLOCK_MONOTONIC_RAW */
   uint64_t ts, freq_hz, cpu_ns_before, cpu_ns_after;
};

struct ngv_drm_bind_op { /* == struct nicogpu_vm_bind_op */
   uint32_t op, flags, bo_handle, pad;
   uint64_t bo_offset, va, range;
};

struct ngv_drm_cmdbuf { /* == struct nicogpu_cmdbuf */
   uint64_t va;
   uint32_t size_dw, flags;
};

struct ngv_drm_sync { /* == struct nicogpu_sync */
   uint32_t handle, flags;
   uint64_t point;
};

struct ngv_drm_fault { /* the fields of struct nicogpu_queue_fault the winsys reports */
   uint32_t status, cause, cp_error, flags;
   int32_t err;
   uint64_t fence_seq;
   uint32_t mmu_code, mmu_access, mmu_level, mmu_vmid;
   uint64_t mmu_va, mmu_pte;
   uint32_t resets, gpu_resets;
};

/* Lists /dev/dri/renderD* nodes whose DRM driver name is "nicogpu". paths[i] is
 * a NUL-terminated absolute path of at most 63 characters. */
int ngv_drm_scan(char (*paths)[64], uint32_t max_paths, uint32_t *count);

/* Checks the driver name and the ABI (NGV_DRM_ABI_MIN <= kernel ABI <= NGV_DRM_ABI_VERSION; -EPROTO otherwise).
 * *abi_out (may be NULL) = the kernel's ABI version. */
int ngv_drm_open(const char *path, int *fd_out, uint32_t *abi_out);
void ngv_drm_close(int fd);
int ngv_drm_get_info(int fd, struct ngv_drm_info *out);
/* ABI v5 only (a v4 kernel answers -EINVAL for the unknown ioctl: do not call it, check the ABI first). */
int ngv_drm_get_gpu_info(int fd, struct ngv_drm_gpu_info *out);
int ngv_drm_dev_status(int fd, struct ngv_drm_dev_status *out);
int ngv_drm_timestamp(int fd, struct ngv_drm_timestamp *io);

int ngv_drm_bo_create(int fd, uint64_t size, uint64_t alignment, uint32_t domain, uint32_t flags,
                      uint32_t *handle, uint64_t *size_out, uint64_t *mmap_offset);
int ngv_drm_bo_free(int fd, uint32_t handle);
void *ngv_drm_bo_mmap(int fd, uint64_t mmap_offset, uint64_t size); /* NULL on failure */
void ngv_drm_bo_munmap(void *ptr, uint64_t size);

int ngv_drm_vm_create(int fd, uint32_t *vm_id);
int ngv_drm_vm_destroy(int fd, uint32_t vm_id);
int ngv_drm_vm_bind(int fd, uint32_t vm_id, const struct ngv_drm_bind_op *ops, uint32_t n);

int ngv_drm_queue_create(int fd, uint32_t vm_id, uint32_t priority, uint32_t *queue_id);
int ngv_drm_queue_destroy(int fd, uint32_t queue_id);
int ngv_drm_queue_status(int fd, uint32_t queue_id, uint32_t *status);
int ngv_drm_queue_fault(int fd, uint32_t queue_id, struct ngv_drm_fault *out);

int ngv_drm_syncobj_create(int fd, uint32_t flags, uint32_t *handle);
int ngv_drm_syncobj_destroy(int fd, uint32_t handle);
int ngv_drm_syncobj_reset(int fd, const uint32_t *handles, uint32_t count);
int ngv_drm_syncobj_signal(int fd, const uint32_t *handles, uint32_t count); /* ABI v2+ */
int ngv_drm_syncobj_wait(int fd, const uint32_t *handles, uint32_t count, int64_t timeout_ns,
                         uint32_t flags, uint32_t *first_signaled);

int ngv_drm_submit(int fd, uint32_t queue_id, const struct ngv_drm_cmdbuf *cmdbufs,
                   uint32_t num_cmdbufs, const struct ngv_drm_sync *waits, uint32_t num_waits,
                   const struct ngv_drm_sync *signals, uint32_t num_signals, uint64_t *fence_seq);

#ifdef __cplusplus
}
#endif
#endif /* NGV_DRM_IOCTL_H */
