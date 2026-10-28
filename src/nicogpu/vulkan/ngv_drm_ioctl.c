/* SPDX-License-Identifier: MIT */
/* ngv_drm_ioctl.c -- ABI layer of the "drm" winsys backend (PROVISIONAL).
 *
 * The only file of the driver that includes nicogpu_uapi.h.  NOT RUN against a
 * real kernel yet (no board in the build loop): it compiles against the ABI v5
 * header and is checked structurally (layout static-asserts below); the first
 * run is the driver's board bring-up item.
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "nicogpu_uapi.h"
#include "ngv_drm_ioctl.h"

#define LAYOUT(a, b) _Static_assert(sizeof(a) == sizeof(b), "layout mismatch: " #a " vs " #b)
LAYOUT(struct ngv_drm_bind_op, struct nicogpu_vm_bind_op);
LAYOUT(struct ngv_drm_cmdbuf, struct nicogpu_cmdbuf);
LAYOUT(struct ngv_drm_sync, struct nicogpu_sync);
_Static_assert(NGV_DRM_ABI_VERSION == NICOGPU_ABI_VERSION, "ngv_drm_ioctl.c written for another ABI");
_Static_assert(NGV_DRM_ABI_MIN <= NGV_DRM_ABI_VERSION, "ABI range");
_Static_assert(NGV_DRM_DEVSTAT_LOST == NICOGPU_DEVSTAT_LOST && NGV_DRM_DEVSTAT_PR_WEDGED == NICOGPU_DEVSTAT_PR_WEDGED &&
               NGV_DRM_DEVSTAT_IO_ERROR == NICOGPU_DEVSTAT_IO_ERROR, "device status bits");

static int
ngv_ioctl(int fd, unsigned long req, void *arg)
{
   int r;
   do {
      r = ioctl(fd, req, arg);
   } while (r == -1 && (errno == EINTR || errno == EAGAIN));
   return r == -1 ? -errno : 0;
}

static int
drm_driver_is_nicogpu(int fd)
{
   char name[16] = {0};
   struct drm_version v;
   memset(&v, 0, sizeof(v));
   v.name = name;
   v.name_len = sizeof(name) - 1;
   if (ngv_ioctl(fd, DRM_IOCTL_VERSION, &v))
      return 0;
   return v.name_len == 7 && !memcmp(name, "nicogpu", 7);
}

int
ngv_drm_scan(char (*paths)[64], uint32_t max_paths, uint32_t *count)
{
   uint32_t n = 0;
   DIR *d = opendir("/dev/dri");
   if (!d) {
      *count = 0;
      return 0; /* no DRM at all: zero devices, not an error */
   }
   struct dirent *e;
   while ((e = readdir(d))) {
      if (strncmp(e->d_name, "renderD", 7))
         continue;
      char p[64];
      snprintf(p, sizeof(p), "/dev/dri/%s", e->d_name);
      int fd = open(p, O_RDWR | O_CLOEXEC);
      if (fd < 0)
         continue;
      int ours = drm_driver_is_nicogpu(fd);
      close(fd);
      if (!ours)
         continue;
      if (paths && n < max_paths)
         memcpy(paths[n], p, sizeof(p));
      n++;
   }
   closedir(d);
   *count = n;
   return 0;
}

int
ngv_drm_open(const char *path, int *fd_out, uint32_t *abi_out)
{
   int fd = open(path, O_RDWR | O_CLOEXEC);
   if (fd < 0)
      return -errno;
   if (!drm_driver_is_nicogpu(fd)) {
      close(fd);
      return -ENODEV;
   }
   struct nicogpu_info info;
   memset(&info, 0, sizeof(info));
   info.struct_size = sizeof(info);
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_GET_INFO, &info);
   /* v5 is purely additive over v4; anything outside [MIN, own] is refused, never guessed */
   if (r || info.abi_version < NGV_DRM_ABI_MIN || info.abi_version > NGV_DRM_ABI_VERSION) {
      close(fd);
      return r ? r : -EPROTO;
   }
   *fd_out = fd;
   if (abi_out)
      *abi_out = info.abi_version;
   return 0;
}

void
ngv_drm_close(int fd)
{
   close(fd);
}

int
ngv_drm_get_info(int fd, struct ngv_drm_info *out)
{
   struct nicogpu_info info;
   memset(&info, 0, sizeof(info));
   info.struct_size = sizeof(info);
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_GET_INFO, &info);
   if (r)
      return r;
   out->abi_version = info.abi_version;
   out->msi_vectors = info.msi_vectors;
   out->shell_id = info.shell_id;
   out->shell_version = info.shell_version;
   for (int i = 0; i < 3; i++)
      out->bar_len[i] = info.bar_len[i];
   return 0;
}

int
ngv_drm_get_gpu_info(int fd, struct ngv_drm_gpu_info *out)
{
   struct nicogpu_gpu_info a;
   memset(&a, 0, sizeof(a));
   a.struct_size = sizeof(a);
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_GET_GPU_INFO, &a);
   if (r)
      return r;
   memset(out, 0, sizeof(*out));
   out->abi_version = a.abi_version;
   out->info_flags = a.info_flags;
   out->sp_units = a.sp_units;
   out->features = a.features;
   out->vendor_id = a.vendor_id;
   out->device_id = a.device_id;
   out->revision = a.revision;
   out->num_cores = a.num_cores;
   out->sp_per_core = a.sp_per_core;
   out->warp_size = a.warp_size;
   out->warp_slots_per_sp = a.warp_slots_per_sp;
   out->max_vregs = a.max_vregs;
   out->num_uregs = a.num_uregs;
   out->max_workgroup_threads = a.max_workgroup_threads;
   out->max_shared_bytes = a.max_shared_bytes;
   out->max_user_uregs = a.max_user_uregs;
   out->va_bits = a.va_bits;
   out->page_sizes = a.page_sizes;
   out->max_vms = a.max_vms;
   out->max_submit_cmdbufs = a.max_submit_cmdbufs;
   out->max_submit_syncs = a.max_submit_syncs;
   out->max_bind_ops = a.max_bind_ops;
   out->va_user_start = a.va_user_start;
   out->va_user_end = a.va_user_end;
   out->vram_size = a.vram_size;
   out->vram_visible_size = a.vram_visible_size;
   out->gtt_size = a.gtt_size;
   out->gtt_bo_max = a.gtt_bo_max;
   out->vram_used = a.vram_used;
   out->gtt_used = a.gtt_used;
   out->timestamp_freq_hz = a.timestamp_freq_hz;
   return 0;
}

int
ngv_drm_dev_status(int fd, struct ngv_drm_dev_status *out)
{
   struct nicogpu_dev_status a;
   memset(&a, 0, sizeof(a));
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_GET_DEV_STATUS, &a);
   if (r)
      return r;
   out->status = a.status;
   out->lost_reason = a.lost_reason;
   out->lost_err = a.lost_err;
   out->lost_time_ns = a.lost_time_ns;
   out->lost_count = a.lost_count;
   out->resets = a.resets;
   out->gpu_resets = a.gpu_resets;
   out->now_ns = a.now_ns;
   return 0;
}

int
ngv_drm_timestamp(int fd, struct ngv_drm_timestamp *io)
{
   struct nicogpu_timestamp a;
   memset(&a, 0, sizeof(a));
   a.clock_id = io->clock_id;
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_GET_TIMESTAMP, &a);
   if (r)
      return r;
   io->ts = a.ts;
   io->freq_hz = a.freq_hz;
   io->cpu_ns_before = a.cpu_ns_before;
   io->cpu_ns_after = a.cpu_ns_after;
   return 0;
}

int
ngv_drm_bo_create(int fd, uint64_t size, uint64_t alignment, uint32_t domain, uint32_t flags,
                  uint32_t *handle, uint64_t *size_out, uint64_t *mmap_offset)
{
   struct nicogpu_bo_create a;
   memset(&a, 0, sizeof(a));
   a.size = size;
   a.alignment = alignment;
   a.domain = domain;
   a.flags = flags;
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_BO_CREATE, &a);
   if (r)
      return r;
   *handle = a.handle;
   *size_out = a.size;
   *mmap_offset = a.mmap_offset;
   return 0;
}

int
ngv_drm_bo_free(int fd, uint32_t handle)
{
   struct nicogpu_bo_free a = {.handle = handle};
   return ngv_ioctl(fd, DRM_IOCTL_NICOGPU_BO_FREE, &a);
}

void *
ngv_drm_bo_mmap(int fd, uint64_t mmap_offset, uint64_t size)
{
   /* ABI v4: the offset is the opaque DRM fake offset, the mapping starts at
    * byte 0 of the BO; caching (WC for VRAM, cached for GTT) is the kernel's. */
   void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)mmap_offset);
   return p == MAP_FAILED ? NULL : p;
}

void
ngv_drm_bo_munmap(void *ptr, uint64_t size)
{
   munmap(ptr, size);
}

int
ngv_drm_vm_create(int fd, uint32_t *vm_id)
{
   struct nicogpu_vm_create a;
   memset(&a, 0, sizeof(a));
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_VM_CREATE, &a);
   if (r)
      return r;
   *vm_id = a.vm_id;
   return 0;
}

int
ngv_drm_vm_destroy(int fd, uint32_t vm_id)
{
   struct nicogpu_vm_destroy a = {.vm_id = vm_id};
   return ngv_ioctl(fd, DRM_IOCTL_NICOGPU_VM_DESTROY, &a);
}

int
ngv_drm_vm_bind(int fd, uint32_t vm_id, const struct ngv_drm_bind_op *ops, uint32_t n)
{
   struct nicogpu_vm_bind a;
   memset(&a, 0, sizeof(a));
   a.vm_id = vm_id;
   a.num_ops = n;
   a.ops = (uint64_t)(uintptr_t)ops;
   return ngv_ioctl(fd, DRM_IOCTL_NICOGPU_VM_BIND, &a);
}

int
ngv_drm_queue_create(int fd, uint32_t vm_id, uint32_t priority, uint32_t *queue_id)
{
   struct nicogpu_queue_create a;
   memset(&a, 0, sizeof(a));
   a.vm_id = vm_id;
   a.priority = priority;
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_QUEUE_CREATE, &a);
   if (r)
      return r;
   *queue_id = a.queue_id;
   return 0;
}

int
ngv_drm_queue_destroy(int fd, uint32_t queue_id)
{
   struct nicogpu_queue_destroy a = {.queue_id = queue_id};
   return ngv_ioctl(fd, DRM_IOCTL_NICOGPU_QUEUE_DESTROY, &a);
}

int
ngv_drm_queue_status(int fd, uint32_t queue_id, uint32_t *status)
{
   struct nicogpu_queue_status a;
   memset(&a, 0, sizeof(a));
   a.queue_id = queue_id;
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_QUEUE_STATUS, &a);
   if (r)
      return r;
   *status = a.status;
   return 0;
}

int
ngv_drm_queue_fault(int fd, uint32_t queue_id, struct ngv_drm_fault *out)
{
   struct nicogpu_queue_fault a;
   memset(&a, 0, sizeof(a));
   a.queue_id = queue_id;
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_QUEUE_FAULT, &a);
   if (r)
      return r;
   out->status = a.status;
   out->cause = a.cause;
   out->cp_error = a.cp_error;
   out->flags = a.flags;
   out->err = a.err;
   out->fence_seq = a.fence_seq;
   out->mmu_code = a.mmu_code;
   out->mmu_access = a.mmu_access;
   out->mmu_level = a.mmu_level;
   out->mmu_vmid = a.mmu_vmid;
   out->mmu_va = a.mmu_va;
   out->mmu_pte = a.mmu_pte;
   out->resets = a.resets;
   out->gpu_resets = a.gpu_resets;
   return 0;
}

int
ngv_drm_syncobj_create(int fd, uint32_t flags, uint32_t *handle)
{
   struct nicogpu_syncobj_create a;
   memset(&a, 0, sizeof(a));
   a.flags = flags;
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_SYNCOBJ_CREATE, &a);
   if (r)
      return r;
   *handle = a.handle;
   return 0;
}

int
ngv_drm_syncobj_destroy(int fd, uint32_t handle)
{
   struct nicogpu_syncobj_destroy a = {.handle = handle};
   return ngv_ioctl(fd, DRM_IOCTL_NICOGPU_SYNCOBJ_DESTROY, &a);
}

int
ngv_drm_syncobj_reset(int fd, const uint32_t *handles, uint32_t count)
{
   struct nicogpu_syncobj_array a;
   memset(&a, 0, sizeof(a));
   a.handles = (uint64_t)(uintptr_t)handles;
   a.count = count;
   return ngv_ioctl(fd, DRM_IOCTL_NICOGPU_SYNCOBJ_RESET, &a);
}

int
ngv_drm_syncobj_signal(int fd, const uint32_t *handles, uint32_t count)
{
   struct nicogpu_syncobj_array a;
   memset(&a, 0, sizeof(a));
   a.handles = (uint64_t)(uintptr_t)handles;
   a.count = count;
   return ngv_ioctl(fd, DRM_IOCTL_NICOGPU_SYNCOBJ_SIGNAL, &a);
}

int
ngv_drm_syncobj_wait(int fd, const uint32_t *handles, uint32_t count, int64_t timeout_ns,
                     uint32_t flags, uint32_t *first_signaled)
{
   struct nicogpu_syncobj_wait a;
   memset(&a, 0, sizeof(a));
   a.handles = (uint64_t)(uintptr_t)handles;
   a.count = count;
   a.timeout_ns = timeout_ns;
   a.flags = flags; /* NICOGPU_SYNCOBJ_WAIT_ALL / _FOR_SUBMIT / _AVAILABLE (v5): same bits as the HAL's */
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_SYNCOBJ_WAIT, &a);
   if (!r && first_signaled)
      *first_signaled = a.first_signaled;
   return r;
}

/* the winsys names DRM core syncobjs with NICOGPU_WS_SYNC_F_DRM (nicogpu_winsys.h) and passes the flags through */
_Static_assert(NICOGPU_SYNC_F_DRM == 1u, "NICOGPU_WS_SYNC_F_DRM");

int
ngv_drm_submit(int fd, uint32_t queue_id, const struct ngv_drm_cmdbuf *cmdbufs,
               uint32_t num_cmdbufs, const struct ngv_drm_sync *waits, uint32_t num_waits,
               const struct ngv_drm_sync *signals, uint32_t num_signals, uint64_t *fence_seq)
{
   struct nicogpu_submit a;
   memset(&a, 0, sizeof(a));
   a.queue_id = queue_id;
   a.cmdbufs = (uint64_t)(uintptr_t)cmdbufs;
   a.num_cmdbufs = num_cmdbufs;
   a.waits = (uint64_t)(uintptr_t)waits;
   a.num_waits = num_waits;
   a.signals = (uint64_t)(uintptr_t)signals;
   a.num_signals = num_signals;
   int r = ngv_ioctl(fd, DRM_IOCTL_NICOGPU_SUBMIT, &a);
   if (!r && fence_seq)
      *fence_seq = a.fence_seq;
   return r;
}
