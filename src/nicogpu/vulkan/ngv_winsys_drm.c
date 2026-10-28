/* SPDX-License-Identifier: MIT */
/* ngv winsys backend "drm": native implementation on the nicogpu.ko render node
 * (PROVISIONAL, NOT RUN against a kernel).
 *
 * Semantics follow the reference winsys implementation call for call (that file is the
 * reference, test_winsys the conformance suite); what this file adds is the
 * policy the kernel does not provide and the HAL "drm" backend would otherwise
 * own: GPU-VA pool (util_vma_heap), allocate==VM_BIND bookkeeping, the sticky
 * device-lost latch, ordered teardown.  Kernel objects are used directly as
 * handles: BO handle = GEM handle, queue/syncobj ids = kernel ids.
 *
 * Kernel ABI v5 closed these gaps:
 *  - nicogpu_device_info comes from GET_GPU_INFO.  The compile-time defaults
 *    (identical to the sim backend) stay as the FALLBACK for a v4 kernel or a
 *    failing query (logged); PROVISIONAL, never trusted on other hardware.
 *  - device-wide lost comes from GET_DEV_STATUS (lost reason / time / count);
 *    queue faults and -ENODEV/-ECANCELED from submit still latch it too.
 *  - read_timestamp / _ex use GET_TIMESTAMP (calibrated timestamps).
 *  - WAIT_AVAILABLE and host syncobj_signal pass through.
 *  A v4 kernel is accepted: the v5-only calls return NICOGPU_ERR_NOTSUP.
 */
#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/hash_table.h"
#include "util/u_math.h"
#include "util/vma.h"
#include "ngv_drm_ioctl.h"
#include "ngv_winsys.h"

struct drm_bo {
   uint32_t handle; /* GEM handle == winsys handle */
   uint64_t va, size, mmap_offset;
   void *map;
   bool bound;
};

struct drm_queue {
   uint32_t id;
};

struct nicogpu_ws_device {
   int fd;
   uint32_t abi; /* kernel ABI version (4 or 5) */
   uint32_t vm;
   struct nicogpu_device_info info;
   struct util_vma_heap va;
   struct hash_table_u64 *bos; /* GEM handle -> struct drm_bo * (malloc'ed); guarded by bo_mu */
   struct drm_queue *queues;
   uint32_t num_queues, cap_queues;
   /* Thread contract (nicogpu_winsys.h): any thread may call any op.  ioctls run unlocked (the kernel serializes
    * them); the user-space state has two small locks and no lock is held across an ioctl except bo_mu, which
    * only serializes the BO ops among themselves:
    *   mu    : queue table, device-lost latch (lost*, device_gone, dst), log hook;
    *   bo_mu : BO table, GPU-VA pool, VM bind/unbind calls of the BO ops. */
   pthread_mutex_t mu;
   pthread_mutex_t bo_mu;
   bool lost;
   uint32_t lost_queue;
   struct nicogpu_queue_fault_info lost_fault;
   bool device_gone;
   struct ngv_drm_dev_status dst; /* last GET_DEV_STATUS answer */
   bool have_dst;
   nicogpu_ws_log_fn log_fn;
   void *log_user;
};

static void
dlog(struct nicogpu_ws_device *d, int level, const char *msg)
{
   pthread_mutex_lock(&d->mu);
   nicogpu_ws_log_fn fn = d->log_fn;
   void *user = d->log_user;
   pthread_mutex_unlock(&d->mu);
   if (fn)
      fn(user, level, msg);
}

static int
drm_enumerate(struct nicogpu_ws_device_desc *descs, uint32_t max_descs, uint32_t *count)
{
   if (!count || (max_descs && !descs))
      return NICOGPU_ERR_INVAL;
   char (*paths)[64] = max_descs ? calloc(max_descs, 64) : NULL;
   uint32_t n = 0;
   int r = ngv_drm_scan(paths, max_descs, &n);
   if (r) {
      free(paths);
      return r;
   }
   *count = n;
   for (uint32_t i = 0; i < n && i < max_descs; i++) {
      uint32_t sz = descs[i].struct_size;
      struct nicogpu_ws_device_desc full;
      memset(&full, 0, sizeof(full));
      full.struct_size = sz;
      snprintf(full.uri, sizeof(full.uri), "drm:%s", paths[i]);
      snprintf(full.backend, sizeof(full.backend), "drm");
      snprintf(full.name, sizeof(full.name), "NicoWorks Heathcliff GPU (%s)", paths[i]);
      full.vendor_id = 0x10ee; /* placeholder until a PCI ID is assigned (as in the sim) */
      full.device_id = 0x9000;
      memcpy(&descs[i], &full, sz < sizeof(full) ? sz : sizeof(full));
   }
   free(paths);
   return NICOGPU_OK;
}

static void
fill_default_info(struct nicogpu_device_info *info)
{
   /* FALLBACK (PROVISIONAL): kernel ABI v4 has no capability query; mirror sim_backend.cpp. */
   memset(info, 0, sizeof(*info));
   info->struct_size = sizeof(*info);
   info->api_version = NICOGPU_HAL_API_VERSION;
   snprintf(info->backend, sizeof(info->backend), "drm");
   info->vendor_id = 0x10ee;
   info->device_id = 0x9000;
   snprintf(info->isa_name, sizeof(info->isa_name), "nicogpu_isa");
   snprintf(info->isa_version, sizeof(info->isa_version), "0.1");
   info->num_cores = 4;
   info->sp_per_core = 4;
   info->warp_size = 32;
   info->warp_slots_per_sp = 16;
   info->max_vregs = 128;
   info->num_uregs = 64;
   info->max_workgroup_threads = 1024;
   info->max_shared_bytes = 64 * 1024;
   info->max_user_uregs = 26;
   info->va_bits = 48;
   info->va_user_start = 1ull << 32;      /* ABI v4: VM_BIND VA in [4 GiB, 2^47) */
   info->va_user_end = 1ull << 47;
   info->page_sizes = NICOGPU_PAGE_4K | NICOGPU_PAGE_64K | NICOGPU_PAGE_2M;
   info->vram_size = 16ull << 30;
   info->vram_visible_size = (256ull << 20) - (16ull << 20);
   info->gtt_size = 64ull << 30;
   info->timestamp_freq_hz = 100000000ull;
   /* the v4 kernel has host SYNCOBJ_SIGNAL and QUEUE_FAULT, nothing else of the v5 features */
   info->features = NICOGPU_FEATURE_SYNCOBJ_SIGNAL | NICOGPU_FEATURE_QUEUE_FAULT;
}

/* The kernel's NICOGPU_FEAT_* and the HAL's NICOGPU_FEATURE_* share bit positions 1..5; bit 0 (GPU_INFO) and 6..7 (PR_LOAD,
 * RESET: operator calls) have no HAL meaning. */
#define NGV_FEATURE_MASK (NICOGPU_FEATURE_TIMESTAMP | NICOGPU_FEATURE_SYNCOBJ_WAIT_AVAILABLE |\
                          NICOGPU_FEATURE_SYNCOBJ_SIGNAL | NICOGPU_FEATURE_DEVICE_STATUS | NICOGPU_FEATURE_QUEUE_FAULT |\
                          NICOGPU_FEATURE_SYNCOBJ_TIMELINE | NICOGPU_FEATURE_DRM_SYNCOBJ | NICOGPU_FEATURE_SYNC_FILE) /* last 3: ABI v6 */

/* nicogpu_device_info: GET_GPU_INFO (ABI v5), else the fallback defaults.  ISA name/version are not a kernel fact (the CP
 * only knows its command stream version), they stay driver constants. */
static void
fill_info(struct nicogpu_ws_device *d)
{
   struct nicogpu_device_info *i = &d->info;
   fill_default_info(i);
   if (d->abi < 5) {
      dlog(d, NICOGPU_WS_LOG_WARN, "kernel ABI < 5 has no GET_GPU_INFO: compile-time default device info (PROVISIONAL)");
      return;
   }
   struct ngv_drm_gpu_info g;
   int r = ngv_drm_get_gpu_info(d->fd, &g);
   if (r) {
      dlog(d, NICOGPU_WS_LOG_WARN, "GET_GPU_INFO failed: compile-time default device info (PROVISIONAL)");
      return;
   }
   i->vendor_id = g.vendor_id;
   i->device_id = g.device_id;
   i->revision = g.revision;
   i->num_cores = g.num_cores;
   i->sp_per_core = g.sp_per_core;
   i->warp_size = g.warp_size;
   i->warp_slots_per_sp = g.warp_slots_per_sp;
   i->max_vregs = g.max_vregs;
   i->num_uregs = g.num_uregs;
   i->max_workgroup_threads = g.max_workgroup_threads;
   i->max_shared_bytes = g.max_shared_bytes;
   i->max_user_uregs = g.max_user_uregs;
   i->va_bits = g.va_bits;
   i->va_user_start = g.va_user_start;
   i->va_user_end = g.va_user_end;
   i->page_sizes = g.page_sizes; /* kernel NICOGPU_GPU_PAGE_* == HAL NICOGPU_PAGE_* */
   i->vram_size = g.vram_size;
   i->vram_visible_size = g.vram_visible_size;
   i->gtt_size = g.gtt_size;
   i->gtt_bo_max = g.gtt_bo_max;
   i->timestamp_freq_hz = g.timestamp_freq_hz;
   i->features = g.features & NGV_FEATURE_MASK;
   i->info_flags = g.info_flags; /* kernel NICOGPU_GPU_INFO_F_* == HAL NICOGPU_INFO_F_* */
   i->sp_units = g.sp_units;     /* kernel NICOGPU_SP_UNIT_* == HAL NICOGPU_SP_UNIT_* (valid iff NICOGPU_INFO_F_SP_UNITS) */
}

static void
drm_device_destroy(struct nicogpu_ws_device *d)
{
   if (!d)
      return;
   for (uint32_t i = 0; i < d->num_queues; i++)
      ngv_drm_queue_destroy(d->fd, d->queues[i].id);
   const uint32_t num_bos = d->bos ? (uint32_t)_mesa_hash_table_u64_num_entries(d->bos) : 0;
   if (num_bos) {
      struct ngv_drm_bind_op *ops = calloc(num_bos, sizeof(*ops));
      uint32_t n = 0;
      hash_table_u64_foreach(d->bos, he)
      {
         const struct drm_bo *b = he.data;
         if (ops && b->bound) {
            ops[n].op = 1; /* NICOGPU_VM_BIND_UNMAP */
            ops[n].va = b->va;
            ops[n].range = b->size;
            n++;
         }
      }
      if (n)
         ngv_drm_vm_bind(d->fd, d->vm, ops, n);
      free(ops);
      hash_table_u64_foreach(d->bos, he)
      {
         struct drm_bo *b = he.data;
         if (b->map)
            ngv_drm_bo_munmap(b->map, b->size);
         ngv_drm_bo_free(d->fd, b->handle);
         free(b);
      }
   }
   if (d->vm)
      ngv_drm_vm_destroy(d->fd, d->vm);
   util_vma_heap_finish(&d->va);
   ngv_drm_close(d->fd);
   pthread_mutex_destroy(&d->mu);
   pthread_mutex_destroy(&d->bo_mu);
   if (d->bos)
      _mesa_hash_table_u64_destroy(d->bos);
   free(d->queues);
   free(d);
}

static int
drm_device_create(const char *uri, const struct nicogpu_ws_device_create_args *args,
                  struct nicogpu_ws_device **out)
{
   (void)args;
   if (!out)
      return NICOGPU_ERR_INVAL;
   *out = NULL;
   char path[64];
   if (uri && !strncmp(uri, "drm:", 4)) {
      snprintf(path, sizeof(path), "%s", uri + 4);
   } else {
      uint32_t n = 0;
      char(*paths)[64] = calloc(1, 64);
      int sr = ngv_drm_scan(paths, 1, &n);
      if (!sr && n)
         snprintf(path, sizeof(path), "%s", paths[0]);
      free(paths);
      if (sr || !n)
         return sr ? sr : NICOGPU_ERR_NODEV;
   }
   struct nicogpu_ws_device *d = calloc(1, sizeof(*d));
   if (!d)
      return NICOGPU_ERR_NOMEM;
   d->bos = _mesa_hash_table_u64_create(NULL);
   if (!d->bos) {
      free(d);
      return NICOGPU_ERR_NOMEM;
   }
   pthread_mutex_init(&d->mu, NULL);
   pthread_mutex_init(&d->bo_mu, NULL);
   int r = ngv_drm_open(path, &d->fd, &d->abi);
   if (r) {
      pthread_mutex_destroy(&d->mu);
      pthread_mutex_destroy(&d->bo_mu);
      _mesa_hash_table_u64_destroy(d->bos);
      free(d);
      return r;
   }
   fill_info(d);
   r = ngv_drm_vm_create(d->fd, &d->vm);
   if (r) {
      ngv_drm_close(d->fd);
      pthread_mutex_destroy(&d->mu);
      pthread_mutex_destroy(&d->bo_mu);
      _mesa_hash_table_u64_destroy(d->bos);
      free(d);
      return r;
   }
   util_vma_heap_init(&d->va, d->info.va_user_start, d->info.va_user_end - d->info.va_user_start);
   *out = d;
   return NICOGPU_OK;
}

static int
drm_device_get_info(struct nicogpu_ws_device *d, struct nicogpu_device_info *info)
{
   if (!d || !info || info->struct_size < sizeof(uint32_t))
      return NICOGPU_ERR_INVAL;
   uint32_t sz = info->struct_size;
   memcpy(info, &d->info, sz < sizeof(d->info) ? sz : sizeof(d->info));
   info->struct_size = sz;
   return NICOGPU_OK;
}

static int
drm_device_read_timestamp_ex(struct nicogpu_ws_device *d, struct nicogpu_timestamp_info *info)
{
   if (!d || !info || info->struct_size < offsetof(struct nicogpu_timestamp_info, ts))
      return NICOGPU_ERR_INVAL;
   if (d->abi < 5)
      return NICOGPU_ERR_NOTSUP; /* no GET_TIMESTAMP before ABI v5 */
   struct ngv_drm_timestamp t = {.clock_id = info->clock_id};
   int r = ngv_drm_timestamp(d->fd, &t);
   if (r)
      return r;
   struct nicogpu_timestamp_info full;
   memset(&full, 0, sizeof(full));
   full.struct_size = info->struct_size;
   full.clock_id = info->clock_id;
   full.ts = t.ts;
   full.freq_hz = t.freq_hz;
   full.cpu_ns_before = t.cpu_ns_before;
   full.cpu_ns_after = t.cpu_ns_after;
   memcpy(info, &full, info->struct_size < sizeof(full) ? info->struct_size : sizeof(full));
   return NICOGPU_OK;
}

static int
drm_device_read_timestamp(struct nicogpu_ws_device *d, uint64_t *ts)
{
   if (!d || !ts)
      return NICOGPU_ERR_INVAL;
   struct nicogpu_timestamp_info t;
   memset(&t, 0, sizeof(t));
   t.struct_size = sizeof(t);
   t.clock_id = 1; /* CLOCK_MONOTONIC */
   int r = drm_device_read_timestamp_ex(d, &t);
   if (r)
      return r;
   *ts = t.ts;
   return NICOGPU_OK;
}

static struct drm_bo *
find_bo(struct nicogpu_ws_device *d, uint32_t h)
{
   return _mesa_hash_table_u64_search(d->bos, h);
}

/* Latches the device-lost state (first guilty queue wins).  The fault record is fetched before taking the lock
 * (ioctl), then everything is published at once, so a concurrent reader never sees "lost" without its record. */
static void
mark_lost(struct nicogpu_ws_device *d, uint32_t queue_id, const char *why)
{
   pthread_mutex_lock(&d->mu);
   const bool already = d->lost;
   pthread_mutex_unlock(&d->mu);
   if (already)
      return;
   struct nicogpu_queue_fault_info rec;
   memset(&rec, 0, sizeof(rec));
   struct ngv_drm_fault f;
   const bool have_rec = !ngv_drm_queue_fault(d->fd, queue_id, &f);
   bool gone = false;
   if (have_rec) {
      struct nicogpu_queue_fault_info *o = &rec;
      o->struct_size = sizeof(*o);
      o->status = f.status;
      o->cause = f.cause;
      o->cp_error = f.cp_error;
      o->flags = f.flags;
      o->err = f.err;
      o->fence_seq = f.fence_seq;
      o->mmu_code = f.mmu_code;
      o->mmu_access = f.mmu_access;
      o->mmu_level = f.mmu_level;
      o->mmu_vmid = f.mmu_vmid;
      o->mmu_va = f.mmu_va;
      o->mmu_pte = f.mmu_pte;
      o->resets = f.resets;
      o->gpu_resets = f.gpu_resets;
      gone = (f.flags & NICOGPU_FAULT_F_DEVICE_LOST) != 0;
   }
   pthread_mutex_lock(&d->mu);
   const bool won = !d->lost;
   if (won) {
      d->lost = true;
      d->lost_queue = queue_id;
      if (have_rec)
         d->lost_fault = rec;
      if (gone)
         d->device_gone = true;
   }
   pthread_mutex_unlock(&d->mu);
   if (won)
      dlog(d, NICOGPU_WS_LOG_ERROR, why);
}

static void
poll_queues(struct nicogpu_ws_device *d)
{
   /* snapshot the table under the lock, ask the kernel without it */
   uint32_t stack[16];
   uint32_t *ids = stack;
   pthread_mutex_lock(&d->mu);
   uint32_t n = d->num_queues;
   if (n > 16)
      ids = malloc(n * sizeof(*ids));
   if (!ids) {
      ids = stack;
      n = 16;
   }
   for (uint32_t i = 0; i < n; i++)
      ids[i] = d->queues[i].id;
   pthread_mutex_unlock(&d->mu);
   for (uint32_t i = 0; i < n; i++) {
      uint32_t st = 0;
      if (!ngv_drm_queue_status(d->fd, ids[i], &st) && st == NICOGPU_QUEUE_STATUS_GUILTY)
         mark_lost(d, ids[i], "queue is guilty of a GPU fault");
   }
   if (ids != stack)
      free(ids);
}

/* Whole-device state (GET_DEV_STATUS, ABI v5): sticky like the queue latch.  -ENODEV from the ioctl means the DRM
 * device was unplugged. */
static void
poll_device(struct nicogpu_ws_device *d)
{
   if (d->abi < 5)
      return;
   struct ngv_drm_dev_status st;
   int r = ngv_drm_dev_status(d->fd, &st);
   bool log_gone = false, log_lost = false;
   pthread_mutex_lock(&d->mu);
   if (r == NICOGPU_ERR_NODEV) {
      log_gone = !d->device_gone;
      d->device_gone = true;
   } else if (!r) {
      d->dst = st;
      d->have_dst = true;
      if ((st.status & NGV_DRM_DEVSTAT_LOST) && !d->device_gone) {
         log_lost = true;
         d->device_gone = true;
      }
   }
   pthread_mutex_unlock(&d->mu);
   if (log_gone)
      dlog(d, NICOGPU_WS_LOG_ERROR, "device lost: the DRM device is gone (ioctl -ENODEV)");
   if (log_lost)
      dlog(d, NICOGPU_WS_LOG_ERROR, "device lost: the kernel reports the whole device gone (see GET_DEV_STATUS)");
}

static int
drm_device_is_lost(struct nicogpu_ws_device *d)
{
   poll_queues(d);
   poll_device(d);
   pthread_mutex_lock(&d->mu);
   const bool lost = d->lost || d->device_gone;
   pthread_mutex_unlock(&d->mu);
   return lost ? 1 : 0;
}

static int
drm_device_get_lost_info(struct nicogpu_ws_device *d, struct nicogpu_ws_lost_info *out)
{
   if (!d || !out || out->struct_size < sizeof(uint32_t))
      return NICOGPU_ERR_INVAL;
   poll_queues(d);
   poll_device(d);
   struct nicogpu_ws_lost_info full;
   memset(&full, 0, sizeof(full));
   full.struct_size = out->struct_size;
   pthread_mutex_lock(&d->mu); /* one consistent snapshot of the latch */
   full.reason = d->device_gone ? NICOGPU_WS_LOST_DEVICE : d->lost ? NICOGPU_WS_LOST_QUEUE_FAULT : NICOGPU_WS_LOST_NONE;
   full.queue = d->lost_queue;
   full.fault = d->lost_fault;
   full.status.struct_size = sizeof(full.status);
   if (d->have_dst) { /* the kernel's own account: reason, error, time, count */
      full.status.flags = d->dst.status & (NICOGPU_DEVICE_STATUS_LOST | NICOGPU_DEVICE_STATUS_PR_WEDGED |
                                           NICOGPU_DEVICE_STATUS_IO_ERROR);
      full.status.resets = d->dst.resets;
      full.status.gpu_resets = d->dst.gpu_resets;
      full.status.lost_reason = d->dst.lost_reason;
      full.status.lost_err = d->dst.lost_err;
      full.status.lost_time_ns = d->dst.lost_time_ns;
      full.status.lost_count = d->dst.lost_count;
      full.status.now_ns = d->dst.now_ns;
   } else {
      full.status.resets = d->lost_fault.resets;
      full.status.gpu_resets = d->lost_fault.gpu_resets;
   }
   if (d->device_gone)
      full.status.flags |= NICOGPU_DEVICE_STATUS_LOST;
   pthread_mutex_unlock(&d->mu);
   memcpy(out, &full, out->struct_size < sizeof(full) ? out->struct_size : sizeof(full));
   return NICOGPU_OK;
}

static int
drm_set_log_fn(struct nicogpu_ws_device *d, nicogpu_ws_log_fn fn, void *user)
{
   pthread_mutex_lock(&d->mu);
   d->log_fn = fn;
   d->log_user = user;
   pthread_mutex_unlock(&d->mu);
   return NICOGPU_OK;
}

static int
drm_bo_alloc_locked(struct nicogpu_ws_device *d, const struct nicogpu_ws_bo_args *a, struct nicogpu_ws_bo_info *out)
{
   if (!d || !a || a->struct_size < sizeof(*a) || !out || out->struct_size < sizeof(*out))
      return NICOGPU_ERR_INVAL;
   if (!a->size || (a->domain != NICOGPU_BO_DOMAIN_VRAM && a->domain != NICOGPU_BO_DOMAIN_GTT) ||
       (a->flags & ~NICOGPU_BO_HOST_VISIBLE) || (a->domain == NICOGPU_BO_DOMAIN_GTT && a->flags) ||
       (a->vm_flags & ~(NICOGPU_VM_READ | NICOGPU_VM_WRITE | NICOGPU_VM_EXEC | NICOGPU_VM_UNCACHED)) ||
       (a->alignment && (a->alignment < NICOGPU_PAGE_SIZE || (a->alignment & (a->alignment - 1)))))
      return NICOGPU_ERR_INVAL;
   const uint64_t size = align64(a->size, NICOGPU_PAGE_SIZE);
   if (size < a->size)
      return NICOGPU_ERR_INVAL;
   /* Same size classes as the model's winsys/HAL placement. */
   const uint64_t va_align = a->alignment ? a->alignment : nicogpu_default_va_align(size);
   uint64_t va = a->va;
   if (va) {
      if ((va % va_align) || va < d->info.va_user_start || va + size > d->info.va_user_end || va + size < va)
         return NICOGPU_ERR_INVAL;
      if (!util_vma_heap_alloc_addr(&d->va, va, size))
         return NICOGPU_ERR_NOSPC;
   } else {
      va = util_vma_heap_alloc(&d->va, size, va_align);
      if (!va)
         return NICOGPU_ERR_NOSPC;
   }

   uint32_t handle = 0;
   uint64_t got_size = 0, mmap_off = 0;
   int r = ngv_drm_bo_create(d->fd, a->size, a->alignment, a->domain, a->flags, &handle, &got_size, &mmap_off);
   if (r) {
      util_vma_heap_free(&d->va, va, size);
      return r;
   }
   struct ngv_drm_bind_op op = {.op = NICOGPU_VM_BIND_MAP, .flags = a->vm_flags, .bo_handle = handle,
                                .va = va, .range = size};
   r = ngv_drm_vm_bind(d->fd, d->vm, &op, 1);
   if (r) {
      ngv_drm_bo_free(d->fd, handle);
      util_vma_heap_free(&d->va, va, size);
      return r;
   }
   struct drm_bo *nb = malloc(sizeof(*nb));
   if (!nb) {
      struct ngv_drm_bind_op un = {.op = 1, .va = va, .range = size};
      ngv_drm_vm_bind(d->fd, d->vm, &un, 1);
      ngv_drm_bo_free(d->fd, handle);
      util_vma_heap_free(&d->va, va, size);
      return NICOGPU_ERR_NOMEM;
   }
   *nb = (struct drm_bo){.handle = handle, .va = va, .size = size, .mmap_offset = mmap_off, .bound = true};
   _mesa_hash_table_u64_insert(d->bos, handle, nb);
   out->handle = handle;
   out->va = va;
   out->size = size;
   return NICOGPU_OK;
}

static int
drm_bo_map_locked(struct nicogpu_ws_device *d, uint32_t h, void **ptr)
{
   struct drm_bo *b = find_bo(d, h);
   if (!b || !ptr)
      return b ? NICOGPU_ERR_INVAL : NICOGPU_ERR_NOENT;
   if (!b->map) {
      if (!b->mmap_offset)
         return NICOGPU_ERR_INVAL; /* VRAM without HOST_VISIBLE is not mappable */
      b->map = ngv_drm_bo_mmap(d->fd, b->mmap_offset, b->size);
      if (!b->map)
         return NICOGPU_ERR_NOMEM;
   }
   *ptr = b->map;
   return NICOGPU_OK;
}

static int
drm_bo_unmap_locked(struct nicogpu_ws_device *d, uint32_t h)
{
   return find_bo(d, h) ? NICOGPU_OK : NICOGPU_ERR_NOENT; /* mapping is kept until free (stable pointer) */
}

static int
drm_bo_free_locked(struct nicogpu_ws_device *d, uint32_t h)
{
   struct drm_bo *b = find_bo(d, h);
   if (!b)
      return NICOGPU_ERR_NOENT;
   if (b->bound) {
      struct ngv_drm_bind_op op = {.op = 1 /* UNMAP */, .va = b->va, .range = b->size};
      int r = ngv_drm_vm_bind(d->fd, d->vm, &op, 1);
      if (r)
         return r;
      util_vma_heap_free(&d->va, b->va, b->size);
      b->bound = false;
   }
   if (b->map) {
      ngv_drm_bo_munmap(b->map, b->size);
      b->map = NULL;
   }
   int r = ngv_drm_bo_free(d->fd, b->handle);
   if (r)
      return r;
   _mesa_hash_table_u64_remove(d->bos, h);
   free(b);
   return NICOGPU_OK;
}

/* BO ops: one lock for the BO table + VA pool (see struct nicogpu_ws_device); the VM_BIND / GEM ioctls inside run
 * under it, which only orders BO ops against each other. */
#define DRM_BO_WRAP(name, sig, args)   \
   static int name sig                 \
   {                                   \
      if (!d)                          \
         return NICOGPU_ERR_INVAL;        \
      pthread_mutex_lock(&d->bo_mu);   \
      int r = name##_locked args;      \
      pthread_mutex_unlock(&d->bo_mu); \
      return r;                        \
   }
DRM_BO_WRAP(drm_bo_alloc, (struct nicogpu_ws_device *d, const struct nicogpu_ws_bo_args *a, struct nicogpu_ws_bo_info *out),
            (d, a, out))
DRM_BO_WRAP(drm_bo_map, (struct nicogpu_ws_device *d, uint32_t h, void **ptr), (d, h, ptr))
DRM_BO_WRAP(drm_bo_unmap, (struct nicogpu_ws_device *d, uint32_t h), (d, h))
DRM_BO_WRAP(drm_bo_free, (struct nicogpu_ws_device *d, uint32_t h), (d, h))

static int
drm_queue_create(struct nicogpu_ws_device *d, uint32_t *h)
{
   uint32_t id = 0;
   int r = ngv_drm_queue_create(d->fd, d->vm, NICOGPU_QUEUE_PRIORITY_NORMAL, &id);
   if (r)
      return r;
   pthread_mutex_lock(&d->mu);
   if (d->num_queues == d->cap_queues) {
      uint32_t ncap = d->cap_queues ? d->cap_queues * 2 : 8;
      struct drm_queue *nq = realloc(d->queues, ncap * sizeof(*nq));
      if (!nq) {
         pthread_mutex_unlock(&d->mu);
         ngv_drm_queue_destroy(d->fd, id);
         return NICOGPU_ERR_NOMEM;
      }
      d->queues = nq;
      d->cap_queues = ncap;
   }
   d->queues[d->num_queues++].id = id;
   pthread_mutex_unlock(&d->mu);
   *h = id;
   return NICOGPU_OK;
}

static int
drm_queue_destroy(struct nicogpu_ws_device *d, uint32_t h)
{
   bool known = false;
   pthread_mutex_lock(&d->mu);
   for (uint32_t i = 0; i < d->num_queues; i++)
      if (d->queues[i].id == h)
         known = true;
   pthread_mutex_unlock(&d->mu);
   if (!known)
      return NICOGPU_ERR_NOENT;
   int r = ngv_drm_queue_destroy(d->fd, h);
   if (r)
      return r;
   pthread_mutex_lock(&d->mu);
   for (uint32_t i = 0; i < d->num_queues; i++)
      if (d->queues[i].id == h) {
         d->queues[i] = d->queues[--d->num_queues];
         break;
      }
   pthread_mutex_unlock(&d->mu);
   return NICOGPU_OK;
}

static int
drm_queue_get_status(struct nicogpu_ws_device *d, uint32_t h, uint32_t *st)
{
   int r = ngv_drm_queue_status(d->fd, h, st);
   if (!r && *st == NICOGPU_QUEUE_STATUS_GUILTY)
      mark_lost(d, h, "queue is guilty of a GPU fault");
   return r;
}

static int
drm_queue_get_fault(struct nicogpu_ws_device *d, uint32_t h, struct nicogpu_queue_fault_info *info)
{
   struct ngv_drm_fault f;
   int r = ngv_drm_queue_fault(d->fd, h, &f);
   if (r)
      return r;
   struct nicogpu_queue_fault_info full;
   memset(&full, 0, sizeof(full));
   full.struct_size = info->struct_size;
   full.status = f.status;
   full.cause = f.cause;
   full.cp_error = f.cp_error;
   full.flags = f.flags;
   full.err = f.err;
   full.fence_seq = f.fence_seq;
   full.mmu_code = f.mmu_code;
   full.mmu_access = f.mmu_access;
   full.mmu_level = f.mmu_level;
   full.mmu_vmid = f.mmu_vmid;
   full.mmu_va = f.mmu_va;
   full.mmu_pte = f.mmu_pte;
   full.resets = f.resets;
   full.gpu_resets = f.gpu_resets;
   memcpy(info, &full, info->struct_size < sizeof(full) ? info->struct_size : sizeof(full));
   if (f.status == NICOGPU_QUEUE_STATUS_GUILTY)
      mark_lost(d, h, "queue is guilty of a GPU fault");
   return NICOGPU_OK;
}

static int
drm_syncobj_create(struct nicogpu_ws_device *d, uint32_t flags, uint32_t *h)
{
   return ngv_drm_syncobj_create(d->fd, flags, h);
}

static int
drm_syncobj_destroy(struct nicogpu_ws_device *d, uint32_t h)
{
   return ngv_drm_syncobj_destroy(d->fd, h);
}

static int
drm_syncobj_reset(struct nicogpu_ws_device *d, const uint32_t *hs, uint32_t n)
{
   return ngv_drm_syncobj_reset(d->fd, hs, n);
}

static int
drm_syncobj_signal(struct nicogpu_ws_device *d, const uint32_t *hs, uint32_t n)
{
   if (!d || !hs || !n)
      return NICOGPU_ERR_INVAL;
   return ngv_drm_syncobj_signal(d->fd, hs, n);
}

static int
drm_syncobj_wait(struct nicogpu_ws_device *d, const uint32_t *hs, uint32_t n, int64_t timeout_ns,
                 uint32_t flags, uint32_t *first)
{
   if (flags & ~(NICOGPU_SYNCOBJ_WAIT_ALL | NICOGPU_SYNCOBJ_WAIT_FOR_SUBMIT | NICOGPU_SYNCOBJ_WAIT_AVAILABLE))
      return NICOGPU_ERR_INVAL;
   if ((flags & NICOGPU_SYNCOBJ_WAIT_AVAILABLE) && d->abi < 5)
      return NICOGPU_ERR_NOTSUP; /* WAIT_AVAILABLE is ABI v5 */
   return ngv_drm_syncobj_wait(d->fd, hs, n, timeout_ns, flags, first);
}

static int
drm_submit(struct nicogpu_ws_device *d, const struct nicogpu_ws_submit_args *a)
{
   if (!d || !a || a->struct_size < sizeof(*a))
      return NICOGPU_ERR_INVAL;
   /* ngv_ws_sync / ngv_drm_sync / nicogpu_cmdbuf: identical layouts (the HAL and
    * kernel types are the same by design); the winsys-vs-kernel handle
    * namespaces are also identical here (kernel ids are used directly). */
   _Static_assert(sizeof(struct nicogpu_ws_sync) == sizeof(struct ngv_drm_sync), "sync layout");
   _Static_assert(sizeof(struct nicogpu_cmdbuf) == sizeof(struct ngv_drm_cmdbuf), "cmdbuf layout");
   for (uint32_t i = 0; i < a->num_waits + a->num_signals; i++) {
      const struct nicogpu_ws_sync *s = i < a->num_waits ? &a->waits[i] : &a->signals[i - a->num_waits];
      if (s->flags & ~NICOGPU_WS_SYNC_F_DRM)
         return NICOGPU_ERR_INVAL;
      if ((s->flags & NICOGPU_WS_SYNC_F_DRM) && !(d->info.features & NICOGPU_FEATURE_DRM_SYNCOBJ))
         return NICOGPU_ERR_NOTSUP; /* ABI < 6 */
      if (!(s->flags & NICOGPU_WS_SYNC_F_DRM) && s->point)
         return NICOGPU_ERR_INVAL;
   }
   int r = ngv_drm_submit(d->fd, a->queue, (const struct ngv_drm_cmdbuf *)a->cmdbufs, a->num_cmdbufs,
                          (const struct ngv_drm_sync *)a->waits, a->num_waits,
                          (const struct ngv_drm_sync *)a->signals, a->num_signals, NULL);
   if (r == NICOGPU_ERR_CANCELED || r == NICOGPU_ERR_NODEV)
      mark_lost(d, a->queue, "submit refused: queue guilty or device gone");
   return r;
}

static int
drm_fd_of(struct nicogpu_ws_device *d)
{
   return d && (d->info.features & NICOGPU_FEATURE_DRM_SYNCOBJ) ? d->fd : -1;
}

const struct ngv_winsys_ops ngv_winsys_drm_ops = {
   .name = "drm",
   .enumerate = drm_enumerate,
   .device_create = drm_device_create,
   .device_destroy = drm_device_destroy,
   .device_get_info = drm_device_get_info,
   .device_read_timestamp = drm_device_read_timestamp,
   .device_read_timestamp_ex = drm_device_read_timestamp_ex,
   .device_is_lost = drm_device_is_lost,
   .device_get_lost_info = drm_device_get_lost_info,
   .set_log_fn = drm_set_log_fn,
   .bo_alloc = drm_bo_alloc,
   .bo_map = drm_bo_map,
   .bo_unmap = drm_bo_unmap,
   .bo_free = drm_bo_free,
   .queue_create = drm_queue_create,
   .queue_destroy = drm_queue_destroy,
   .queue_get_status = drm_queue_get_status,
   .queue_get_fault = drm_queue_get_fault,
   .syncobj_create = drm_syncobj_create,
   .syncobj_destroy = drm_syncobj_destroy,
   .syncobj_reset = drm_syncobj_reset,
   .syncobj_signal = drm_syncobj_signal,
   .syncobj_wait = drm_syncobj_wait,
   .submit = drm_submit,
   .drm_fd = drm_fd_of,
};
