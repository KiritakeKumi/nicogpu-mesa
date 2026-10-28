/* SPDX-License-Identifier: MIT */
/* ngv winsys backend "stub": a fake device with nothing behind it.
 *
 * Purpose: let the Mesa skeleton be built, loaded by the Vulkan loader and
 * smoke-tested (vulkaninfo, vkCreateDevice, vkAllocateMemory, fences) on a
 * machine that has neither the model nor the board.  Select explicitly with
 * NGV_WINSYS=stub.  Since batch 2 the stub has a software CP (ngv_stub_cp.c):
 * command buffers are interpreted at submit time (DISPATCH only for the
 * placeholder shader blob; real code objects and unknown packets stop the CP
 * with an error, they are never silently skipped).  Every op is serialized by
 * a per-device mutex and syncobj waits block (condvar), so the Vulkan
 * runtime's submit thread can be used against it.  Device facts mirror the "sim"
 * HAL backend of the reference simulator, so the properties and memory
 * heaps the driver reports are the ones the model reports.
 */
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ngv_stub_cp.h"
#include "ngv_winsys.h"

#define STUB_MAX_OBJS 4096
#define STUB_VA_START 0x100000000ull
#define STUB_VA_END   0x800000000000ull

struct stub_bo {
   int used;
   uint64_t size, va;
   void *mem;
};

struct stub_sync {
   int used;
   int has_fence;
};

struct nicogpu_ws_device {
   pthread_mutex_t mu;
   pthread_cond_t cv; /* signaled whenever a syncobj gains a fence */
   struct stub_bo *bos;
   struct stub_sync *syncs;
   int *queues;
   uint64_t next_va;
   uint64_t ts;
   struct ngv_stub_cp cp;
   int lost;
   uint32_t guilty_queue;
   struct nicogpu_queue_fault_info fault;
};

static int
stub_enumerate(struct nicogpu_ws_device_desc *descs, uint32_t max_descs, uint32_t *count)
{
   if (!count || (max_descs && !descs))
      return NICOGPU_ERR_INVAL;
   *count = 1;
   if (max_descs) {
      memset(&descs[0], 0, sizeof(descs[0]));
      descs[0].struct_size = sizeof(descs[0]);
      snprintf(descs[0].uri, sizeof(descs[0].uri), "stub");
      snprintf(descs[0].backend, sizeof(descs[0].backend), "stub");
      snprintf(descs[0].name, sizeof(descs[0].name), "NicoWorks Heathcliff GPU (stub winsys, no device)");
      descs[0].vendor_id = 0x10ee;
      descs[0].device_id = 0x9000;
   }
   return NICOGPU_OK;
}

/* VA -> host pointer over the BO table (the stub's "MMU"). */
static void *
stub_xlate(void *ctx, uint64_t va, uint64_t *avail)
{
   struct nicogpu_ws_device *d = ctx;
   for (int i = 1; i < STUB_MAX_OBJS; i++) {
      const struct stub_bo *b = &d->bos[i];
      if (b->used && va >= b->va && va < b->va + b->size) {
         *avail = b->va + b->size - va;
         return (char *)b->mem + (va - b->va);
      }
   }
   return NULL;
}

static int
stub_device_create(const char *uri, const struct nicogpu_ws_device_create_args *args,
                   struct nicogpu_ws_device **out)
{
   (void)uri;
   (void)args;
   struct nicogpu_ws_device *d = calloc(1, sizeof(*d));
   if (!d)
      return NICOGPU_ERR_NOMEM;
   d->bos = calloc(STUB_MAX_OBJS, sizeof(*d->bos));
   d->syncs = calloc(STUB_MAX_OBJS, sizeof(*d->syncs));
   d->queues = calloc(STUB_MAX_OBJS, sizeof(*d->queues));
   if (!d->bos || !d->syncs || !d->queues) {
      free(d->bos);
      free(d->syncs);
      free(d->queues);
      free(d);
      return NICOGPU_ERR_NOMEM;
   }
   pthread_mutex_init(&d->mu, NULL);
   pthread_condattr_t ca;
   pthread_condattr_init(&ca);
   pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
   pthread_cond_init(&d->cv, &ca);
   pthread_condattr_destroy(&ca);
   d->next_va = STUB_VA_START;
   d->cp.xlate = stub_xlate;
   d->cp.ctx = d;
   *out = d;
   return NICOGPU_OK;
}

static void
stub_device_destroy(struct nicogpu_ws_device *d)
{
   if (!d)
      return;
   for (int i = 0; i < STUB_MAX_OBJS; i++)
      if (d->bos[i].used)
         free(d->bos[i].mem);
   pthread_cond_destroy(&d->cv);
   pthread_mutex_destroy(&d->mu);
   free(d->bos);
   free(d->syncs);
   free(d->queues);
   free(d);
}

static int
stub_device_get_info(struct nicogpu_ws_device *d, struct nicogpu_device_info *info)
{
   (void)d;
   if (!info || info->struct_size < sizeof(uint32_t))
      return NICOGPU_ERR_INVAL;
   struct nicogpu_device_info full;
   memset(&full, 0, sizeof(full));
   full.struct_size = sizeof(full);
   full.api_version = NICOGPU_HAL_API_VERSION;
   snprintf(full.backend, sizeof(full.backend), "stub");
   full.vendor_id = 0x10ee;
   full.device_id = 0x9000;
   snprintf(full.isa_name, sizeof(full.isa_name), "nicogpu_isa");
   snprintf(full.isa_version, sizeof(full.isa_version), "0.1");
   full.num_cores = 4;
   full.sp_per_core = 4;
   full.warp_size = 32;
   full.warp_slots_per_sp = 16;
   full.max_vregs = 128;
   full.num_uregs = 64;
   full.max_workgroup_threads = 1024;
   full.max_shared_bytes = 64 * 1024;
   full.max_user_uregs = 26;
   full.va_bits = 48;
   full.va_user_start = STUB_VA_START;
   full.va_user_end = STUB_VA_END;
   full.page_sizes = NICOGPU_PAGE_4K | NICOGPU_PAGE_64K | NICOGPU_PAGE_2M;
   full.vram_size = 16ull << 30;
   full.vram_visible_size = (256ull << 20) - (16ull << 20); /* BAR2 minus the kernel C3 window */
   full.gtt_size = 64ull << 30;
   full.timestamp_freq_hz = 100000000ull;
   full.features = NICOGPU_FEATURE_TIMESTAMP | NICOGPU_FEATURE_SYNCOBJ_WAIT_AVAILABLE | NICOGPU_FEATURE_SYNCOBJ_SIGNAL;
   /* test hook: $NGV_STUB_FEATURES_MASK (hex) is ANDed in, to exercise the driver's features-bit degradation */
   const char *mask = getenv("NGV_STUB_FEATURES_MASK");
   if (mask)
      full.features &= strtoull(mask, NULL, 16);
   memcpy(info, &full, info->struct_size < sizeof(full) ? info->struct_size : sizeof(full));
   return NICOGPU_OK;
}

static int
stub_device_read_timestamp_l(struct nicogpu_ws_device *d, uint64_t *ts)
{
   *ts = d->cp.timestamp += 1000;
   return NICOGPU_OK;
}

static int
stub_device_read_timestamp_ex_l(struct nicogpu_ws_device *d, struct nicogpu_timestamp_info *info)
{
   if (!info || info->struct_size < offsetof(struct nicogpu_timestamp_info, ts))
      return NICOGPU_ERR_INVAL;
   if (info->clock_id != 0 && info->clock_id != 1 && info->clock_id != 4)
      return NICOGPU_ERR_INVAL;
   struct nicogpu_timestamp_info full;
   memset(&full, 0, sizeof(full));
   full.struct_size = info->struct_size;
   full.clock_id = info->clock_id;
   full.ts = d->cp.timestamp += 1000;
   full.freq_hz = 100000000ull;
   struct timespec t0, t1;
   clock_gettime(info->clock_id == 4 ? CLOCK_MONOTONIC_RAW : CLOCK_MONOTONIC, &t0);
   clock_gettime(info->clock_id == 4 ? CLOCK_MONOTONIC_RAW : CLOCK_MONOTONIC, &t1);
   full.cpu_ns_before = (uint64_t)t0.tv_sec * 1000000000ull + (uint64_t)t0.tv_nsec;
   full.cpu_ns_after = (uint64_t)t1.tv_sec * 1000000000ull + (uint64_t)t1.tv_nsec;
   memcpy(info, &full, info->struct_size < sizeof(full) ? info->struct_size : sizeof(full));
   return NICOGPU_OK;
}

static int
stub_device_is_lost_l(struct nicogpu_ws_device *d)
{
   return d->lost;
}

static int
stub_device_get_lost_info_l(struct nicogpu_ws_device *d, struct nicogpu_ws_lost_info *out)
{
   (void)d;
   uint32_t sz = out->struct_size;
   struct nicogpu_ws_lost_info full;
   memset(&full, 0, sizeof(full));
   full.struct_size = sz;
   if (d->lost) {
      full.reason = NICOGPU_WS_LOST_QUEUE_FAULT;
      full.queue = d->guilty_queue;
      full.fault = d->fault;
   }
   memcpy(out, &full, sz < sizeof(full) ? sz : sizeof(full));
   return NICOGPU_OK;
}

static int
stub_set_log_fn(struct nicogpu_ws_device *d, nicogpu_ws_log_fn fn, void *user)
{
   (void)d, (void)fn, (void)user;
   return NICOGPU_OK;
}

static int
stub_bo_alloc_l(struct nicogpu_ws_device *d, const struct nicogpu_ws_bo_args *a, struct nicogpu_ws_bo_info *out)
{
   if (!a->size)
      return NICOGPU_ERR_INVAL;
   uint64_t size = (a->size + NICOGPU_PAGE_SIZE - 1) & ~(NICOGPU_PAGE_SIZE - 1);
   uint64_t align = a->alignment ? a->alignment : NICOGPU_PAGE_SIZE;
   for (int i = 1; i < STUB_MAX_OBJS; i++) {
      struct stub_bo *b = &d->bos[i];
      if (b->used)
         continue;
      if (posix_memalign(&b->mem, align < NICOGPU_PAGE_SIZE ? NICOGPU_PAGE_SIZE : align, size))
         return NICOGPU_ERR_NOMEM;
      memset(b->mem, 0, size);
      b->va = a->va ? a->va : ((d->next_va + align - 1) & ~(align - 1));
      d->next_va = b->va + size;
      b->size = size;
      b->used = 1;
      out->handle = i;
      out->va = b->va;
      out->size = size;
      return NICOGPU_OK;
   }
   return NICOGPU_ERR_NOMEM;
}

static struct stub_bo *
stub_bo(struct nicogpu_ws_device *d, uint32_t h)
{
   return (h > 0 && h < STUB_MAX_OBJS && d->bos[h].used) ? &d->bos[h] : NULL;
}

static int
stub_bo_map_l(struct nicogpu_ws_device *d, uint32_t h, void **ptr)
{
   struct stub_bo *b = stub_bo(d, h);
   if (!b)
      return NICOGPU_ERR_NOENT;
   *ptr = b->mem;
   return NICOGPU_OK;
}

static int
stub_bo_unmap_l(struct nicogpu_ws_device *d, uint32_t h)
{
   return stub_bo(d, h) ? NICOGPU_OK : NICOGPU_ERR_NOENT;
}

static int
stub_bo_free_l(struct nicogpu_ws_device *d, uint32_t h)
{
   struct stub_bo *b = stub_bo(d, h);
   if (!b)
      return NICOGPU_ERR_NOENT;
   free(b->mem);
   memset(b, 0, sizeof(*b));
   return NICOGPU_OK;
}

static int
stub_queue_create_l(struct nicogpu_ws_device *d, uint32_t *h)
{
   /* test hook: $NGV_STUB_NO_CP=1 behaves like a device without a command processor (queue_create -> -ENODEV) */
   const char *nocp = getenv("NGV_STUB_NO_CP");
   if (nocp && *nocp && strcmp(nocp, "0"))
      return NICOGPU_ERR_NODEV;
   for (int i = 1; i < STUB_MAX_OBJS; i++)
      if (!d->queues[i]) {
         d->queues[i] = 1;
         *h = i;
         return NICOGPU_OK;
      }
   return NICOGPU_ERR_NOMEM;
}

static int
stub_queue_destroy_l(struct nicogpu_ws_device *d, uint32_t h)
{
   if (h == 0 || h >= STUB_MAX_OBJS || !d->queues[h])
      return NICOGPU_ERR_NOENT;
   d->queues[h] = 0;
   return NICOGPU_OK;
}

static int
stub_queue_get_status_l(struct nicogpu_ws_device *d, uint32_t h, uint32_t *st)
{
   if (h == 0 || h >= STUB_MAX_OBJS || !d->queues[h])
      return NICOGPU_ERR_NOENT;
   *st = (d->lost && d->guilty_queue == h) ? d->fault.status : NICOGPU_QUEUE_STATUS_OK;
   return NICOGPU_OK;
}

static int
stub_queue_get_fault_l(struct nicogpu_ws_device *d, uint32_t h, struct nicogpu_queue_fault_info *info)
{
   if (h == 0 || h >= STUB_MAX_OBJS || !d->queues[h])
      return NICOGPU_ERR_NOENT;
   uint32_t sz = info->struct_size;
   struct nicogpu_queue_fault_info full;
   memset(&full, 0, sizeof(full));
   if (d->lost && d->guilty_queue == h)
      full = d->fault;
   full.struct_size = sz;
   memcpy(info, &full, sz < sizeof(full) ? sz : sizeof(full));
   return NICOGPU_OK;
}

static struct stub_sync *
stub_sync(struct nicogpu_ws_device *d, uint32_t h)
{
   return (h > 0 && h < STUB_MAX_OBJS && d->syncs[h].used) ? &d->syncs[h] : NULL;
}

static int
stub_syncobj_create_l(struct nicogpu_ws_device *d, uint32_t flags, uint32_t *h)
{
   for (int i = 1; i < STUB_MAX_OBJS; i++)
      if (!d->syncs[i].used) {
         d->syncs[i].used = 1;
         d->syncs[i].has_fence = !!(flags & NICOGPU_SYNCOBJ_CREATE_SIGNALED);
         *h = i;
         return NICOGPU_OK;
      }
   return NICOGPU_ERR_NOMEM;
}

static int
stub_syncobj_destroy_l(struct nicogpu_ws_device *d, uint32_t h)
{
   if (!stub_sync(d, h))
      return NICOGPU_ERR_NOENT;
   d->syncs[h].used = 0;
   return NICOGPU_OK;
}

static int
stub_syncobj_reset_l(struct nicogpu_ws_device *d, const uint32_t *hs, uint32_t n)
{
   for (uint32_t i = 0; i < n; i++)
      if (!stub_sync(d, hs[i]))
         return NICOGPU_ERR_NOENT;
   for (uint32_t i = 0; i < n; i++)
      d->syncs[hs[i]].has_fence = 0;
   return NICOGPU_OK;
}

static int
stub_device_read_timestamp(struct nicogpu_ws_device *d, uint64_t *ts)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_device_read_timestamp_l(d, ts);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_device_read_timestamp_ex(struct nicogpu_ws_device *d, struct nicogpu_timestamp_info *info)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_device_read_timestamp_ex_l(d, info);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_device_is_lost(struct nicogpu_ws_device *d)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_device_is_lost_l(d);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_device_get_lost_info(struct nicogpu_ws_device *d, struct nicogpu_ws_lost_info *out)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_device_get_lost_info_l(d, out);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_bo_alloc(struct nicogpu_ws_device *d, const struct nicogpu_ws_bo_args *a, struct nicogpu_ws_bo_info *out)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_bo_alloc_l(d, a, out);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_bo_map(struct nicogpu_ws_device *d, uint32_t h, void **ptr)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_bo_map_l(d, h, ptr);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_bo_unmap(struct nicogpu_ws_device *d, uint32_t h)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_bo_unmap_l(d, h);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_bo_free(struct nicogpu_ws_device *d, uint32_t h)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_bo_free_l(d, h);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_queue_create(struct nicogpu_ws_device *d, uint32_t *h)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_queue_create_l(d, h);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_queue_destroy(struct nicogpu_ws_device *d, uint32_t h)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_queue_destroy_l(d, h);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_queue_get_status(struct nicogpu_ws_device *d, uint32_t h, uint32_t *st)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_queue_get_status_l(d, h, st);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_queue_get_fault(struct nicogpu_ws_device *d, uint32_t h, struct nicogpu_queue_fault_info *info)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_queue_get_fault_l(d, h, info);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_syncobj_create(struct nicogpu_ws_device *d, uint32_t flags, uint32_t *h)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_syncobj_create_l(d, flags, h);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_syncobj_destroy(struct nicogpu_ws_device *d, uint32_t h)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_syncobj_destroy_l(d, h);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_syncobj_reset(struct nicogpu_ws_device *d, const uint32_t *hs, uint32_t n)
{
   pthread_mutex_lock(&d->mu);
   int r = stub_syncobj_reset_l(d, hs, n);
   pthread_mutex_unlock(&d->mu);
   return r;
}

static int
stub_syncobj_signal(struct nicogpu_ws_device *d, const uint32_t *hs, uint32_t n)
{
   pthread_mutex_lock(&d->mu);
   int r = NICOGPU_OK;
   for (uint32_t i = 0; i < n; i++)
      if (!stub_sync(d, hs[i]))
         r = NICOGPU_ERR_NOENT;
   if (!r) {
      for (uint32_t i = 0; i < n; i++)
         d->syncs[hs[i]].has_fence = 1;
      pthread_cond_broadcast(&d->cv);
   }
   pthread_mutex_unlock(&d->mu);
   return r;
}

/* Is the wait condition met?  Sets *err for a hard error. */
static bool
wait_ready(struct nicogpu_ws_device *d, const uint32_t *hs, uint32_t n, uint32_t flags, uint32_t *first, int *err)
{
   int all = 1, first_idx = -1;
   *err = NICOGPU_OK;
   for (uint32_t i = 0; i < n; i++) {
      struct stub_sync *s = stub_sync(d, hs[i]);
      if (!s) {
         *err = NICOGPU_ERR_NOENT;
         return false;
      }
      if (!s->has_fence && !(flags & NICOGPU_SYNCOBJ_WAIT_FOR_SUBMIT)) {
         *err = NICOGPU_ERR_INVAL;
         return false;
      }
      if (s->has_fence && first_idx < 0)
         first_idx = (int)i;
      all &= s->has_fence;
   }
   if ((flags & NICOGPU_SYNCOBJ_WAIT_ALL) ? all : first_idx >= 0) {
      if (first)
         *first = first_idx < 0 ? 0 : (uint32_t)first_idx;
      return true;
   }
   return false;
}

/* Blocking wait with the HAL semantics: a fence-less handle is INVAL, or "not signaled yet" with WAIT_FOR_SUBMIT
 * (then the wait blocks until a submit attaches a fence).  WAIT_AVAILABLE == signaled here, because the stub
 * attaches a fence and completes the job in the same locked section. */
static int
stub_syncobj_wait(struct nicogpu_ws_device *d, const uint32_t *hs, uint32_t n, int64_t timeout_ns, uint32_t flags,
                  uint32_t *first)
{
   if (flags & ~(NICOGPU_SYNCOBJ_WAIT_ALL | NICOGPU_SYNCOBJ_WAIT_FOR_SUBMIT | NICOGPU_SYNCOBJ_WAIT_AVAILABLE))
      return NICOGPU_ERR_INVAL;
   if (flags & NICOGPU_SYNCOBJ_WAIT_AVAILABLE)
      flags |= NICOGPU_SYNCOBJ_WAIT_FOR_SUBMIT;
   struct timespec deadline;
   bool infinite = timeout_ns == NICOGPU_TIMEOUT_INFINITE || timeout_ns < 0;
   if (!infinite) {
      clock_gettime(CLOCK_MONOTONIC, &deadline);
      uint64_t ns = (uint64_t)deadline.tv_nsec + ((uint64_t)timeout_ns % 1000000000ull);
      deadline.tv_sec += (time_t)((uint64_t)timeout_ns / 1000000000ull + ns / 1000000000ull);
      deadline.tv_nsec = (long)(ns % 1000000000ull);
   }
   int ret = NICOGPU_OK;
   pthread_mutex_lock(&d->mu);
   for (;;) {
      if (wait_ready(d, hs, n, flags, first, &ret) || ret)
         break;
      if (timeout_ns == 0) {
         ret = NICOGPU_ERR_TIME;
         break;
      }
      if (infinite) {
         pthread_cond_wait(&d->cv, &d->mu);
      } else if (pthread_cond_timedwait(&d->cv, &d->mu, &deadline) == ETIMEDOUT) {
         /* last look: a signal may have raced the timeout */
         if (!wait_ready(d, hs, n, flags, first, &ret) && !ret)
            ret = NICOGPU_ERR_TIME;
         break;
      }
   }
   pthread_mutex_unlock(&d->mu);
   return ret;
}

/* Runs the IBs on the software CP and attaches the fence.  A faulted job still signals its fences (kernel
 * semantics, section 7 gap 5): the queue becomes GUILTY and the device latches "lost". */
static int
stub_submit(struct nicogpu_ws_device *d, const struct nicogpu_ws_submit_args *a)
{
   pthread_mutex_lock(&d->mu);
   int ret = NICOGPU_OK;
   if (a->queue == 0 || a->queue >= STUB_MAX_OBJS || !d->queues[a->queue])
      ret = NICOGPU_ERR_NOENT;
   for (uint32_t i = 0; !ret && i < a->num_waits; i++) {
      struct stub_sync *s = stub_sync(d, a->waits[i].handle);
      if (!s)
         ret = NICOGPU_ERR_NOENT;
      else if (!s->has_fence)
         ret = NICOGPU_ERR_INVAL;
   }
   for (uint32_t i = 0; !ret && i < a->num_signals; i++)
      if (!stub_sync(d, a->signals[i].handle))
         ret = NICOGPU_ERR_NOENT;
   if (!ret && d->lost && d->guilty_queue == a->queue)
      ret = NICOGPU_ERR_CANCELED;
   if (!ret) {
      for (uint32_t i = 0; i < a->num_cmdbufs && !d->lost; i++) {
         int r = ngv_stub_cp_exec_ib(&d->cp, a->cmdbufs[i].va, a->cmdbufs[i].size_dw);
         if (r) {
            d->lost = 1;
            d->guilty_queue = a->queue;
            memset(&d->fault, 0, sizeof(d->fault));
            d->fault.struct_size = sizeof(d->fault);
            d->fault.status = NICOGPU_QUEUE_STATUS_GUILTY;
            d->fault.cp_error = d->cp.cp_error;
            d->fault.err = r;
         }
      }
      for (uint32_t i = 0; i < a->num_signals; i++)
         d->syncs[a->signals[i].handle].has_fence = 1;
      pthread_cond_broadcast(&d->cv);
   }
   pthread_mutex_unlock(&d->mu);
   return ret;
}

const struct ngv_winsys_ops ngv_winsys_stub_ops = {
   .name = "stub",
   .enumerate = stub_enumerate,
   .device_create = stub_device_create,
   .device_destroy = stub_device_destroy,
   .device_get_info = stub_device_get_info,
   .device_read_timestamp = stub_device_read_timestamp,
   .device_read_timestamp_ex = stub_device_read_timestamp_ex,
   .device_is_lost = stub_device_is_lost,
   .device_get_lost_info = stub_device_get_lost_info,
   .set_log_fn = stub_set_log_fn,
   .bo_alloc = stub_bo_alloc,
   .bo_map = stub_bo_map,
   .bo_unmap = stub_bo_unmap,
   .bo_free = stub_bo_free,
   .queue_create = stub_queue_create,
   .queue_destroy = stub_queue_destroy,
   .queue_get_status = stub_queue_get_status,
   .queue_get_fault = stub_queue_get_fault,
   .syncobj_create = stub_syncobj_create,
   .syncobj_destroy = stub_syncobj_destroy,
   .syncobj_reset = stub_syncobj_reset,
   .syncobj_signal = stub_syncobj_signal,
   .syncobj_wait = stub_syncobj_wait,
   .submit = stub_submit,
};
