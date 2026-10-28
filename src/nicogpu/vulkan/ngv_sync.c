/* SPDX-License-Identifier: MIT */
/* ngv_sync.c -- vk_sync type over the winsys' binary syncobj (skeleton).
 * Timeline semaphores are emulated by the runtime on top of this type
 * (vk_sync_timeline); section 7 lists the known gaps. */
#include "util/os_time.h"
#include "ngv_private.h"

static VkResult
ngv_sync_init(struct vk_device *vk_dev, struct vk_sync *vk_sync, uint64_t initial_value)
{
   struct ngv_device *dev = container_of(vk_dev, struct ngv_device, vk);
   struct ngv_sync *s = container_of(vk_sync, struct ngv_sync, vk);
   int r = dev->ws.ops->syncobj_create(dev->ws.dev, initial_value ? NICOGPU_SYNCOBJ_CREATE_SIGNALED : 0, &s->handle);
   return r ? ngv_winsys_result(r) : VK_SUCCESS;
}

static void
ngv_sync_finish(struct vk_device *vk_dev, struct vk_sync *vk_sync)
{
   struct ngv_device *dev = container_of(vk_dev, struct ngv_device, vk);
   struct ngv_sync *s = container_of(vk_sync, struct ngv_sync, vk);
   dev->ws.ops->syncobj_destroy(dev->ws.dev, s->handle);
}

static VkResult
ngv_sync_reset(struct vk_device *vk_dev, struct vk_sync *vk_sync)
{
   struct ngv_device *dev = container_of(vk_dev, struct ngv_device, vk);
   struct ngv_sync *s = container_of(vk_sync, struct ngv_sync, vk);
   int r = dev->ws.ops->syncobj_reset(dev->ws.dev, &s->handle, 1);
   return r ? ngv_winsys_result(r) : VK_SUCCESS;
}

static int64_t
rel_timeout_ns(uint64_t abs_timeout_ns)
{
   if (abs_timeout_ns == UINT64_MAX)
      return NICOGPU_TIMEOUT_INFINITE;
   uint64_t now = os_time_get_nano();
   return abs_timeout_ns > now ? (int64_t)(abs_timeout_ns - now) : 0;
}

static VkResult
ngv_sync_wait_many(struct vk_device *vk_dev, uint32_t count, const struct vk_sync_wait *waits,
                   enum vk_sync_wait_flags wait_flags, uint64_t abs_timeout_ns)
{
   struct ngv_device *dev = container_of(vk_dev, struct ngv_device, vk);
   uint32_t stack[16] = {0};
   uint32_t *h = count <= 16 ? stack : malloc(count * sizeof(*h));
   if (!h)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   for (uint32_t i = 0; i < count; i++)
      h[i] = container_of(waits[i].sync, struct ngv_sync, vk)->handle;

   /* Like vk_drm_syncobj, every wait passes WAIT_FOR_SUBMIT: a fence-less
    * object is "not signaled yet" (HAL v1, D8 #2), never EINVAL. */
   uint32_t flags = 0;
   if (!(wait_flags & VK_SYNC_WAIT_ANY))
      flags |= NICOGPU_SYNCOBJ_WAIT_ALL;
   flags |= NICOGPU_SYNCOBJ_WAIT_FOR_SUBMIT;
   /* VK_SYNC_WAIT_PENDING = return once *submitted* = DRM WAIT_AVAILABLE (HAL v2 / kernel ABI v5).  The runtime only
    * passes the flag when the type declared VK_SYNC_FEATURE_WAIT_PENDING, which ngv_sync_type_init does only if the
    * winsys reports NICOGPU_FEATURE_SYNCOBJ_WAIT_AVAILABLE. */
   if (wait_flags & VK_SYNC_WAIT_PENDING)
      flags |= NICOGPU_SYNCOBJ_WAIT_AVAILABLE;
   /* A faulted job still signals its fence (kernel semantics) and a lost device may never signal at all, so the
 * wait runs in slices and looks at the device state in between and after (section 7 gap 5). */
   const int64_t slice_ns = 100 * 1000 * 1000;
   int r;
   for (;;) {
      int64_t rel = rel_timeout_ns(abs_timeout_ns);
      int64_t t = (rel == NICOGPU_TIMEOUT_INFINITE || rel > slice_ns) ? slice_ns : rel;
      r = dev->ws.ops->syncobj_wait(dev->ws.dev, h, count, t, flags, NULL);
      if (r != NICOGPU_ERR_TIME)
         break;
      if (dev->ws.ops->device_is_lost(dev->ws.dev) == 1) {
         /* A PENDING wait is the runtime's submit thread waiting for a submit to arrive.  On a lost device it must
          * not fail: that would kill the thread with submits still queued (upstream asserts on that in
          * vkDestroyDevice).  Let it through; ngv_queue_submit then signals and drops the submit. */
         r = (wait_flags & VK_SYNC_WAIT_PENDING) ? NICOGPU_OK : NICOGPU_ERR_CANCELED;
         break;
      }
      if (rel != NICOGPU_TIMEOUT_INFINITE && rel <= slice_ns)
         break; /* the caller's timeout is up */
   }
   if (r == NICOGPU_OK && !(wait_flags & VK_SYNC_WAIT_PENDING) && dev->ws.ops->device_is_lost(dev->ws.dev) == 1)
      r = NICOGPU_ERR_CANCELED;
   if (h != stack)
      free(h);
   if (r == NICOGPU_ERR_TIME)
      return VK_TIMEOUT;
   if (r == NICOGPU_ERR_CANCELED)
      return VK_ERROR_DEVICE_LOST;
   return r ? ngv_winsys_result(r) : VK_SUCCESS;
}

static VkResult
ngv_sync_wait(struct vk_device *vk_dev, struct vk_sync *vk_sync, uint64_t wait_value,
              enum vk_sync_wait_flags wait_flags, uint64_t abs_timeout_ns)
{
   struct vk_sync_wait w = {.sync = vk_sync, .wait_value = wait_value};
   return ngv_sync_wait_many(vk_dev, 1, &w, wait_flags, abs_timeout_ns);
}

static VkResult
ngv_sync_signal(struct vk_device *vk_dev, struct vk_sync *vk_sync, uint64_t value)
{
   struct ngv_device *dev = container_of(vk_dev, struct ngv_device, vk);
   struct ngv_sync *s = container_of(vk_sync, struct ngv_sync, vk);
   int r = dev->ws.ops->syncobj_signal(dev->ws.dev, &s->handle, 1);
   return r ? ngv_winsys_result(r) : VK_SUCCESS;
}

/* dst takes src's payload, src ends up reset.  The runtime requires `move` of every binary type once threaded submit
 * is on (vk_common_CreateSemaphore asserts it; the private type enables threaded submit when the winsys has
 * WAIT_AVAILABLE, i.e. stub).  The payload is the winsys syncobj itself, so the two handles are swapped and the
 * object src now holds (dst's old one) is reset. */
static VkResult
ngv_sync_move(struct vk_device *vk_dev, struct vk_sync *dst, struct vk_sync *src)
{
   struct ngv_sync *d = container_of(dst, struct ngv_sync, vk);
   struct ngv_sync *s = container_of(src, struct ngv_sync, vk);
   uint32_t h = d->handle;
   d->handle = s->handle;
   s->handle = h;
   return ngv_sync_reset(vk_dev, src);
}

/* The template: every callback the type can ever have.  CPU_SIGNAL / WAIT_PENDING are declared per device by
 * ngv_sync_type_init from the winsys' feature bits (leftover 2 + 3). */
const struct vk_sync_type ngv_sync_type = {
   .size = sizeof(struct ngv_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_GPU_WAIT | VK_SYNC_FEATURE_GPU_MULTI_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_RESET | VK_SYNC_FEATURE_WAIT_ANY |
               VK_SYNC_FEATURE_CPU_SIGNAL | VK_SYNC_FEATURE_WAIT_PENDING,
   .init = ngv_sync_init,
   .finish = ngv_sync_finish,
   .signal = ngv_sync_signal,
   .reset = ngv_sync_reset,
   .move = ngv_sync_move,
   .wait = ngv_sync_wait,
   .wait_many = ngv_sync_wait_many,
 /* no export/import (sync file / opaque fd): gap in section 7 */
};

void
ngv_sync_type_init(struct vk_sync_type *out, uint64_t hal_features)
{
   *out = ngv_sync_type;
   if (!(hal_features & NICOGPU_FEATURE_SYNCOBJ_SIGNAL)) {
      out->features &= ~(uint32_t)VK_SYNC_FEATURE_CPU_SIGNAL;
      out->signal = NULL;
   }
   if (!(hal_features & NICOGPU_FEATURE_SYNCOBJ_WAIT_AVAILABLE))
      out->features &= ~(uint32_t)VK_SYNC_FEATURE_WAIT_PENDING;
}
