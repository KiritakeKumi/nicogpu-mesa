/* SPDX-License-Identifier: MIT */
/* ngv_instance.c -- instance, physical device enumeration, queue families,
 * memory properties, device and queue (skeleton). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ngv_private.h"

#define NGV_SUBMIT_MAX_SYNCS 64u

static VkResult
ngv_physical_device_init(struct ngv_physical_device *pdev, struct ngv_instance *instance,
                         const struct ngv_winsys_ops *ops, const struct nicogpu_ws_device_desc *desc)
{
   struct vk_physical_device_dispatch_table dispatch;
   vk_physical_device_dispatch_table_from_entrypoints(&dispatch, &ngv_physical_device_entrypoints, true);
   vk_physical_device_dispatch_table_from_entrypoints(&dispatch, &vk_common_physical_device_entrypoints, false);

   VkResult result = vk_physical_device_init(&pdev->vk, &instance->vk, NULL, NULL, NULL, &dispatch);
   if (result != VK_SUCCESS)
      return result;

   pdev->ops = ops;
   pdev->desc = *desc;

   /* Probe: open, read the device facts, close.  Every VkDevice opens its own. */
   struct nicogpu_ws_device *probe = NULL;
   int r = ops->device_create(desc->uri, NULL, &probe);
   if (r) {
      vk_physical_device_finish(&pdev->vk);
      return ngv_winsys_result(r);
   }
   pdev->info.struct_size = sizeof(pdev->info);
   r = ops->device_get_info(probe, &pdev->info);
   if (r) {
      ops->device_destroy(probe);
      vk_physical_device_finish(&pdev->vk);
      return ngv_winsys_result(r);
   }

 /* Sync : on the drm winsys of a kernel with DRM core syncobjs (ABI >= 6) the runtime's vk_drm_syncobj does
    * everything -- timeline points, WAIT_PENDING, host signal, sync_file / opaque fd import and export -- on the
    * device's DRM file.  Anything else (stub, hal, older kernels, NGV_SYNC=private) keeps the private binary type
 * with the timeline emulated on top (section 7). The type probe creates and destroys a syncobj on the
    * probe's fd, so it has to run before the probe goes. */
   const char *sync_env = getenv("NGV_SYNC");
   int drm_fd = ops->drm_fd ? ops->drm_fd(probe) : -1;
   pdev->sync_drm = false;
   if (drm_fd >= 0 && (pdev->info.features & NICOGPU_FEATURE_DRM_SYNCOBJ) && (!sync_env || strcmp(sync_env, "private"))) {
      struct vk_sync_type t = vk_drm_syncobj_get_type(drm_fd);
      /* the kernel really honours points: require what the runtime needs of a type that is both CPU- and GPU-waitable */
      if ((t.features & VK_SYNC_FEATURE_BINARY) && (t.features & VK_SYNC_FEATURE_CPU_WAIT) &&
          (t.features & VK_SYNC_FEATURE_GPU_WAIT)) {
         pdev->sync_type = t;
         pdev->sync_drm = true;
      }
   }
   ops->device_destroy(probe);
   if (!pdev->sync_drm)
      ngv_sync_type_init(&pdev->sync_type, pdev->info.features);
   uint32_t nt = 0;
   pdev->sync_types[nt++] = &pdev->sync_type;
   if (!(pdev->sync_type.features & VK_SYNC_FEATURE_TIMELINE)) {
      pdev->timeline_type = vk_sync_timeline_get_type(&pdev->sync_type);
      pdev->sync_types[nt++] = &pdev->timeline_type.sync;
   }
   pdev->sync_types[nt] = NULL;
   pdev->vk.supported_sync_types = pdev->sync_types;

   ngv_get_device_extensions(pdev, &pdev->vk.supported_extensions);
   ngv_get_features(pdev, &pdev->vk.supported_features);
   ngv_get_properties(pdev, &pdev->vk.properties);
   snprintf(pdev->vk.properties.deviceName, sizeof(pdev->vk.properties.deviceName), "%s", desc->name);
   ngv_physical_device_init_uuids(pdev); /* driverName / driverInfo / driverID / the three UUIDs (ngv_uuid.c) */
   ngv_shader_cache_init(pdev);
   return VK_SUCCESS;
}

static void
ngv_destroy_physical_device(struct vk_physical_device *vk)
{
   struct vk_instance *instance = vk->instance;
   ngv_shader_cache_finish((struct ngv_physical_device *)vk);
   vk_physical_device_finish(vk);
   vk_free(&instance->alloc, vk);
}

static VkResult
ngv_enumerate_physical_devices(struct vk_instance *vk_instance)
{
   struct ngv_instance *instance = container_of(vk_instance, struct ngv_instance, vk);
   const struct ngv_winsys_ops *ops = ngv_winsys_select();
   if (!ops)
      return VK_SUCCESS; /* unknown NGV_WINSYS: no devices */

   struct nicogpu_ws_device_desc descs[8];
   uint32_t count = 0;
   for (unsigned k = 0; k < 8; k++) {
      memset(&descs[k], 0, sizeof(descs[k]));
      descs[k].struct_size = sizeof(descs[k]);
   }
   if (ops->enumerate(descs, 8, &count) != NICOGPU_OK)
      return VK_SUCCESS;
   if (count > 8)
      count = 8;
   for (uint32_t i = 0; i < count; i++) {
      struct ngv_physical_device *pdev =
         vk_zalloc(&vk_instance->alloc, sizeof(*pdev), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
      if (!pdev)
         return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
      VkResult result = ngv_physical_device_init(pdev, instance, ops, &descs[i]);
      if (result != VK_SUCCESS) {
         vk_free(&vk_instance->alloc, pdev);
         return result;
      }
      list_addtail(&pdev->vk.link, &vk_instance->physical_devices.list);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                   VkInstance *pInstance)
{
   if (!pAllocator)
      pAllocator = vk_default_allocator();
   struct ngv_instance *instance = vk_zalloc(pAllocator, sizeof(*instance), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_instance_dispatch_table dispatch;
   vk_instance_dispatch_table_from_entrypoints(&dispatch, &ngv_instance_entrypoints, true);
   vk_instance_dispatch_table_from_entrypoints(&dispatch, &vk_common_instance_entrypoints, false);
   VkResult result = vk_instance_init(&instance->vk, &ngv_instance_extensions, &dispatch, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, instance);
      return vk_error(NULL, result);
   }
   instance->vk.physical_devices.enumerate = ngv_enumerate_physical_devices;
   instance->vk.physical_devices.destroy = ngv_destroy_physical_device;
   *pInstance = ngv_instance_to_handle(instance);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyInstance(VkInstance _instance, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_instance, instance, _instance);
   if (!instance)
      return;
   vk_instance_finish(&instance->vk);
   vk_free(&instance->vk.alloc, instance);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
ngv_GetInstanceProcAddr(VkInstance _instance, const char *pName)
{
   VK_FROM_HANDLE(vk_instance, instance, _instance);
   return vk_instance_get_proc_addr(instance, &ngv_instance_entrypoints, pName);
}

PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName)
{
   return ngv_GetInstanceProcAddr(instance, pName);
}

/* ---- queue family / memory properties ----------------------------------------- */

/* Queues per family (ngv_private.h) */
static const uint32_t ngv_queue_family_queues[NGV_QUEUE_FAMILY_COUNT] = {
   NGV_QUEUE_FAMILY_UNIVERSAL_QUEUES,
   NGV_QUEUE_FAMILY_COMPUTE_QUEUES,
};

VKAPI_ATTR void VKAPI_CALL
ngv_GetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                            VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_FROM_HANDLE(ngv_physical_device, pdev_, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out, pQueueFamilyProperties, pCount);
 /* Two families with identical capabilities (; section 5 had one family with one queue): compute+transfer, no
    * graphics until P2.  Family 0 is the "universal" one, family 1 the distinct compute-only family the CTS
    * *_compute_only_queue cases look for.  All queues go to the one CP ring and the one device clock, so the
    * timestamp bits are the same everywhere. */
   for (uint32_t f = 0; f < NGV_QUEUE_FAMILY_COUNT; f++) {
      vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p)
      {
         p->queueFamilyProperties = (VkQueueFamilyProperties){
            .queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
            .queueCount = ngv_queue_family_queues[f],
 /* TIMESTAMP packets need the device clock (section 9.3): without it no timestamp queries */
            .timestampValidBits = (pdev_->info.features & NICOGPU_FEATURE_TIMESTAMP) ? 64 : 0,
            .minImageTransferGranularity = {1, 1, 1},
         };
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties *m)
{
   VK_FROM_HANDLE(ngv_physical_device, pdev, physicalDevice);
   const struct nicogpu_device_info *i = &pdev->info;
   memset(m, 0, sizeof(*m));
   /* Two layouts of the device-local memory, by what the winsys reports:
    *  - visible window smaller than VRAM (the model: 16 GiB behind a 240 MiB BAR window): two device-local heaps, the
    *    window is its own heap carved out of vram_size;
    *  - all of VRAM visible (kernel ABI v5, nicogpu_gpu_info.vram_visible_size == vram_size: the allocator only manages the
    *    BAR2 heap): ONE device-local heap, and the DEVICE_LOCAL|HOST_VISIBLE type points at it like type 0 does
    *    (resizable-BAR style), so the same memory is never advertised twice.
    * No window at all (vram_visible_size == 0): no HOST_VISIBLE VRAM type. */
   if (NGV_NO_VRAM(i)) {
      /* RM without a VRAM path: one host-memory heap/type (see ngv_private.h); VRAM types would fail BO_CREATE */
      m->memoryHeapCount = 1;
      m->memoryHeaps[0] = (VkMemoryHeap){.size = i->gtt_size, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
      m->memoryTypeCount = 1;
      m->memoryTypes[0] = (VkMemoryType){
         .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                          ((i->info_flags & NICOGPU_INFO_F_GTT_UNCACHED) ? 0 : VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
         .heapIndex = 0};
      return;
   }
   const bool all_visible = i->vram_visible_size && i->vram_visible_size >= i->vram_size;
   const uint64_t vis = (!all_visible && i->vram_visible_size < i->vram_size) ? i->vram_visible_size : 0;
   uint32_t heap_vram = 0, heap_vis = 0, heap_gtt = 0, nheaps = 0;
   heap_vram = nheaps++;
   m->memoryHeaps[heap_vram] = (VkMemoryHeap){.size = i->vram_size - vis, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
   heap_vis = heap_vram;
   if (vis) {
      heap_vis = nheaps++;
      m->memoryHeaps[heap_vis] = (VkMemoryHeap){.size = vis, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
   }
   heap_gtt = nheaps++;
   m->memoryHeaps[heap_gtt] = (VkMemoryHeap){.size = i->gtt_size, .flags = 0};
   m->memoryHeapCount = nheaps;
   m->memoryTypeCount = NGV_MEM_TYPE_COUNT;
   m->memoryTypes[NGV_MEM_TYPE_VRAM] =
      (VkMemoryType){.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .heapIndex = heap_vram};
   /* Non-coherent host (kernel NICOGPU_GPU_INFO_F_GTT_UNCACHED): the GTT mapping is CPU-uncached/write-combined, so it is
    * not HOST_CACHED (slow CPU reads); it needs no cache maintenance, hence it stays HOST_COHERENT (the spec wants one
    * HOST_VISIBLE|HOST_COHERENT type; the submit makes host writes visible).  PROVISIONAL. */
   m->memoryTypes[NGV_MEM_TYPE_GTT] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       ((i->info_flags & NICOGPU_INFO_F_GTT_UNCACHED) ? 0 : VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
      .heapIndex = heap_gtt};
   m->memoryTypes[NGV_MEM_TYPE_VRAM_VIS] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
      .heapIndex = heap_vis};
   if (!vis && !all_visible)
      m->memoryTypeCount = 2; /* no CPU-visible window: only VRAM and GTT */
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties2 *p)
{
   ngv_GetPhysicalDeviceMemoryProperties(physicalDevice, &p->memoryProperties);
}

/* ---- queue -------------------------------------------------------------------- */

/* The device is (or just became) lost and this submit cannot run.  Latch the loss in the runtime, signal everything
 * the submit would have signaled (so no waiter on a dead device blocks for good: the same "faulted job still
 * signals its fence" contract the kernel gives) and report SUCCESS.  Returning an error from driver_submit would
 * kill the runtime's submit thread with submits still queued, and the upstream vk_queue_stop_submit_thread asserts
 * on that in vkDestroyDevice (section 13, PROVISIONAL 19, resolved). The loss is reported by
 * vkQueueWaitIdle / the next submit / the waits, all VK_ERROR_DEVICE_LOST. */
static VkResult
ngv_queue_submit_lost(struct ngv_queue *queue, const struct vk_queue_submit *submit, const char *why, int err)
{
   struct ngv_device *dev = queue->device;
   vk_queue_set_lost(&queue->vk, "%s (%d)", why, err);
   /* best effort: a sync type without host signal (NICOGPU_FEATURE_SYNCOBJ_SIGNAL absent) cannot do more */
   for (uint32_t i = 0; i < submit->signal_count; i++) {
      struct vk_sync *sync = submit->signals[i].sync;
      if (sync->type->features & VK_SYNC_FEATURE_CPU_SIGNAL) {
         VkResult sr = vk_sync_signal(&dev->vk, sync, submit->signals[i].signal_value);
         (void)sr;
      }
   }
   return VK_SUCCESS;
}

static VkResult
ngv_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct ngv_queue *queue = container_of(vk_queue, struct ngv_queue, vk);
   struct ngv_device *dev = queue->device;
   const struct ngv_winsys_ops *ops = dev->ws.ops;

   if (vk_queue_submit_has_bind(submit))
      return vk_error(queue, VK_ERROR_FEATURE_NOT_PRESENT); /* sparse: P2 */
   if (vk_device_is_lost(&dev->vk))
      return ngv_queue_submit_lost(queue, submit, "submit on a lost device", 0);

   /* Every recorded IB chunk of every command buffer becomes one cmdbuf entry; the kernel wraps each in an INDIRECT
 * and the FENCE(IRQ) of the job (section 5). */
   uint32_t ib_cap = 0;
   for (uint32_t i = 0; i < submit->command_buffer_count; i++)
      ib_cap += container_of(submit->command_buffers[i], struct ngv_cmd_buffer, vk)->chunk_count;

   size_t nsync = (size_t)submit->wait_count + submit->signal_count;
   struct nicogpu_ws_sync *waits = calloc(nsync + 1, sizeof(*waits));
   struct nicogpu_cmdbuf *ibs = calloc((size_t)ib_cap + 1, sizeof(*ibs));
   if (!waits || !ibs) {
      free(waits);
      free(ibs);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   struct nicogpu_ws_sync *signals = waits + submit->wait_count;
   for (uint32_t i = 0; i < submit->wait_count; i++)
      ngv_ws_sync_from_vk(&waits[i], submit->waits[i].sync, submit->waits[i].wait_value);
   /* The kernel rejects a job that signals one syncobj twice, but the runtime merges queued submits
    * (vk_queue_merge_submit), so a timeline semaphore can show up at several values (CTS
    * synchronization.implicit.timeline_semaphore).  They all signal when the job retires: keep the highest value. */
   uint32_t n_sig = 0;
   for (uint32_t i = 0; i < submit->signal_count; i++) {
      const struct vk_sync_signal *si = &submit->signals[i];
      bool superseded = false;
      for (uint32_t k = 0; k < submit->signal_count && !superseded; k++) {
         const struct vk_sync_signal *sk = &submit->signals[k];
         superseded = k != i && sk->sync == si->sync &&
                      (sk->signal_value > si->signal_value || (sk->signal_value == si->signal_value && k > i));
      }
      if (!superseded)
         ngv_ws_sync_from_vk(&signals[n_sig++], si->sync, si->signal_value);
   }
   uint32_t n_ib = 0;
   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct ngv_cmd_buffer *cmd = container_of(submit->command_buffers[i], struct ngv_cmd_buffer, vk);
      VkResult vr = ngv_cmd_buffer_collect_ibs(cmd, ibs, &n_ib, ib_cap);
      if (vr != VK_SUCCESS) {
         free(waits);
         free(ibs);
         return vr;
      }
   }
   /* The kernel takes at most NICOGPU_SUBMIT_MAX_SYNCS (64, nicogpu_uapi.h; not included here) waits and as many
    * signals per submit; Vulkan has no such limit (CTS synchronization.timeline_semaphore.misc.ignore_timeline_semaphore_info signals 65).  The kernel keeps a
    * queue's jobs in order, so the extra waits go into leading bare-fence jobs (no command buffers, waits only) and the
 * extra signals into trailing bare-fence jobs: the semaphores still signal only after the command buffers. */
   const uint32_t max_sy = NGV_SUBMIT_MAX_SYNCS;
   int r = 0;
   for (uint32_t off = max_sy; r == 0 && off < submit->wait_count; off += max_sy) {
      struct nicogpu_ws_submit_args w = {
         .struct_size = sizeof(w),
         .queue = queue->ws_queue,
         .num_waits = MIN2(submit->wait_count - off, max_sy),
         .waits = waits + off,
      };
      r = ops->submit(dev->ws.dev, &w);
   }
   if (r == 0) {
      struct nicogpu_ws_submit_args a = {
         .struct_size = sizeof(a),
         .queue = queue->ws_queue,
         .num_cmdbufs = n_ib,
         .cmdbufs = ibs,
         .num_waits = MIN2(submit->wait_count, max_sy),
         .waits = waits,
         .num_signals = MIN2(n_sig, max_sy),
         .signals = signals,
      };
      r = ops->submit(dev->ws.dev, &a);
   }
   for (uint32_t off = max_sy; r == 0 && off < n_sig; off += max_sy) {
      struct nicogpu_ws_submit_args t = {
         .struct_size = sizeof(t),
         .queue = queue->ws_queue,
         .num_signals = MIN2(n_sig - off, max_sy),
         .signals = signals + off,
      };
      r = ops->submit(dev->ws.dev, &t);
   }
   free(waits);
   free(ibs);
   if (r)
      return ngv_queue_submit_lost(queue, submit, "winsys submit failed", r);
   /* A faulted job still signals its fence (error-signaled): look at the device state after every submit
 * (section 7 gap 5). Nothing left to signal here; the loss surfaces as described above. */
   if (ops->device_is_lost(dev->ws.dev) == 1)
      vk_queue_set_lost(&queue->vk, "device lost after submit");
   return VK_SUCCESS;
}

/* ---- device -------------------------------------------------------------------- */

/* vk_device::check_status: the runtime calls it after every wait and in vkGetFenceStatus / vkQueueWaitIdle.  With the DRM
 * syncobj sync type the waits go straight to the kernel and succeed once the (faulted) job's fence has signalled, so
 * this is the place where a GUILTY queue / lost device (winsys device_is_lost: GET_DEV_STATUS + queue status) turns
 * into VK_ERROR_DEVICE_LOST. */
static VkResult
ngv_check_status(struct vk_device *vk)
{
   struct ngv_device *dev = container_of(vk, struct ngv_device, vk);
   if (dev->ws.ops && dev->ws.dev && dev->ws.ops->device_is_lost(dev->ws.dev) == 1)
      return vk_device_set_lost(vk, "the winsys reports a lost device (GPU fault / device gone)");
   return VK_SUCCESS;
}

/* Tears down device->queues[0 .. queue_count).  vk_queue_finish of EVERY queue first: it drains and joins that
 * queue's runtime submit thread, which is a winsys caller (submit, waits) until it returns, and a submit thread may be
 * waiting for a signal that another queue's thread delivers; only after all threads are gone may the winsys queues
 * go.  Works after a device loss too, because ngv_queue_submit never fails the thread (ngv_queue_submit_lost). */
static void
ngv_queues_finish(struct ngv_device *device)
{
   for (uint32_t i = 0; i < device->queue_count; i++)
      vk_queue_finish(&device->queues[i].vk);
   for (uint32_t i = 0; i < device->queue_count; i++)
      device->ws.ops->queue_destroy(device->ws.dev, device->queues[i].ws_queue);
   device->queue_count = 0;
}

/* One runtime queue + one winsys queue (= one kernel queue on the device's VM) per requested VkQueue. The
 * kernel orders jobs per queue and keeps queues independent (a job blocked on a dependency does not hold back other
 * queues), and marks GUILTY only the queue whose job faulted; the device-wide loss on top of that
 * is the winsys latch (device_is_lost: any guilty queue), so a fault on one VkQueue is VK_ERROR_DEVICE_LOST on all
 * of them -- as the Vulkan spec wants a lost device to be. */
static VkResult
ngv_queues_init(struct ngv_device *device, const VkDeviceCreateInfo *pCreateInfo)
{
   struct ngv_physical_device *pdev = device->pdev;
   const VkDeviceQueueCreateInfo *pick[NGV_QUEUE_FAMILY_COUNT] = {NULL};

   /* Everything below is a valid-usage rule (unique family, 1..queueCount queues, no PROTECTED: no family reports
    * VK_QUEUE_PROTECTED_BIT); checked anyway, a bad request must not overrun device->queues. */
   if (pCreateInfo->queueCreateInfoCount == 0)
      return VK_ERROR_INITIALIZATION_FAILED;
   for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      const VkDeviceQueueCreateInfo *qci = &pCreateInfo->pQueueCreateInfos[i];
      const uint32_t f = qci->queueFamilyIndex;
      if (f >= NGV_QUEUE_FAMILY_COUNT || qci->flags || qci->queueCount == 0 || qci->queueCount > ngv_queue_family_queues[f])
         return VK_ERROR_INITIALIZATION_FAILED;
      /* A family listed twice is a valid-usage violation (queueFamilyIndex must be unique) that the CTS commits when its
       * universal and compute queue family are the same index (dEQP-VK.texture.mipmap.*image_view_min_lod creates its
 * device through createRobustBufferAccessDevice this way): the entries of one family merge into one, with
       * the largest queueCount, so vkGetDeviceQueue(family, i) is valid for every i any entry asked for. */
      if (!pick[f] || qci->queueCount > pick[f]->queueCount)
         pick[f] = qci;
   }

   device->queue_count = 0;
   uint32_t made = 0;
   for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      const uint32_t f = pCreateInfo->pQueueCreateInfos[i].queueFamilyIndex;
      if (made & (1u << f))
         continue; /* repeated family: merged above */
      made |= 1u << f;
      const VkDeviceQueueCreateInfo *qci = pick[f];
      for (uint32_t q = 0; q < qci->queueCount; q++) {
         struct ngv_queue *queue = &device->queues[device->queue_count];
         VkResult result = vk_queue_init(&queue->vk, &device->vk, qci, q);
         if (result != VK_SUCCESS) {
            ngv_queues_finish(device);
            return result;
         }
         queue->device = device;
         queue->vk.driver_submit = ngv_queue_submit;
         int r = pdev->ops->queue_create(device->ws.dev, &queue->ws_queue);
         if (r) {
            vk_queue_finish(&queue->vk);
            ngv_queues_finish(device);
            /* -ENODEV from queue_create = no command processor behind the device (a bring-up board without CP
             * firmware, a sim without CP): the device cannot be brought up, but it is not lost -- there was never a
             * working device to lose.  Everything else keeps the generic mapping (ngv_winsys_result). */
            return r == NICOGPU_ERR_NODEV ? VK_ERROR_INITIALIZATION_FAILED : ngv_winsys_result(r);
         }
         device->queue_count++;
      }
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
   VK_FROM_HANDLE(ngv_physical_device, pdev, physicalDevice);
   struct ngv_device *device = vk_zalloc2(&pdev->vk.instance->alloc, pAllocator, sizeof(*device), 8,
                                          VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_device_dispatch_table dispatch;
   vk_device_dispatch_table_from_entrypoints(&dispatch, &ngv_device_entrypoints, true);
   /* everything the driver does not implement itself comes from the Mesa runtime (vk_common_*) */
   vk_device_dispatch_table_from_entrypoints(&dispatch, &vk_common_device_entrypoints, false);
   VkResult result = vk_device_init(&device->vk, &pdev->vk, &dispatch, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free2(&pdev->vk.instance->alloc, pAllocator, device);
      return result;
   }
   device->pdev = pdev;
   device->vk.command_buffer_ops = &ngv_cmd_buffer_ops;
   device->vk.check_status = ngv_check_status;
   /* Wait-before-submit needs the runtime's submit thread, which needs WAIT_PENDING on every GPU-wait sync type
 * (leftover 2). NGV_THREADED_SUBMIT=0 turns it off. The winsys is thread-safe (thread
    * contract in nicogpu_winsys.h: every backend serializes internally), the submit thread is just one more caller. */
   {
      const char *ts = getenv("NGV_THREADED_SUBMIT");
      if ((!ts || strcmp(ts, "0")) && (pdev->sync_type.features & VK_SYNC_FEATURE_WAIT_PENDING))
         vk_device_enable_threaded_submit(&device->vk);
   }
   simple_mtx_init(&device->mem_lock, mtx_plain);
   simple_mtx_init(&device->scratch_lock, mtx_plain);
   simple_mtx_init(&device->meta_lock, mtx_plain);

   device->ws.ops = pdev->ops;
   struct nicogpu_ws_device_create_args ca = {.struct_size = sizeof(ca)};
   snprintf(ca.client, sizeof(ca.client), "mesa-ngv");
   int r = pdev->ops->device_create(pdev->desc.uri, &ca, &device->ws.dev);
   if (r) {
      result = ngv_winsys_result(r);
      goto fail_init;
   }
   if (pdev->sync_drm) {
      /* the runtime's vk_drm_syncobj creates / waits / exports syncobjs on this file; the winsys owns the fd */
      int fd = pdev->ops->drm_fd ? pdev->ops->drm_fd(device->ws.dev) : -1;
      if (fd < 0) {
         result = VK_ERROR_INITIALIZATION_FAILED;
         goto fail_ws;
      }
      vk_device_set_drm_fd(&device->vk, fd);
   }

   result = ngv_queues_init(device, pCreateInfo);
   if (result != VK_SUCCESS)
      goto fail_ws;

   *pDevice = ngv_device_to_handle(device);
   return VK_SUCCESS;

fail_ws:
   pdev->ops->device_destroy(device->ws.dev);
fail_init:
   simple_mtx_destroy(&device->meta_lock);
   simple_mtx_destroy(&device->scratch_lock);
   simple_mtx_destroy(&device->mem_lock);
   vk_device_finish(&device->vk);
   vk_free2(&pdev->vk.instance->alloc, pAllocator, device);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   if (!device)
      return;
   ngv_queues_finish(device);
   ngv_scratch_finish(device);
   ngv_meta_finish(device);
   simple_mtx_destroy(&device->meta_lock);
   simple_mtx_destroy(&device->scratch_lock);
   simple_mtx_destroy(&device->mem_lock);
   vk_device_finish(&device->vk);
   /* the DRM file (vk_drm_syncobj) must outlive everything the runtime frees in vk_device_finish */
   device->ws.ops->device_destroy(device->ws.dev);
   vk_free2(&device->vk.alloc, pAllocator, device);
}

/* ---- format and image-format queries : the format table comes from the format description source via ngv_formats_gen.h ---- */

VKAPI_ATTR void VKAPI_CALL
ngv_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice, VkFormat format,
                                       VkFormatProperties2 *pFormatProperties)
{
   const struct ngv_vkfmt *f = ngv_format_lookup(format);
   const VkFormatProperties fp = {
      .linearTilingFeatures = ngv_format_image_features(f, VK_IMAGE_TILING_LINEAR),
      .optimalTilingFeatures = ngv_format_image_features(f, VK_IMAGE_TILING_OPTIMAL),
      .bufferFeatures = ngv_format_buffer_features(f),
   };
   pFormatProperties->formatProperties = fp;
   vk_foreach_struct(ext, pFormatProperties->pNext) {
      if (ext->sType == VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3) {
         VkFormatProperties3 *p3 = (VkFormatProperties3 *)ext;
         p3->linearTilingFeatures = fp.linearTilingFeatures;
         p3->optimalTilingFeatures = fp.optimalTilingFeatures;
         p3->bufferFeatures = fp.bufferFeatures;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_GetPhysicalDeviceImageFormatProperties2(VkPhysicalDevice physicalDevice,
                                            const VkPhysicalDeviceImageFormatInfo2 *pInfo,
                                            VkImageFormatProperties2 *pProps)
{
   /* The spec: when VK_ERROR_FORMAT_NOT_SUPPORTED is returned, all members of imageFormatProperties are zero (and so
    * are the members of the chained output structures that report capabilities).  vkGetPhysicalDeviceImageFormat-
    * Properties (1.0) is the runtime shim over this function and copies the struct as it is: without the clear it
    * returns stack garbage (dEQP api.info.image_format_properties2.* compares the two queries). */
   vk_foreach_struct(ext, pProps->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES:
         ((VkExternalImageFormatProperties *)ext)->externalMemoryProperties = (VkExternalMemoryProperties){0};
         break;
      case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES:
         ((VkSamplerYcbcrConversionImageFormatProperties *)ext)->combinedImageSamplerDescriptorCount = 0;
         break;
      default:
         break;
      }
   }
   /* external memory handle types are not supported */
   vk_foreach_struct_const(ext, pInfo->pNext) {
      if (ext->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO) {
         const VkPhysicalDeviceExternalImageFormatInfo *e = (const void *)ext;
         if (e->handleType) {
            pProps->imageFormatProperties = (VkImageFormatProperties){0};
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
         }
      }
   }
   return ngv_image_format_properties(pInfo, pProps);
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetPhysicalDeviceSparseImageFormatProperties2(VkPhysicalDevice physicalDevice,
                                                  const VkPhysicalDeviceSparseImageFormatInfo2 *pInfo,
                                                  uint32_t *pPropertyCount,
                                                  VkSparseImageFormatProperties2 *pProperties)
{
   *pPropertyCount = 0;
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetPhysicalDeviceExternalBufferProperties(VkPhysicalDevice physicalDevice,
                                              const VkPhysicalDeviceExternalBufferInfo *pInfo,
                                              VkExternalBufferProperties *pProps)
{
   pProps->externalMemoryProperties = (VkExternalMemoryProperties){0}; /* no dma-buf yet (HAL v1 #4) */
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetPhysicalDeviceExternalSemaphoreProperties(VkPhysicalDevice physicalDevice,
                                                 const VkPhysicalDeviceExternalSemaphoreInfo *pInfo,
                                                 VkExternalSemaphoreProperties *pProps)
{
   pProps->exportFromImportedHandleTypes = 0;
   pProps->compatibleHandleTypes = 0;
   pProps->externalSemaphoreFeatures = 0;
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetPhysicalDeviceExternalFenceProperties(VkPhysicalDevice physicalDevice,
                                             const VkPhysicalDeviceExternalFenceInfo *pInfo,
                                             VkExternalFenceProperties *pProps)
{
   pProps->exportFromImportedHandleTypes = 0;
   pProps->compatibleHandleTypes = 0;
   pProps->externalFenceFeatures = 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_EnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pPropertyCount,
                                         VkExtensionProperties *pProperties)
{
   if (pLayerName)
      return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);
   return vk_enumerate_instance_extension_properties(&ngv_instance_extensions, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_EnumerateInstanceLayerProperties(uint32_t *pPropertyCount, VkLayerProperties *pProperties)
{
   if (!pProperties) {
      *pPropertyCount = 0;
      return VK_SUCCESS;
   }
   return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT); /* no layers */
}
