/* SPDX-License-Identifier: MIT */
/* ngv_memory.c -- VkDeviceMemory and VkBuffer (skeleton).
 * Mapping of memory types to winsys domains: section 4. */
#include <stdlib.h>
#include <string.h>

#include "ngv_private.h"

static void
type_to_bo(const struct nicogpu_device_info *info, uint32_t type, uint32_t *domain, uint32_t *flags)
{
   *flags = 0;
   if (NGV_NO_VRAM(info)) {
      *domain = NICOGPU_BO_DOMAIN_GTT; /* the only memory type is host memory */
      return;
   }
   switch (type) {
   case NGV_MEM_TYPE_GTT:
      *domain = NICOGPU_BO_DOMAIN_GTT;
      break;
   case NGV_MEM_TYPE_VRAM_VIS:
      *domain = NICOGPU_BO_DOMAIN_VRAM;
      *flags = NICOGPU_BO_HOST_VISIBLE;
      break;
   default:
      *domain = NICOGPU_BO_DOMAIN_VRAM;
      break;
   }
}

/* ---- driver-internal BOs (IBs, descriptor pools, code, queries) ------------------------------ */

VkResult
ngv_bo_create(struct ngv_device *dev, uint64_t size, enum ngv_bo_kind kind, struct ngv_bo *out)
{
   memset(out, 0, sizeof(*out));
   struct nicogpu_ws_bo_args a = {
      .struct_size = sizeof(a),
      .size = size,
      .vm_flags = NICOGPU_VM_READ | NICOGPU_VM_WRITE | NICOGPU_VM_EXEC,
   };
   if (kind == NGV_BO_CODE && dev->pdev->info.vram_visible_size) {
      a.domain = NICOGPU_BO_DOMAIN_VRAM;
      a.flags = NICOGPU_BO_HOST_VISIBLE;
   } else {
      a.domain = NICOGPU_BO_DOMAIN_GTT;
   }
   struct nicogpu_ws_bo_info info = {.struct_size = sizeof(info)};
   simple_mtx_lock(&dev->mem_lock);
   int r = dev->ws.ops->bo_alloc(dev->ws.dev, &a, &info);
   if (!r) {
      void *map = NULL;
      r = dev->ws.ops->bo_map(dev->ws.dev, info.handle, &map);
      if (r)
         dev->ws.ops->bo_free(dev->ws.dev, info.handle);
      else {
         out->handle = info.handle;
         out->va = info.va;
         out->size = info.size;
         out->map = map;
      }
   }
   simple_mtx_unlock(&dev->mem_lock);
   return r ? ngv_winsys_result(r) : VK_SUCCESS;
}

void
ngv_bo_destroy(struct ngv_device *dev, struct ngv_bo *bo)
{
   if (!bo->handle)
      return;
   simple_mtx_lock(&dev->mem_lock);
   dev->ws.ops->bo_free(dev->ws.dev, bo->handle);
   simple_mtx_unlock(&dev->mem_lock);
   memset(bo, 0, sizeof(*bo));
}

/* ---- per-lane scratch ------------------------------------------------------------------ */

/* Lane-private scratch layout (the errata notes (v0.1.6), "scratch 按 dword 交错排列",; it replaced the
 * linear v0.1.2 layout): byte o of lane L of the warp in global warp slot G is at
 * ScratchBase + G * 32 * ScratchBytes + (o / 4) * 128 + L * 4 + o % 4, i.e. every warp still owns 32 * ScratchBytes
 * bytes -- the layout is invisible here (and to ngcomp, which addresses scratch by lane byte offset only).  A dispatch
 * never uses more than (all resident warp slots of the device) * lanes * ScratchBytes, so one BO per device sized for
 * the largest
 * ScratchBytes seen so far backs every dispatch (the stride is the dispatch's own SET_SHADER scratch_bytes).
 *
 * The device must report the number of resident warps (ERRATA v0.1.2 / ISA REFERENCE 10 item 3); the upper bound used
 * here is num_cores * sp_per_core * warp_slots_per_sp from GET_GPU_INFO (: warp_slots_per_sp is an
 * architecture constant, 16; the defaults 4 cores x 4 SPs are the same numbers the winsys falls back to).
 *
 * Growth: a bigger BO replaces the current one; the old BO is NOT freed but kept on a retired list until the device
 * dies, because command buffers recorded earlier (or jobs in flight) still hold its VA.  The new size is at least twice
 * the old one, so the retired BOs add up to less than the current one. */
VkResult
ngv_scratch_get(struct ngv_device *dev, uint32_t bytes_per_lane, uint64_t *va)
{
   const struct nicogpu_device_info *i = &dev->pdev->info;
   const uint64_t cores = i->num_cores ? i->num_cores : 4, sps = i->sp_per_core ? i->sp_per_core : 4;
   const uint64_t slots = cores * sps * (i->warp_slots_per_sp ? i->warp_slots_per_sp : 16);
   const uint64_t lanes = i->warp_size ? i->warp_size : 32;
   const uint64_t need = align64((uint64_t)bytes_per_lane * slots * lanes, NICOGPU_PAGE_SIZE);

   if (i->gtt_bo_max && need > i->gtt_bo_max)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   simple_mtx_lock(&dev->scratch_lock);
   VkResult r = VK_SUCCESS;
   if (dev->scratch_bo.size < need) {
      uint64_t size = dev->scratch_bo.size * 2 > need ? dev->scratch_bo.size * 2 : need;
      if (i->gtt_bo_max && size > i->gtt_bo_max)
         size = i->gtt_bo_max;
      if (dev->scratch_retired_count == dev->scratch_retired_cap) {
         uint32_t cap = dev->scratch_retired_cap ? dev->scratch_retired_cap * 2 : 4;
         struct ngv_bo *n = realloc(dev->scratch_retired, cap * sizeof(*n));
         if (!n) {
            r = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto out;
         }
         dev->scratch_retired = n;
         dev->scratch_retired_cap = cap;
      }
      struct ngv_bo bo;
      r = ngv_bo_create(dev, size, NGV_BO_HOST, &bo);
      if (r != VK_SUCCESS)
         goto out;
      if (dev->scratch_bo.handle)
         dev->scratch_retired[dev->scratch_retired_count++] = dev->scratch_bo;
      dev->scratch_bo = bo;
   }
   *va = dev->scratch_bo.va; /* BO VAs are page aligned; SET_SHADER wants 256 B */
out:
   simple_mtx_unlock(&dev->scratch_lock);
   return r;
}

void
ngv_scratch_finish(struct ngv_device *dev)
{
   for (uint32_t k = 0; k < dev->scratch_retired_count; k++)
      ngv_bo_destroy(dev, &dev->scratch_retired[k]);
   free(dev->scratch_retired);
   dev->scratch_retired = NULL;
   dev->scratch_retired_count = dev->scratch_retired_cap = 0;
   ngv_bo_destroy(dev, &dev->scratch_bo);
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_AllocateMemory(VkDevice _device, const VkMemoryAllocateInfo *pAllocateInfo,
                   const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   struct ngv_device_memory *mem = vk_device_memory_create(&device->vk, pAllocateInfo, pAllocator, sizeof(*mem));
   if (!mem)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   if (mem->vk.import_handle_type || mem->vk.export_handle_types || mem->vk.host_ptr) {
      vk_device_memory_destroy(&device->vk, pAllocator, &mem->vk);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT); /* no import/export yet (HAL v1 #4 is later) */
   }

   struct nicogpu_ws_bo_args a = {
      .struct_size = sizeof(a),
      .size = pAllocateInfo->allocationSize,
      /* Code, descriptors and data share the heap: all mappings R/W/X.  Cache
       * policy is the PTE default (MTYPE CC); UC is never requested here. */
      .vm_flags = NICOGPU_VM_READ | NICOGPU_VM_WRITE | NICOGPU_VM_EXEC,
 /* : a T64K image needs its base on a 64 KiB tile boundary (the descriptor spec section 6.7); memory objects of that size
       * are placed there, whatever the size class of the winsys would pick */
      .alignment = pAllocateInfo->allocationSize >= 65536 ? 65536 : 0,
   };
   type_to_bo(&device->pdev->info, pAllocateInfo->memoryTypeIndex, &a.domain, &a.flags);
   struct nicogpu_ws_bo_info info = {.struct_size = sizeof(info)};

   simple_mtx_lock(&device->mem_lock);
   int r = device->ws.ops->bo_alloc(device->ws.dev, &a, &info);
   simple_mtx_unlock(&device->mem_lock);
   if (r) {
      vk_device_memory_destroy(&device->vk, pAllocator, &mem->vk);
      return vk_error(device, ngv_winsys_result(r));
   }
   mem->bo = info.handle;
   mem->va = info.va;
   *pMem = ngv_device_memory_to_handle(mem);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_FreeMemory(VkDevice _device, VkDeviceMemory _mem, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_device_memory, mem, _mem);
   if (!mem)
      return;
   simple_mtx_lock(&device->mem_lock);
   device->ws.ops->bo_free(device->ws.dev, mem->bo);
   simple_mtx_unlock(&device->mem_lock);
   vk_device_memory_destroy(&device->vk, pAllocator, &mem->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_MapMemory2(VkDevice _device, const VkMemoryMapInfo *pInfo, void **ppData)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_device_memory, mem, pInfo->memory);
   if (!mem->map) {
      simple_mtx_lock(&device->mem_lock);
      int r = device->ws.ops->bo_map(device->ws.dev, mem->bo, &mem->map);
      simple_mtx_unlock(&device->mem_lock);
      if (r)
         return vk_error(device, VK_ERROR_MEMORY_MAP_FAILED);
   }
   *ppData = (char *)mem->map + pInfo->offset;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_UnmapMemory2(VkDevice _device, const VkMemoryUnmapInfo *pInfo)
{
   /* The winsys mapping is stable until free; "unmap" only forgets nothing. */
   return VK_SUCCESS;
}

/* VRAM|HOST_VISIBLE is not HOST_COHERENT (write-combined BAR window): flush
 * drains the CPU write-combining buffers, invalidate has nothing to do
 * (reads of a WC mapping are uncached).  GTT is HOST_COHERENT (cached on a coherent host, uncached/WC on a
 * non-coherent one: no cache maintenance either way) and never gets here. */
VKAPI_ATTR VkResult VKAPI_CALL
ngv_FlushMappedMemoryRanges(VkDevice _device, uint32_t count, const VkMappedMemoryRange *ranges)
{
   __sync_synchronize();
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_InvalidateMappedMemoryRanges(VkDevice _device, uint32_t count, const VkMappedMemoryRange *ranges)
{
   __sync_synchronize();
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetDeviceMemoryCommitment(VkDevice device, VkDeviceMemory memory, VkDeviceSize *pCommittedMemoryInBytes)
{
   *pCommittedMemoryInBytes = 0;
}

/* ---- buffers ------------------------------------------------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateBuffer(VkDevice _device, const VkBufferCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   if (pCreateInfo->flags & (VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
                             VK_BUFFER_CREATE_SPARSE_ALIASED_BIT))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   /* VkPhysicalDeviceMaintenance4Properties::maxBufferSize bounds a buffer.  Without this a size near UINT64_MAX
    * wraps in align64() of the memory requirements and the buffer reports a size smaller than asked (dEQP
    * api.buffer.basic.size_max_uint64 wants success with size >= requested, or one of the two OOM errors). */
   if (pCreateInfo->size > device->vk.physical->properties.maxBufferSize)
      return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   struct ngv_buffer *buffer = vk_buffer_create(&device->vk, pCreateInfo, pAllocator, sizeof(*buffer));
   if (!buffer)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   *pBuffer = ngv_buffer_to_handle(buffer);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyBuffer(VkDevice _device, VkBuffer _buffer, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_buffer, buffer, _buffer);
   if (buffer)
      vk_buffer_destroy(&device->vk, pAllocator, &buffer->vk);
}

/* ---- buffer views (texel buffers) ----------------------------------------------------------------
 * A view is the runtime object plus the image descriptor (view_type = buffer, the descriptor spec section 4.6) that
 * vkUpdateDescriptorSets copies into UNIFORM_TEXEL_BUFFER / STORAGE_TEXEL_BUFFER descriptors.  A format the texture unit
 * does not know leaves the descriptor null (reads return 0, writes are dropped). */

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateBufferView(VkDevice _device, const VkBufferViewCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator, VkBufferView *pView)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   struct ngv_buffer_view *view = vk_buffer_view_create(&device->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   const struct ngv_vkfmt *f = ngv_format_lookup(pCreateInfo->format);
   VK_FROM_HANDLE(ngv_buffer, buf, pCreateInfo->buffer);
   memset(view->desc, 0, sizeof(view->desc));
   if (f && !(f->flags & NGV_FMT_COMPRESSED) && view->vk.elements && view->vk.elements <= (1u << 28)) {
      uint32_t *d = view->desc;
      nicogpu_image_set_base_va(d, buf->vk.device_address + view->vk.offset);
      nicogpu_image_set_data_format(d, f->data_format);
      nicogpu_image_set_num_format(d, f->num_format);
      nicogpu_image_set_order(d, f->order);
      nicogpu_image_set_view_type(d, NICOGPU_VIEW_TYPE_BUFFER);
      nicogpu_image_set_elements_m1(d, view->vk.elements - 1);
      nicogpu_image_set_swizzle_r(d, NICOGPU_SWIZZLE_R);
      nicogpu_image_set_swizzle_g(d, NICOGPU_SWIZZLE_G);
      nicogpu_image_set_swizzle_b(d, NICOGPU_SWIZZLE_B);
      nicogpu_image_set_swizzle_a(d, NICOGPU_SWIZZLE_A);
   }
   *pView = vk_buffer_view_to_handle(&view->vk);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyBufferView(VkDevice _device, VkBufferView _view, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(vk_buffer_view, view, _view);
   if (view)
      vk_buffer_view_destroy(&device->vk, pAllocator, view);
}

#define NGV_BUFFER_ALIGNMENT 256 /* PROVISIONAL: >= every descriptor / cacheline requirement */

VKAPI_ATTR void VKAPI_CALL
ngv_GetBufferMemoryRequirements2(VkDevice _device, const VkBufferMemoryRequirementsInfo2 *pInfo,
                                 VkMemoryRequirements2 *pReqs)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_buffer, buffer, pInfo->buffer);
   uint32_t types = (1u << NGV_MEM_TYPE_COUNT) - 1;
   if (NGV_NO_VRAM(&device->pdev->info))
      types = 1u; /* single host-memory type */
   else if (!device->pdev->info.vram_visible_size)
      types &= ~(1u << NGV_MEM_TYPE_VRAM_VIS);
   pReqs->memoryRequirements = (VkMemoryRequirements){
      .size = align64(buffer->vk.size, NGV_BUFFER_ALIGNMENT),
      .alignment = NGV_BUFFER_ALIGNMENT,
      .memoryTypeBits = types,
   };
   vk_foreach_struct(ext, pReqs->pNext) {
      if (ext->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS) {
         VkMemoryDedicatedRequirements *d = (void *)ext;
         d->prefersDedicatedAllocation = false;
         d->requiresDedicatedAllocation = false;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_BindBufferMemory2(VkDevice _device, uint32_t bindInfoCount, const VkBindBufferMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; i++) {
      VK_FROM_HANDLE(ngv_buffer, buffer, pBindInfos[i].buffer);
      VK_FROM_HANDLE(ngv_device_memory, mem, pBindInfos[i].memory);
      /* bufferDeviceAddress comes for free: the winsys VA is the device address. */
      buffer->vk.device_address = mem->va + pBindInfos[i].memoryOffset;
   }
   return VK_SUCCESS;
}
