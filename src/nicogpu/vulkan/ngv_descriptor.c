/* SPDX-License-Identifier: MIT */
/* ngv_descriptor.c -- descriptor set layouts, pools and sets (batch 2).
 *
 * Descriptors are plain memory (the descriptor spec section 8), written by the host into a pool BO the GPU reads through
 * the set base address in u2..u9 (section 6.2). Slot numbering: a binding's byte offset in the set / 16; the
 * binding offsets below are what ngv_nir_lower_pipeline_layout will turn into compile-time slots (NIR path: later).
 *
 * Buffer descriptors are built from the buffer address (below); image / sampler / texel-buffer descriptors are the 32 / 16 /
 * 32 byte images the view and sampler objects built when they were created (ngv_image.c, ngv_memory.c;), copied in.
 * A combined image sampler is the image descriptor at +0 and the sampler descriptor at +32.  Immutable samplers are written
 * when the set is allocated and ignored by later writes.
 */
#include <string.h>

#include "vk_descriptor_update_template.h"
#include "vk_descriptors.h"
#include "ngv_private.h"

uint32_t
ngv_descriptor_size(VkDescriptorType t)
{
   switch (t) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_SAMPLER:
      return 16;
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      return 32;
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return 64;
   default:
      return 0; /* inline uniform block, acceleration structure, mutable ...: later */
   }
}

static uint32_t
desc_align(VkDescriptorType t)
{
   return ngv_descriptor_size(t) >= 32 ? 32 : 16;
}

static bool
is_dynamic(VkDescriptorType t)
{
   return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}

/* ---- layouts ---------------------------------------------------------------------------------- */

static void
ngv_descriptor_set_layout_destroy(struct vk_device *device, struct vk_descriptor_set_layout *layout)
{
   free(container_of(layout, struct ngv_descriptor_set_layout, vk)->imm);
   vk_descriptor_set_layout_destroy(device, layout);
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateDescriptorSetLayout(VkDevice _device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                              const VkAllocationCallbacks *pAllocator, VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   if (pCreateInfo->flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT); /* push descriptors: leftover */

   VkDescriptorSetLayoutBinding *sorted = NULL;
   VkResult result = vk_create_sorted_bindings(pCreateInfo->pBindings, pCreateInfo->bindingCount, &sorted);
   if (result != VK_SUCCESS)
      return vk_error(device, result);

   struct ngv_descriptor_set_layout *layout =
      vk_descriptor_set_layout_zalloc(&device->vk, sizeof(*layout) + pCreateInfo->bindingCount * sizeof(layout->bindings[0]),
                                      pCreateInfo);
   if (!layout) {
      free(sorted);
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   layout->vk.destroy = ngv_descriptor_set_layout_destroy;
   layout->binding_count = pCreateInfo->bindingCount;

   uint32_t offset = 0, dyn = 0;
   for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *b = &sorted[i];
      uint32_t sz = ngv_descriptor_size(b->descriptorType);
      if (!sz) {
         free(sorted);
         vk_descriptor_set_layout_destroy(&device->vk, &layout->vk);
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      offset = align(offset, desc_align(b->descriptorType));
      struct ngv_binding_layout *bl = &layout->bindings[i];
      bl->binding = b->binding;
      bl->type = b->descriptorType;
      bl->count = b->descriptorCount;
      bl->offset = offset;
      bl->stride = sz;
      bl->dyn_start = -1;
      bl->imm_start = -1;
      if (is_dynamic(b->descriptorType)) {
         bl->dyn_start = (int32_t)dyn;
         dyn += b->descriptorCount;
      }
      offset += sz * b->descriptorCount;
   }
   layout->size = offset;
   layout->dynamic_count = dyn;
   layout->vk.dynamic_descriptor_count = dyn;
   if (dyn > NGV_MAX_DYNAMIC) {
      free(sorted);
      vk_descriptor_set_layout_destroy(&device->vk, &layout->vk);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   /* immutable samplers: their descriptors are copied out of the sampler objects now, so the layout does not depend on
    * the objects' lifetime */
   uint32_t imm_total = 0;
   for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++)
      if (sorted[i].pImmutableSamplers && (sorted[i].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                                           sorted[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER))
         imm_total += sorted[i].descriptorCount;
   if (imm_total) {
      layout->imm = calloc(imm_total, 16);
      if (!layout->imm) {
         free(sorted);
         vk_descriptor_set_layout_destroy(&device->vk, &layout->vk);
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      for (uint32_t i = 0, at = 0; i < pCreateInfo->bindingCount; i++) {
         const VkDescriptorSetLayoutBinding *b = &sorted[i];
         if (!b->pImmutableSamplers || (b->descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER &&
                                        b->descriptorType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER))
            continue;
         layout->bindings[i].imm_start = (int32_t)at;
         for (uint32_t k = 0; k < b->descriptorCount; k++, at++) {
            VK_FROM_HANDLE(ngv_sampler, smp, b->pImmutableSamplers[k]);
            if (smp)
               memcpy(&layout->imm[at * 4], smp->desc, 16);
         }
      }
      layout->imm_count = imm_total;
   }
   free(sorted);
   *pSetLayout = ngv_descriptor_set_layout_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyDescriptorSetLayout(VkDevice _device, VkDescriptorSetLayout _layout, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(vk_descriptor_set_layout, layout, _layout);
   if (layout)
      vk_descriptor_set_layout_unref(&device->vk, layout);
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetDescriptorSetLayoutSupport(VkDevice _device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                  VkDescriptorSetLayoutSupport *pSupport)
{
   bool ok = !(pCreateInfo->flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);
   uint32_t dyn = 0;
   for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
      ok &= ngv_descriptor_size(pCreateInfo->pBindings[i].descriptorType) != 0;
      if (is_dynamic(pCreateInfo->pBindings[i].descriptorType))
         dyn += pCreateInfo->pBindings[i].descriptorCount;
   }
   pSupport->supported = ok && dyn <= NGV_MAX_DYNAMIC;
}

/* ---- pools ------------------------------------------------------------------------------------ */

#define SET_ALIGN 32

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateDescriptorPool(VkDevice _device, const VkDescriptorPoolCreateInfo *pCreateInfo,
                         const VkAllocationCallbacks *pAllocator, VkDescriptorPool *pPool)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   struct ngv_descriptor_pool *pool = vk_object_zalloc(&device->vk, pAllocator, sizeof(*pool), VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   uint64_t size = 0;
   for (uint32_t i = 0; i < pCreateInfo->poolSizeCount; i++) {
      uint32_t sz = ngv_descriptor_size(pCreateInfo->pPoolSizes[i].type);
      if (!sz) {
         vk_object_free(&device->vk, pAllocator, pool);
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      size += (uint64_t)sz * pCreateInfo->pPoolSizes[i].descriptorCount;
   }
   /* worst-case alignment slack of every set + binding */
   size += (uint64_t)pCreateInfo->maxSets * 2 * SET_ALIGN + NICOGPU_PAGE_SIZE;
   VkResult r = ngv_bo_create(device, align64(size, NICOGPU_PAGE_SIZE), NGV_BO_HOST, &pool->bo);
   if (r != VK_SUCCESS) {
      vk_object_free(&device->vk, pAllocator, pool);
      return vk_error(device, r);
   }
   pool->max_sets = pCreateInfo->maxSets;
   list_inithead(&pool->sets);
   *pPool = ngv_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

static void
pool_free_set(struct ngv_device *device, struct ngv_descriptor_set *set)
{
   list_del(&set->link);
   set->pool->set_count--;
   vk_descriptor_set_layout_unref(&device->vk, &set->layout->vk);
   vk_object_free(&device->vk, NULL, set);
}

static void
pool_free_all(struct ngv_device *device, struct ngv_descriptor_pool *pool)
{
   list_for_each_entry_safe(struct ngv_descriptor_set, set, &pool->sets, link) pool_free_set(device, set);
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyDescriptorPool(VkDevice _device, VkDescriptorPool _pool, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_descriptor_pool, pool, _pool);
   if (!pool)
      return;
   pool_free_all(device, pool);
   ngv_bo_destroy(device, &pool->bo);
   vk_object_free(&device->vk, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_ResetDescriptorPool(VkDevice _device, VkDescriptorPool _pool, VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_descriptor_pool, pool, _pool);
   pool_free_all(device, pool);
   return VK_SUCCESS;
}

/* First fit over the offset-sorted set list. */
static VkResult
pool_alloc_set(struct ngv_device *device, struct ngv_descriptor_pool *pool, struct ngv_descriptor_set_layout *layout,
               struct ngv_descriptor_set **out)
{
   if (pool->set_count >= pool->max_sets)
      return VK_ERROR_OUT_OF_POOL_MEMORY;

   uint32_t size = align(layout->size ? layout->size : 1, SET_ALIGN);
   uint32_t pos = 0;
   struct list_head *prev = &pool->sets; /* the new set goes after 'prev' (sorted by offset) */
   bool found = false;
   list_for_each_entry(struct ngv_descriptor_set, s, &pool->sets, link) {
      if (s->offset >= pos + size) {
         found = true;
         break;
      }
      pos = s->offset + align(s->size ? s->size : 1, SET_ALIGN);
      prev = &s->link;
   }
   if (!found && (uint64_t)pos + size > pool->bo.size)
      return VK_ERROR_OUT_OF_POOL_MEMORY;

   struct ngv_descriptor_set *set = vk_object_zalloc(&device->vk, NULL, sizeof(*set), VK_OBJECT_TYPE_DESCRIPTOR_SET);
   if (!set)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   set->pool = pool;
   set->layout = layout;
   vk_descriptor_set_layout_ref(&layout->vk);
   set->offset = pos;
   set->size = layout->size;
   set->va = pool->bo.va + pos;
   set->map = (char *)pool->bo.map + pos;
   memset(set->map, 0, size); /* null descriptors */
   for (uint32_t i = 0; i < layout->binding_count; i++) {
      const struct ngv_binding_layout *b = &layout->bindings[i];
      if (b->imm_start < 0)
         continue;
      for (uint32_t k = 0; k < b->count; k++) {
         uint32_t *d = (uint32_t *)((char *)set->map + b->offset + (size_t)k * b->stride);
         memcpy(b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? d + 8 : d, &layout->imm[(b->imm_start + k) * 4], 16);
      }
   }
   list_add(&set->link, prev);
   pool->set_count++;
   *out = set;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_AllocateDescriptorSets(VkDevice _device, const VkDescriptorSetAllocateInfo *pInfo, VkDescriptorSet *pSets)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_descriptor_pool, pool, pInfo->descriptorPool);
   VkResult result = VK_SUCCESS;
   uint32_t i;
   for (i = 0; i < pInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(ngv_descriptor_set_layout, layout, pInfo->pSetLayouts[i]);
      struct ngv_descriptor_set *set;
      result = pool_alloc_set(device, pool, layout, &set);
      if (result != VK_SUCCESS)
         break;
      pSets[i] = ngv_descriptor_set_to_handle(set);
   }
   if (result != VK_SUCCESS) {
      for (uint32_t k = 0; k < i; k++) {
         pool_free_set(device, ngv_descriptor_set_from_handle(pSets[k]));
         pSets[k] = VK_NULL_HANDLE;
      }
      for (uint32_t k = i; k < pInfo->descriptorSetCount; k++)
         pSets[k] = VK_NULL_HANDLE;
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_FreeDescriptorSets(VkDevice _device, VkDescriptorPool _pool, uint32_t count, const VkDescriptorSet *pSets)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(ngv_descriptor_set, set, pSets[i]);
      if (set)
         pool_free_set(device, set);
   }
   return VK_SUCCESS;
}

/* ---- updates ---------------------------------------------------------------------------------- */

/* buffer descriptor: base_va dw0[31:0]+dw1[15:0], size dw2, dw3 = 0 (the descriptor spec section 2.1) */
static void
write_buffer_desc(uint32_t *d, uint64_t va, uint32_t size)
{
   d[0] = (uint32_t)va;
   d[1] = (uint32_t)(va >> 32) & 0xffffu;
   d[2] = size;
   d[3] = 0;
}

static const struct ngv_binding_layout *
find_binding(const struct ngv_descriptor_set_layout *l, uint32_t binding)
{
   for (uint32_t i = 0; i < l->binding_count; i++)
      if (l->bindings[i].binding == binding)
         return &l->bindings[i];
   return NULL;
}

/* The descriptor at (binding, element) with the walk-on-overflow rule of vkUpdateDescriptorSets: an update that
 * runs past the last element of a binding continues in the next binding number that exists. */
static const struct ngv_binding_layout *
walk(const struct ngv_descriptor_set_layout *l, const struct ngv_binding_layout *b, uint32_t *elem)
{
   while (b && *elem >= b->count) {
      *elem -= b->count;
      b = (b + 1 < l->bindings + l->binding_count) ? b + 1 : NULL;
   }
   return b;
}

/* Writes one descriptor of binding layout `b` at `d` from `info`: a VkDescriptorBufferInfo (buffer types), a
 * VkDescriptorImageInfo (sampler / image types) or a VkBufferView (texel buffers), as the binding's type says. */
static void
write_descriptor(const struct ngv_binding_layout *b, uint32_t *d, const void *info)
{
   switch (b->type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
      const VkDescriptorBufferInfo *bi = info;
      VK_FROM_HANDLE(ngv_buffer, buf, bi->buffer);
      if (!buf) {
         memset(d, 0, 16); /* null descriptor */
         break;
      }
      uint64_t range = bi->range == VK_WHOLE_SIZE ? buf->vk.size - bi->offset : bi->range;
      write_buffer_desc(d, buf->vk.device_address + bi->offset, range > UINT32_MAX ? UINT32_MAX : (uint32_t)range);
      /* Dynamic descriptors are never read by the GPU directly (ngv_cmd_buffer.c patches a copy), so the
       * reserved dw3 carries "distance to the buffer end" for WHOLE_SIZE ranges, whose size must shrink by
       * the dynamic offset. */
      if (is_dynamic(b->type) && bi->range == VK_WHOLE_SIZE)
         d[3] = range > UINT32_MAX ? UINT32_MAX : (uint32_t)range;
      break;
   }
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
      VK_FROM_HANDLE(ngv_image_view, view, ((const VkDescriptorImageInfo *)info)->imageView);
      if (view)
         memcpy(d, view->sampled, 32);
      else
         memset(d, 0, 32);
      break;
   }
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
      VK_FROM_HANDLE(ngv_image_view, view, ((const VkDescriptorImageInfo *)info)->imageView);
      if (view)
         memcpy(d, view->storage, 32);
      else
         memset(d, 0, 32);
      break;
   }
   case VK_DESCRIPTOR_TYPE_SAMPLER: {
      if (b->imm_start >= 0)
         break; /* immutable: written at allocation */
      VK_FROM_HANDLE(ngv_sampler, smp, ((const VkDescriptorImageInfo *)info)->sampler);
      if (smp)
         memcpy(d, smp->desc, 16);
      else
         memset(d, 0, 16);
      break;
   }
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
      const VkDescriptorImageInfo *ii = info;
      VK_FROM_HANDLE(ngv_image_view, view, ii->imageView);
      if (view)
         memcpy(d, view->sampled, 32);
      else
         memset(d, 0, 32);
      if (b->imm_start < 0) {
         VK_FROM_HANDLE(ngv_sampler, smp, ii->sampler);
         memset(d + 8, 0, 32);
         if (smp)
            memcpy(d + 8, smp->desc, 16);
      }
      break;
   }
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
      VK_FROM_HANDLE(ngv_buffer_view, bv, *(const VkBufferView *)info);
      if (bv)
         memcpy(d, bv->desc, 32);
      else
         memset(d, 0, 32);
      break;
   }
   default:
      memset(d, 0, b->stride);
      break;
   }
}

/* `count` descriptors starting at (`binding`, `elem`) of `set`, with the walk-on-overflow rule; element i is described by
 * the info struct at (const char *)info + i * info_stride.  The binding is looked up once and then followed along the
 * walk (ngv_UpdateDescriptorSets and the template path share this). */
static void
update_descriptors(struct ngv_descriptor_set *set, uint32_t binding, uint32_t elem, uint32_t count, const void *info,
                   size_t info_stride)
{
   const struct ngv_binding_layout *b = find_binding(set->layout, binding);
   for (uint32_t i = 0; i < count; i++, elem++) {
      uint32_t e = elem;
      b = walk(set->layout, b, &e);
      if (!b)
         break;
      elem = e; /* normalized into the (possibly next) binding */
      write_descriptor(b, (uint32_t *)((char *)set->map + b->offset + (size_t)e * b->stride),
                       (const char *)info + (size_t)i * info_stride);
   }
}

VKAPI_ATTR void VKAPI_CALL
ngv_UpdateDescriptorSets(VkDevice _device, uint32_t writeCount, const VkWriteDescriptorSet *pWrites,
                         uint32_t copyCount, const VkCopyDescriptorSet *pCopies)
{
   for (uint32_t w = 0; w < writeCount; w++) {
      const VkWriteDescriptorSet *wr = &pWrites[w];
      VK_FROM_HANDLE(ngv_descriptor_set, set, wr->dstSet);
      const void *info;
      size_t stride;
      switch (wr->descriptorType) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         info = wr->pBufferInfo;
         stride = sizeof(VkDescriptorBufferInfo);
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         info = wr->pTexelBufferView;
         stride = sizeof(VkBufferView);
         break;
      default:
         info = wr->pImageInfo;
         stride = sizeof(VkDescriptorImageInfo);
         break;
      }
      update_descriptors(set, wr->dstBinding, wr->dstArrayElement, wr->descriptorCount, info, stride);
   }
   for (uint32_t c = 0; c < copyCount; c++) {
      const VkCopyDescriptorSet *cp = &pCopies[c];
      VK_FROM_HANDLE(ngv_descriptor_set, src, cp->srcSet);
      VK_FROM_HANDLE(ngv_descriptor_set, dst, cp->dstSet);
      uint32_t se = cp->srcArrayElement, de = cp->dstArrayElement;
      const struct ngv_binding_layout *sb = find_binding(src->layout, cp->srcBinding);
      const struct ngv_binding_layout *db = find_binding(dst->layout, cp->dstBinding);
      for (uint32_t i = 0; i < cp->descriptorCount; i++, se++, de++) {
         sb = walk(src->layout, sb, &se);
         db = walk(dst->layout, db, &de);
         if (!sb || !db)
            break;
         memcpy((char *)dst->map + db->offset + (size_t)de * db->stride,
                (const char *)src->map + sb->offset + (size_t)se * sb->stride, db->stride);
      }
   }
}

/* vkUpdateDescriptorSetWithTemplate (Vulkan 1.1 core, was missing:). The template object is the runtime's
 * (vk_common_CreateDescriptorUpdateTemplate); every entry is one run of descriptors of a binding, written by the same
 * code as vkUpdateDescriptorSets (update_descriptors: same descriptor encoding, same walk-on-overflow rule, which
 * templates follow too) with the entry's binding resolved once.  Inline uniform blocks have no descriptor size here
 * (ngv_descriptor_size), so no layout can contain one. */
VKAPI_ATTR void VKAPI_CALL
ngv_UpdateDescriptorSetWithTemplate(VkDevice _device, VkDescriptorSet descriptorSet,
                                    VkDescriptorUpdateTemplate descriptorUpdateTemplate, const void *pData)
{
   VK_FROM_HANDLE(ngv_descriptor_set, set, descriptorSet);
   VK_FROM_HANDLE(vk_descriptor_update_template, templ, descriptorUpdateTemplate);
   for (uint32_t i = 0; i < templ->entry_count; i++) {
      const struct vk_descriptor_template_entry *entry = &templ->entries[i];
      if (entry->type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)
         continue;
      update_descriptors(set, entry->binding, entry->array_element, entry->array_count,
                         (const char *)pData + entry->offset, entry->stride);
   }
}
