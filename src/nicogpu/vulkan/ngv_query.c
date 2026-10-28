/* SPDX-License-Identifier: MIT */
/* ngv_query.c -- timestamp query pools (batch 2).
 *
 * Layout of the pool BO (GTT, host-visible): `count` u64 timestamp values, then `count` u64 availability words.
 * vkCmdWriteTimestamp2 = TIMESTAMP (end-of-pipe, 64 bit device clock, command-set spec section 4.1) followed by an in-IB FENCE64
 * that publishes availability = 1 (both packets wait for all earlier async work).  Results are raw device ticks; the
 * period is timestampPeriod = 1e9 / timestamp_freq_hz.  Occlusion pools exist but never receive results (compute-only
 * queue family); pipeline statistics pools: graphics batch.
 */
#include <time.h>
#include <string.h>

#include "ngv_cmd_stream.h"
#include "ngv_private.h"

static uint64_t
avail_va(const struct ngv_query_pool *p, uint32_t q)
{
   return p->bo.va + (uint64_t)(p->vk.query_count + q) * 8;
}

static uint64_t
value_va(const struct ngv_query_pool *p, uint32_t q)
{
   return p->bo.va + (uint64_t)q * 8;
}

static volatile uint64_t *
value_ptr(const struct ngv_query_pool *p, uint32_t q)
{
   return (volatile uint64_t *)p->bo.map + q;
}

static volatile uint64_t *
avail_ptr(const struct ngv_query_pool *p, uint32_t q)
{
   return (volatile uint64_t *)p->bo.map + p->vk.query_count + q;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateQueryPool(VkDevice _device, const VkQueryPoolCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                    VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   /* TIMESTAMP is written by the CP.  OCCLUSION pools can be created (every device must allow it; dEQP
    * api.object_management.*query_pool) but nothing can write them: vkCmdBeginQuery / EndQuery are graphics-queue
    * commands, so the queries of such a pool stay unavailable (ngv_CmdBeginQueryIndexedEXT in ngv_cmd_buffer.c).
    * PIPELINE_STATISTICS needs the pipelineStatisticsQuery feature (off), the rest needs extensions we do not expose. */
   if (pCreateInfo->queryType == VK_QUERY_TYPE_TIMESTAMP) {
      if (!(device->pdev->info.features & NICOGPU_FEATURE_TIMESTAMP))
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT); /* no device clock (features-bit degradation) */
   } else if (pCreateInfo->queryType != VK_QUERY_TYPE_OCCLUSION) {
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   struct ngv_query_pool *pool = vk_query_pool_create(&device->vk, pCreateInfo, pAllocator, sizeof(*pool));
   if (!pool)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   VkResult r = ngv_bo_create(device, align64((uint64_t)pCreateInfo->queryCount * 16, NICOGPU_PAGE_SIZE), NGV_BO_HOST, &pool->bo);
   if (r != VK_SUCCESS) {
      vk_query_pool_destroy(&device->vk, pAllocator, &pool->vk);
      return vk_error(device, r);
   }
   memset(pool->bo.map, 0, pool->bo.size);
   *pQueryPool = ngv_query_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyQueryPool(VkDevice _device, VkQueryPool _pool, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_query_pool, pool, _pool);
   if (!pool)
      return;
   ngv_bo_destroy(device, &pool->bo);
   vk_query_pool_destroy(&device->vk, pAllocator, &pool->vk);
}

VKAPI_ATTR void VKAPI_CALL
ngv_ResetQueryPool(VkDevice _device, VkQueryPool _pool, uint32_t first, uint32_t count)
{
   VK_FROM_HANDLE(ngv_query_pool, pool, _pool);
   for (uint32_t i = 0; i < count; i++) {
      *avail_ptr(pool, first + i) = 0;
      *value_ptr(pool, first + i) = 0;
   }
   __sync_synchronize();
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_GetQueryPoolResults(VkDevice _device, VkQueryPool _pool, uint32_t first, uint32_t count, size_t dataSize,
                        void *pData, VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_query_pool, pool, _pool);
   VkResult result = VK_SUCCESS;
   const bool b64 = flags & VK_QUERY_RESULT_64_BIT;
   for (uint32_t i = 0; i < count; i++) {
      uint64_t avail = *avail_ptr(pool, first + i);
      if (!avail && (flags & VK_QUERY_RESULT_WAIT_BIT)) {
         /* poll; give up after ~10 s (a lost device never publishes) */
         for (int k = 0; k < 200000 && !(avail = *avail_ptr(pool, first + i)); k++) {
            struct timespec ts = {0, 50000};
            nanosleep(&ts, NULL);
            if (device->ws.ops->device_is_lost(device->ws.dev) == 1)
               return VK_ERROR_DEVICE_LOST;
         }
      }
      char *out = (char *)pData + (size_t)i * stride;
      if (avail) {
         uint64_t v = *value_ptr(pool, first + i);
         if (b64)
            *(uint64_t *)out = v;
         else
            *(uint32_t *)out = (uint32_t)v;
      } else {
         result = VK_NOT_READY;
      }
      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
         if (b64)
            *(uint64_t *)(out + 8) = !!avail;
         else
            *(uint32_t *)(out + 4) = !!avail;
      }
   }
   return result;
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool _pool, uint32_t first, uint32_t count)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_query_pool, pool, _pool);
   if (!count)
      return;
   /* values then availability are two contiguous ranges: two FILLs.  FILL is async: any later end-of-pipe packet
    * (TIMESTAMP / FENCE) waits for it. */
   uint32_t *dw = ngv_cmd_reserve(cmd, 12);
   if (!dw)
      return;
   uint32_t n = ngv_emit_fill(dw, value_va(pool, first), (uint64_t)count * 8, 0);
   n += ngv_emit_fill(dw + n, avail_va(pool, first), (uint64_t)count * 8, 0);
   ngv_cmd_commit(cmd, n);
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdWriteTimestamp2(VkCommandBuffer commandBuffer, VkPipelineStageFlags2 stage, VkQueryPool _pool, uint32_t query)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_query_pool, pool, _pool);
   uint32_t *dw = ngv_cmd_reserve(cmd, 3 + 6);
   if (!dw)
      return;
   uint32_t n = ngv_emit_timestamp(dw, value_va(pool, query));
   n += ngv_emit_fence64(dw + n, avail_va(pool, query), 1);
   ngv_cmd_commit(cmd, n);
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool _pool, uint32_t first, uint32_t count,
                            VkBuffer _dst, VkDeviceSize dstOffset, VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_query_pool, pool, _pool);
   VK_FROM_HANDLE(ngv_buffer, dst, _dst);
   /* 32-bit results (batch 3): the pool stores little-endian u64 values / availability words, and a 32-bit
    * result is the low 32 bits of the value (host path: GetQueryPoolResults truncates the same way), i.e. the first
    * 4 bytes -- the COPY engine does it with a 4-byte copy, no meta shader needed. */
   const bool b64 = flags & VK_QUERY_RESULT_64_BIT;
   const uint64_t esz = b64 ? 8 : 4;
   for (uint32_t i = 0; i < count; i++) {
      uint32_t q = first + i;
      uint64_t out = dst->vk.device_address + dstOffset + (uint64_t)i * stride;
      uint32_t *dw = ngv_cmd_reserve(cmd, 8 + 8 + 8);
      if (!dw)
         return;
      uint32_t n = 0;
      if (flags & VK_QUERY_RESULT_WAIT_BIT)
         n += ngv_emit_wait_mem(dw + n, avail_va(pool, q), 1, ~0ull, NICOGPU_CMD_WAIT_FUNC_EQ, true);
      n += ngv_emit_copy(dw + n, value_va(pool, q), out, esz);
      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
         n += ngv_emit_copy(dw + n, avail_va(pool, q), out + esz, esz);
      ngv_cmd_commit(cmd, n);
   }
}
