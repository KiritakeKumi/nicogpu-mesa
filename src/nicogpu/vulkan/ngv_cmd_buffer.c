/* SPDX-License-Identifier: MIT */
/* ngv_cmd_buffer.c -- command buffer recording (batch 2).
 *
 * Recording writes the CP packet stream (command-set spec v0.2, encoders in ngv_cmd_stream.h) straight into
 * host-mapped GTT BOs ("IB chunks", 64 KiB each).  Every chunk is one entry of nicogpu_ws_submit_args.cmdbufs; the kernel
 * wraps each in an INDIRECT, and per INDIRECT the CP state (shader, user registers) starts empty, so the first
 * dispatch of every chunk re-emits SET_SHADER + SET_USER_REGS (section 6.4).
 *
 * What is supported: compute pipelines, descriptor sets (incl. dynamic buffers via a patched copy in an upload
 * area), push constants (16 dwords in u10..u25, the rest in the root table behind u0:u1), dispatch / indirect,
 * CopyBuffer / FillBuffer / UpdateBuffer, pipeline barriers, events, timestamp queries (ngv_query.c).
 * Base-group dispatch (DISPATCH_BASE pipelines) and 32-bit query results are supported since batch 3.
 * Per-lane scratch (shaders with scratch / spills) is backed by one BO per device (ngv_scratch_get).
 * secondary command buffers (inlined by ngv_CmdExecuteCommands) and gl_NumWorkGroups (a pointer to the grid in
 * two driver push dwords, flush_compute_state).
 * copies between buffers and images are in ngv_image.c.
 * Not supported (the command buffer enters an error state, VK_ERROR_FEATURE_NOT_PRESENT): everything else that is
 * graphics or image related, see the stubs at the end of this file.
 */
#include <string.h>

#include "ngv_cmd_stream.h"
#include "ngv_private.h"

static struct ngv_device *
cmd_device(struct ngv_cmd_buffer *cmd)
{
   return container_of(cmd->vk.base.device, struct ngv_device, vk);
}

/* ---- IB chunks / uploads ---------------------------------------------------------------------- */

/* Frees every BO and the bookkeeping arrays (destroy, and reset with RELEASE_RESOURCES). */
static void
cmd_free_storage(struct ngv_cmd_buffer *cmd)
{
   struct ngv_device *dev = cmd_device(cmd);
   for (uint32_t i = 0; i < cmd->chunk_alloc; i++)
      ngv_bo_destroy(dev, &cmd->chunks[i].bo);
   for (uint32_t i = 0; i < cmd->upload_alloc; i++)
      ngv_bo_destroy(dev, &cmd->uploads[i]);
   free(cmd->chunks);
   free(cmd->uploads);
   cmd->chunks = NULL;
   cmd->uploads = NULL;
   cmd->chunk_count = cmd->chunk_alloc = cmd->chunk_cap = 0;
   cmd->upload_count = cmd->upload_alloc = cmd->upload_cap = 0;
   cmd->upload_used = 0;
}

/* Plain reset: the BOs stay, the next recording starts again at chunk 0 / upload 0.  Nothing of the old recording can
 * reach the GPU: a submit only sees chunks[0 .. chunk_count) and, of each, the first used_dw dwords (INDIRECT size_dw;
 * ngv_cmd_buffer_collect_ibs); used_dw restarts at 0 when a chunk is taken into use again (cmd_new_chunk).  Upload
 * areas are zeroed when they are handed out again (upload_alloc), so they read like a fresh BO. */
static void
cmd_reset_storage(struct ngv_cmd_buffer *cmd)
{
   cmd->chunk_count = 0;
   cmd->upload_count = 0;
   cmd->upload_used = 0;
}

static void
cmd_reset_state(struct ngv_cmd_buffer *cmd)
{
   cmd->pipeline = NULL;
   memset(cmd->sets, 0, sizeof(cmd->sets));
   memset(cmd->dyn_offsets, 0, sizeof(cmd->dyn_offsets));
   memset(cmd->push, 0, sizeof(cmd->push));
   cmd->emitted_pipeline = NULL;
   cmd->u_valid = false;
   cmd->sets_valid = cmd->root_valid = false;
   memset(cmd->set_va, 0, sizeof(cmd->set_va));
   cmd->root_va = 0;
   cmd->numwg_va = 0;
   memset(cmd->numwg_grid, 0, sizeof(cmd->numwg_grid));
}

static bool
cmd_new_chunk(struct ngv_cmd_buffer *cmd)
{
   if (cmd->chunk_count == cmd->chunk_alloc) {
      /* no spare BO from an earlier recording: allocate one */
      if (cmd->chunk_alloc == cmd->chunk_cap) {
         uint32_t cap = cmd->chunk_cap ? cmd->chunk_cap * 2 : 4;
         struct ngv_ib_chunk *n = realloc(cmd->chunks, cap * sizeof(*n));
         if (!n) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
            return false;
         }
         cmd->chunks = n;
         cmd->chunk_cap = cap;
      }
      VkResult r = ngv_bo_create(cmd_device(cmd), NGV_IB_CHUNK_BYTES, NGV_BO_HOST, &cmd->chunks[cmd->chunk_alloc].bo);
      if (r != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, r);
         return false;
      }
      cmd->chunk_alloc++;
   }
   cmd->chunks[cmd->chunk_count].used_dw = 0;
   cmd->chunk_count++;
   /* new INDIRECT: CP state starts empty */
   cmd->emitted_pipeline = NULL;
   cmd->u_valid = false;
   return true;
}

uint32_t *
ngv_cmd_reserve(struct ngv_cmd_buffer *cmd, uint32_t n_dw)
{
   if (cmd->vk.record_result != VK_SUCCESS)
      return NULL;
   if (!cmd->chunk_count || cmd->chunks[cmd->chunk_count - 1].used_dw + n_dw > NGV_IB_CHUNK_DWORDS) {
      if (!cmd_new_chunk(cmd))
         return NULL;
   }
   struct ngv_ib_chunk *c = &cmd->chunks[cmd->chunk_count - 1];
   return (uint32_t *)c->bo.map + c->used_dw;
}

#define cmd_commit ngv_cmd_commit

/* One packet of n dwords: reserve, encode with `f`-style code at the call site, commit. */
#define EMIT(cmd, maxdw, expr)                      \
   do {                                             \
      uint32_t *dw_ = ngv_cmd_reserve(cmd, maxdw);  \
      if (dw_) {                                    \
         uint32_t n_ = (expr);                      \
         cmd_commit(cmd, n_);                       \
      }                                             \
   } while (0)

static bool
upload_alloc(struct ngv_cmd_buffer *cmd, uint32_t size, uint32_t alignment, uint64_t *va, void **ptr)
{
   size = align(size ? size : 1, alignment);
   if (!cmd->upload_count || align(cmd->upload_used, alignment) + size > cmd->uploads[cmd->upload_count - 1].size) {
      /* Never touch what is already handed out: take the next BO into use.  After a plain reset that is one kept from
       * the earlier recording (replaced if it is too small for an oversized request), else a new one. */
      const uint64_t bsz = size > NGV_UPLOAD_CHUNK_BYTES ? align64(size, NICOGPU_PAGE_SIZE) : NGV_UPLOAD_CHUNK_BYTES;
      if (cmd->upload_count == cmd->upload_alloc) {
         if (cmd->upload_alloc == cmd->upload_cap) {
            uint32_t cap = cmd->upload_cap ? cmd->upload_cap * 2 : 4;
            struct ngv_bo *n = realloc(cmd->uploads, cap * sizeof(*n));
            if (!n) {
               vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
               return false;
            }
            cmd->uploads = n;
            cmd->upload_cap = cap;
         }
         VkResult r = ngv_bo_create(cmd_device(cmd), bsz, NGV_BO_HOST, &cmd->uploads[cmd->upload_alloc]);
         if (r != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, r);
            return false;
         }
         cmd->upload_alloc++;
      } else if (cmd->uploads[cmd->upload_count].size < bsz) {
         struct ngv_bo *slot = &cmd->uploads[cmd->upload_count];
         struct ngv_bo bigger;
         VkResult r = ngv_bo_create(cmd_device(cmd), bsz, NGV_BO_HOST, &bigger);
         if (r != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, r);
            return false;
         }
         ngv_bo_destroy(cmd_device(cmd), slot);
         *slot = bigger;
      }
      cmd->upload_count++;
      cmd->upload_used = 0;
   }
   struct ngv_bo *bo = &cmd->uploads[cmd->upload_count - 1];
   uint32_t off = align(cmd->upload_used, alignment);
   *va = bo->va + off;
   *ptr = (char *)bo->map + off;
   memset(*ptr, 0, size); /* a kept BO holds the previous recording's bytes; hand out zeros like a fresh one */
   cmd->upload_used = off + size;
   return true;
}

bool
ngv_cmd_upload(struct ngv_cmd_buffer *cmd, uint32_t size, uint32_t alignment, uint64_t *va, void **ptr)
{
   return upload_alloc(cmd, size, alignment, va, ptr);
}

VkResult
ngv_cmd_buffer_collect_ibs(struct ngv_cmd_buffer *cmd, struct nicogpu_cmdbuf *out, uint32_t *n, uint32_t cap)
{
   if (cmd->vk.record_result != VK_SUCCESS)
      return cmd->vk.record_result;
   for (uint32_t i = 0; i < cmd->chunk_count; i++) {
      if (!cmd->chunks[i].used_dw)
         continue;
      if (*n >= cap)
         return VK_ERROR_UNKNOWN;
      out[*n].va = cmd->chunks[i].bo.va;
      out[*n].size_dw = cmd->chunks[i].used_dw;
      out[*n].flags = 0;
      (*n)++;
   }
   return VK_SUCCESS;
}

/* ---- command buffer object ------------------------------------------------------------------- */

static void
ngv_cmd_buffer_destroy(struct vk_command_buffer *vk)
{
   struct ngv_cmd_buffer *cmd = container_of(vk, struct ngv_cmd_buffer, vk);
   struct vk_command_pool *pool = vk->pool;
   cmd_free_storage(cmd);
   vk_command_buffer_finish(&cmd->vk);
   vk_free(&pool->alloc, cmd);
}

/* Also the path of vkResetCommandPool (the runtime forwards its RELEASE_RESOURCES flag), of a recycled vkFreeCommandBuffers
 * (always RELEASE_RESOURCES; this driver does not recycle, so a freed buffer goes through ngv_cmd_buffer_destroy) and of
 * the implicit reset in vkBeginCommandBuffer (flags 0): only RELEASE_RESOURCES gives the BOs back.  vkTrimCommandPool
 * only drops recycled buffers, of which there are none. */
static void
ngv_cmd_buffer_reset(struct vk_command_buffer *vk, VkCommandBufferResetFlags flags)
{
   struct ngv_cmd_buffer *cmd = container_of(vk, struct ngv_cmd_buffer, vk);
   if (flags & VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)
      cmd_free_storage(cmd);
   else
      cmd_reset_storage(cmd);
   cmd_reset_state(cmd);
   vk_command_buffer_reset(vk);
}

static VkResult
ngv_cmd_buffer_create(struct vk_command_pool *pool, VkCommandBufferLevel level, struct vk_command_buffer **out)
{
   struct ngv_cmd_buffer *cmd = vk_zalloc(&pool->alloc, sizeof(*cmd), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);
   VkResult r = vk_command_buffer_init(pool, &cmd->vk, &ngv_cmd_buffer_ops, level);
   if (r != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return r;
   }
   *out = &cmd->vk;
   return VK_SUCCESS;
}

const struct vk_command_buffer_ops ngv_cmd_buffer_ops = {
   .create = ngv_cmd_buffer_create,
   .reset = ngv_cmd_buffer_reset,
   .destroy = ngv_cmd_buffer_destroy,
};

VKAPI_ATTR VkResult VKAPI_CALL
ngv_BeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   vk_command_buffer_begin(&cmd->vk, pBeginInfo); /* resets a previously recorded buffer through ops->reset */
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   return vk_command_buffer_end(&cmd->vk);
}

/* Secondary command buffers are inlined: the packets of every IB chunk of each secondary are copied into the
 * primary's current chunk (or a new one).  That is valid because
 *   - the packets are position independent (no IB-relative addresses; INDIRECT cannot appear in an IB anyway),
 *   - a secondary chunk never relies on CP state from before its start: its first dispatch emits SET_SHADER and all
 *     of u0..u25 (a chunk starts with emitted_pipeline = NULL / u_valid = false), and a whole secondary chunk fits in
 *     an empty primary chunk (both NGV_IB_CHUNK_DWORDS), so no packet is split across two INDIRECTs,
 *   - what the packets point at (the secondary's upload area: dynamic-set copies, root tables, gl_NumWorkGroups grids;
 *     pipelines, query pools, events, scratch) stays alive as long as the secondary may be executed: the Vulkan spec
 *     makes the primary invalid when a secondary it records is reset, re-recorded or freed.
 * Compute state is not inherited in either direction (Vulkan: after vkCmdExecuteCommands the primary's bound state
 * is undefined); the primary's CP-state shadow is invalidated so its next dispatch re-emits everything.
 * PROVISIONAL: a copy, not a reference to the secondary's BOs (which would need one more nicogpu_cmdbuf entry per
 * secondary chunk and the kernel's per-submit cmdbuf limit); fine for the sizes dEQP and applications use. */
VKAPI_ATTR void VKAPI_CALL
ngv_CmdExecuteCommands(VkCommandBuffer commandBuffer, uint32_t count, const VkCommandBuffer *pCmds)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(ngv_cmd_buffer, sec, pCmds[i]);
      if (sec->vk.record_result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, sec->vk.record_result);
         return;
      }
      for (uint32_t c = 0; c < sec->chunk_count; c++) {
         const struct ngv_ib_chunk *ch = &sec->chunks[c];
         if (!ch->used_dw)
            continue;
         uint32_t *dw = ngv_cmd_reserve(cmd, ch->used_dw);
         if (!dw)
            return;
         memcpy(dw, ch->bo.map, (size_t)ch->used_dw * 4);
         cmd_commit(cmd, ch->used_dw);
         /* the copied packets changed the CP shader / user-register state of this chunk */
         cmd->emitted_pipeline = NULL;
         cmd->u_valid = false;
      }
   }
}

/* ---- bound state ------------------------------------------------------------------------------- */

VKAPI_ATTR void VKAPI_CALL
ngv_CmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint bindPoint, VkPipeline _pipeline)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_pipeline, p, _pipeline);
   if (bindPoint != VK_PIPELINE_BIND_POINT_COMPUTE) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (cmd->pipeline && cmd->pipeline->layout != p->layout)
      cmd->sets_valid = cmd->root_valid = false; /* dynamic-set copies / root table depend on the layout */
   cmd->pipeline = p;
}

static void
bind_sets(struct ngv_cmd_buffer *cmd, uint32_t first, uint32_t count, const VkDescriptorSet *sets, uint32_t dyn_count,
          const uint32_t *dyn)
{
   uint32_t d = 0;
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(ngv_descriptor_set, set, sets[i]);
      uint32_t idx = first + i;
      if (idx >= NGV_MAX_SETS) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      cmd->sets[idx] = set;
      for (uint32_t k = 0; k < set->layout->dynamic_count; k++)
         cmd->dyn_offsets[idx][k] = d < dyn_count ? dyn[d++] : 0;
   }
   cmd->sets_valid = cmd->root_valid = false; /* the root table holds the set addresses */
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint bindPoint, VkPipelineLayout layout,
                          uint32_t firstSet, uint32_t count, const VkDescriptorSet *pSets, uint32_t dynCount,
                          const uint32_t *pDyn)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   if (bindPoint != VK_PIPELINE_BIND_POINT_COMPUTE)
      return; /* no graphics queue: nothing consumes other bind points */
   bind_sets(cmd, firstSet, count, pSets, dynCount, pDyn);
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdBindDescriptorSets2(VkCommandBuffer commandBuffer, const VkBindDescriptorSetsInfo *info)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   if (!(info->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT))
      return;
   bind_sets(cmd, info->firstSet, info->descriptorSetCount, info->pDescriptorSets, info->dynamicOffsetCount,
             info->pDynamicOffsets);
}

static void
push_constants(struct ngv_cmd_buffer *cmd, uint32_t offset, uint32_t size, const void *values)
{
   if (offset + size > NGV_MAX_PUSH) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   memcpy((char *)cmd->push + offset, values, size);
   /* push[0..15] (bytes 0..63) travel in u10..u25 on every flush; only bytes 64.. live in the root table */
   if (size && offset + size > 64)
      cmd->root_valid = false;
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags,
                     uint32_t offset, uint32_t size, const void *pValues)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   push_constants(cmd, offset, size, pValues);
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdPushConstants2(VkCommandBuffer commandBuffer, const VkPushConstantsInfo *info)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   push_constants(cmd, info->offset, info->size, info->pValues);
}

/* ---- dispatch ---------------------------------------------------------------------------------- */

#define ROOT_TABLE_BYTES 128 /* +0x00 u64 set_va[8]; +0x40 u32 push_tail[16] (section 6.3) */

static bool
needs_root_table(const struct vk_pipeline_layout *l)
{
   if (l->set_count > 4)
      return true;
   for (uint32_t i = 0; i < l->push_range_count; i++)
      if (l->push_ranges[i].offset + l->push_ranges[i].size > 64)
         return true;
   return false;
}

/* Rebuild the GPU-visible copies of the bound sets (sets with dynamic descriptors get a patched copy).
 *
 * Upload lifetime: every call allocates fresh upload space (bump allocator, upload_alloc) and never writes into space it
 * handed out before, so a dispatch recorded earlier keeps pointing at exactly the bytes it was recorded with: the u2..u9 /
 * u0:u1 values it emitted into the IB are addresses of immutable data.  Only the driver's *current* addresses
 * (set_va[], root_va) are replaced; the IB already holds the old ones. */
static bool
build_set_uploads(struct ngv_cmd_buffer *cmd, const struct vk_pipeline_layout *layout)
{
   for (uint32_t i = 0; i < NGV_MAX_SETS; i++)
      cmd->set_va[i] = 0;
   for (uint32_t i = 0; i < layout->set_count; i++) {
      struct ngv_descriptor_set *set = cmd->sets[i];
      if (!set)
         continue;
      if (!set->layout->dynamic_count) {
         cmd->set_va[i] = set->va;
         continue;
      }
      /* dynamic buffers: patched copy in the upload area (descriptors are plain memory; descriptor spec section 8) */
      void *ptr;
      if (!upload_alloc(cmd, set->layout->size, 32, &cmd->set_va[i], &ptr))
         return false;
      memcpy(ptr, set->map, set->layout->size);
      for (uint32_t b = 0; b < set->layout->binding_count; b++) {
         const struct ngv_binding_layout *bl = &set->layout->bindings[b];
         if (bl->dyn_start < 0)
            continue;
         for (uint32_t e = 0; e < bl->count; e++) {
            uint32_t *d = (uint32_t *)((char *)ptr + bl->offset + (size_t)e * bl->stride);
            uint64_t off = cmd->dyn_offsets[i][bl->dyn_start + e];
            uint64_t base = (uint64_t)d[0] | ((uint64_t)(d[1] & 0xffffu) << 32);
            uint32_t size = d[2];
            if (d[3]) { /* WHOLE_SIZE write: d[3] is the distance to the buffer end (see ngv_UpdateDescriptorSets) */
               size = d[3] > off ? (uint32_t)(d[3] - off) : 0;
            }
            base += off;
            d[0] = (uint32_t)base;
            d[1] = (uint32_t)(base >> 32) & 0xffffu;
            d[2] = size;
            d[3] = 0;
         }
      }
   }
   cmd->sets_valid = true;
   cmd->root_valid = false; /* the table embeds set_va[] */
   return true;
}

/* The root table (u0:u1): set_va[8] and push[16..31] in a fresh 128 B upload.  Built from set_va[] as it is now, so it
 * needs build_set_uploads to have run for the current sets / layout.  A push-constant write that does not touch
 * bytes 64.. leaves it valid; one that does costs this 128 B allocation only (no dynamic-set copy is redone).  The tables
 * of earlier dispatches are never modified (see build_set_uploads). */
static bool
build_root_table(struct ngv_cmd_buffer *cmd, const struct vk_pipeline_layout *layout)
{
   cmd->root_va = 0;
   if (needs_root_table(layout)) {
      void *ptr;
      if (!upload_alloc(cmd, ROOT_TABLE_BYTES, 64, &cmd->root_va, &ptr))
         return false;
      uint64_t *tbl = ptr;
      for (uint32_t i = 0; i < NGV_MAX_SETS; i++)
         tbl[i] = cmd->set_va[i];
      memcpy((char *)ptr + 0x40, &cmd->push[16], 64);
   }
   cmd->root_valid = true;
   return true;
}

/* Everything a dispatch needs in the CP state: SET_SHADER (if the pipeline changed in this chunk) and the user
 * registers (if they differ from what the chunk already holds).  Reserves the whole bundle at once so that no chunk
 * switch can happen between the packets.  grid = the workgroup counts of a direct dispatch, NULL for an indirect one
 * whose VkDispatchIndirectCommand is at indirect_va (both only matter to shaders that read gl_NumWorkGroups). */
static bool
flush_compute_state(struct ngv_cmd_buffer *cmd, const uint32_t base[3], const uint32_t *grid, uint64_t indirect_va)
{
   struct ngv_pipeline *p = cmd->pipeline;
   if (!p || !p->layout) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return false;
   }
 /* Scratch : ScratchBytes is the per-lane stride (a multiple of 16, command-set spec SET_SHADER) and the VA a
    * 256 B aligned device BO big enough for every resident lane (ngv_scratch_get).  SCRATCH_BYTES = 0 needs VA = 0.  If
    * the BO grows later, a SET_SHADER already in this chunk keeps the old VA: still valid (the old BO is kept alive) and
    * big enough, since it was sized for this very pipeline. */
   uint32_t scratch_bytes = 0;
   uint64_t scratch_va = 0;
   if (p->scratch_bytes) {
      scratch_bytes = align(p->scratch_bytes, 16);
      VkResult r = ngv_scratch_get(cmd_device(cmd), scratch_bytes, &scratch_va);
      if (r != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, r);
         return false;
      }
   }
   if (!cmd->sets_valid && !build_set_uploads(cmd, p->layout))
      return false;
   if (!cmd->root_valid && !build_root_table(cmd, p->layout))
      return false;

   uint32_t u[NGV_NUM_UREGS] = {0};
   u[0] = (uint32_t)cmd->root_va;
   u[1] = (uint32_t)(cmd->root_va >> 32);
   for (uint32_t i = 0; i < 4; i++) {
      u[2 + 2 * i] = (uint32_t)cmd->set_va[i];
      u[3 + 2 * i] = (uint32_t)(cmd->set_va[i] >> 32);
   }
   memcpy(&u[10], cmd->push, 16 * sizeof(uint32_t));
   if (p->base_push_offset >= 0) {
      /* DISPATCH_BASE pipeline: the compiled code adds these three dwords to the hardware workgroup id (they sit
       * behind the layout's push range, ngv_pipeline.c); written for every dispatch so a stale base never leaks */
      for (uint32_t k = 0; k < 3; k++)
         u[10 + p->base_push_offset / 4 + k] = base[k];
   } else if (base[0] | base[1] | base[2]) {
      /* VUID-vkCmdDispatchBase-baseGroupX-00421: the pipeline must have been created with DISPATCH_BASE */
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return false;
   }
   if (p->numwg_push_offset >= 0) {
 /* gl_NumWorkGroups (PROVISIONAL): the compiled code loads {x, y, z} from this VA (ngv_nir.c). Indirect:
       * the parameter buffer itself (the CP reads the same 12 bytes for the dispatch; a barrier that orders an earlier
       * GPU write of them also invalidates L1, ngv_CmdPipelineBarrier2).  Direct: a copy in the upload area, reused
       * while the grid stays the same. */
      uint64_t va = indirect_va;
      if (grid) {
         if (!cmd->numwg_va || memcmp(cmd->numwg_grid, grid, sizeof(cmd->numwg_grid))) {
            void *ptr;
            if (!upload_alloc(cmd, sizeof(cmd->numwg_grid), 16, &cmd->numwg_va, &ptr)) {
               cmd->numwg_va = 0;
               return false;
            }
            memcpy(ptr, grid, sizeof(cmd->numwg_grid));
            memcpy(cmd->numwg_grid, grid, sizeof(cmd->numwg_grid));
         }
         va = cmd->numwg_va;
      }
      u[10 + p->numwg_push_offset / 4] = (uint32_t)va;
      u[11 + p->numwg_push_offset / 4] = (uint32_t)(va >> 32);
   }

   uint32_t *dw = ngv_cmd_reserve(cmd, NGV_IB_RESERVE_DW); /* may open a chunk: resets emitted_pipeline / u_valid */
   if (!dw)
      return false;
   uint32_t n = 0;
   if (cmd->emitted_pipeline != p) {
      n += ngv_emit_set_shader(dw + n, p->code.va, p->entry, p->vregs, p->shared_bytes, scratch_bytes, scratch_va,
                               p->wg[0], p->wg[1], p->wg[2]);
      cmd->emitted_pipeline = p;
   }
   if (!cmd->u_valid || memcmp(u, cmd->u_shadow, sizeof(u))) {
      n += ngv_emit_set_user_regs(dw + n, 0, u, NGV_NUM_UREGS);
      memcpy(cmd->u_shadow, u, sizeof(u));
      cmd->u_valid = true;
   }
   cmd_commit(cmd, n);
   return true;
}

static const uint32_t no_base[3] = {0, 0, 0};

VKAPI_ATTR void VKAPI_CALL
ngv_CmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   if (!x || !y || !z)
      return; /* empty dispatch */
   const uint32_t grid[3] = {x, y, z};
   if (!flush_compute_state(cmd, no_base, grid, 0))
      return;
   EMIT(cmd, 4, ngv_emit_dispatch(dw_, x, y, z));
}

/* base != 0 (batch 3): the hardware numbers workgroups from 0 and u30/u31 cannot be written by the driver, so
 * the base travels as three push-constant dwords of the pipeline (flush_compute_state / ngv_pipeline.c). */
VKAPI_ATTR void VKAPI_CALL
ngv_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y,
                    uint32_t z)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   if (!x || !y || !z)
      return;
   const uint32_t base[3] = {bx, by, bz};
   const uint32_t grid[3] = {x, y, z}; /* gl_NumWorkGroups = the counts, without the base */
   if (!flush_compute_state(cmd, base, grid, 0))
      return;
   EMIT(cmd, 4, ngv_emit_dispatch(dw_, x, y, z));
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer, VkDeviceSize offset)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_buffer, buffer, _buffer);
   const uint64_t args_va = buffer->vk.device_address + offset;
   if (!flush_compute_state(cmd, no_base, NULL, args_va))
      return;
   EMIT(cmd, 3, ngv_emit_dispatch_indirect(dw_, args_va));
}

/* ---- synchronization --------------------------------------------------------------------------- */

#define CACHE_FULL                                                                                     \
   (NICOGPU_CMD_CACHE_FLAGS_WAIT_IDLE_MASK | NICOGPU_CMD_CACHE_FLAGS_INV_L1_MASK | NICOGPU_CMD_CACHE_FLAGS_INV_K_MASK)

static bool
dep_has_src_stage(const VkDependencyInfo *d)
{
   VkPipelineStageFlags2 s = 0;
   for (uint32_t i = 0; i < d->memoryBarrierCount; i++)
      s |= d->pMemoryBarriers[i].srcStageMask;
   for (uint32_t i = 0; i < d->bufferMemoryBarrierCount; i++)
      s |= d->pBufferMemoryBarriers[i].srcStageMask;
   for (uint32_t i = 0; i < d->imageMemoryBarrierCount; i++)
      s |= d->pImageMemoryBarriers[i].srcStageMask;
   return s != 0;
}

/* One conservative rule (command-set spec section 3.5: "the implementation may be more conservative"): any barrier with a source
 * stage waits for all async work (DISPATCH/COPY/FILL) and invalidates L1/K so descriptors / data written by the GPU
 * are re-read.  Per-stage tuning (e.g. no INV for transfer->transfer) is a later optimization. */
VKAPI_ATTR void VKAPI_CALL
ngv_CmdPipelineBarrier2(VkCommandBuffer commandBuffer, const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   if (dep_has_src_stage(pDependencyInfo))
      EMIT(cmd, 2, ngv_emit_cache(dw_, CACHE_FULL));
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateEvent(VkDevice _device, const VkEventCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                VkEvent *pEvent)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   struct ngv_event *ev = vk_object_zalloc(&device->vk, pAllocator, sizeof(*ev), VK_OBJECT_TYPE_EVENT);
   if (!ev)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   VkResult r = ngv_bo_create(device, NICOGPU_PAGE_SIZE, NGV_BO_HOST, &ev->bo);
   if (r != VK_SUCCESS) {
      vk_object_free(&device->vk, pAllocator, ev);
      return vk_error(device, r);
   }
   *(volatile uint32_t *)ev->bo.map = 0;
   *pEvent = ngv_event_to_handle(ev);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyEvent(VkDevice _device, VkEvent _event, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_event, ev, _event);
   if (!ev)
      return;
   ngv_bo_destroy(device, &ev->bo);
   vk_object_free(&device->vk, pAllocator, ev);
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_GetEventStatus(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(ngv_event, ev, _event);
   return *(volatile uint32_t *)ev->bo.map ? VK_EVENT_SET : VK_EVENT_RESET;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_SetEvent(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(ngv_event, ev, _event);
   *(volatile uint32_t *)ev->bo.map = 1;
   __sync_synchronize();
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_ResetEvent(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(ngv_event, ev, _event);
   *(volatile uint32_t *)ev->bo.map = 0;
   __sync_synchronize();
   return VK_SUCCESS;
}

/* Event set/reset from the device: WRITE_DATA does not wait for earlier work, so wait for idle first (set must
 * happen after the work it covers; reset likewise, conservatively). */
static void
event_write(struct ngv_cmd_buffer *cmd, struct ngv_event *ev, uint32_t value)
{
   uint32_t *dw = ngv_cmd_reserve(cmd, 2 + 4);
   if (!dw)
      return;
   uint32_t n = ngv_emit_cache(dw, NICOGPU_CMD_CACHE_FLAGS_WAIT_IDLE_MASK);
   n += ngv_emit_write_data(dw + n, ev->bo.va, &value, 1);
   cmd_commit(cmd, n);
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent _event, const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_event, ev, _event);
   event_write(cmd, ev, 1);
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent _event, VkPipelineStageFlags2 stageMask)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_event, ev, _event);
   event_write(cmd, ev, 0);
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent *pEvents,
                   const VkDependencyInfo *pDependencyInfos)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   for (uint32_t i = 0; i < eventCount; i++) {
      VK_FROM_HANDLE(ngv_event, ev, pEvents[i]);
      uint32_t *dw = ngv_cmd_reserve(cmd, 8 + 2);
      if (!dw)
         return;
      uint32_t n = ngv_emit_wait_mem(dw, ev->bo.va, 1, 0xffffffffu, NICOGPU_CMD_WAIT_FUNC_EQ, false);
      /* the event's covered work is complete; make its writes visible to the shaders that follow */
      n += ngv_emit_cache(dw + n, NICOGPU_CMD_CACHE_FLAGS_INV_L1_MASK | NICOGPU_CMD_CACHE_FLAGS_INV_K_MASK);
      cmd_commit(cmd, n);
   }
}

/* ---- transfer ---------------------------------------------------------------------------------- */

VKAPI_ATTR void VKAPI_CALL
ngv_CmdCopyBuffer2(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *info)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_buffer, src, info->srcBuffer);
   VK_FROM_HANDLE(ngv_buffer, dst, info->dstBuffer);
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkBufferCopy2 *r = &info->pRegions[i];
      if (r->size)
         EMIT(cmd, 8, ngv_emit_copy(dw_, src->vk.device_address + r->srcOffset, dst->vk.device_address + r->dstOffset,
                                    r->size));
   }
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer _dst, VkDeviceSize offset, VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_buffer, dst, _dst);
   size = vk_buffer_range(&dst->vk, offset, size);
   size &= ~(VkDeviceSize)3; /* WHOLE_SIZE rounds down to a multiple of 4 */
   if (size)
      EMIT(cmd, 6, ngv_emit_fill(dw_, dst->vk.device_address + offset, size, data));
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer _dst, VkDeviceSize offset, VkDeviceSize size,
                    const void *pData)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_buffer, dst, _dst);
   const uint32_t *data = pData;
   uint64_t va = dst->vk.device_address + offset;
   uint32_t remaining = (uint32_t)(size / 4); /* size is a multiple of 4 (spec) */
   while (remaining) {
      uint32_t k = remaining > 4096 ? 4096 : remaining;
      uint32_t *dw = ngv_cmd_reserve(cmd, 3 + k);
      if (!dw)
         return;
      cmd_commit(cmd, ngv_emit_write_data(dw, va, data, k));
      va += (uint64_t)k * 4;
      data += k;
      remaining -= k;
   }
}

/* ---- commands the device cannot execute -------------------------------------------------------------
 * The device has a compute queue family only and no image hardware (graphics and images are P2).  A core command that
 * the driver does not define has a NULL entry in the dispatch table: an application (and the CTS) that calls it
 * through the pointer it got from vkGetDeviceProcAddr then dies with SIGSEGV, and the Mesa runtime shims of the older
 * entry points (vkCmdBlitImage -> CmdBlitImage2, vkCmdBeginQuery -> CmdBeginQueryIndexedEXT, vkCmdBeginRenderPass ->
 * CmdBeginRendering, ...) call straight through the table.  Every such command is defined here and puts the command
 * buffer into an error state, so the failure surfaces as VK_ERROR_FEATURE_NOT_PRESENT from vkEndCommandBuffer.
 * Draws, blit, resolve, ClearAttachments, vertex / index binds and queries are graphics-queue commands; they are
 * not valid on the compute-only queue family anyway.  vkCmdClearColorImage / vkCmdClearDepthStencilImage are valid
 * on a compute queue and run as compute kernels with the image path (ngv_meta.c). */
#define NGV_CMD_UNSUPPORTED(name, proto)                                  \
   VKAPI_ATTR void VKAPI_CALL ngv_##name proto                            \
   {                                                                      \
      VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);                 \
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT); \
   }

NGV_CMD_UNSUPPORTED(CmdDraw, (VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                              uint32_t firstVertex, uint32_t firstInstance))
NGV_CMD_UNSUPPORTED(CmdDrawIndexed, (VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                                     uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance))
NGV_CMD_UNSUPPORTED(CmdDrawIndirect, (VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                      uint32_t drawCount, uint32_t stride))
NGV_CMD_UNSUPPORTED(CmdDrawIndexedIndirect, (VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                             uint32_t drawCount, uint32_t stride))
NGV_CMD_UNSUPPORTED(CmdDrawIndirectCount, (VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                           VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                           uint32_t maxDrawCount, uint32_t stride))
NGV_CMD_UNSUPPORTED(CmdDrawIndexedIndirectCount, (VkCommandBuffer commandBuffer, VkBuffer buffer,
                                                  VkDeviceSize offset, VkBuffer countBuffer,
                                                  VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                  uint32_t stride))
NGV_CMD_UNSUPPORTED(CmdClearAttachments, (VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                                          const VkClearAttachment *pAttachments, uint32_t rectCount,
                                          const VkClearRect *pRects))
NGV_CMD_UNSUPPORTED(CmdBlitImage2, (VkCommandBuffer commandBuffer, const VkBlitImageInfo2 *pBlitImageInfo))
NGV_CMD_UNSUPPORTED(CmdResolveImage2, (VkCommandBuffer commandBuffer, const VkResolveImageInfo2 *pResolveImageInfo))
NGV_CMD_UNSUPPORTED(CmdBeginRendering, (VkCommandBuffer commandBuffer, const VkRenderingInfo *pRenderingInfo))
NGV_CMD_UNSUPPORTED(CmdBindVertexBuffers2, (VkCommandBuffer commandBuffer, uint32_t firstBinding,
                                            uint32_t bindingCount, const VkBuffer *pBuffers,
                                            const VkDeviceSize *pOffsets, const VkDeviceSize *pSizes,
                                            const VkDeviceSize *pStrides))
NGV_CMD_UNSUPPORTED(CmdBeginQueryIndexedEXT, (VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query,
                                              VkQueryControlFlags flags, uint32_t index))
NGV_CMD_UNSUPPORTED(CmdEndQueryIndexedEXT, (VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query,
                                            uint32_t index))
NGV_CMD_UNSUPPORTED(CmdBindIndexBuffer2, (VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                          VkDeviceSize size, VkIndexType indexType))
