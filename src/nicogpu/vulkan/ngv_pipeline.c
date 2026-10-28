/* SPDX-License-Identifier: MIT */
/* ngv_pipeline.c -- compute pipelines and the shader compiler stage (batch 2/3).
 *
 * ngv_shader_compile turns a SPIR-V entry point into a code object (struct ngv_shader_object).  Sources, in order
 * (section 14.2):
 *
 * 1. $NGV_XO_DIR/<entry-point-name>.xo -- a ready ngcomp object ("NGCO" container; "XGCO" objects of earlier stages are still read, ngcomp/DESIGN.md 6)
 *      produced offline; a debugging override, off unless the variable is set.
 *   2. the real path (batch 3): SPIR-V -> NIR (ngv_nir.c, incl. ngv_nir_lower_pipeline_layout) -> ngcomp text
 * (ngv_nir_print.c) -> `python3 ngcomp/driver.py isel --regalloc --exact-waitcnt` (ngv_ngcomp.c) ->.xo.
 *   3. a placeholder blob: 16 bytes {"NGVPHKRN", kernel id}, kernel id from an entry point named "ngv_ph_*".  Only the
 *      stub winsys' CP emulation (ngv_stub_cp.c) can run it (it is not ISA code); it is what the stub backend gets
 *      for such entry names, because the stub is not an ISA interpreter.  $NGV_SHADER_COMPILER = ngcomp |
 *      placeholder | auto (default) overrides the choice.
 *
 * Everything the driver needs from a code object is in struct ngv_shader_object, so the pipeline / command stream
 * code does not care where it came from.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/spirv/spirv.h"
#include "util/ralloc.h"
#include "vk_pipeline_cache.h"

#include "ngv_private.h"
#include "ngv_xo.h"

/* Entry point name and LocalSize of the GLCompute entry point called `name`.  LocalSizeId / WorkgroupSize built-in
 * (specialization) are not handled: the local size then stays 1x1x1 -- PROVISIONAL. */
static bool
spirv_scan(const uint32_t *w, uint32_t n, const char *name, uint32_t wg[3])
{
   if (n < 5 || w[0] != SpvMagicNumber)
      return false;
   uint32_t entry_id = 0;
   bool found = false;
   wg[0] = wg[1] = wg[2] = 1;
   for (uint32_t i = 5; i < n;) {
      uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
      if (!wc || i + wc > n)
         return false;
      if (op == SpvOpEntryPoint && wc >= 4 && w[i + 1] == SpvExecutionModelGLCompute) {
         const char *s = (const char *)&w[i + 3];
         size_t maxlen = (size_t)(wc - 3) * 4;
         if (strnlen(s, maxlen) < maxlen && !strcmp(s, name)) {
            entry_id = w[i + 2];
            found = true;
         }
      } else if (op == SpvOpExecutionMode && wc >= 6 && found && w[i + 1] == entry_id &&
                 w[i + 2] == SpvExecutionModeLocalSize) {
         wg[0] = w[i + 3];
         wg[1] = w[i + 4];
         wg[2] = w[i + 5];
      }
      i += wc;
   }
   return found;
}

/* Read a whole file (the .xo container); NULL when it is missing or implausibly small / large. */
uint8_t *
ngv_read_file(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      return NULL;
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint8_t *buf = (sz > 32 && sz < (64 << 20)) ? malloc((size_t)sz) : NULL;
   if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
      free(buf);
      fclose(f);
      return NULL;
   }
   fclose(f);
   *size = (size_t)sz;
   return buf;
}

/* Parse an in-memory .xo container (also the format of the shader cache entries, ngv_ngcomp.c).  ABI 0.1 and ABI 0.2
 * objects are both accepted (ngv_xo.h): the metadata fields read here -- entry, vgprs, scratch -- are the same in both;
 * ABI 0.2 adds the stage mask, which a compute pipeline must find to be exactly "compute" (ngv_xo_is_compute). */
VkResult
ngv_parse_xo(const uint8_t *buf, size_t sz, struct ngv_shader_object *o)
{
   struct ngv_xo_info xi;
   if (!ngv_xo_check(buf, sz, &xi))
      return VK_ERROR_UNKNOWN;
   o->xo_abi = xi.abi;
   o->xo_stage_mask = xi.stage_mask;
   const char *meta = (const char *)buf + xi.meta_off;
   uint64_t v;
   o->entry = ngv_xo_json_int(meta, xi.meta_len, "entry", false, &v) ? (uint32_t)v : 0;
   o->vregs = ngv_xo_json_int(meta, xi.meta_len, "vgprs", true, &v) ? (uint32_t)v : 0;
   /* SET_SHADER.VREGS is 1..max_vregs and v0 holds the packed local id at entry (ABI section 2), so a shader
    * that touches no vector register (dEQP-VK.compute.pipeline.basic.empty_shader) still needs one */
   if (o->vregs == 0)
      o->vregs = 1;
   o->scratch_bytes = ngv_xo_json_int(meta, xi.meta_len, "scratch", true, &v) ? (uint32_t)v : 0;
   o->code_size = xi.code_len;
   o->code = malloc(xi.code_len);
   if (!o->code)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   memcpy(o->code, buf + xi.code_off, xi.code_len);
   return VK_SUCCESS;
}

VkResult
ngv_load_xo(const char *path, struct ngv_shader_object *o)
{
   size_t sz;
   uint8_t *buf = ngv_read_file(path, &sz);
   if (!buf)
      return VK_ERROR_UNKNOWN;
   VkResult r = ngv_parse_xo(buf, sz, o);
   free(buf);
   return r;
}

/* Compiler-stage failures are always reported on stderr (the Vulkan result alone says nothing; the runtime's vk_log
 * only reaches a debug messenger the application may not have). */
VkResult
ngv_shader_error(struct ngv_device *dev, VkResult res, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   fprintf(stderr, "ngv: ");
   vfprintf(stderr, fmt, ap);
   fputc(10, stderr);
   va_end(ap);
   return vk_error(dev, res);
}

static bool
use_placeholder(struct ngv_device *dev, const char *entry_name)
{
   const char *m = getenv("NGV_SHADER_COMPILER");
   if (m && !strcmp(m, "placeholder"))
      return true;
   if (m && !strcmp(m, "ngcomp"))
      return false;
   return dev->pdev->ops == &ngv_winsys_stub_ops && !strncmp(entry_name, "ngv_ph_", 7);
}

/* (PROVISIONAL): a workgroup lives in one half core (<= 2 SPs) and every resident warp holds `vregs`
 * vector registers of its SP's register file, so a compiled shader must satisfy
 *    vregs * ceil(warps / min(sp_per_core, 2)) <= vreg slots per SP
 * (: 128 KiB / (warp_size * 4 B) = 1024 slots at warp_size 32; ng_wd rejects the dispatch otherwise). The
 * register file size is not in the device info, so it is derived from the architectural 128 KiB per SP. */
static uint32_t
ngv_vreg_slots_per_sp(const struct ngv_device *dev)
{
   const uint32_t warp_size = dev->pdev->info.warp_size ? dev->pdev->info.warp_size : 32;
   return (128u * 1024u) / (warp_size * 4u);
}

/* warps per SP a workgroup of wg[] needs when spread over one half core */
uint32_t
ngv_warps_per_sp(const struct ngv_device *dev, const uint32_t wg[3])
{
   const uint32_t warp_size = dev->pdev->info.warp_size ? dev->pdev->info.warp_size : 32;
   const uint32_t sp = dev->pdev->info.sp_per_core ? dev->pdev->info.sp_per_core : 1;
   const uint32_t half = sp < 2 ? sp : 2;
   const uint64_t threads = (uint64_t)wg[0] * wg[1] * wg[2];
   const uint64_t warps = (threads + warp_size - 1) / warp_size;
   return (uint32_t)((warps + half - 1) / half);
}

static VkResult
ngv_check_regfile_budget(struct ngv_device *dev, const char *entry_name, const struct ngv_shader_object *so)
{
   if (so->placeholder)
      return VK_SUCCESS;
   /* ABI 0.2 object header (ngv_xo.h): this driver runs compute only, so graphics segments (fragment, tile programs,
    * DRAW-launched geometry) are refused; every ABI 0.1 object is compute */
   if (!ngv_xo_is_compute(so->xo_stage_mask))
      return ngv_shader_error(dev, VK_ERROR_UNKNOWN, "shader '%s': code object has non-compute stages (ABI %u, stage mask 0x%x)",
                              entry_name, so->xo_abi, so->xo_stage_mask);
   const uint32_t per_sp = ngv_warps_per_sp(dev, so->wg);
   const uint64_t need = (uint64_t)so->vregs * per_sp;
   if (need > ngv_vreg_slots_per_sp(dev))
      return ngv_shader_error(dev, VK_ERROR_UNKNOWN,
                              "shader '%s': %u vector registers x %u warps/SP = %llu exceeds the %u register slots of an SP "
                              "(use a smaller local size or fewer live values)",
                              entry_name, so->vregs, per_sp, (unsigned long long)need, ngv_vreg_slots_per_sp(dev));
   return VK_SUCCESS;
}

VkResult
ngv_shader_compile(struct ngv_device *dev, struct vk_pipeline_cache *pc, const VkPipelineShaderStageCreateInfo *stage,
                   VkPipelineCreateFlags2KHR pipeline_flags, const uint32_t *spirv, uint32_t spirv_words,
                   const struct vk_pipeline_layout *layout, int32_t base_push_offset, int32_t numwg_push_offset,
                   struct ngv_shader_object *out)
{
   const char *entry_name = stage->pName;
   memset(out, 0, sizeof(*out));
   if (!spirv_scan(spirv, spirv_words, entry_name, out->wg))
      return ngv_shader_error(dev, VK_ERROR_UNKNOWN, "no GLCompute entry point '%s' in the SPIR-V module", entry_name);
   if (!out->wg[0] || !out->wg[1] || !out->wg[2] ||
       (uint64_t)out->wg[0] * out->wg[1] * out->wg[2] > dev->pdev->info.max_workgroup_threads)
      return ngv_shader_error(dev, VK_ERROR_UNKNOWN, "bad LocalSize %ux%ux%u", out->wg[0], out->wg[1], out->wg[2]);

   const char *dir = getenv("NGV_XO_DIR");
   if (dir && *dir) {
      char path[512];
      snprintf(path, sizeof(path), "%s/%s.xo", dir, entry_name);
      VkResult r = ngv_load_xo(path, out);
      if (r == VK_SUCCESS)
         return ngv_check_regfile_budget(dev, entry_name, out);
      if (r == VK_ERROR_OUT_OF_HOST_MEMORY)
         return r;
      /* not found / malformed: fall through */
   }

   if (!use_placeholder(dev, entry_name)) {
      /* SPIR-V level cache (VkPipelineCache + disk cache) before any translation; a hit still passes the register file
       * budget check below like a fresh compile */
      uint8_t spv_key[20];
      const bool use_spv_cache = ngv_ngcomp_spirv_key(dev, pipeline_flags, stage, spirv, spirv_words, layout,
                                                      base_push_offset, numwg_push_offset, spv_key);
      if (use_spv_cache) {
         VkResult cr = ngv_ngcomp_spirv_lookup(dev, pc, spv_key, out);
         if (cr == VK_SUCCESS)
            return ngv_check_regfile_budget(dev, entry_name, out);
         if (cr != VK_INCOMPLETE)
            return cr;
      }
      nir_shader *nir = NULL;
      bool uses_numwg = false;
      VkResult r = ngv_shader_to_nir(dev, pipeline_flags, stage, layout, base_push_offset, numwg_push_offset,
                                     &uses_numwg, &nir);
      if (r != VK_SUCCESS)
         return r;
      char err[256];
      char *text = ngv_nir_print_ngcomp(nir, err, sizeof(err));
      /* the local size and shared size of the lowered NIR are authoritative (specialization constants) */
      uint32_t wg[3] = {nir->info.workgroup_size[0], nir->info.workgroup_size[1], nir->info.workgroup_size[2]};
      uint32_t shared = nir->info.shared_size;
      ralloc_free(nir);
      if (!text)
         return ngv_shader_error(dev, VK_ERROR_UNKNOWN, "shader '%s': cannot hand it to ngcomp: %s", entry_name, err);
      if (!wg[0] || !wg[1] || !wg[2] || (uint64_t)wg[0] * wg[1] * wg[2] > dev->pdev->info.max_workgroup_threads ||
          shared > dev->pdev->info.max_shared_bytes) {
         free(text);
         return ngv_shader_error(dev, VK_ERROR_UNKNOWN, "shader '%s': bad local size %ux%ux%u / shared %u B", entry_name, wg[0],
                          wg[1], wg[2], shared);
      }
      /* occupancy target: warps per SP this workgroup needs; ngcomp lowers its register budget to fit (spills) */
      const uint32_t occupancy = ngv_warps_per_sp(dev, wg);
      uint8_t *xo = NULL;
      size_t xo_size = 0;
      r = ngv_ngcomp_compile(dev, text, occupancy, out, &xo, &xo_size);
      free(text);
      if (r != VK_SUCCESS)
         return r;
      memcpy(out->wg, wg, sizeof(wg));
      out->shared_bytes = shared;
      out->uses_num_workgroups = uses_numwg;
      r = ngv_check_regfile_budget(dev, entry_name, out);
      if (r == VK_SUCCESS && use_spv_cache)
         ngv_ngcomp_spirv_store(dev, pc, spv_key, out, occupancy, xo, xo_size);
      free(xo);
      return r;
   }

   uint32_t kernel = 0;
   if (!strcmp(entry_name, "ngv_ph_scale_bias"))
      kernel = base_push_offset >= 0 ? NGV_PH_KERNEL_SCALE_BIAS_BASE : NGV_PH_KERNEL_SCALE_BIAS;
   if (!kernel)
      return ngv_shader_error(dev, VK_ERROR_UNKNOWN, "shader '%s': no placeholder kernel of that name", entry_name);
   struct ngv_ph_header hdr = {.kernel = kernel};
   memcpy(hdr.magic, NGV_PH_MAGIC, 8);
   out->code = malloc(sizeof(hdr));
   if (!out->code)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   memcpy(out->code, &hdr, sizeof(hdr));
   out->code_size = sizeof(hdr);
   out->vregs = 8;
   out->placeholder = true;
   return VK_SUCCESS;
}

/* ---- pipelines ---------------------------------------------------------------------------------- */

static VkResult
create_compute_pipeline(struct ngv_device *device, struct vk_pipeline_cache *pc, const VkComputePipelineCreateInfo *ci,
                        const VkAllocationCallbacks *pAllocator, VkPipeline *out)
{
   const VkPipelineShaderStageCreateInfo *st = &ci->stage;
   if (st->stage != VK_SHADER_STAGE_COMPUTE_BIT)
      return vk_error(device, VK_ERROR_UNKNOWN);

   const uint32_t *words = NULL;
   uint32_t nwords = 0;
   if (st->module != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(vk_shader_module, mod, st->module);
      words = (const uint32_t *)mod->data;
      nwords = mod->size / 4;
   } else {
      const VkShaderModuleCreateInfo *smci = vk_find_struct_const(st->pNext, SHADER_MODULE_CREATE_INFO);
      if (!smci)
         return vk_error(device, VK_ERROR_UNKNOWN);
      words = smci->pCode;
      nwords = (uint32_t)(smci->codeSize / 4);
   }

   struct ngv_pipeline *p = vk_object_zalloc(&device->vk, pAllocator, sizeof(*p), VK_OBJECT_TYPE_PIPELINE);
   if (!p)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_pipeline_layout *layout = ci->layout ? vk_pipeline_layout_from_handle(ci->layout) : NULL;
   if (!layout) {
      vk_object_free(&device->vk, pAllocator, p);
      return vk_error(device, VK_ERROR_UNKNOWN);
   }
   /* VK_PIPELINE_CREATE_DISPATCH_BASE: vkCmdDispatchBase with a non-zero base is only legal on such pipelines.  The
    * base workgroup id has no hardware register (ABI u26..u28 count from 0, u30/u31 cannot be set), so the compiled
    * code reads it from three push-constant dwords placed right behind the layout's push range; they have to be
 * inside u10..u25 (the first 16 dwords). PROVISIONAL (.md 14.3). */
   VkPipelineCreateFlags2KHR flags2 = ci->flags;
   const VkPipelineCreateFlags2CreateInfoKHR *f2 =
      vk_find_struct_const(ci->pNext, PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR);
   if (f2)
      flags2 = f2->flags;
   p->base_push_offset = -1;
   uint32_t end = 0;
   for (uint32_t i = 0; i < layout->push_range_count; i++)
      end = MAX2(end, layout->push_ranges[i].offset + layout->push_ranges[i].size);
   end = align(end, 4);
   if (flags2 & VK_PIPELINE_CREATE_DISPATCH_BASE_BIT) {
      if (end + 12 > 64) {
         vk_object_free(&device->vk, pAllocator, p);
         return ngv_shader_error(device, VK_ERROR_FEATURE_NOT_PRESENT,
                          "DISPATCH_BASE pipeline: a push range of %u B leaves no room for the base workgroup id in "
                          "the first 16 push dwords",
                          end);
      }
      p->base_push_offset = (int32_t)end;
      end += 12;
   }
 /* gl_NumWorkGroups (PROVISIONAL): a 64-bit VA of the grid in the two push dwords behind that; only taken if
    * the shader reads it (ngv_nir.c), -1 to the compiler = no room (a shader that reads it then fails to compile) */
   const int32_t numwg_candidate = end + 8 <= 64 ? (int32_t)end : -1;

   struct ngv_shader_object so;
   VkResult r = ngv_shader_compile(device, pc, st, flags2, words, nwords, layout, p->base_push_offset, numwg_candidate,
                                   &so);
   if (r != VK_SUCCESS) {
      vk_object_free(&device->vk, pAllocator, p);
      return r;
   }
   p->numwg_push_offset = so.uses_num_workgroups ? numwg_candidate : -1;
   r = ngv_bo_create(device, align(so.code_size, NICOGPU_PAGE_SIZE), NGV_BO_CODE, &p->code);
   if (r != VK_SUCCESS) {
      free(so.code);
      vk_object_free(&device->vk, pAllocator, p);
      return vk_error(device, r);
   }
   memcpy(p->code.map, so.code, so.code_size);
   free(so.code);
   __sync_synchronize(); /* WC window: drain before the GPU can fetch it */

   p->entry = so.entry;
   p->vregs = so.vregs;
   p->shared_bytes = so.shared_bytes;
   p->scratch_bytes = so.scratch_bytes;
   memcpy(p->wg, so.wg, sizeof(p->wg));
   p->layout = vk_pipeline_layout_ref(layout);
   *out = ngv_pipeline_to_handle(p);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateComputePipelines(VkDevice _device, VkPipelineCache cache, uint32_t count,
                           const VkComputePipelineCreateInfo *pInfos, const VkAllocationCallbacks *pAllocator,
                           VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(vk_pipeline_cache, pc, cache); /* NULL for VK_NULL_HANDLE */
   VkResult result = VK_SUCCESS;
   uint32_t i;
   for (i = 0; i < count; i++) {
      VkResult r = create_compute_pipeline(device, pc, &pInfos[i], pAllocator, &pPipelines[i]);
      if (r != VK_SUCCESS) {
         result = r;
         pPipelines[i] = VK_NULL_HANDLE;
         if (pInfos[i].flags & VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT) {
            for (i++; i < count; i++)
               pPipelines[i] = VK_NULL_HANDLE;
            break;
         }
      }
   }
   return result;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyPipeline(VkDevice _device, VkPipeline _pipeline, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_pipeline, p, _pipeline);
   if (!p)
      return;
   ngv_bo_destroy(device, &p->code);
   if (p->layout)
      vk_pipeline_layout_unref(&device->vk, p->layout);
   vk_object_free(&device->vk, pAllocator, p);
}
