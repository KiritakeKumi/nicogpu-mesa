/* SPDX-License-Identifier: MIT */
/* ngv_ngcomp.c -- run ngcomp as a subprocess (batch 3, section 8 / 14.2).
 *
 *   python3 $NGV_NGCOMP_DIR/driver.py isel in.nir out.xo --regalloc --exact-waitcnt --units sfu
 *
 * The compiler is the Python tool chain (the ngcomp tool chain); the driver is C.  Interface: files in $NGV_TMPDIR (else
 * $TMPDIR, else /tmp) -- the NIR-subset text in, the `.xo` container out, stdout+stderr of the compiler into a log
 * that becomes the error message on failure. $NGV_NGCOMP_DIR must point at ngcomp (the directory with
 * driver.py; the ISA description is found relative to it:../../nicogpu_isa.yaml); $NGV_PYTHON overrides
 * "python3".  $NGV_KEEP_TMP=1 keeps the files (they are the quickest way to reproduce a compiler problem by hand).
 *
 * Shader cache : the finished `.xo` is stored in Mesa's disk cache (util/disk_cache.h, so
 * MESA_SHADER_CACHE_DISABLE, MESA_SHADER_CACHE_DIR, MESA_SHADER_CACHE_MAX_SIZE apply) under SHA-1(NIR text,
 * occupancy target, compiler options, ngcomp stamp). The stamp is a hash of every.py file of ngcomp and of the ISA
 * table, so editing the compiler invalidates the entries by itself.  $NGV_SHADER_CACHE=0 turns the cache off for this
 * driver only; $NGV_CACHE_VERBOSE=1 reports hit / miss on stderr.  Above that sits a SPIR-V level (ngv_ngcomp_spirv_key /
 * _lookup / _store, below): a pipeline create first asks for a finished object under a key of the SPIR-V module, entry
 * point, specialization, pipeline layout, create flags and compiler identity, in the VkPipelineCache and the disk cache,
 * and only on a miss translates SPIR-V -> NIR -> text and comes here.
 *
 * Target units : every compile gets `--units <set>`, the optional execution units the SP build has (all|none|
 * list of sfu,xlane,fp64; ngcomp/units.py). ngcomp never emits an instruction of a missing unit (uniform spills
 * take an xlane-free path, anything else fails the compile).  The set comes from the device: the RM publishes the units
 * it was built with in CAPS[26:24] (CAPS[27] UNITS_VALID), the kernel passes them as nicogpu_gpu_info.sp_units, the HAL as
 * nicogpu_device_info.sp_units (info_flags NICOGPU_INFO_F_SP_UNITS).  $NGV_NGCOMP_UNITS (e.g. "sfu,xlane", "all") overrides
 * it.  An RM that does not report them (b4 .. b11, an old kernel) gets the legacy default "sfu" = what every shipped
 * rm_sp has: a clear NICOGPU_INFO_F_SP_UNITS means "unknown", not "none" (ngv_units.h). The set is part
 * of the shader-cache key.
 *
 * fp32 contraction : $NGV_NGCOMP_FMA = full | basic | off is passed as `--fma-contract <mode>` and hashed into
 * the cache key; unset / empty = ngcomp's default (full: a*b+c -> ffma also for multi-use products, through fneg and
 * across blocks; NoContraction, printed as {exact=1}, is honoured by every mode).  "off" gives the unfused fmul + fadd
 * (bit-exact double rounding) for A/B comparisons against a CPU reference; an unknown value fails the compile.
 *
 * PROVISIONAL: process start-up per shader (tens of ms plus the YAML load), Python and PyYAML must exist on the
 * machine running the driver. Long-term replacement:.md 8.
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "util/disk_cache.h"
#include "util/mesa-sha1.h"
#include "vk_pipeline_cache.h"

#include "ngv_private.h"
#include "ngv_units.h"

_Static_assert(NGV_SPU_SFU == NICOGPU_SP_UNIT_SFU && NGV_SPU_XLANE == NICOGPU_SP_UNIT_XLANE && NGV_SPU_FP64 == NICOGPU_SP_UNIT_FP64 &&
                  NGV_SPU_F16CVT == NICOGPU_SP_UNIT_F16CVT,
               "ngv_units.h: unit bits != HAL NICOGPU_SP_UNIT_*");

extern char **environ;

#define NGV_CACHE_ID "ngv-xo-1" /* bump when the cache entry format (the .xo container) changes */
#define NGV_NGCOMP_OPTIONS "isel --regalloc --exact-waitcnt"

/* the unit set handed to ngcomp (--units) and hashed into the cache key: $NGV_NGCOMP_UNITS, else what the device reports,
 * else the legacy default (ngv_units.h).  `buf` backs the returned string.  $NGV_CACHE_VERBOSE=1 prints it once. */
static const char *
ngv_target_units(const struct ngv_physical_device *pdev, char buf[NGV_UNITS_BUFSZ])
{
   const char *src;
   const char *u = ngv_units_resolve(getenv("NGV_NGCOMP_UNITS"), (pdev->info.info_flags & NICOGPU_INFO_F_SP_UNITS) != 0,
                                     pdev->info.sp_units, buf, &src);
   static bool logged;
   const char *v = getenv("NGV_CACHE_VERBOSE");
   if (!logged && v && strcmp(v, "0")) {
      logged = true;
      fprintf(stderr, "ngv: ngcomp target units \"%s\" (%s)\n", u, src);
   }
   return u;
}

/* : the target executes the binary16 conversions (unit "f16cvt" in the resolved set): ngv_shader_to_nir then keeps
 * pack_half_2x16_split / unpack_half_2x16_split_x/y for ngcomp (pkcvt.f16.f32 / fcvt.f32.f16.lo/hi) instead of the
 * integer expansion.  The same resolved string goes to --units and the cache key, and the NIR text differs, so a
 * cached .xo never crosses the switch. */
bool
ngv_target_has_f16cvt(const struct ngv_physical_device *pdev)
{
   char buf[NGV_UNITS_BUFSZ];
   return ngv_units_has(ngv_target_units(pdev, buf), "f16cvt");
}

/* : the target has XLANE (xred / xscan / shfl): ngv_nir_lower_subgroups keeps the whole-warp integer reduce and iadd
 * scans as intrinsics for ngcomp's native forms; without it they are lowered to the bit-serial-readable shuffle code. */
bool
ngv_target_has_xlane(const struct ngv_physical_device *pdev)
{
   char buf[NGV_UNITS_BUFSZ];
   return ngv_units_has(ngv_target_units(pdev, buf), "xlane");
}

void
ngv_shader_cache_init(struct ngv_physical_device *pdev)
{
   const char *off = getenv("NGV_SHADER_CACHE");
   pdev->xo_cache = (off && !strcmp(off, "0")) ? NULL : disk_cache_create("nicogpu", NGV_CACHE_ID, 0);
}

void
ngv_shader_cache_finish(struct ngv_physical_device *pdev)
{
   if (pdev->xo_cache)
      disk_cache_destroy(pdev->xo_cache);
   pdev->xo_cache = NULL;
}

static int
name_cmp(const void *a, const void *b)
{
   return strcmp(*(char *const *)a, *(char *const *)b);
}

static bool
sha1_file(struct mesa_sha1 *ctx, const char *path, const char *label)
{
   size_t sz;
   uint8_t *buf = ngv_read_file(path, &sz);
   if (!buf)
      return false;
   _mesa_sha1_update(ctx, label, strlen(label) + 1);
   _mesa_sha1_update(ctx, buf, sz);
   free(buf);
   return true;
}

/* Hash of the compiler: every.py of ngcomp (sorted by name) + the ISA table. Computed once per process. */
static bool
ngcomp_stamp(const char *xdir, uint8_t stamp[20])
{
   static simple_mtx_t lock = SIMPLE_MTX_INITIALIZER;
   static uint8_t cached[20];
   static bool have, tried;
   simple_mtx_lock(&lock);
   if (!tried) {
      tried = true;
      struct mesa_sha1 ctx;
      _mesa_sha1_init(&ctx);
      char *names[256];
      unsigned n = 0;
      bool ok = false;
      DIR *d = opendir(xdir);
      if (d) {
         struct dirent *e;
         while ((e = readdir(d)) && n < 256) {
            size_t l = strlen(e->d_name);
            if (l > 3 && !strcmp(e->d_name + l - 3, ".py") && (names[n] = strdup(e->d_name)))
               n++;
         }
         closedir(d);
         qsort(names, n, sizeof(names[0]), name_cmp);
         ok = n > 0;
         for (unsigned i = 0; i < n; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", xdir, names[i]);
            ok = ok && sha1_file(&ctx, path, names[i]);
            free(names[i]);
         }
      }
      char yaml[1024];
 snprintf(yaml, sizeof(yaml), "%s/../../nicogpu_isa.yaml", xdir);
      ok = ok && sha1_file(&ctx, yaml, "nicogpu_isa.yaml");
      if (ok) {
         _mesa_sha1_final(&ctx, cached);
         have = true;
      }
   }
   if (have)
      memcpy(stamp, cached, 20);
   simple_mtx_unlock(&lock);
   return have;
}

/* $NGV_NGCOMP_FMA : "" = ngcomp's default contraction mode */
static const char *
ngv_fma_mode(void)
{
   const char *m = getenv("NGV_NGCOMP_FMA");
   return (m && *m) ? m : "";
}

/* Everything outside the driver binary that decides what code a SPIR-V module compiles to: the cache entry format and
 * ngcomp command line, the compiler stamp (every .py of $NGV_NGCOMP_DIR + the ISA table; a marker when the directory is
 * not set -- compiles fail then anyway), the unit set and fp32 contraction mode that go to ngcomp, the Vulkan memory model
 * switch (selects SPIR-V capabilities and the memory-model pass) and the 16-bit conversion choice (a function of the
 * unit set).  The single source of the compiler-side inputs of pipelineCacheUUID (ngv_ngcomp_identity), of the cache key of
 * a compiled NIR text (ngcomp_cache_key) and of the cache key of a SPIR-V compile (ngv_ngcomp_spirv_key).  Returns false
 * when the ngcomp stamp is not available (no key built on it may be cached). */
static bool
hash_compiler_identity(const struct ngv_physical_device *pdev, struct mesa_sha1 *ctx)
{
   char buf[NGV_UNITS_BUFSZ];
   const char *units = ngv_target_units(pdev, buf);
   const char *xdir = getenv("NGV_NGCOMP_DIR");
   uint8_t stamp[20] = {0};
   const char *state = "ngcomp-stamp";
   bool ok = true;
   if (!xdir || !*xdir) {
      state = "ngcomp-unset";
      ok = false;
   } else if (!ngcomp_stamp(xdir, stamp)) {
      state = "ngcomp-unreadable";
      ok = false;
   }
   _mesa_sha1_update(ctx, NGV_CACHE_ID, sizeof(NGV_CACHE_ID));
   _mesa_sha1_update(ctx, NGV_NGCOMP_OPTIONS, sizeof(NGV_NGCOMP_OPTIONS));
   _mesa_sha1_update(ctx, state, strlen(state) + 1);
   _mesa_sha1_update(ctx, stamp, sizeof(stamp));
   _mesa_sha1_update(ctx, units, strlen(units) + 1);
   const char *fma = ngv_fma_mode();
   _mesa_sha1_update(ctx, fma, strlen(fma) + 1);
   const uint8_t mm = ngv_vulkan_memory_model_enabled();
   _mesa_sha1_update(ctx, &mm, sizeof(mm));
   return ok;
}

/* hashed into pipelineCacheUUID (ngv_uuid.c, section 16) */
void
ngv_ngcomp_identity(const struct ngv_physical_device *pdev, struct mesa_sha1 *ctx)
{
   (void)hash_compiler_identity(pdev, ctx);
}

/* key of a finished .xo: the compiler identity, the occupancy target (the clamped value the command line gets) and the
 * NIR text */
static bool
ngcomp_cache_key(const struct ngv_physical_device *pdev, const char *nir_text, uint32_t occupancy, cache_key key)
{
   struct mesa_sha1 ctx;
   _mesa_sha1_init(&ctx);
   const bool ok = hash_compiler_identity(pdev, &ctx);
   _mesa_sha1_update(&ctx, &occupancy, sizeof(occupancy));
   _mesa_sha1_update(&ctx, nir_text, strlen(nir_text));
   _mesa_sha1_final(&ctx, key);
   return ok;
}

/* ---- SPIR-V level cache --------------------------------------------------------------------------------------------
 * ngv_shader_compile asks for a finished code object before it translates anything: the key is made of the inputs of the
 * whole SPIR-V -> NIR -> ngcomp chain, and the entry holds what that chain produced (the .xo plus the three values the NIR
 * lowering decides: local size, shared bytes, whether gl_NumWorkGroups is read).  Entries live in the application's
 * VkPipelineCache (as raw-data objects, so vkGetPipelineCacheData / vkMergePipelineCaches / pInitialData carry them) and
 * in the driver's disk cache.  The NIR-text level (ngv_ngcomp_compile) stays underneath: different SPIR-V can lower to
 * the same text. */

#define NGV_SPV_CACHE_ID "ngv-spv-1" /* bump when the key inputs or the entry layout change */
#define NGV_SPV_ENTRY_MAGIC 0x3153474eu /* "NGS1" */

struct ngv_spv_entry_hdr {
   uint32_t magic;
   uint32_t wg[3];
   uint32_t shared_bytes;
   uint32_t uses_num_workgroups;
   uint32_t occupancy; /* the target the code was register-allocated for (ngv_warps_per_sp of wg on that device) */
   uint32_t xo_size;   /* bytes of the .xo container behind the header */
};

static void
h_u32(struct mesa_sha1 *ctx, uint32_t v)
{
   _mesa_sha1_update(ctx, &v, sizeof(v));
}

static void
h_str(struct mesa_sha1 *ctx, const char *s)
{
   _mesa_sha1_update(ctx, s, strlen(s) + 1); /* with the NUL: ("ab","c") != ("a","bc") */
}

/* everything ngv_nir_lower_pipeline_layout / ngv_nir_lower_tex_resources read of the layout (set count, every binding of
 * every set layout: slot offset, stride, type, ...), and the push ranges */
static void
hash_pipeline_layout(struct mesa_sha1 *ctx, const struct vk_pipeline_layout *l)
{
   h_u32(ctx, l->set_count);
   for (uint32_t i = 0; i < l->set_count; i++) {
      if (!l->set_layouts[i]) {
         h_u32(ctx, 0);
         continue;
      }
      h_u32(ctx, 1);
      const struct ngv_descriptor_set_layout *sl = container_of(l->set_layouts[i], struct ngv_descriptor_set_layout, vk);
      h_u32(ctx, sl->binding_count);
      h_u32(ctx, sl->size);
      h_u32(ctx, sl->dynamic_count);
      h_u32(ctx, sl->imm_count);
      for (uint32_t b = 0; b < sl->binding_count; b++) {
         const struct ngv_binding_layout *bl = &sl->bindings[b];
         h_u32(ctx, bl->binding);
         h_u32(ctx, (uint32_t)bl->type);
         h_u32(ctx, bl->count);
         h_u32(ctx, bl->offset);
         h_u32(ctx, bl->stride);
         h_u32(ctx, (uint32_t)bl->dyn_start);
         h_u32(ctx, (uint32_t)bl->imm_start);
      }
   }
   h_u32(ctx, l->push_range_count);
   for (uint32_t i = 0; i < l->push_range_count; i++) {
      h_u32(ctx, l->push_ranges[i].stageFlags);
      h_u32(ctx, l->push_ranges[i].offset);
      h_u32(ctx, l->push_ranges[i].size);
   }
}

/* The SPIR-V level key (20 bytes).  Inputs, each of which can change the code:
 *   - the SPIR-V module (all words), the entry point name, the stage flags / stage / required subgroup size, the
 *     specialization map and data, the pipeline create flags (64 bit; DISPATCH_BASE, VIEW_INDEX_FROM_DEVICE_INDEX, ...);
 *   - the driver build (driverUUID = build id of libvulkan_nicogpu.so: the SPIR-V -> NIR lowering and the NIR printer live in
 *     it) and the compile-relevant device description (ISA name / version, warp size, vreg / ureg / shared / workgroup
 *     limits, SPs per core: the occupancy target derives from it) -- the same fields as pipelineCacheUUID plus the ones that
 *     only matter at compile time;
 *   - the compiler identity (hash_compiler_identity, read live: stamp, units, fma mode, memory model), $NGV_SUBGROUP_ARITH
 *     (it selects the SPIR-V capabilities the runtime accepts);
 *   - the pipeline layout (hash_pipeline_layout) and the driver push dwords (base workgroup id / num workgroups offsets).
 * false: not cacheable (no ngcomp stamp). */
bool
ngv_ngcomp_spirv_key(const struct ngv_device *dev, VkPipelineCreateFlags2KHR pipeline_flags,
                     const VkPipelineShaderStageCreateInfo *stage, const uint32_t *spirv, uint32_t spirv_words,
                     const struct vk_pipeline_layout *layout, int32_t base_push_offset, int32_t numwg_push_offset,
                     uint8_t key[20])
{
   const struct ngv_physical_device *pdev = dev->pdev;
   const struct nicogpu_device_info *i = &pdev->info;
   struct mesa_sha1 ctx;
   _mesa_sha1_init(&ctx);
   h_str(&ctx, NGV_SPV_CACHE_ID);
   _mesa_sha1_update(&ctx, pdev->vk.properties.driverUUID, VK_UUID_SIZE);
   h_str(&ctx, i->isa_name);
   h_str(&ctx, i->isa_version);
   h_u32(&ctx, i->warp_size);
   h_u32(&ctx, i->max_vregs);
   h_u32(&ctx, i->num_uregs);
   h_u32(&ctx, i->max_shared_bytes);
   h_u32(&ctx, i->max_workgroup_threads);
   h_u32(&ctx, i->sp_per_core);
   const bool ok = hash_compiler_identity(pdev, &ctx);
   const char *arith = getenv("NGV_SUBGROUP_ARITH");
   h_str(&ctx, arith ? arith : "");

   h_u32(&ctx, (uint32_t)pipeline_flags);
   h_u32(&ctx, (uint32_t)(pipeline_flags >> 32));
   h_u32(&ctx, stage->flags);
   h_u32(&ctx, stage->stage);
   const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo *rss =
      vk_find_struct_const(stage->pNext, PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO);
   h_u32(&ctx, rss ? rss->requiredSubgroupSize : 0);
   h_str(&ctx, stage->pName);
   const VkSpecializationInfo *spec = stage->pSpecializationInfo;
   h_u32(&ctx, spec ? 1 : 0);
   if (spec) {
      h_u32(&ctx, spec->mapEntryCount);
      for (uint32_t k = 0; k < spec->mapEntryCount; k++) {
         h_u32(&ctx, spec->pMapEntries[k].constantID);
         h_u32(&ctx, spec->pMapEntries[k].offset);
         h_u32(&ctx, (uint32_t)spec->pMapEntries[k].size);
      }
      h_u32(&ctx, (uint32_t)spec->dataSize);
      if (spec->dataSize)
         _mesa_sha1_update(&ctx, spec->pData, spec->dataSize);
   }
   h_u32(&ctx, (uint32_t)base_push_offset);
   h_u32(&ctx, (uint32_t)numwg_push_offset);
   hash_pipeline_layout(&ctx, layout);
   h_u32(&ctx, spirv_words);
   _mesa_sha1_update(&ctx, spirv, (size_t)spirv_words * 4);
   _mesa_sha1_final(&ctx, key);
   return ok;
}

/* Decodes an entry into `out` (code object + the lowering results) and checks it against this device; false = unusable
 * (corrupt, other layout, limits / occupancy differ): the caller compiles. */
static VkResult
spv_entry_decode(const struct ngv_device *dev, const void *data, size_t size, struct ngv_shader_object *out)
{
   struct ngv_spv_entry_hdr h;
   if (size < sizeof(h))
      return VK_ERROR_UNKNOWN;
   memcpy(&h, data, sizeof(h));
   if (h.magic != NGV_SPV_ENTRY_MAGIC || h.xo_size != size - sizeof(h) || !h.wg[0] || !h.wg[1] || !h.wg[2])
      return VK_ERROR_UNKNOWN;
   const struct nicogpu_device_info *i = &dev->pdev->info;
   /* the same limits ngv_shader_compile checks after the lowering, and the occupancy the entry was built for */
   if ((uint64_t)h.wg[0] * h.wg[1] * h.wg[2] > i->max_workgroup_threads || h.shared_bytes > i->max_shared_bytes ||
       h.occupancy != ngv_warps_per_sp(dev, h.wg))
      return VK_ERROR_UNKNOWN;
   VkResult r = ngv_parse_xo((const uint8_t *)data + sizeof(h), h.xo_size, out);
   if (r != VK_SUCCESS)
      return r;
   memcpy(out->wg, h.wg, sizeof(out->wg));
   out->shared_bytes = h.shared_bytes;
   out->uses_num_workgroups = h.uses_num_workgroups != 0;
   return VK_SUCCESS;
}

/* VK_SUCCESS: `out` is the finished object.  VK_INCOMPLETE: no usable entry.  VK_ERROR_OUT_OF_HOST_MEMORY. */
VkResult
ngv_ngcomp_spirv_lookup(struct ngv_device *dev, struct vk_pipeline_cache *pc, const uint8_t key[20],
                        struct ngv_shader_object *out)
{
   const bool verbose = getenv("NGV_CACHE_VERBOSE") && strcmp(getenv("NGV_CACHE_VERBOSE"), "0");
   if (pc) {
      struct vk_pipeline_cache_object *obj =
         vk_pipeline_cache_lookup_object(pc, key, 20, &vk_raw_data_cache_object_ops, NULL);
      if (obj) {
         const struct vk_raw_data_cache_object *raw = container_of(obj, struct vk_raw_data_cache_object, base);
         VkResult r = spv_entry_decode(dev, raw->data, raw->data_size, out);
         vk_pipeline_cache_object_unref(&dev->vk, obj);
         if (r == VK_SUCCESS || r == VK_ERROR_OUT_OF_HOST_MEMORY) {
            if (verbose && r == VK_SUCCESS)
               fprintf(stderr, "ngv: pipeline cache hit (SPIR-V key)\n");
            return r;
         }
      }
   }
   struct disk_cache *dc = dev->pdev->xo_cache;
   if (dc) {
      size_t size = 0;
      void *blob = disk_cache_get(dc, (const uint8_t *)key, &size);
      if (blob) {
         VkResult r = spv_entry_decode(dev, blob, size, out);
         if (r == VK_SUCCESS && pc) {
            struct vk_pipeline_cache_object *obj =
               vk_pipeline_cache_create_and_insert_object(pc, key, 20, blob, size, &vk_raw_data_cache_object_ops);
            if (obj)
               vk_pipeline_cache_object_unref(&dev->vk, obj);
         }
         free(blob);
         if (r == VK_SUCCESS || r == VK_ERROR_OUT_OF_HOST_MEMORY) {
            if (verbose && r == VK_SUCCESS)
               fprintf(stderr, "ngv: shader cache hit (SPIR-V key)\n");
            return r;
         }
         disk_cache_remove(dc, (const uint8_t *)key); /* unusable entry: drop it */
      }
   }
   return VK_INCOMPLETE;
}

/* Stores the result of a complete compile: `so` carries the lowering results, (xo, xo_size) is the .xo it came from
 * (ngv_ngcomp_compile), `occupancy` the target it was built for. */
void
ngv_ngcomp_spirv_store(struct ngv_device *dev, struct vk_pipeline_cache *pc, const uint8_t key[20],
                       const struct ngv_shader_object *so, uint32_t occupancy, const void *xo, size_t xo_size)
{
   struct disk_cache *dc = dev->pdev->xo_cache;
   if ((!dc && !pc) || !xo || xo_size > UINT32_MAX - sizeof(struct ngv_spv_entry_hdr))
      return;
   const struct ngv_spv_entry_hdr h = {
      .magic = NGV_SPV_ENTRY_MAGIC,
      .wg = {so->wg[0], so->wg[1], so->wg[2]},
      .shared_bytes = so->shared_bytes,
      .uses_num_workgroups = so->uses_num_workgroups,
      .occupancy = occupancy,
      .xo_size = (uint32_t)xo_size,
   };
   uint8_t *blob = malloc(sizeof(h) + xo_size);
   if (!blob)
      return;
   memcpy(blob, &h, sizeof(h));
   memcpy(blob + sizeof(h), xo, xo_size);
   if (dc) {
      disk_cache_put(dc, (const uint8_t *)key, blob, sizeof(h) + xo_size, NULL);
      disk_cache_wait_for_idle(dc); /* the process may exit without destroying the device */
   }
   if (pc) {
      struct vk_pipeline_cache_object *obj =
         vk_pipeline_cache_create_and_insert_object(pc, key, 20, blob, sizeof(h) + xo_size, &vk_raw_data_cache_object_ops);
      if (obj)
         vk_pipeline_cache_object_unref(&dev->vk, obj);
   }
   free(blob);
}

static int
tmpfile_make(const char *dir, const char *suffix, char *path, size_t n)
{
   snprintf(path, n, "%s/xgvXXXXXX%s", dir, suffix);
   return mkstemps(path, (int)strlen(suffix));
}

/* xo / xo_size (both or neither, either may be NULL): receive the finished .xo container (malloc'd, the caller frees it;
 * left NULL / 0 on failure), so that the caller can store it under its own cache key (ngv_ngcomp_spirv_store). */
VkResult
ngv_ngcomp_compile(struct ngv_device *dev, const char *nir_text, uint32_t target_occupancy,
                   struct ngv_shader_object *out, uint8_t **xo, size_t *xo_size)
{
   /* only what the .xo holds is filled in below (code, entry, vregs, scratch, ABI); the caller owns the rest (local size,
    * shared bytes, uses_num_workgroups), so it must not start as stack garbage */
   memset(out, 0, sizeof(*out));
   if (xo)
      *xo = NULL;
   if (xo_size)
      *xo_size = 0;
   const char *xdir = getenv("NGV_NGCOMP_DIR");
   if (!xdir || !*xdir)
      return ngv_shader_error(dev, VK_ERROR_INITIALIZATION_FAILED,
 "$NGV_NGCOMP_DIR is not set (directory of ngcomp with driver.py)");
   const char *py = getenv("NGV_PYTHON");
   if (!py || !*py)
      py = "python3";
   const char *tdir = getenv("NGV_TMPDIR");
   if (!tdir || !*tdir)
      tdir = getenv("TMPDIR");
   if (!tdir || !*tdir)
      tdir = "/tmp";
   const bool keep = getenv("NGV_KEEP_TMP") && strcmp(getenv("NGV_KEEP_TMP"), "0");

   /* cache lookup; the occupancy is hashed as the clamped value the command line gets */
   struct disk_cache *dc = dev->pdev->xo_cache;
   const bool verbose = getenv("NGV_CACHE_VERBOSE") && strcmp(getenv("NGV_CACHE_VERBOSE"), "0");
   char units_buf[NGV_UNITS_BUFSZ];
   const char *units = ngv_target_units(dev->pdev, units_buf);
   const char *fma = ngv_fma_mode();
   cache_key ckey;
   const bool use_cache = dc && ngcomp_cache_key(dev->pdev, nir_text, target_occupancy > 16 ? 16u : target_occupancy, ckey);
   if (use_cache) {
      size_t csz = 0;
      void *blob = disk_cache_get(dc, ckey, &csz);
      if (blob) {
         VkResult cr = ngv_parse_xo(blob, csz, out);
         if (cr == VK_SUCCESS && xo && xo_size) {
            *xo = blob; /* handed over */
            *xo_size = csz;
         } else {
            free(blob);
         }
         if (cr == VK_SUCCESS || cr == VK_ERROR_OUT_OF_HOST_MEMORY) {
            if (verbose && cr == VK_SUCCESS)
               fprintf(stderr, "ngv: shader cache hit\n");
            return cr;
         }
         disk_cache_remove(dc, ckey); /* unreadable entry: drop it and compile */
      }
      if (verbose)
         fprintf(stderr, "ngv: shader cache miss\n");
   }

   char nir_path[512], xo_path[520], log_path[520], script[512];
   int fd = tmpfile_make(tdir, ".nir", nir_path, sizeof(nir_path));
   if (fd < 0)
      return ngv_shader_error(dev, VK_ERROR_UNKNOWN, "cannot create a temporary file in %s: %s", tdir, strerror(errno));
   size_t len = strlen(nir_text), off = 0;
   while (off < len) {
      ssize_t w = write(fd, nir_text + off, len - off);
      if (w < 0 && errno == EINTR)
         continue;
      if (w <= 0)
         break;
      off += (size_t)w;
   }
   close(fd);
   snprintf(xo_path, sizeof(xo_path), "%s.xo", nir_path);
   snprintf(log_path, sizeof(log_path), "%s.log", nir_path);
   snprintf(script, sizeof(script), "%s/driver.py", xdir);

   VkResult result = VK_ERROR_UNKNOWN;
   if (off != len) {
      result = ngv_shader_error(dev, VK_ERROR_UNKNOWN, "short write to %s", nir_path);
      goto out;
   }

   {
 /* --occupancy N: warps/SP the register budget must allow (ngcomp/regalloc.py target_occupancy); only
       * passed when it constrains (one warp per SP may use all 128 registers) */
      char occ[16];
      snprintf(occ, sizeof(occ), "%u", target_occupancy > 16 ? 16u : target_occupancy);
      char *argv[14] = {(char *)py, script, "isel", nir_path, xo_path, "--regalloc", "--exact-waitcnt",
                        (char *)"--units", (char *)units};
      unsigned argc = 9;
      if (target_occupancy > 1) {
         argv[argc++] = (char *)"--occupancy";
         argv[argc++] = occ;
      }
      if (*fma) {
         argv[argc++] = (char *)"--fma-contract";
         argv[argc++] = (char *)fma;
      }
      argv[argc] = NULL;
      posix_spawn_file_actions_t fa;
      posix_spawn_file_actions_init(&fa);
      posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
      posix_spawn_file_actions_addopen(&fa, 1, log_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
      posix_spawn_file_actions_adddup2(&fa, 1, 2);
      pid_t pid;
      int rc = posix_spawnp(&pid, py, &fa, NULL, argv, environ);
      posix_spawn_file_actions_destroy(&fa);
      if (rc) {
         result = ngv_shader_error(dev, VK_ERROR_INITIALIZATION_FAILED, "cannot run '%s': %s", py, strerror(rc));
         goto out;
      }
      int status = 0;
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR)
         ;
      if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
         char msg[1024] = "";
         FILE *lf = fopen(log_path, "r");
         if (lf) {
            size_t n = fread(msg, 1, sizeof(msg) - 1, lf);
            msg[n] = 0;
            fclose(lf);
         }
         result = ngv_shader_error(dev, VK_ERROR_UNKNOWN, "ngcomp failed (%s %d): %s",
                            WIFEXITED(status) ? "exit" : "signal",
                            WIFEXITED(status) ? WEXITSTATUS(status) : WTERMSIG(status), msg);
         goto out;
      }
   }
   {
      size_t xsz = 0;
      uint8_t *xbuf = ngv_read_file(xo_path, &xsz);
      result = xbuf ? ngv_parse_xo(xbuf, xsz, out) : VK_ERROR_UNKNOWN;
      if (result == VK_SUCCESS && use_cache) {
         disk_cache_put(dc, ckey, xbuf, xsz, NULL);
         disk_cache_wait_for_idle(dc); /* the process may exit without destroying the device */
      }
      if (result == VK_SUCCESS && xo && xo_size) {
         *xo = xbuf; /* handed over */
         *xo_size = xsz;
      } else {
         free(xbuf);
      }
   }
   if (result != VK_SUCCESS && result != VK_ERROR_OUT_OF_HOST_MEMORY)
      result = ngv_shader_error(dev, VK_ERROR_UNKNOWN, "ngcomp produced an unreadable object %s", xo_path);

out:
   if (!keep) {
      unlink(nir_path);
      unlink(xo_path);
      unlink(log_path);
   } else {
      fprintf(stderr, "ngv: kept %s (+ .xo, .log)\n", nir_path);
   }
   return result;
}
