/* SPDX-License-Identifier: MIT */
/* ngv_private.h -- Nicogpu Vulkan driver "ngv" (skeleton). */
#ifndef NGV_PRIVATE_H
#define NGV_PRIVATE_H

#include "util/list.h"
#include "util/simple_mtx.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_buffer_view.h"
#include "vk_image.h"
#include "vk_sampler.h"
#include "vk_command_buffer.h"
#include "vk_common_entrypoints.h"
#include "vk_descriptor_set_layout.h"
#include "vk_pipeline_layout.h"
#include "vk_query_pool.h"
#include "vk_shader_module.h"
#include "vk_command_pool.h"
#include "vk_device.h"
#include "vk_device_memory.h"
#include "vk_instance.h"
#include "vk_log.h"
#include "vk_object.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_sync.h"
#include "vk_sync_timeline.h"
#include "vk_drm_syncobj.h"
#include "vk_util.h"

#include "ngv_entrypoints.h"
#include "ngv_winsys.h"
#include "nicogpu_cmd.h" /* generated (cmdgen): CP packet constants */

#include "ngv_formats_gen.h" /* generated, checked in: VkFormat -> image descriptor format */
#include "ngv_layout.h" /* image memory layout (the descriptor spec section 6) */

/* The advertised version is what the driver really implements (user ruling Q5 2026-10-24): Vulkan 1.1, with
 * the 1.2 / 1.3 functionality that works exposed as extensions. The design target stays 1.3 (section 2);
 * lists what is missing for 1.2 and 1.3. The ICD manifest (meson.build, --api-version) must agree. */
#define NGV_API_VERSION VK_MAKE_API_VERSION(0, 1, 1, VK_HEADER_VERSION)

/* Memory layout (section 4). */
#define NGV_MEM_TYPE_VRAM 0     /* DEVICE_LOCAL, domain VRAM */
#define NGV_MEM_TYPE_GTT 1      /* HOST_VISIBLE|COHERENT|CACHED, domain GTT */
#define NGV_MEM_TYPE_VRAM_VIS 2 /* DEVICE_LOCAL|HOST_VISIBLE, VRAM|HOST_VISIBLE (BAR2 window) */
#define NGV_MEM_TYPE_COUNT 3
#define NGV_MEM_HEAP_VRAM 0
/* The kernel reports a RM without a VRAM path (rm_cp / rm_sp; GET_GPU_INFO info_flags NICOGPU_GPU_INFO_F_NO_VRAM, HAL
 * NICOGPU_INFO_F_NO_VRAM, additive, ABI v6 stays) or no VRAM heap at all: then only host (GTT) memory is exposed -- ONE
 * heap and ONE memory type (index 0): DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT[|HOST_CACHED], domain GTT, like a UMA GPU. */
#ifndef NICOGPU_INFO_F_NO_VRAM
#define NICOGPU_INFO_F_NO_VRAM (1u << 5)
#endif
#define NGV_NO_VRAM(info) ((((info)->info_flags & NICOGPU_INFO_F_NO_VRAM) != 0) || (info)->vram_size == 0)
#define NGV_MEM_HEAP_VRAM_VIS 1
#define NGV_MEM_HEAP_GTT 2
#define NGV_MEM_HEAP_COUNT 3

struct ngv_instance {
   struct vk_instance vk;
};

struct ngv_physical_device {
   struct vk_physical_device vk;
   const struct ngv_winsys_ops *ops;
   struct nicogpu_ws_device_desc desc;
   struct nicogpu_device_info info;
   /* The sync type of the device: the Mesa runtime's vk_drm_syncobj on the kernel's DRM core syncobjs (sync_drm; drm
    * winsys, ABI >= 6) or a per-device copy of ngv_sync_type with the feature bits the winsys cannot back removed
    * (ngv_sync_type_init).  sync_types[1] is the timeline emulation, only when the base type has no TIMELINE. */
   struct vk_sync_type sync_type;
   bool sync_drm;
   const struct vk_sync_type *sync_types[3];
   struct vk_sync_timeline_type timeline_type;
 struct disk_cache *xo_cache; /* compiled shaders (ngv_ngcomp.c); NULL = off */
};

/* Queue families (; replaces the single family / single queue of section 5). Both families are
 * COMPUTE|TRANSFER (no GRAPHICS until P2) with the same timestamp bits; family 1 exists so that applications (and the
 * CTS *_compute_only_queue / concurrent_compute cases) find a compute family distinct from the "universal" family 0.
 * Every VkQueue is its own winsys queue (= kernel queue, nicogpu_queue_create) on the device's one VM, and every kernel
 * queue feeds the one CP ring: no new UAPI, no new packet.  Family 0 keeps one queue, so applications that create
 * "all queues of the universal family" behave as before. */
#define NGV_QUEUE_FAMILY_COUNT 2
#define NGV_QUEUE_FAMILY_UNIVERSAL_QUEUES 1 /* family 0 */
#define NGV_QUEUE_FAMILY_COMPUTE_QUEUES 2   /* family 1 */
#define NGV_MAX_QUEUES (NGV_QUEUE_FAMILY_UNIVERSAL_QUEUES + NGV_QUEUE_FAMILY_COMPUTE_QUEUES)

struct ngv_queue {
   struct vk_queue vk;
   struct ngv_device *device;
   uint32_t ws_queue;
};

/* A winsys BO with its VA and (for host-visible domains) a persistent CPU mapping. */
struct ngv_bo {
   uint32_t handle; /* 0 = none */
   uint64_t va;
   uint64_t size;
   void *map;
};

struct ngv_meta_cache;

struct ngv_device {
   struct vk_device vk;
   struct ngv_physical_device *pdev;
   struct ngv_winsys ws;
   simple_mtx_t mem_lock; /* serializes winsys BO calls (device calls are not thread-safe) */
   /* the queues of VkDeviceCreateInfo, in creation order (queue_count of them); vkGetDeviceQueue* is the runtime's
    * (vk_common_GetDeviceQueue2 walks vk_device::queues) */
   struct ngv_queue queues[NGV_MAX_QUEUES];
   uint32_t queue_count;
 /* per-lane scratch (ngv_scratch_get): one BO for all dispatches, grown on demand; replaced BOs stay alive
    * (scratch_retired) until the device is destroyed because earlier command buffers / jobs in flight still use them.
    * Shared by all queues: jobs of different queues meet in the one CP ring and every job ends with an end-of-pipe
 * FENCE (command-set spec section 3.5), exactly like consecutive jobs of one queue. */
   simple_mtx_t scratch_lock;
   struct ngv_bo scratch_bo;
   struct ngv_bo *scratch_retired;
   uint32_t scratch_retired_count, scratch_retired_cap;
 /* compute kernels of the image transfer commands (ngv_meta.c): compiled on first use, kept until the device dies */
   simple_mtx_t meta_lock;
   struct ngv_meta_cache *meta;
};

struct ngv_device_memory {
   struct vk_device_memory vk;
   uint32_t bo;
   uint64_t va;
   void *map;
};

struct ngv_buffer {
   struct vk_buffer vk;
};

/* ngv_image.c: images, views and samplers (: hardware layout + image / sampler descriptors).
 * `fmt` == NULL: a format the texture unit does not know (combined depth/stencil, multi-planar, ETC2/ASTC ...): the image keeps
 * the layout (level by level, layer by layer, rows tightly packed) and the CP COPY engine byte-range transfers. */
struct ngv_image {
   struct vk_image vk;
   uint64_t size;
   uint64_t va;
   const struct ngv_vkfmt *fmt;
   struct ngv_layout lay; /* valid when fmt != NULL */
   bool linear;           /* tiling = linear in the descriptor (VK_IMAGE_TILING_LINEAR) */
 /* fmt == NULL (layout), filled at creation: bytes of one array layer, and the byte offset of each level inside a
    * layer (levels past the 16th are summed on demand, ngv_image.c legacy_level_offset) */
   uint64_t legacy_layer;
   uint64_t legacy_off[NGV_LAYOUT_MAX_LEVELS];
};

struct ngv_image_view {
   struct vk_image_view vk;
   uint32_t sampled[8]; /* image descriptor for SAMPLED_IMAGE / COMBINED_IMAGE_SAMPLER / INPUT_ATTACHMENT (view swizzle) */
   uint32_t storage[8]; /* image descriptor for STORAGE_IMAGE (identity swizzle, one level) */
};

struct ngv_sampler {
   struct vk_sampler vk;
 uint32_t desc[4]; /* sampler descriptor (the descriptor spec section 2.1) */
};

struct ngv_buffer_view {
   struct vk_buffer_view vk;
 uint32_t desc[8]; /* image descriptor with view_type = buffer (the descriptor spec section 4.6) */
};

/* ---- descriptors (section 6.2): buffer 16 B/16, image-like 32/32, sampler 16/16, combined 64/32 ---- */
#define NGV_MAX_SETS 8
#define NGV_MAX_DYNAMIC 32 /* per set: maxDescriptorSet{Uniform,Storage}BufferDynamic = 16 + 16 */
#define NGV_MAX_PUSH 128
/* u0..u25 are the user registers SET_USER_REGS can write: the generated command header is the single source */
#define NGV_NUM_UREGS NICOGPU_CMD_LIMIT_USER_UREGS
_Static_assert(NGV_NUM_UREGS == NICOGPU_CMD_SET_USER_REGS_COUNT_MAX - 1, "SET_USER_REGS: 1 start dword + the u-registers");

struct ngv_binding_layout {
   uint32_t binding;
   VkDescriptorType type;
   uint32_t count;
   uint32_t offset; /* byte offset in the set */
   uint32_t stride; /* descriptor size in bytes */
   int32_t dyn_start; /* index of the first dynamic descriptor of this binding, -1 if not dynamic */
 /* immutable samplers : index of the first of `count` sampler descriptors (4 dwords each) in
    * ngv_descriptor_set_layout::imm; -1 = none */
   int32_t imm_start;
};

struct ngv_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;
   uint32_t binding_count;
   uint32_t size; /* bytes */
   uint32_t dynamic_count;
   uint32_t *imm;     /* immutable sampler descriptors, 4 dwords each (malloc'd, NULL if none) */
   uint32_t imm_count;
   struct ngv_binding_layout bindings[]; /* sorted by binding number */
};

struct ngv_descriptor_pool {
   struct vk_object_base base;
   struct ngv_bo bo;
   uint32_t max_sets;
   uint32_t set_count;    /* length of `sets` */
   struct list_head sets; /* sorted by offset */
};

struct ngv_descriptor_set {
   struct vk_object_base base;
   struct ngv_descriptor_pool *pool;
   struct ngv_descriptor_set_layout *layout;
   struct list_head link;
   uint32_t offset;
   uint32_t size;
   uint64_t va;
   void *map;
};

/* ---- pipelines ---- */
struct ngv_pipeline {
   struct vk_object_base base;
   struct vk_pipeline_layout *layout;
   struct ngv_bo code;
   uint32_t entry; /* byte offset of the entry point in the code object */
   uint32_t vregs;
   uint32_t shared_bytes;
   uint32_t scratch_bytes;
   uint32_t wg[3];
 /* DISPATCH_BASE pipelines (.md 14.3): byte offset in the push-constant block where the driver stores the
    * three base-workgroup dwords the compiled code reads (load_base_workgroup_id); -1 = not a DISPATCH_BASE pipeline */
   int32_t base_push_offset;
 /* Shaders that read gl_NumWorkGroups (PROVISIONAL): byte offset in the push-constant block of a 64-bit VA
    * the driver stores per dispatch; the compiled code loads the grid {x, y, z} (3 dwords) from it.  Direct dispatch:
    * the grid in the command buffer's upload area; indirect dispatch: the VkDispatchIndirectCommand itself.
    * -1 = the shader does not read gl_NumWorkGroups. */
   int32_t numwg_push_offset;
};

/* Result of the shader "compiler" stage. */
struct ngv_shader_object {
   void *code;
   uint32_t code_size;
   uint32_t entry, vregs, shared_bytes, scratch_bytes;
   uint32_t wg[3];
   bool placeholder;
   bool uses_num_workgroups; /* the code reads the num-workgroups pointer (struct ngv_pipeline::numwg_push_offset) */
   /* .xo header (ngv_xo.h): ABI code of the object (1 = ABI 0.1, 2 = ABI 0.2) and its stage mask (bit k = stage k; ABI 0.1
    * objects are compute only).  0 / 0 for the placeholder kernels. */
   uint32_t xo_abi, xo_stage_mask;
};

/* ---- queries (timestamp only) ---- */
struct ngv_query_pool {
   struct vk_query_pool vk;
   struct ngv_bo bo; /* [count] u64 values, then [count] u64 availability */
};

/* ---- events: one 4 B word in a (page-sized, PROVISIONAL) GTT BO; 0 = reset, 1 = set ---- */
struct ngv_event {
   struct vk_object_base base;
   struct ngv_bo bo;
};

/* ---- command buffer ---- */
#define NGV_IB_CHUNK_BYTES (64u * 1024u)
#define NGV_IB_CHUNK_DWORDS (NGV_IB_CHUNK_BYTES / 4)
#define NGV_UPLOAD_CHUNK_BYTES (16u * 1024u)
#define NGV_IB_RESERVE_DW 64 /* every emit helper first ensures this much space; no bundle is larger */

struct ngv_ib_chunk {
   struct ngv_bo bo;
   uint32_t used_dw;
};

struct ngv_cmd_buffer {
   struct vk_command_buffer vk;

   /* IB chunks (GTT BOs).  A chunk is one INDIRECT in the kernel's submission, so all shader / user-register state
 * is re-emitted at the start of each (section 6.4). chunks[0.. chunk_alloc) own a BO, chunks[0.. chunk_count)
    * are the ones the current recording uses (and what a submit sees); a plain reset only zeroes chunk_count and the next
    * recording reuses the BOs, vkResetCommandBuffer(RELEASE_RESOURCES) / destroy free them. */
   struct ngv_ib_chunk *chunks;
   uint32_t chunk_count, chunk_alloc, chunk_cap;
   /* bump-allocated data (copied descriptor sets, root table, grids): uploads[0 .. upload_alloc) own a BO,
    * uploads[0 .. upload_count) are in use by the current recording, upload_used = bytes used of the last of those */
   struct ngv_bo *uploads;
   uint32_t upload_count, upload_alloc, upload_cap;
   uint32_t upload_used;

   /* bound state */
   struct ngv_pipeline *pipeline;
   struct ngv_descriptor_set *sets[NGV_MAX_SETS];
   uint32_t dyn_offsets[NGV_MAX_SETS][NGV_MAX_DYNAMIC];
   uint32_t push[NGV_MAX_PUSH / 4];

   /* what the current IB chunk already has */
   struct ngv_pipeline *emitted_pipeline;
   bool u_valid;
   uint32_t u_shadow[NGV_NUM_UREGS];
   /* GPU-visible copies produced from the bound state.  sets_valid: set_va[] (and the dynamic-set copies behind it)
    * match the bound sets / dynamic offsets / layout; root_valid: root_va matches set_va[] and the push-constant tail
    * (push[16..31]).  A push-constant write invalidates only the root table. */
   uint64_t set_va[NGV_MAX_SETS];
   uint64_t root_va;
   bool sets_valid, root_valid;
   /* last direct-dispatch grid written to the upload area for gl_NumWorkGroups (0 = none yet) */
   uint64_t numwg_va;
   uint32_t numwg_grid[3];
};

struct ngv_sync {
   struct vk_sync vk;
   uint32_t handle; /* winsys syncobj */
};

/* Fills a winsys submit entry from a runtime sync (wait_value / signal_value 0 for binary syncs). */
static inline void
ngv_ws_sync_from_vk(struct nicogpu_ws_sync *out, const struct vk_sync *sync, uint64_t value)
{
   const struct vk_drm_syncobj *d = vk_sync_as_drm_syncobj((struct vk_sync *)sync);
   if (d) {
      out->handle = d->syncobj;
      out->flags = NICOGPU_WS_SYNC_F_DRM;
      out->point = (sync->flags & VK_SYNC_IS_TIMELINE) ? value : 0;
   } else {
      out->handle = container_of(sync, struct ngv_sync, vk)->handle;
      out->flags = 0;
      out->point = 0;
   }
}

VK_DEFINE_HANDLE_CASTS(ngv_instance, vk.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(ngv_physical_device, vk.base, VkPhysicalDevice, VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(ngv_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)
VK_DEFINE_HANDLE_CASTS(ngv_queue, vk.base, VkQueue, VK_OBJECT_TYPE_QUEUE)
VK_DEFINE_HANDLE_CASTS(ngv_cmd_buffer, vk.base, VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_device_memory, vk.base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_buffer, vk.base, VkBuffer, VK_OBJECT_TYPE_BUFFER)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_image_view, vk.base, VkImageView, VK_OBJECT_TYPE_IMAGE_VIEW)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_sampler, vk.base, VkSampler, VK_OBJECT_TYPE_SAMPLER)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_buffer_view, vk.base, VkBufferView, VK_OBJECT_TYPE_BUFFER_VIEW)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_descriptor_set_layout, vk.base, VkDescriptorSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_descriptor_pool, base, VkDescriptorPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_descriptor_set, base, VkDescriptorSet, VK_OBJECT_TYPE_DESCRIPTOR_SET)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_event, base, VkEvent, VK_OBJECT_TYPE_EVENT)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_pipeline, base, VkPipeline, VK_OBJECT_TYPE_PIPELINE)
VK_DEFINE_NONDISP_HANDLE_CASTS(ngv_query_pool, vk.base, VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL)

extern const struct vk_sync_type ngv_sync_type; /* template; use ngv_sync_type_init for the per-device copy */
void ngv_sync_type_init(struct vk_sync_type *out, uint64_t hal_features);
extern const struct vk_command_buffer_ops ngv_cmd_buffer_ops;

/* ngv_device.c */
void ngv_get_properties(const struct ngv_physical_device *pdev, struct vk_properties *p);
void ngv_get_features(const struct ngv_physical_device *pdev, struct vk_features *f);
void ngv_get_device_extensions(const struct ngv_physical_device *pdev, struct vk_device_extension_table *e);
bool ngv_vulkan_memory_model_enabled(void);
extern const struct vk_instance_extension_table ngv_instance_extensions;

/* ngv_memory.c */
uint32_t ngv_memory_type_domain(uint32_t type, uint32_t *bo_flags);
enum ngv_bo_kind {
   NGV_BO_HOST,  /* GTT, CPU-mapped: IBs, descriptors, uploads, queries */
   NGV_BO_CODE,  /* VRAM|HOST_VISIBLE if the device has a window, else GTT; CPU-mapped */
};
VkResult ngv_bo_create(struct ngv_device *dev, uint64_t size, enum ngv_bo_kind kind, struct ngv_bo *out);
void ngv_bo_destroy(struct ngv_device *dev, struct ngv_bo *bo);
/* Scratch backing store for a dispatch whose shader needs bytes_per_lane (multiple of 16) per lane: returns the VA
 * (256 B aligned) of a BO that holds bytes_per_lane x every resident lane of the device. */
VkResult ngv_scratch_get(struct ngv_device *dev, uint32_t bytes_per_lane, uint64_t *va);
void ngv_scratch_finish(struct ngv_device *dev); /* DestroyDevice only */

/* ngv_descriptor.c */
extern const uint32_t ngv_descriptor_type_size[];
uint32_t ngv_descriptor_size(VkDescriptorType t); /* 0 = unsupported */

/* ngv_image.c */
const struct ngv_vkfmt *ngv_format_lookup(VkFormat f);
/* Descriptor of the raw (unsigned integer, element-sized) view of one mip level of a hardware-layout image, all layers /
 * slices: what the transfer kernels (ngv_meta.c) read and write.  false: not describable. */
bool ngv_image_raw_desc(uint32_t d[8], const struct ngv_image *img, uint32_t level);
/* Descriptor of the image's own format for img.store of the transfer kernels' clear (storage semantics, one level) */
bool ngv_image_typed_desc(uint32_t d[8], const struct ngv_image *img, uint32_t level);
/* ngv_instance.c format queries use these */
VkFormatFeatureFlags ngv_format_image_features(const struct ngv_vkfmt *f, VkImageTiling tiling);
VkFormatFeatureFlags ngv_format_buffer_features(const struct ngv_vkfmt *f);
unsigned ngv_format_raw_mult(const struct ngv_vkfmt *f); /* 3 for 3-component formats (moved as 3 narrow elements), else 1 */
VkResult ngv_image_format_properties(const VkPhysicalDeviceImageFormatInfo2 *info, VkImageFormatProperties2 *props);

/* ngv_meta.c */
void ngv_meta_finish(struct ngv_device *dev);
void ngv_meta_buffer_image(struct ngv_cmd_buffer *cmd, struct ngv_buffer *buf, struct ngv_image *img, uint32_t count,
                           const VkBufferImageCopy2 *regions, bool to_image);
void ngv_meta_image_image(struct ngv_cmd_buffer *cmd, struct ngv_image *src, struct ngv_image *dst, uint32_t count,
                          const VkImageCopy2 *regions);

/* ngv_pipeline.c */
struct vk_pipeline_cache;
VkResult ngv_shader_compile(struct ngv_device *dev, struct vk_pipeline_cache *pc, const VkPipelineShaderStageCreateInfo *stage,
                            VkPipelineCreateFlags2KHR pipeline_flags, const uint32_t *spirv, uint32_t spirv_words,
                            const struct vk_pipeline_layout *layout, int32_t base_push_offset,
                            int32_t numwg_push_offset, struct ngv_shader_object *out);
VkResult ngv_shader_error(struct ngv_device *dev, VkResult res, const char *fmt, ...) PRINTFLIKE(3, 4);
VkResult ngv_load_xo(const char *path, struct ngv_shader_object *o);
VkResult ngv_parse_xo(const uint8_t *buf, size_t size, struct ngv_shader_object *o);
uint8_t *ngv_read_file(const char *path, size_t *size);
void ngv_shader_cache_init(struct ngv_physical_device *pdev);
void ngv_shader_cache_finish(struct ngv_physical_device *pdev);
/* : the ngcomp target units of pdev contain "f16cvt" (hardware binary16 conversions, RM CAPS[29]) */
bool ngv_target_has_f16cvt(const struct ngv_physical_device *pdev);
bool ngv_target_has_xlane(const struct ngv_physical_device *pdev); /* : units contain "xlane" */
/* ngv_ngcomp.c / ngv_uuid.c: the compiler-side inputs of pipelineCacheUUID, and the three UUIDs (section 16) */
struct mesa_sha1;
void ngv_ngcomp_identity(const struct ngv_physical_device *pdev, struct mesa_sha1 *ctx);
void ngv_physical_device_init_uuids(struct ngv_physical_device *pdev);

/* ngv_nir.c / ngv_nir_print.c / ngv_ngcomp.c: SPIR-V -> NIR -> ngcomp text -> `python3 driver.py isel` -> .xo
 * (section 14.2). */
struct nir_shader;
extern const struct nir_shader_compiler_options ngv_nir_options;
VkResult ngv_shader_to_nir(struct ngv_device *dev, VkPipelineCreateFlags2KHR pipeline_flags,
                           const VkPipelineShaderStageCreateInfo *sinfo, const struct vk_pipeline_layout *layout,
                           int32_t base_push_offset, int32_t numwg_push_offset, bool *uses_num_workgroups,
                           struct nir_shader **out);
bool ngv_nir_lower_pipeline_layout(struct nir_shader *nir, const struct vk_pipeline_layout *layout,
                                   int32_t base_push_offset, int32_t numwg_push_offset, bool *uses_num_workgroups,
                                   const char **error);
char *ngv_nir_print_ngcomp(struct nir_shader *nir, char *err, size_t err_len);
/* xo / xo_size: optional, receive the finished .xo container (malloc'd; the caller frees) */
VkResult ngv_ngcomp_compile(struct ngv_device *dev, const char *nir_text, uint32_t target_occupancy,
                            struct ngv_shader_object *out, uint8_t **xo, size_t *xo_size);
/* SPIR-V level shader cache (ngv_ngcomp.c): key of the whole compile chain, lookup (VK_SUCCESS = `out` is the finished
 * object, VK_INCOMPLETE = miss) in the VkPipelineCache `pc` (may be NULL) and the disk cache, and store after a compile.
 * The key function returns false when the compile must not be cached. */
struct vk_pipeline_cache;
bool ngv_ngcomp_spirv_key(const struct ngv_device *dev, VkPipelineCreateFlags2KHR pipeline_flags,
                          const VkPipelineShaderStageCreateInfo *stage, const uint32_t *spirv, uint32_t spirv_words,
                          const struct vk_pipeline_layout *layout, int32_t base_push_offset, int32_t numwg_push_offset,
                          uint8_t key[20]);
VkResult ngv_ngcomp_spirv_lookup(struct ngv_device *dev, struct vk_pipeline_cache *pc, const uint8_t key[20],
                                 struct ngv_shader_object *out);
void ngv_ngcomp_spirv_store(struct ngv_device *dev, struct vk_pipeline_cache *pc, const uint8_t key[20],
                            const struct ngv_shader_object *so, uint32_t occupancy, const void *xo, size_t xo_size);
/* warps per SP a workgroup of wg[] needs when spread over one half core: the ngcomp occupancy target */
uint32_t ngv_warps_per_sp(const struct ngv_device *dev, const uint32_t wg[3]);

/* ngv_cmd_buffer.c */
/* Returns n_dw writable dwords at the end of the current IB chunk (opening a new chunk, which invalidates the
 * emitted shader / user-register state, when it does not fit); NULL + the command buffer's record error on failure.
 * n_dw <= NGV_IB_CHUNK_DWORDS. */
uint32_t *ngv_cmd_reserve(struct ngv_cmd_buffer *cmd, uint32_t n_dw);
static inline void
ngv_cmd_commit(struct ngv_cmd_buffer *cmd, uint32_t n_dw)
{
   cmd->chunks[cmd->chunk_count - 1].used_dw += n_dw;
}
/* Bump-allocates `size` bytes of GPU-visible, CPU-written memory that lives as long as the command buffer's recording. */
bool ngv_cmd_upload(struct ngv_cmd_buffer *cmd, uint32_t size, uint32_t alignment, uint64_t *va, void **ptr);
VkResult ngv_cmd_buffer_collect_ibs(struct ngv_cmd_buffer *cmd, struct nicogpu_cmdbuf *out, uint32_t *n, uint32_t cap);

/* placeholder code object recognized by the stub winsys' CP emulation (ngv_stub_cp.c) */
#define NGV_PH_MAGIC "NGVPHKRN"
#define NGV_PH_KERNEL_SCALE_BIAS 1      /* entry name "ngv_ph_scale_bias" */
#define NGV_PH_KERNEL_SCALE_BIAS_BASE 2 /* same entry name on a DISPATCH_BASE pipeline: element = (base_x + wg) * local + lid,
                                         * base_x = u12 (push dword 2: the base sits right behind the 8-byte push
                                         * range of the test layout, ngv_pipeline.c) */
struct ngv_ph_header {
   char magic[8];
   uint32_t kernel;
   uint32_t reserved;
};

#endif
