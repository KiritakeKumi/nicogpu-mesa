/* SPDX-License-Identifier: MIT */
/* ngv_image.c -- VkImage / VkImageView / VkSampler objects, format and image-format queries.
 *
 * Images whose format the texture unit knows (ngv_formats_gen.h, generated from the format description source) have the
 * hardware layout of the descriptor spec section 6 (ngv_layout.h): t64k for VK_IMAGE_TILING_OPTIMAL, linear for
 * VK_IMAGE_TILING_LINEAR.  Image views and samplers carry the descriptors the shader loads (image 32 B, sampler 16 B);
 * vkUpdateDescriptorSets copies them into the set (ngv_descriptor.c).  Transfers (copies, clears) of such images are
 * compute kernels (ngv_meta.c): the CP copy engine addresses byte ranges and does not know the tiled layout.
 *
 * Images in a format that is not in the table (combined depth/stencil, multi-planar, emulated ETC2/ASTC) keep the 
 * layout (level by level, layer by layer, rows tightly packed) and the CP COPY byte-range transfers; they have no
 * descriptors (views of them are null descriptors) and no format features, so applications only get them if they create
 * the format without asking vkGetPhysicalDeviceFormatProperties first.  Single-aspect formats only; blit / resolve are in
 * ngv_cmd_buffer.c (unsupported: no graphics queue). */
#include "ngv_cmd_stream.h"
#include "ngv_private.h"

#include "vk_format.h"
#include "vk_image.h"
#include "vk_sampler.h"

#define NGV_IMAGE_ALIGNMENT 256

/* limits reported by the device (ngv_device.c: maxImageDimension* / maxImageArrayLayers); descriptor bounds: width/height
 * 14 bits, depth/layers 14 bits, levels 16 */
#define NGV_MAX_DIM_1D_2D 16384u
#define NGV_MAX_DIM_3D 2048u
#define NGV_MAX_LAYERS 2048u
#define NGV_MAX_RESOURCE_SIZE (1ull << 31)

/* ---- formats --------------------------------------------------------------------------------------------------- */

const struct ngv_vkfmt *
ngv_format_lookup(VkFormat f)
{
   for (unsigned i = 0; i < ARRAY_SIZE(ngv_vkfmts); i++)
      if (ngv_vkfmts[i].vk == (int32_t)f)
         return &ngv_vkfmts[i];
   return NULL;
}

VkFormatFeatureFlags
ngv_format_image_features(const struct ngv_vkfmt *f, VkImageTiling tiling)
{
   if (!f)
      return 0;
   /* 3-component formats cannot be tiled (6.8 malformed:tiling); compressed and depth/stencil formats have no linear
    * layout (6.8: linear = 1 level, 1 sample, and the format table's linear_only is about the opposite direction) */
   if (tiling == VK_IMAGE_TILING_OPTIMAL && (f->flags & NGV_FMT_LINEAR_ONLY))
      return 0;
   if (tiling == VK_IMAGE_TILING_LINEAR && (f->flags & (NGV_FMT_COMPRESSED | NGV_FMT_DEPTH | NGV_FMT_STENCIL)))
      return 0;
   VkFormatFeatureFlags r = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                            VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   if (f->flags & NGV_FMT_FILTERABLE)
      r |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
   if (f->flags & NGV_FMT_STORABLE)
      r |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
   if (f->flags & NGV_FMT_ATOMIC)
      r |= VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT;
   return r;
}

VkFormatFeatureFlags
ngv_format_buffer_features(const struct ngv_vkfmt *f)
{
   if (!f || (f->flags & (NGV_FMT_COMPRESSED | NGV_FMT_DEPTH | NGV_FMT_STENCIL)))
      return 0;
   VkFormatFeatureFlags r = VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT;
   if (f->flags & NGV_FMT_STORABLE)
      r |= VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;
   if (f->flags & NGV_FMT_ATOMIC)
      r |= VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   return r;
}

/* ---- layout -------------------------------------------------------------------------------------------------- */

static uint64_t
legacy_level_size(const struct vk_image *img, uint32_t level)
{
   const VkExtent3D e = vk_image_mip_level_extent(img, level);
   return (uint64_t)vk_format_get_blocksize(img->format) * e.width * e.height * e.depth;
}

static uint64_t
legacy_size(const struct vk_image *img, uint64_t *layer_stride)
{
   uint64_t layer = 0;
   for (uint32_t l = 0; l < img->mip_levels; l++)
      layer += legacy_level_size(img, l);
   if (layer_stride)
      *layer_stride = layer;
   return layer * img->array_layers;
}

/* Fills the cached legacy layout of an image without hardware layout (see struct ngv_image). */
static void
legacy_init(struct ngv_image *image)
{
   uint64_t off = 0;
   for (uint32_t l = 0; l < ARRAY_SIZE(image->legacy_off); l++) {
      image->legacy_off[l] = off;
      off += legacy_level_size(&image->vk, l);
   }
   (void)legacy_size(&image->vk, &image->legacy_layer);
}

/* byte offset of `level` inside one array layer of an image without hardware layout */
static uint64_t
legacy_level_offset(const struct ngv_image *img, uint32_t level)
{
   const uint32_t last = ARRAY_SIZE(img->legacy_off) - 1;
   if (level <= last)
      return img->legacy_off[level];
   uint64_t off = img->legacy_off[last];
   for (uint32_t l = last; l < level; l++)
      off += legacy_level_size(&img->vk, l);
   return off;
}

/* Where (level, layer) of an image without hardware layout lives, and the strides inside that level (in texel blocks). */
struct img_loc {
   uint64_t off;   /* byte offset of (level, layer) from the start of the image */
   uint32_t elem;  /* bytes per block */
   uint64_t row;   /* bytes per row of blocks */
   uint64_t slice; /* bytes per slice (z step of a 3D image) */
   uint64_t layer; /* bytes per array layer */
};

static struct img_loc
img_loc(const struct ngv_image *img, uint32_t level, uint32_t layer)
{
   const struct vk_image *vk = &img->vk;
   const VkExtent3D e = vk_image_mip_level_extent(vk, level);
   const uint32_t bs = vk_format_get_blocksize(vk->format);
   const uint64_t row = (uint64_t)bs * DIV_ROUND_UP(e.width, vk_format_get_blockwidth(vk->format));
   return (struct img_loc){
      .off = legacy_level_offset(img, level) + img->legacy_layer * layer,
      .elem = bs,
      .row = row,
      .slice = row * DIV_ROUND_UP(e.height, vk_format_get_blockheight(vk->format)),
      .layer = img->legacy_layer,
   };
}

static enum ngv_res_type
res_type_of(VkImageType t)
{
   return t == VK_IMAGE_TYPE_1D ? NGV_RES_1D : t == VK_IMAGE_TYPE_3D ? NGV_RES_3D : NGV_RES_2D;
}

static unsigned
samples_log2(VkSampleCountFlagBits s)
{
   return s == VK_SAMPLE_COUNT_4_BIT ? 2 : s == VK_SAMPLE_COUNT_2_BIT ? 1 : 0;
}

/* The hardware layout of an image, or false if the image cannot be described (then CreateImage fails). */
static bool
hw_layout(const struct vk_image *img, const struct ngv_vkfmt *f, struct ngv_layout *lay)
{
   if (img->image_type == VK_IMAGE_TYPE_3D && img->array_layers != 1)
      return false;
   const bool linear = img->tiling == VK_IMAGE_TILING_LINEAR;
   if (linear && (f->flags & (NGV_FMT_COMPRESSED | NGV_FMT_DEPTH | NGV_FMT_STENCIL)))
      return false;
   if (!linear && (f->flags & NGV_FMT_LINEAR_ONLY))
      return false;
   if (img->extent.width > 65536 || img->extent.height > 65536 || img->extent.depth > 65536)
      return false;
   const struct ngv_layout_in in = {
      .res_type = res_type_of(img->image_type),
      .linear = linear,
      .elem_bytes = f->elem,
      .block = f->block,
      .samples_log2 = samples_log2(img->samples),
      .width = img->extent.width,
      .height = img->extent.height,
      .depth = img->extent.depth,
      .levels = img->mip_levels,
      .layers = img->array_layers,
   };
   return ngv_layout_compute(&in, lay);
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateImage(VkDevice _device, const VkImageCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                VkImage *pImage)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   if (pCreateInfo->flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                             VK_IMAGE_CREATE_SPARSE_ALIASED_BIT))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   if (vk_format_get_blocksize(pCreateInfo->format) == 0)
      return vk_error(device, VK_ERROR_FORMAT_NOT_SUPPORTED);
   struct ngv_image *image = vk_image_create(&device->vk, pCreateInfo, pAllocator, sizeof(*image));
   if (!image)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   image->fmt = ngv_format_lookup(pCreateInfo->format);
   if (image->fmt) {
      image->linear = image->vk.tiling == VK_IMAGE_TILING_LINEAR;
      if (!hw_layout(&image->vk, image->fmt, &image->lay)) {
         vk_image_destroy(&device->vk, pAllocator, &image->vk);
         return vk_error(device, VK_ERROR_FORMAT_NOT_SUPPORTED);
      }
   }
   if (image->fmt) {
      image->size = image->lay.total;
   } else {
      legacy_init(image);
      image->size = image->legacy_layer * image->vk.array_layers;
   }
   *pImage = ngv_image_to_handle(image);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyImage(VkDevice _device, VkImage _image, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_image, image, _image);
   if (image)
      vk_image_destroy(&device->vk, pAllocator, &image->vk);
}

static void
image_mem_reqs(struct ngv_device *device, uint64_t size, uint32_t align, VkMemoryRequirements2 *pReqs)
{
   uint32_t types = (1u << NGV_MEM_TYPE_COUNT) - 1;
   if (NGV_NO_VRAM(&device->pdev->info))
      types = 1u;
   else if (!device->pdev->info.vram_visible_size)
      types &= ~(1u << NGV_MEM_TYPE_VRAM_VIS);
   pReqs->memoryRequirements = (VkMemoryRequirements){
      .size = align64(size ? size : 1, align),
      .alignment = align,
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

VKAPI_ATTR void VKAPI_CALL
ngv_GetImageMemoryRequirements2(VkDevice _device, const VkImageMemoryRequirementsInfo2 *pInfo,
                                VkMemoryRequirements2 *pReqs)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_image, image, pInfo->image);
   image_mem_reqs(device, image->size, image->fmt ? image->lay.base_align : NGV_IMAGE_ALIGNMENT, pReqs);
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetDeviceImageMemoryRequirements(VkDevice _device, const VkDeviceImageMemoryRequirements *pInfo,
                                     VkMemoryRequirements2 *pReqs)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   struct vk_image tmp;
   vk_image_init(&device->vk, &tmp, pInfo->pCreateInfo);
   struct ngv_layout lay;
   const struct ngv_vkfmt *f = ngv_format_lookup(tmp.format);
   const bool ok = f && hw_layout(&tmp, f, &lay);
   image_mem_reqs(device, ok ? lay.total : legacy_size(&tmp, NULL), ok ? lay.base_align : NGV_IMAGE_ALIGNMENT, pReqs);
   vk_image_finish(&tmp);
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetImageSparseMemoryRequirements2(VkDevice device, const VkImageSparseMemoryRequirementsInfo2 *pInfo,
                                      uint32_t *pCount, VkSparseImageMemoryRequirements2 *pReqs)
{
   *pCount = 0;
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetDeviceImageSparseMemoryRequirements(VkDevice device, const VkDeviceImageMemoryRequirements *pInfo,
                                           uint32_t *pCount, VkSparseImageMemoryRequirements2 *pReqs)
{
   *pCount = 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_BindImageMemory2(VkDevice _device, uint32_t bindInfoCount, const VkBindImageMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; i++) {
      VK_FROM_HANDLE(ngv_image, image, pBindInfos[i].image);
      VK_FROM_HANDLE(ngv_device_memory, mem, pBindInfos[i].memory);
      image->va = mem->va + pBindInfos[i].memoryOffset;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_GetImageSubresourceLayout(VkDevice _device, VkImage _image, const VkImageSubresource *pSubresource,
                              VkSubresourceLayout *pLayout)
{
   VK_FROM_HANDLE(ngv_image, image, _image);
   if (image->fmt) {
      /* Linear images: row-major with the descriptor's pitch.  Tiled images are opaque (the spec only defines offset /
       * size for them): offset and size locate the level, the pitches are the layer size. */
      const struct ngv_layout *lay = &image->lay;
      const struct ngv_level *lv = &lay->level[pSubresource->mipLevel];
      const uint64_t layer_stride = image->linear ? lay->lin_stride : lay->layer_size;
      *pLayout = (VkSubresourceLayout){
         .offset = lv->offset + (image->vk.image_type == VK_IMAGE_TYPE_3D ? 0 : layer_stride * pSubresource->arrayLayer),
         .size = lv->size,
         .rowPitch = image->linear ? lay->pitch : 0,
         .arrayPitch = image->vk.image_type == VK_IMAGE_TYPE_3D ? 0 : layer_stride,
         .depthPitch = image->vk.image_type == VK_IMAGE_TYPE_3D ? layer_stride : 0,
      };
      return;
   }
   const struct img_loc loc = img_loc(image, pSubresource->mipLevel, pSubresource->arrayLayer);
   *pLayout = (VkSubresourceLayout){
      .offset = loc.off,
      .size = legacy_level_size(&image->vk, pSubresource->mipLevel),
      .rowPitch = loc.row,
      .arrayPitch = loc.layer,
      .depthPitch = loc.slice,
   };
}

/* ---- image descriptors ------------------------------------------------------------------------------------------ */

struct desc_view {
   unsigned view_type;                       /* enum nicogpu_view_type */
   unsigned data_format, num_format, order;  /* how the texel is interpreted */
   unsigned base_level, level_count;         /* 0 / 1 and levels_m1 = 0 when `single` */
   unsigned base_layer, layer_count;
   unsigned swizzle[4];                      /* enum nicogpu_swizzle */
   unsigned min_lod;                         /* ufix4.8, absolute level */
   bool single;                              /* reinterpret one level as a resource of its own (raw / block-texel views) */
   unsigned level;                           /* that level when `single` */
   unsigned mult;                            /* `single`: view elements per image element along x (3-component raw views) */
   unsigned block;                           /* `single`: block edge of the view's format in texels */
};

static void
desc_fill(uint32_t d[8], const struct ngv_image *img, const struct desc_view *v)
{
   const struct ngv_layout *lay = &img->lay;
   memset(d, 0, 32);
   uint64_t va = img->va;
   uint32_t w, h, depth, levels;
   const bool is3d = img->vk.image_type == VK_IMAGE_TYPE_3D;
   if (v->single) {
      const struct ngv_level *lv = &lay->level[v->level];
      va += lv->offset;
      w = lv->ew * v->mult * v->block;
      h = lv->eh * v->block;
      depth = is3d ? lv->ed : img->vk.array_layers;
      levels = 1;
   } else {
      w = img->vk.extent.width;
      h = img->vk.extent.height;
      depth = is3d ? img->vk.extent.depth : img->vk.array_layers;
      levels = img->vk.mip_levels;
   }
   nicogpu_image_set_base_va(d, va);
   nicogpu_image_set_data_format(d, v->data_format);
   nicogpu_image_set_num_format(d, v->num_format);
   nicogpu_image_set_order(d, v->order);
   nicogpu_image_set_view_type(d, v->view_type);
   nicogpu_image_set_width_m1(d, w - 1);
   nicogpu_image_set_height_m1(d, img->vk.image_type == VK_IMAGE_TYPE_1D ? 0 : h - 1);
   nicogpu_image_set_tiling(d, img->linear ? NICOGPU_TILING_LINEAR : NICOGPU_TILING_T64K);
   nicogpu_image_set_samples_log2(d, samples_log2(img->vk.samples));
   nicogpu_image_set_depth_m1(d, depth - 1);
   nicogpu_image_set_levels_m1(d, levels - 1);
   nicogpu_image_set_res_type(d, res_type_of(img->vk.image_type));
   nicogpu_image_set_min_lod(d, v->min_lod);
   nicogpu_image_set_base_layer(d, v->base_layer);
   nicogpu_image_set_layer_count_m1(d, v->layer_count - 1);
   nicogpu_image_set_base_level(d, v->single ? 0 : v->base_level);
   nicogpu_image_set_level_count_m1(d, v->single ? 0 : v->level_count - 1);
   nicogpu_image_set_swizzle_r(d, v->swizzle[0]);
   nicogpu_image_set_swizzle_g(d, v->swizzle[1]);
   nicogpu_image_set_swizzle_b(d, v->swizzle[2]);
   nicogpu_image_set_swizzle_a(d, v->swizzle[3]);
   if (img->linear) {
      nicogpu_image_set_pitch(d, lay->pitch / 16);
      nicogpu_image_set_layer_stride(d, lay->lin_stride / 256);
   } else {
      nicogpu_image_set_layer_stride(d, lay->layer_size / 256);
   }
}

static const unsigned identity_swizzle[4] = {NICOGPU_SWIZZLE_R, NICOGPU_SWIZZLE_G, NICOGPU_SWIZZLE_B, NICOGPU_SWIZZLE_A};

/* Element-sized unsigned integer format used by the transfer kernels.  3-component formats (3, 6, 12 B) are read as
 * 3 elements of the component size; `*mult` is that factor. */
static bool
raw_format(const struct ngv_vkfmt *f, unsigned *df, unsigned *mult)
{
   *mult = 1;
   switch (f->elem) {
   case 1: *df = NICOGPU_DATA_FORMAT_R8; return true;
   case 2: *df = NICOGPU_DATA_FORMAT_R16; return true;
   case 4: *df = NICOGPU_DATA_FORMAT_R32; return true;
   case 8: *df = NICOGPU_DATA_FORMAT_R32G32; return true;
   case 16: *df = NICOGPU_DATA_FORMAT_R32G32B32A32; return true;
   case 3: *df = NICOGPU_DATA_FORMAT_R8; *mult = 3; return true;
   case 6: *df = NICOGPU_DATA_FORMAT_R16; *mult = 3; return true;
   case 12: *df = NICOGPU_DATA_FORMAT_R32; *mult = 3; return true;
   default: return false;
   }
}

unsigned
ngv_format_raw_mult(const struct ngv_vkfmt *f)
{
   unsigned df, m = 1;
   return raw_format(f, &df, &m) ? m : 1;
}

/* view type a transfer kernel uses for the image: 1D / 2D images as arrays, 3D as 3D */
static unsigned
xfer_view_type(const struct ngv_image *img)
{
   return img->vk.image_type == VK_IMAGE_TYPE_1D   ? NICOGPU_VIEW_TYPE_1DARRAY
          : img->vk.image_type == VK_IMAGE_TYPE_3D ? NICOGPU_VIEW_TYPE_3D
                                                   : NICOGPU_VIEW_TYPE_2DARRAY;
}

bool
ngv_image_raw_desc(uint32_t d[8], const struct ngv_image *img, uint32_t level)
{
   unsigned df, mult;
   if (!img->fmt || level >= img->vk.mip_levels || !raw_format(img->fmt, &df, &mult))
      return false;
   const struct ngv_level *lv = &img->lay.level[level];
   const bool is3d = img->vk.image_type == VK_IMAGE_TYPE_3D;
   struct desc_view v = {
      .view_type = xfer_view_type(img),
      .data_format = df,
      .num_format = NICOGPU_NUM_FORMAT_UINT,
      .order = NICOGPU_ORDER_RGBA,
      .level_count = 1,
      .base_layer = 0,
      .layer_count = is3d ? lv->ed : img->vk.array_layers,
      .single = true,
      .level = level,
      .mult = mult,
      .block = 1,
   };
   memcpy(v.swizzle, identity_swizzle, sizeof(v.swizzle));
   desc_fill(d, img, &v);
   return true;
}

bool
ngv_image_typed_desc(uint32_t d[8], const struct ngv_image *img, uint32_t level)
{
   const struct ngv_vkfmt *f = img->fmt;
   if (!f || level >= img->vk.mip_levels || img->vk.samples != VK_SAMPLE_COUNT_1_BIT)
      return false;
   const bool is3d = img->vk.image_type == VK_IMAGE_TYPE_3D;
   struct desc_view v = {
      .view_type = xfer_view_type(img),
      .data_format = f->data_format,
      .num_format = f->num_format,
      .order = f->order,
      .base_level = level,
      .level_count = 1,
      .layer_count = is3d ? MAX2(1u, img->vk.extent.depth >> level) : img->vk.array_layers,
      .mult = 1,
      .block = 1,
   };
   memcpy(v.swizzle, identity_swizzle, sizeof(v.swizzle));
   desc_fill(d, img, &v);
   return true;
}

static unsigned
swizzle_code(VkComponentSwizzle s)
{
   switch (s) {
   case VK_COMPONENT_SWIZZLE_ZERO: return NICOGPU_SWIZZLE_ZERO;
   case VK_COMPONENT_SWIZZLE_ONE: return NICOGPU_SWIZZLE_ONE;
   case VK_COMPONENT_SWIZZLE_G: return NICOGPU_SWIZZLE_G;
   case VK_COMPONENT_SWIZZLE_B: return NICOGPU_SWIZZLE_B;
   case VK_COMPONENT_SWIZZLE_A: return NICOGPU_SWIZZLE_A;
   default: return NICOGPU_SWIZZLE_R;
   }
}

static unsigned
view_type_of(const struct ngv_image *img, VkImageViewType t, bool storage)
{
   const bool ms = img->vk.samples != VK_SAMPLE_COUNT_1_BIT;
   switch (t) {
   case VK_IMAGE_VIEW_TYPE_1D: return NICOGPU_VIEW_TYPE_1D;
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return NICOGPU_VIEW_TYPE_1DARRAY;
   case VK_IMAGE_VIEW_TYPE_2D: return ms ? NICOGPU_VIEW_TYPE_2DMS : NICOGPU_VIEW_TYPE_2D;
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? NICOGPU_VIEW_TYPE_2DMSARRAY : NICOGPU_VIEW_TYPE_2DARRAY;
   case VK_IMAGE_VIEW_TYPE_3D: return NICOGPU_VIEW_TYPE_3D;
   case VK_IMAGE_VIEW_TYPE_CUBE: return storage ? NICOGPU_VIEW_TYPE_2DARRAY : NICOGPU_VIEW_TYPE_CUBE;
   default: return storage ? NICOGPU_VIEW_TYPE_2DARRAY : NICOGPU_VIEW_TYPE_CUBEARRAY;
   }
}

static unsigned
min_lod_fix(float v)
{
   if (!(v > 0.0f))
      return 0;
   if (v >= 16.0f)
      return 0xfff;
   return (unsigned)(v * 256.0f + 0.5f);
}

/* Fills the image descriptors of a view; leaves them zero (null descriptor) for combinations the table cannot describe. */
static void
view_fill(struct ngv_image_view *view, const struct ngv_image *img)
{
   memset(view->sampled, 0, sizeof(view->sampled));
   memset(view->storage, 0, sizeof(view->storage));
   const struct ngv_vkfmt *f = img->fmt;
   const struct ngv_vkfmt *vf = ngv_format_lookup(view->vk.view_format);
   if (!f || !vf)
      return;
   const bool mixed = (vf->block != f->block) || (vf->elem != f->elem);
   if (vf->elem != f->elem && vf->block == f->block)
      return; /* not a texel-compatible view */
   const struct vk_image_view *vk = &view->vk;
   const bool is3d = img->vk.image_type == VK_IMAGE_TYPE_3D;
   struct desc_view v = {
      .data_format = vf->data_format,
      .num_format = vf->num_format,
      .order = vf->order,
      .base_level = vk->base_mip_level,
      .level_count = vk->level_count,
      .base_layer = vk->base_array_layer,
      .layer_count = vk->layer_count,
      .min_lod = min_lod_fix(vk->min_lod),
      .mult = 1,
      .block = 1,
   };
   for (unsigned i = 0; i < 4; i++)
      v.swizzle[i] = swizzle_code((&vk->swizzle.r)[i]);
   if (mixed) {
      /* block-texel-view-compatible: one level of the image seen with the other format's element size */
      v.single = true;
      v.level = vk->base_mip_level;
      v.level_count = 1;
      v.block = vf->block;
   }
   if (is3d) {
      if (vk->view_type == VK_IMAGE_VIEW_TYPE_3D) {
         v.base_layer = vk->storage.z_slice_offset;
         v.layer_count = vk->storage.z_slice_count ? vk->storage.z_slice_count
                                                   : MAX2(1u, img->vk.extent.depth >> vk->base_mip_level);
      }
   }
   v.view_type = view_type_of(img, vk->view_type, false);
   desc_fill(view->sampled, img, &v);
   /* storage: identity swizzle, view type without cube, one level */
   struct desc_view s = v;
   memcpy(s.swizzle, identity_swizzle, sizeof(s.swizzle));
   s.view_type = view_type_of(img, vk->view_type, true);
   s.level_count = 1;
   s.min_lod = 0;
   desc_fill(view->storage, img, &s);
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateImageView(VkDevice _device, const VkImageViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_image, image, pCreateInfo->image);
   struct ngv_image_view *view = vk_image_view_create(&device->vk, false, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   view_fill(view, image);
   *pView = ngv_image_view_to_handle(view);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroyImageView(VkDevice _device, VkImageView _view, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_image_view, view, _view);
   if (view)
      vk_image_view_destroy(&device->vk, pAllocator, &view->vk);
}

/* ---- samplers ----------------------------------------------------------------------------------------------------- */

static uint32_t
lod_fix(float v, unsigned bits)
{
   const float lim = (float)((1u << bits) - 1u);
   float x = v * 256.0f;
   if (x < 0.0f)
      x = 0.0f;
   if (x > lim)
      x = lim;
   return (uint32_t)(x + 0.5f);
}

static unsigned
aniso_log2(float a)
{
   if (a < 2.0f)
      return 0;
   if (a < 4.0f)
      return 1;
   if (a < 8.0f)
      return 2;
   if (a < 16.0f)
      return 3;
   return 4;
}

VKAPI_ATTR VkResult VKAPI_CALL
ngv_CreateSampler(VkDevice _device, const VkSamplerCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                  VkSampler *pSampler)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   struct ngv_sampler *sampler = vk_sampler_create(&device->vk, pCreateInfo, pAllocator, sizeof(*sampler));
   if (!sampler)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   uint32_t *d = sampler->desc;
   memset(d, 0, sizeof(sampler->desc));
   nicogpu_sampler_set_mag_filter(d, pCreateInfo->magFilter == VK_FILTER_LINEAR);
   nicogpu_sampler_set_min_filter(d, pCreateInfo->minFilter == VK_FILTER_LINEAR);
   nicogpu_sampler_set_mip_filter(d, pCreateInfo->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR);
   nicogpu_sampler_set_addr_u(d, pCreateInfo->addressModeU);
   nicogpu_sampler_set_addr_v(d, pCreateInfo->addressModeV);
   nicogpu_sampler_set_addr_w(d, pCreateInfo->addressModeW);
   nicogpu_sampler_set_max_aniso_log2(d, pCreateInfo->anisotropyEnable ? aniso_log2(pCreateInfo->maxAnisotropy) : 0);
   nicogpu_sampler_set_compare_enable(d, pCreateInfo->compareEnable);
   nicogpu_sampler_set_compare_op(d, pCreateInfo->compareEnable ? pCreateInfo->compareOp : 0);
   nicogpu_sampler_set_reduction(d, sampler->vk.reduction_mode);
   nicogpu_sampler_set_unnormalized(d, pCreateInfo->unnormalizedCoordinates);
   nicogpu_sampler_set_nonseamless_cube(d, !!(pCreateInfo->flags & VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT));
   unsigned border = NICOGPU_BORDER_MODE_TRANSPARENT_BLACK;
   switch (pCreateInfo->borderColor) {
   case VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK:
   case VK_BORDER_COLOR_INT_OPAQUE_BLACK:
      border = NICOGPU_BORDER_MODE_OPAQUE_BLACK;
      break;
   case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
   case VK_BORDER_COLOR_INT_OPAQUE_WHITE:
      border = NICOGPU_BORDER_MODE_OPAQUE_WHITE;
      break;
   default:
      break;
   }
   nicogpu_sampler_set_border_mode(d, border);
   nicogpu_sampler_set_min_lod(d, lod_fix(pCreateInfo->minLod, 12));
   nicogpu_sampler_set_max_lod(d, pCreateInfo->maxLod >= 16.0f ? 0xfff : lod_fix(pCreateInfo->maxLod, 12));
   /* sfix4.8, 13 bits two's complement, range [-16, 16) */
   float b = pCreateInfo->mipLodBias * 256.0f;
   b = b < -4096.0f ? -4096.0f : b > 4095.0f ? 4095.0f : b;
   const int32_t bi = (int32_t)(b < 0.0f ? b - 0.5f : b + 0.5f);
   nicogpu_sampler_set_lod_bias(d, (uint32_t)bi & 0x1fffu);
   *pSampler = ngv_sampler_to_handle(sampler);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
ngv_DestroySampler(VkDevice _device, VkSampler _sampler, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(ngv_device, device, _device);
   VK_FROM_HANDLE(ngv_sampler, sampler, _sampler);
   if (sampler)
      vk_sampler_destroy(&device->vk, pAllocator, &sampler->vk);
}

/* ---- image format properties -------------------------------------------------------------------------------------- */

VkResult
ngv_image_format_properties(const VkPhysicalDeviceImageFormatInfo2 *info, VkImageFormatProperties2 *props)
{
   VkImageFormatProperties *p = &props->imageFormatProperties;
   *p = (VkImageFormatProperties){0};
   const struct ngv_vkfmt *f = ngv_format_lookup(info->format);
   if (!f || (info->tiling != VK_IMAGE_TILING_OPTIMAL && info->tiling != VK_IMAGE_TILING_LINEAR))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   const VkFormatFeatureFlags feat = ngv_format_image_features(f, info->tiling);
   if (!feat)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   /* usage: no attachments (no graphics queue), no transient, no fragment shading rate / density map ... */
   VkImageUsageFlags usage = info->usage;
   const VkImageUsageFlags supported = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
   if (usage & ~supported)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if ((usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) && !(feat & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if ((usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) && !(feat & VK_FORMAT_FEATURE_TRANSFER_DST_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) && !(feat & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) && !(feat & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   const VkImageCreateFlags ok_flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT |
                                       VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT |
                                       VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT |
                                       VK_IMAGE_CREATE_EXTENDED_USAGE_BIT | VK_IMAGE_CREATE_ALIAS_BIT;
   if (info->flags & ~ok_flags)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if ((info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) && info->type != VK_IMAGE_TYPE_2D)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if ((info->flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT) && info->type != VK_IMAGE_TYPE_3D)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if ((info->flags & VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT) && !(f->flags & NGV_FMT_COMPRESSED))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   const bool linear = info->tiling == VK_IMAGE_TILING_LINEAR;
   switch (info->type) {
   case VK_IMAGE_TYPE_1D:
      if ((f->flags & (NGV_FMT_COMPRESSED | NGV_FMT_DEPTH | NGV_FMT_STENCIL)) || linear)
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      p->maxExtent = (VkExtent3D){NGV_MAX_DIM_1D_2D, 1, 1};
      p->maxMipLevels = 15;
      p->maxArrayLayers = NGV_MAX_LAYERS;
      break;
   case VK_IMAGE_TYPE_2D:
      p->maxExtent = (VkExtent3D){NGV_MAX_DIM_1D_2D, NGV_MAX_DIM_1D_2D, 1};
      p->maxMipLevels = linear ? 1 : 15;
      p->maxArrayLayers = linear ? 1 : NGV_MAX_LAYERS;
      break;
   case VK_IMAGE_TYPE_3D:
      if ((f->flags & (NGV_FMT_DEPTH | NGV_FMT_STENCIL)) || linear || f->elem > 16)
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      p->maxExtent = (VkExtent3D){NGV_MAX_DIM_3D, NGV_MAX_DIM_3D, NGV_MAX_DIM_3D};
      p->maxMipLevels = 12;
      p->maxArrayLayers = 1;
      break;
   default:
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }
   /* the Vulkan CTS (and the spec's intent) tie sampleCounts > 1 to attachment support: formats without COLOR_ATTACHMENT /
    * DEPTH_STENCIL_ATTACHMENT report exactly VK_SAMPLE_COUNT_1_BIT.  This device has no attachments (compute-only, no
    * graphics queue yet) and no multisample storage images, so it reports 1; the multisample tex.* views (2dms) exist in
    * the descriptors but no application can create such an image until P2 adds the graphics queue. */
   p->sampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->maxResourceSize = NGV_MAX_RESOURCE_SIZE;
   vk_foreach_struct(ext, props->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES:
         ((VkSamplerYcbcrConversionImageFormatProperties *)ext)->combinedImageSamplerDescriptorCount = 1;
         break;
      default:
         break;
      }
   }
   return VK_SUCCESS;
}

/* ---- transfer: copies between buffers and images, images and images ------------------------------------------ */

static bool
img_single_aspect(const struct vk_image *img, VkImageAspectFlags aspect)
{
   const VkImageAspectFlags all = vk_format_aspects(img->format);
   return util_is_power_of_two_nonzero(all) && aspect == all;
}

/* Adjacent (src, dst) ranges are merged into one COPY packet. */
struct copy_run {
   uint64_t src, dst, len;
};

static void
run_flush(struct ngv_cmd_buffer *cmd, struct copy_run *r)
{
   if (!r->len)
      return;
   uint32_t *dw = ngv_cmd_reserve(cmd, 8);
   if (dw)
      ngv_cmd_commit(cmd, ngv_emit_copy(dw, r->src, r->dst, r->len));
   r->len = 0;
}

static void
run_add(struct ngv_cmd_buffer *cmd, struct copy_run *r, uint64_t src, uint64_t dst, uint64_t len)
{
   if (r->len && r->src + r->len == src && r->dst + r->len == dst) {
      r->len += len;
      return;
   }
   run_flush(cmd, r);
   *r = (struct copy_run){src, dst, len};
}

/* `rows` rows of `bytes` each between two layouts with the given row strides. */
static void
copy_rows(struct ngv_cmd_buffer *cmd, struct copy_run *run, uint64_t src, uint64_t src_row, uint64_t dst,
          uint64_t dst_row, uint32_t rows, uint64_t bytes)
{
   for (uint32_t y = 0; y < rows; y++)
      run_add(cmd, run, src + y * src_row, dst + y * dst_row, bytes);
}

static bool
xfer_unsupported(struct ngv_cmd_buffer *cmd, bool bad)
{
   if (bad)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return bad;
}

/* Byte address of block (x, y) of slice s of a subresource range: 3D images step through the planes of the level,
 * all other images through the array layers. */
static uint64_t
img_slice_addr(const struct ngv_image *img, const struct img_loc *loc, VkOffset3D off, uint32_t s)
{
   const uint64_t slice = img->vk.image_type == VK_IMAGE_TYPE_3D ? (uint64_t)(off.z + s) * loc->slice
                                                                  : (uint64_t)s * loc->layer;
   return img->va + loc->off + slice + (uint64_t)off.y * loc->row + (uint64_t)off.x * loc->elem;
}

static void
buffer_image_copy(struct ngv_cmd_buffer *cmd, struct ngv_buffer *buf, struct ngv_image *img, uint32_t count,
                  const VkBufferImageCopy2 *regions, bool to_image)
{
   struct copy_run run = {0};
   for (uint32_t i = 0; i < count; i++) {
      const VkBufferImageCopy2 *r = &regions[i];
      const VkImageSubresourceLayers *sub = &r->imageSubresource;
      if (xfer_unsupported(cmd, !img_single_aspect(&img->vk, sub->aspectMask)))
         return;
      const struct vk_image_buffer_layout bl = vk_image_buffer_copy_layout(&img->vk, r);
      const VkExtent3D ext = vk_image_extent_to_elements(&img->vk, r->imageExtent);
      const VkOffset3D off = vk_image_offset_to_elements(&img->vk, r->imageOffset);
      const uint32_t layers = vk_image_subresource_layer_count(&img->vk, sub);
      const uint32_t slices = MAX2(ext.depth, layers);
      const struct img_loc loc = img_loc(img, sub->mipLevel, sub->baseArrayLayer);
      const uint64_t bytes = (uint64_t)ext.width * loc.elem;
      for (uint32_t s = 0; s < slices; s++) {
         const uint64_t ia = img_slice_addr(img, &loc, off, s);
         const uint64_t ba = buf->vk.device_address + r->bufferOffset + (uint64_t)s * bl.image_stride_B;
         if (to_image)
            copy_rows(cmd, &run, ba, bl.row_stride_B, ia, loc.row, ext.height, bytes);
         else
            copy_rows(cmd, &run, ia, loc.row, ba, bl.row_stride_B, ext.height, bytes);
      }
   }
   run_flush(cmd, &run);
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdCopyBufferToImage2(VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2 *info)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_buffer, src, info->srcBuffer);
   VK_FROM_HANDLE(ngv_image, dst, info->dstImage);
   if (dst->fmt)
      ngv_meta_buffer_image(cmd, src, dst, info->regionCount, info->pRegions, true);
   else
      buffer_image_copy(cmd, src, dst, info->regionCount, info->pRegions, true);
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdCopyImageToBuffer2(VkCommandBuffer commandBuffer, const VkCopyImageToBufferInfo2 *info)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_image, src, info->srcImage);
   VK_FROM_HANDLE(ngv_buffer, dst, info->dstBuffer);
   if (src->fmt)
      ngv_meta_buffer_image(cmd, dst, src, info->regionCount, info->pRegions, false);
   else
      buffer_image_copy(cmd, dst, src, info->regionCount, info->pRegions, false);
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdCopyImage2(VkCommandBuffer commandBuffer, const VkCopyImageInfo2 *info)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_image, src, info->srcImage);
   VK_FROM_HANDLE(ngv_image, dst, info->dstImage);
   const struct vk_image *s = &src->vk, *d = &dst->vk;
   if (src->fmt && dst->fmt) {
      ngv_meta_image_image(cmd, src, dst, info->regionCount, info->pRegions);
      return;
   }
   if (src->fmt || dst->fmt) {
      /* hardware layout on one side, byte-linear on the other: no kernel for that */
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   /* same block size and block shape on both sides (compatible formats of one class) */
   const bool shape_differs = vk_format_get_blocksize(s->format) != vk_format_get_blocksize(d->format) ||
                              vk_format_get_blockwidth(s->format) != vk_format_get_blockwidth(d->format) ||
                              vk_format_get_blockheight(s->format) != vk_format_get_blockheight(d->format);
   struct copy_run run = {0};
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageCopy2 *r = &info->pRegions[i];
      if (xfer_unsupported(cmd, shape_differs || !img_single_aspect(s, r->srcSubresource.aspectMask) ||
                                   !img_single_aspect(d, r->dstSubresource.aspectMask)))
         return;
      const VkExtent3D ext = vk_image_extent_to_elements(s, r->extent);
      const VkOffset3D so = vk_image_offset_to_elements(s, r->srcOffset);
      const VkOffset3D dof = vk_image_offset_to_elements(d, r->dstOffset);
      const uint32_t layers = vk_image_subresource_layer_count(s, &r->srcSubresource);
      const uint32_t slices = MAX2(ext.depth, layers);
      const struct img_loc sl = img_loc(src, r->srcSubresource.mipLevel, r->srcSubresource.baseArrayLayer);
      const struct img_loc dl = img_loc(dst, r->dstSubresource.mipLevel, r->dstSubresource.baseArrayLayer);
      const uint64_t bytes = (uint64_t)ext.width * sl.elem;
      for (uint32_t k = 0; k < slices; k++)
         copy_rows(cmd, &run, img_slice_addr(src, &sl, so, k), sl.row, img_slice_addr(dst, &dl, dof, k), dl.row,
                   ext.height, bytes);
   }
   run_flush(cmd, &run);
}
