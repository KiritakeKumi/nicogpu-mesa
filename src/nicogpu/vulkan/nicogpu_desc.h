/* SPDX-License-Identifier: MIT */
/* Generated file: image descriptor constants and hardware format enums (table v0.1).  Do not edit; regenerate from the format description source. */
#ifndef NICOGPU_DESC_H
#define NICOGPU_DESC_H

#include <stdbool.h>
#include <stdint.h>

#define NICOGPU_DESC_VERSION "0.1"
#define NICOGPU_DESC_VA_BITS 48

enum nicogpu_filter {
  NICOGPU_FILTER_NEAREST = 0,
  NICOGPU_FILTER_LINEAR = 1,
};
enum nicogpu_address_mode {
  NICOGPU_ADDRESS_MODE_REPEAT = 0,
  NICOGPU_ADDRESS_MODE_MIRRORED_REPEAT = 1,
  NICOGPU_ADDRESS_MODE_CLAMP_TO_EDGE = 2,
  NICOGPU_ADDRESS_MODE_CLAMP_TO_BORDER = 3,
  NICOGPU_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE = 4,
  NICOGPU_ADDRESS_MODE_GL_CLAMP = 5,
};
enum nicogpu_compare_op {
  NICOGPU_COMPARE_OP_NEVER = 0,
  NICOGPU_COMPARE_OP_LESS = 1,
  NICOGPU_COMPARE_OP_EQUAL = 2,
  NICOGPU_COMPARE_OP_LESS_OR_EQUAL = 3,
  NICOGPU_COMPARE_OP_GREATER = 4,
  NICOGPU_COMPARE_OP_NOT_EQUAL = 5,
  NICOGPU_COMPARE_OP_GREATER_OR_EQUAL = 6,
  NICOGPU_COMPARE_OP_ALWAYS = 7,
};
enum nicogpu_reduction {
  NICOGPU_REDUCTION_WEIGHTED_AVERAGE = 0,
  NICOGPU_REDUCTION_MIN = 1,
  NICOGPU_REDUCTION_MAX = 2,
};
enum nicogpu_border_mode {
  NICOGPU_BORDER_MODE_TRANSPARENT_BLACK = 0,
  NICOGPU_BORDER_MODE_OPAQUE_BLACK = 1,
  NICOGPU_BORDER_MODE_OPAQUE_WHITE = 2,
  NICOGPU_BORDER_MODE_CUSTOM = 3,
};
enum nicogpu_num_format {
  NICOGPU_NUM_FORMAT_UNORM = 0,
  NICOGPU_NUM_FORMAT_SNORM = 1,
  NICOGPU_NUM_FORMAT_USCALED = 2,
  NICOGPU_NUM_FORMAT_SSCALED = 3,
  NICOGPU_NUM_FORMAT_UINT = 4,
  NICOGPU_NUM_FORMAT_SINT = 5,
  NICOGPU_NUM_FORMAT_FLOAT = 6,
  NICOGPU_NUM_FORMAT_SRGB = 7,
};
enum nicogpu_order {
  NICOGPU_ORDER_RGBA = 0,
  NICOGPU_ORDER_BGRA = 1,
  NICOGPU_ORDER_ABGR = 2,
  NICOGPU_ORDER_ARGB = 3,
  NICOGPU_ORDER_GR = 4,
  NICOGPU_ORDER_A = 5,
};
enum nicogpu_swizzle {
  NICOGPU_SWIZZLE_R = 0,
  NICOGPU_SWIZZLE_G = 1,
  NICOGPU_SWIZZLE_B = 2,
  NICOGPU_SWIZZLE_A = 3,
  NICOGPU_SWIZZLE_ZERO = 4,
  NICOGPU_SWIZZLE_ONE = 5,
};
enum nicogpu_view_type {
  NICOGPU_VIEW_TYPE_1D = 0,
  NICOGPU_VIEW_TYPE_2D = 1,
  NICOGPU_VIEW_TYPE_3D = 2,
  NICOGPU_VIEW_TYPE_CUBE = 3,
  NICOGPU_VIEW_TYPE_1DARRAY = 4,
  NICOGPU_VIEW_TYPE_2DARRAY = 5,
  NICOGPU_VIEW_TYPE_CUBEARRAY = 6,
  NICOGPU_VIEW_TYPE_2DMS = 7,
  NICOGPU_VIEW_TYPE_2DMSARRAY = 8,
  NICOGPU_VIEW_TYPE_BUFFER = 9,
};
enum nicogpu_res_type {
  NICOGPU_RES_TYPE_1D = 0,
  NICOGPU_RES_TYPE_2D = 1,
  NICOGPU_RES_TYPE_3D = 2,
};
enum nicogpu_tiling {
  NICOGPU_TILING_LINEAR = 0,
  NICOGPU_TILING_T64K = 1,
};
enum nicogpu_data_format {
  NICOGPU_DATA_FORMAT_INVALID = 0,
  NICOGPU_DATA_FORMAT_R8 = 1,
  NICOGPU_DATA_FORMAT_R8G8 = 2,
  NICOGPU_DATA_FORMAT_R8G8B8 = 3,
  NICOGPU_DATA_FORMAT_R8G8B8A8 = 4,
  NICOGPU_DATA_FORMAT_R16 = 5,
  NICOGPU_DATA_FORMAT_R16G16 = 6,
  NICOGPU_DATA_FORMAT_R16G16B16 = 7,
  NICOGPU_DATA_FORMAT_R16G16B16A16 = 8,
  NICOGPU_DATA_FORMAT_R32 = 9,
  NICOGPU_DATA_FORMAT_R32G32 = 10,
  NICOGPU_DATA_FORMAT_R32G32B32 = 11,
  NICOGPU_DATA_FORMAT_R32G32B32A32 = 12,
  NICOGPU_DATA_FORMAT_R4G4 = 13,
  NICOGPU_DATA_FORMAT_R4G4B4A4 = 14,
  NICOGPU_DATA_FORMAT_R5G6B5 = 15,
  NICOGPU_DATA_FORMAT_R5G5B5A1 = 16,
  NICOGPU_DATA_FORMAT_R1G5B5A5 = 17,
  NICOGPU_DATA_FORMAT_R10G10B10A2 = 18,
  NICOGPU_DATA_FORMAT_R11G11B10 = 19,
  NICOGPU_DATA_FORMAT_R9G9B9E5 = 20,
  NICOGPU_DATA_FORMAT_D24X8 = 21,
  NICOGPU_DATA_FORMAT_X24S8 = 22,
  NICOGPU_DATA_FORMAT_R64 = 23,
  NICOGPU_DATA_FORMAT_BC1_RGB = 32,
  NICOGPU_DATA_FORMAT_BC1_RGBA = 33,
  NICOGPU_DATA_FORMAT_BC2 = 34,
  NICOGPU_DATA_FORMAT_BC3 = 35,
  NICOGPU_DATA_FORMAT_BC4 = 36,
  NICOGPU_DATA_FORMAT_BC5 = 37,
  NICOGPU_DATA_FORMAT_BC6H_UF16 = 38,
  NICOGPU_DATA_FORMAT_BC6H_SF16 = 39,
  NICOGPU_DATA_FORMAT_BC7 = 40,
};

/* ---- buffer descriptor (4 dwords) ---- */
#define NICOGPU_BUFFER_DWORDS 4
static const uint32_t nicogpu_buffer_reserved_mask[4] = {0x00000000u, 0xffff0000u, 0x00000000u, 0xffffffffu};
#define NICOGPU_BUFFER_BASE_VA_BITS 48
static inline uint64_t nicogpu_buffer_get_base_va(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 0) & 0xffffffffu) << 0;
  v |= (uint64_t)((d[1] >> 0) & 0xffffu) << 32;
  return v;
}
static inline void nicogpu_buffer_set_base_va(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0xffffffffu) | (uint32_t)((v & 0xffffffffu) << 0);
  v >>= 32;
  d[1] = (d[1] & ~0x0000ffffu) | (uint32_t)((v & 0xffffu) << 0);
  v >>= 16;
  (void)v;
}
#define NICOGPU_BUFFER_SIZE_BITS 32
#define NICOGPU_BUFFER_SIZE_DW 2
#define NICOGPU_BUFFER_SIZE_SHIFT 0
#define NICOGPU_BUFFER_SIZE_MASK 0xffffffffu
static inline uint64_t nicogpu_buffer_get_size(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[2] >> 0) & 0xffffffffu) << 0;
  return v;
}
static inline void nicogpu_buffer_set_size(uint32_t *d, uint64_t v)
{
  d[2] = (d[2] & ~0xffffffffu) | (uint32_t)((v & 0xffffffffu) << 0);
  v >>= 32;
  (void)v;
}
static inline bool nicogpu_buffer_is_zero(const uint32_t *d)
{
  uint32_t o = 0;
  for (int i = 0; i < 4; ++i) o |= d[i];
  return o == 0;
}
static inline bool nicogpu_buffer_reserved_ok(const uint32_t *d)
{
  for (int i = 0; i < 4; ++i)
    if (d[i] & nicogpu_buffer_reserved_mask[i]) return false;
  return true;
}

/* ---- sampler descriptor (4 dwords) ---- */
#define NICOGPU_SAMPLER_DWORDS 4
static const uint32_t nicogpu_sampler_reserved_mask[4] = {0xfc000008u, 0xff000000u, 0x0000e000u, 0xf0000000u};
#define NICOGPU_SAMPLER_MAG_FILTER_BITS 1
#define NICOGPU_SAMPLER_MAG_FILTER_DW 0
#define NICOGPU_SAMPLER_MAG_FILTER_SHIFT 0
#define NICOGPU_SAMPLER_MAG_FILTER_MASK 0x00000001u
static inline uint64_t nicogpu_sampler_get_mag_filter(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 0) & 0x1u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_mag_filter(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x00000001u) | (uint32_t)((v & 0x1u) << 0);
  v >>= 1;
  (void)v;
}
#define NICOGPU_SAMPLER_MIN_FILTER_BITS 1
#define NICOGPU_SAMPLER_MIN_FILTER_DW 0
#define NICOGPU_SAMPLER_MIN_FILTER_SHIFT 1
#define NICOGPU_SAMPLER_MIN_FILTER_MASK 0x00000002u
static inline uint64_t nicogpu_sampler_get_min_filter(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 1) & 0x1u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_min_filter(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x00000002u) | (uint32_t)((v & 0x1u) << 1);
  v >>= 1;
  (void)v;
}
#define NICOGPU_SAMPLER_MIP_FILTER_BITS 1
#define NICOGPU_SAMPLER_MIP_FILTER_DW 0
#define NICOGPU_SAMPLER_MIP_FILTER_SHIFT 2
#define NICOGPU_SAMPLER_MIP_FILTER_MASK 0x00000004u
static inline uint64_t nicogpu_sampler_get_mip_filter(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 2) & 0x1u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_mip_filter(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x00000004u) | (uint32_t)((v & 0x1u) << 2);
  v >>= 1;
  (void)v;
}
#define NICOGPU_SAMPLER_ADDR_U_BITS 3
#define NICOGPU_SAMPLER_ADDR_U_DW 0
#define NICOGPU_SAMPLER_ADDR_U_SHIFT 4
#define NICOGPU_SAMPLER_ADDR_U_MASK 0x00000070u
static inline uint64_t nicogpu_sampler_get_addr_u(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 4) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_addr_u(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x00000070u) | (uint32_t)((v & 0x7u) << 4);
  v >>= 3;
  (void)v;
}
#define NICOGPU_SAMPLER_ADDR_V_BITS 3
#define NICOGPU_SAMPLER_ADDR_V_DW 0
#define NICOGPU_SAMPLER_ADDR_V_SHIFT 7
#define NICOGPU_SAMPLER_ADDR_V_MASK 0x00000380u
static inline uint64_t nicogpu_sampler_get_addr_v(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 7) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_addr_v(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x00000380u) | (uint32_t)((v & 0x7u) << 7);
  v >>= 3;
  (void)v;
}
#define NICOGPU_SAMPLER_ADDR_W_BITS 3
#define NICOGPU_SAMPLER_ADDR_W_DW 0
#define NICOGPU_SAMPLER_ADDR_W_SHIFT 10
#define NICOGPU_SAMPLER_ADDR_W_MASK 0x00001c00u
static inline uint64_t nicogpu_sampler_get_addr_w(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 10) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_addr_w(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x00001c00u) | (uint32_t)((v & 0x7u) << 10);
  v >>= 3;
  (void)v;
}
#define NICOGPU_SAMPLER_MAX_ANISO_LOG2_BITS 3
#define NICOGPU_SAMPLER_MAX_ANISO_LOG2_DW 0
#define NICOGPU_SAMPLER_MAX_ANISO_LOG2_SHIFT 13
#define NICOGPU_SAMPLER_MAX_ANISO_LOG2_MASK 0x0000e000u
static inline uint64_t nicogpu_sampler_get_max_aniso_log2(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 13) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_max_aniso_log2(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x0000e000u) | (uint32_t)((v & 0x7u) << 13);
  v >>= 3;
  (void)v;
}
#define NICOGPU_SAMPLER_COMPARE_ENABLE_BITS 1
#define NICOGPU_SAMPLER_COMPARE_ENABLE_DW 0
#define NICOGPU_SAMPLER_COMPARE_ENABLE_SHIFT 16
#define NICOGPU_SAMPLER_COMPARE_ENABLE_MASK 0x00010000u
static inline uint64_t nicogpu_sampler_get_compare_enable(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 16) & 0x1u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_compare_enable(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x00010000u) | (uint32_t)((v & 0x1u) << 16);
  v >>= 1;
  (void)v;
}
#define NICOGPU_SAMPLER_COMPARE_OP_BITS 3
#define NICOGPU_SAMPLER_COMPARE_OP_DW 0
#define NICOGPU_SAMPLER_COMPARE_OP_SHIFT 17
#define NICOGPU_SAMPLER_COMPARE_OP_MASK 0x000e0000u
static inline uint64_t nicogpu_sampler_get_compare_op(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 17) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_compare_op(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x000e0000u) | (uint32_t)((v & 0x7u) << 17);
  v >>= 3;
  (void)v;
}
#define NICOGPU_SAMPLER_REDUCTION_BITS 2
#define NICOGPU_SAMPLER_REDUCTION_DW 0
#define NICOGPU_SAMPLER_REDUCTION_SHIFT 20
#define NICOGPU_SAMPLER_REDUCTION_MASK 0x00300000u
static inline uint64_t nicogpu_sampler_get_reduction(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 20) & 0x3u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_reduction(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x00300000u) | (uint32_t)((v & 0x3u) << 20);
  v >>= 2;
  (void)v;
}
#define NICOGPU_SAMPLER_UNNORMALIZED_BITS 1
#define NICOGPU_SAMPLER_UNNORMALIZED_DW 0
#define NICOGPU_SAMPLER_UNNORMALIZED_SHIFT 22
#define NICOGPU_SAMPLER_UNNORMALIZED_MASK 0x00400000u
static inline uint64_t nicogpu_sampler_get_unnormalized(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 22) & 0x1u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_unnormalized(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x00400000u) | (uint32_t)((v & 0x1u) << 22);
  v >>= 1;
  (void)v;
}
#define NICOGPU_SAMPLER_NONSEAMLESS_CUBE_BITS 1
#define NICOGPU_SAMPLER_NONSEAMLESS_CUBE_DW 0
#define NICOGPU_SAMPLER_NONSEAMLESS_CUBE_SHIFT 23
#define NICOGPU_SAMPLER_NONSEAMLESS_CUBE_MASK 0x00800000u
static inline uint64_t nicogpu_sampler_get_nonseamless_cube(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 23) & 0x1u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_nonseamless_cube(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x00800000u) | (uint32_t)((v & 0x1u) << 23);
  v >>= 1;
  (void)v;
}
#define NICOGPU_SAMPLER_BORDER_MODE_BITS 2
#define NICOGPU_SAMPLER_BORDER_MODE_DW 0
#define NICOGPU_SAMPLER_BORDER_MODE_SHIFT 24
#define NICOGPU_SAMPLER_BORDER_MODE_MASK 0x03000000u
static inline uint64_t nicogpu_sampler_get_border_mode(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 24) & 0x3u) << 0;
  return v;
}
static inline void nicogpu_sampler_set_border_mode(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0x03000000u) | (uint32_t)((v & 0x3u) << 24);
  v >>= 2;
  (void)v;
}
#define NICOGPU_SAMPLER_MIN_LOD_BITS 12
#define NICOGPU_SAMPLER_MIN_LOD_DW 1
#define NICOGPU_SAMPLER_MIN_LOD_SHIFT 0
#define NICOGPU_SAMPLER_MIN_LOD_MASK 0x00000fffu
static inline uint64_t nicogpu_sampler_get_min_lod(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[1] >> 0) & 0xfffu) << 0;
  return v;
}
static inline void nicogpu_sampler_set_min_lod(uint32_t *d, uint64_t v)
{
  d[1] = (d[1] & ~0x00000fffu) | (uint32_t)((v & 0xfffu) << 0);
  v >>= 12;
  (void)v;
}
#define NICOGPU_SAMPLER_MAX_LOD_BITS 12
#define NICOGPU_SAMPLER_MAX_LOD_DW 1
#define NICOGPU_SAMPLER_MAX_LOD_SHIFT 12
#define NICOGPU_SAMPLER_MAX_LOD_MASK 0x00fff000u
static inline uint64_t nicogpu_sampler_get_max_lod(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[1] >> 12) & 0xfffu) << 0;
  return v;
}
static inline void nicogpu_sampler_set_max_lod(uint32_t *d, uint64_t v)
{
  d[1] = (d[1] & ~0x00fff000u) | (uint32_t)((v & 0xfffu) << 12);
  v >>= 12;
  (void)v;
}
#define NICOGPU_SAMPLER_LOD_BIAS_BITS 13
#define NICOGPU_SAMPLER_LOD_BIAS_DW 2
#define NICOGPU_SAMPLER_LOD_BIAS_SHIFT 0
#define NICOGPU_SAMPLER_LOD_BIAS_MASK 0x00001fffu
static inline uint64_t nicogpu_sampler_get_lod_bias(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[2] >> 0) & 0x1fffu) << 0;
  return v;
}
static inline void nicogpu_sampler_set_lod_bias(uint32_t *d, uint64_t v)
{
  d[2] = (d[2] & ~0x00001fffu) | (uint32_t)((v & 0x1fffu) << 0);
  v >>= 13;
  (void)v;
}
#define NICOGPU_SAMPLER_BORDER_VA_BITS 44
static inline uint64_t nicogpu_sampler_get_border_va(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[2] >> 16) & 0xffffu) << 0;
  v |= (uint64_t)((d[3] >> 0) & 0xfffffffu) << 16;
  return v;
}
static inline void nicogpu_sampler_set_border_va(uint32_t *d, uint64_t v)
{
  d[2] = (d[2] & ~0xffff0000u) | (uint32_t)((v & 0xffffu) << 16);
  v >>= 16;
  d[3] = (d[3] & ~0x0fffffffu) | (uint32_t)((v & 0xfffffffu) << 0);
  v >>= 28;
  (void)v;
}
static inline bool nicogpu_sampler_is_zero(const uint32_t *d)
{
  uint32_t o = 0;
  for (int i = 0; i < 4; ++i) o |= d[i];
  return o == 0;
}
static inline bool nicogpu_sampler_reserved_ok(const uint32_t *d)
{
  for (int i = 0; i < 4; ++i)
    if (d[i] & nicogpu_sampler_reserved_mask[i]) return false;
  return true;
}

/* ---- image descriptor (8 dwords) ---- */
#define NICOGPU_IMAGE_DWORDS 8
static const uint32_t nicogpu_image_reserved_mask[8] = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0xffffffffu};
#define NICOGPU_IMAGE_BASE_VA_BITS 48
static inline uint64_t nicogpu_image_get_base_va(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 0) & 0xffffffffu) << 0;
  v |= (uint64_t)((d[1] >> 0) & 0xffffu) << 32;
  return v;
}
static inline void nicogpu_image_set_base_va(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0xffffffffu) | (uint32_t)((v & 0xffffffffu) << 0);
  v >>= 32;
  d[1] = (d[1] & ~0x0000ffffu) | (uint32_t)((v & 0xffffu) << 0);
  v >>= 16;
  (void)v;
}
#define NICOGPU_IMAGE_DATA_FORMAT_BITS 6
#define NICOGPU_IMAGE_DATA_FORMAT_DW 1
#define NICOGPU_IMAGE_DATA_FORMAT_SHIFT 16
#define NICOGPU_IMAGE_DATA_FORMAT_MASK 0x003f0000u
static inline uint64_t nicogpu_image_get_data_format(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[1] >> 16) & 0x3fu) << 0;
  return v;
}
static inline void nicogpu_image_set_data_format(uint32_t *d, uint64_t v)
{
  d[1] = (d[1] & ~0x003f0000u) | (uint32_t)((v & 0x3fu) << 16);
  v >>= 6;
  (void)v;
}
#define NICOGPU_IMAGE_NUM_FORMAT_BITS 3
#define NICOGPU_IMAGE_NUM_FORMAT_DW 1
#define NICOGPU_IMAGE_NUM_FORMAT_SHIFT 22
#define NICOGPU_IMAGE_NUM_FORMAT_MASK 0x01c00000u
static inline uint64_t nicogpu_image_get_num_format(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[1] >> 22) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_image_set_num_format(uint32_t *d, uint64_t v)
{
  d[1] = (d[1] & ~0x01c00000u) | (uint32_t)((v & 0x7u) << 22);
  v >>= 3;
  (void)v;
}
#define NICOGPU_IMAGE_ORDER_BITS 3
#define NICOGPU_IMAGE_ORDER_DW 1
#define NICOGPU_IMAGE_ORDER_SHIFT 25
#define NICOGPU_IMAGE_ORDER_MASK 0x0e000000u
static inline uint64_t nicogpu_image_get_order(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[1] >> 25) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_image_set_order(uint32_t *d, uint64_t v)
{
  d[1] = (d[1] & ~0x0e000000u) | (uint32_t)((v & 0x7u) << 25);
  v >>= 3;
  (void)v;
}
#define NICOGPU_IMAGE_VIEW_TYPE_BITS 4
#define NICOGPU_IMAGE_VIEW_TYPE_DW 1
#define NICOGPU_IMAGE_VIEW_TYPE_SHIFT 28
#define NICOGPU_IMAGE_VIEW_TYPE_MASK 0xf0000000u
static inline uint64_t nicogpu_image_get_view_type(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[1] >> 28) & 0xfu) << 0;
  return v;
}
static inline void nicogpu_image_set_view_type(uint32_t *d, uint64_t v)
{
  d[1] = (d[1] & ~0xf0000000u) | (uint32_t)((v & 0xfu) << 28);
  v >>= 4;
  (void)v;
}
#define NICOGPU_IMAGE_WIDTH_M1_BITS 14
#define NICOGPU_IMAGE_WIDTH_M1_DW 2
#define NICOGPU_IMAGE_WIDTH_M1_SHIFT 0
#define NICOGPU_IMAGE_WIDTH_M1_MASK 0x00003fffu
static inline uint64_t nicogpu_image_get_width_m1(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[2] >> 0) & 0x3fffu) << 0;
  return v;
}
static inline void nicogpu_image_set_width_m1(uint32_t *d, uint64_t v)
{
  d[2] = (d[2] & ~0x00003fffu) | (uint32_t)((v & 0x3fffu) << 0);
  v >>= 14;
  (void)v;
}
#define NICOGPU_IMAGE_HEIGHT_M1_BITS 14
#define NICOGPU_IMAGE_HEIGHT_M1_DW 2
#define NICOGPU_IMAGE_HEIGHT_M1_SHIFT 14
#define NICOGPU_IMAGE_HEIGHT_M1_MASK 0x0fffc000u
static inline uint64_t nicogpu_image_get_height_m1(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[2] >> 14) & 0x3fffu) << 0;
  return v;
}
static inline void nicogpu_image_set_height_m1(uint32_t *d, uint64_t v)
{
  d[2] = (d[2] & ~0x0fffc000u) | (uint32_t)((v & 0x3fffu) << 14);
  v >>= 14;
  (void)v;
}
#define NICOGPU_IMAGE_TILING_BITS 2
#define NICOGPU_IMAGE_TILING_DW 2
#define NICOGPU_IMAGE_TILING_SHIFT 28
#define NICOGPU_IMAGE_TILING_MASK 0x30000000u
static inline uint64_t nicogpu_image_get_tiling(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[2] >> 28) & 0x3u) << 0;
  return v;
}
static inline void nicogpu_image_set_tiling(uint32_t *d, uint64_t v)
{
  d[2] = (d[2] & ~0x30000000u) | (uint32_t)((v & 0x3u) << 28);
  v >>= 2;
  (void)v;
}
#define NICOGPU_IMAGE_SAMPLES_LOG2_BITS 2
#define NICOGPU_IMAGE_SAMPLES_LOG2_DW 2
#define NICOGPU_IMAGE_SAMPLES_LOG2_SHIFT 30
#define NICOGPU_IMAGE_SAMPLES_LOG2_MASK 0xc0000000u
static inline uint64_t nicogpu_image_get_samples_log2(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[2] >> 30) & 0x3u) << 0;
  return v;
}
static inline void nicogpu_image_set_samples_log2(uint32_t *d, uint64_t v)
{
  d[2] = (d[2] & ~0xc0000000u) | (uint32_t)((v & 0x3u) << 30);
  v >>= 2;
  (void)v;
}
#define NICOGPU_IMAGE_DEPTH_M1_BITS 14
#define NICOGPU_IMAGE_DEPTH_M1_DW 3
#define NICOGPU_IMAGE_DEPTH_M1_SHIFT 0
#define NICOGPU_IMAGE_DEPTH_M1_MASK 0x00003fffu
static inline uint64_t nicogpu_image_get_depth_m1(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[3] >> 0) & 0x3fffu) << 0;
  return v;
}
static inline void nicogpu_image_set_depth_m1(uint32_t *d, uint64_t v)
{
  d[3] = (d[3] & ~0x00003fffu) | (uint32_t)((v & 0x3fffu) << 0);
  v >>= 14;
  (void)v;
}
#define NICOGPU_IMAGE_LEVELS_M1_BITS 4
#define NICOGPU_IMAGE_LEVELS_M1_DW 3
#define NICOGPU_IMAGE_LEVELS_M1_SHIFT 14
#define NICOGPU_IMAGE_LEVELS_M1_MASK 0x0003c000u
static inline uint64_t nicogpu_image_get_levels_m1(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[3] >> 14) & 0xfu) << 0;
  return v;
}
static inline void nicogpu_image_set_levels_m1(uint32_t *d, uint64_t v)
{
  d[3] = (d[3] & ~0x0003c000u) | (uint32_t)((v & 0xfu) << 14);
  v >>= 4;
  (void)v;
}
#define NICOGPU_IMAGE_RES_TYPE_BITS 2
#define NICOGPU_IMAGE_RES_TYPE_DW 3
#define NICOGPU_IMAGE_RES_TYPE_SHIFT 18
#define NICOGPU_IMAGE_RES_TYPE_MASK 0x000c0000u
static inline uint64_t nicogpu_image_get_res_type(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[3] >> 18) & 0x3u) << 0;
  return v;
}
static inline void nicogpu_image_set_res_type(uint32_t *d, uint64_t v)
{
  d[3] = (d[3] & ~0x000c0000u) | (uint32_t)((v & 0x3u) << 18);
  v >>= 2;
  (void)v;
}
#define NICOGPU_IMAGE_MIN_LOD_BITS 12
#define NICOGPU_IMAGE_MIN_LOD_DW 3
#define NICOGPU_IMAGE_MIN_LOD_SHIFT 20
#define NICOGPU_IMAGE_MIN_LOD_MASK 0xfff00000u
static inline uint64_t nicogpu_image_get_min_lod(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[3] >> 20) & 0xfffu) << 0;
  return v;
}
static inline void nicogpu_image_set_min_lod(uint32_t *d, uint64_t v)
{
  d[3] = (d[3] & ~0xfff00000u) | (uint32_t)((v & 0xfffu) << 20);
  v >>= 12;
  (void)v;
}
#define NICOGPU_IMAGE_BASE_LAYER_BITS 14
#define NICOGPU_IMAGE_BASE_LAYER_DW 4
#define NICOGPU_IMAGE_BASE_LAYER_SHIFT 0
#define NICOGPU_IMAGE_BASE_LAYER_MASK 0x00003fffu
static inline uint64_t nicogpu_image_get_base_layer(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[4] >> 0) & 0x3fffu) << 0;
  return v;
}
static inline void nicogpu_image_set_base_layer(uint32_t *d, uint64_t v)
{
  d[4] = (d[4] & ~0x00003fffu) | (uint32_t)((v & 0x3fffu) << 0);
  v >>= 14;
  (void)v;
}
#define NICOGPU_IMAGE_LAYER_COUNT_M1_BITS 14
#define NICOGPU_IMAGE_LAYER_COUNT_M1_DW 4
#define NICOGPU_IMAGE_LAYER_COUNT_M1_SHIFT 14
#define NICOGPU_IMAGE_LAYER_COUNT_M1_MASK 0x0fffc000u
static inline uint64_t nicogpu_image_get_layer_count_m1(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[4] >> 14) & 0x3fffu) << 0;
  return v;
}
static inline void nicogpu_image_set_layer_count_m1(uint32_t *d, uint64_t v)
{
  d[4] = (d[4] & ~0x0fffc000u) | (uint32_t)((v & 0x3fffu) << 14);
  v >>= 14;
  (void)v;
}
#define NICOGPU_IMAGE_BASE_LEVEL_BITS 4
#define NICOGPU_IMAGE_BASE_LEVEL_DW 4
#define NICOGPU_IMAGE_BASE_LEVEL_SHIFT 28
#define NICOGPU_IMAGE_BASE_LEVEL_MASK 0xf0000000u
static inline uint64_t nicogpu_image_get_base_level(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[4] >> 28) & 0xfu) << 0;
  return v;
}
static inline void nicogpu_image_set_base_level(uint32_t *d, uint64_t v)
{
  d[4] = (d[4] & ~0xf0000000u) | (uint32_t)((v & 0xfu) << 28);
  v >>= 4;
  (void)v;
}
#define NICOGPU_IMAGE_LEVEL_COUNT_M1_BITS 4
#define NICOGPU_IMAGE_LEVEL_COUNT_M1_DW 5
#define NICOGPU_IMAGE_LEVEL_COUNT_M1_SHIFT 0
#define NICOGPU_IMAGE_LEVEL_COUNT_M1_MASK 0x0000000fu
static inline uint64_t nicogpu_image_get_level_count_m1(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[5] >> 0) & 0xfu) << 0;
  return v;
}
static inline void nicogpu_image_set_level_count_m1(uint32_t *d, uint64_t v)
{
  d[5] = (d[5] & ~0x0000000fu) | (uint32_t)((v & 0xfu) << 0);
  v >>= 4;
  (void)v;
}
#define NICOGPU_IMAGE_SWIZZLE_R_BITS 3
#define NICOGPU_IMAGE_SWIZZLE_R_DW 5
#define NICOGPU_IMAGE_SWIZZLE_R_SHIFT 4
#define NICOGPU_IMAGE_SWIZZLE_R_MASK 0x00000070u
static inline uint64_t nicogpu_image_get_swizzle_r(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[5] >> 4) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_image_set_swizzle_r(uint32_t *d, uint64_t v)
{
  d[5] = (d[5] & ~0x00000070u) | (uint32_t)((v & 0x7u) << 4);
  v >>= 3;
  (void)v;
}
#define NICOGPU_IMAGE_SWIZZLE_G_BITS 3
#define NICOGPU_IMAGE_SWIZZLE_G_DW 5
#define NICOGPU_IMAGE_SWIZZLE_G_SHIFT 7
#define NICOGPU_IMAGE_SWIZZLE_G_MASK 0x00000380u
static inline uint64_t nicogpu_image_get_swizzle_g(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[5] >> 7) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_image_set_swizzle_g(uint32_t *d, uint64_t v)
{
  d[5] = (d[5] & ~0x00000380u) | (uint32_t)((v & 0x7u) << 7);
  v >>= 3;
  (void)v;
}
#define NICOGPU_IMAGE_SWIZZLE_B_BITS 3
#define NICOGPU_IMAGE_SWIZZLE_B_DW 5
#define NICOGPU_IMAGE_SWIZZLE_B_SHIFT 10
#define NICOGPU_IMAGE_SWIZZLE_B_MASK 0x00001c00u
static inline uint64_t nicogpu_image_get_swizzle_b(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[5] >> 10) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_image_set_swizzle_b(uint32_t *d, uint64_t v)
{
  d[5] = (d[5] & ~0x00001c00u) | (uint32_t)((v & 0x7u) << 10);
  v >>= 3;
  (void)v;
}
#define NICOGPU_IMAGE_SWIZZLE_A_BITS 3
#define NICOGPU_IMAGE_SWIZZLE_A_DW 5
#define NICOGPU_IMAGE_SWIZZLE_A_SHIFT 13
#define NICOGPU_IMAGE_SWIZZLE_A_MASK 0x0000e000u
static inline uint64_t nicogpu_image_get_swizzle_a(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[5] >> 13) & 0x7u) << 0;
  return v;
}
static inline void nicogpu_image_set_swizzle_a(uint32_t *d, uint64_t v)
{
  d[5] = (d[5] & ~0x0000e000u) | (uint32_t)((v & 0x7u) << 13);
  v >>= 3;
  (void)v;
}
#define NICOGPU_IMAGE_PITCH_BITS 16
#define NICOGPU_IMAGE_PITCH_DW 5
#define NICOGPU_IMAGE_PITCH_SHIFT 16
#define NICOGPU_IMAGE_PITCH_MASK 0xffff0000u
static inline uint64_t nicogpu_image_get_pitch(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[5] >> 16) & 0xffffu) << 0;
  return v;
}
static inline void nicogpu_image_set_pitch(uint32_t *d, uint64_t v)
{
  d[5] = (d[5] & ~0xffff0000u) | (uint32_t)((v & 0xffffu) << 16);
  v >>= 16;
  (void)v;
}
#define NICOGPU_IMAGE_LAYER_STRIDE_BITS 32
#define NICOGPU_IMAGE_LAYER_STRIDE_DW 6
#define NICOGPU_IMAGE_LAYER_STRIDE_SHIFT 0
#define NICOGPU_IMAGE_LAYER_STRIDE_MASK 0xffffffffu
static inline uint64_t nicogpu_image_get_layer_stride(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[6] >> 0) & 0xffffffffu) << 0;
  return v;
}
static inline void nicogpu_image_set_layer_stride(uint32_t *d, uint64_t v)
{
  d[6] = (d[6] & ~0xffffffffu) | (uint32_t)((v & 0xffffffffu) << 0);
  v >>= 32;
  (void)v;
}
#define NICOGPU_IMAGE_ELEMENTS_M1_BITS 28
#define NICOGPU_IMAGE_ELEMENTS_M1_DW 2
#define NICOGPU_IMAGE_ELEMENTS_M1_SHIFT 0
#define NICOGPU_IMAGE_ELEMENTS_M1_MASK 0x0fffffffu
static inline uint64_t nicogpu_image_get_elements_m1(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[2] >> 0) & 0xfffffffu) << 0;
  return v;
}
static inline void nicogpu_image_set_elements_m1(uint32_t *d, uint64_t v)
{
  d[2] = (d[2] & ~0x0fffffffu) | (uint32_t)((v & 0xfffffffu) << 0);
  v >>= 28;
  (void)v;
}
static inline bool nicogpu_image_is_zero(const uint32_t *d)
{
  uint32_t o = 0;
  for (int i = 0; i < 8; ++i) o |= d[i];
  return o == 0;
}
static inline bool nicogpu_image_reserved_ok(const uint32_t *d)
{
  for (int i = 0; i < 8; ++i)
    if (d[i] & nicogpu_image_reserved_mask[i]) return false;
  return true;
}

/* ---- border_color descriptor (4 dwords) ---- */
#define NICOGPU_BORDER_COLOR_DWORDS 4
static const uint32_t nicogpu_border_color_reserved_mask[4] = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
#define NICOGPU_BORDER_COLOR_R_BITS 32
#define NICOGPU_BORDER_COLOR_R_DW 0
#define NICOGPU_BORDER_COLOR_R_SHIFT 0
#define NICOGPU_BORDER_COLOR_R_MASK 0xffffffffu
static inline uint64_t nicogpu_border_color_get_r(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[0] >> 0) & 0xffffffffu) << 0;
  return v;
}
static inline void nicogpu_border_color_set_r(uint32_t *d, uint64_t v)
{
  d[0] = (d[0] & ~0xffffffffu) | (uint32_t)((v & 0xffffffffu) << 0);
  v >>= 32;
  (void)v;
}
#define NICOGPU_BORDER_COLOR_G_BITS 32
#define NICOGPU_BORDER_COLOR_G_DW 1
#define NICOGPU_BORDER_COLOR_G_SHIFT 0
#define NICOGPU_BORDER_COLOR_G_MASK 0xffffffffu
static inline uint64_t nicogpu_border_color_get_g(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[1] >> 0) & 0xffffffffu) << 0;
  return v;
}
static inline void nicogpu_border_color_set_g(uint32_t *d, uint64_t v)
{
  d[1] = (d[1] & ~0xffffffffu) | (uint32_t)((v & 0xffffffffu) << 0);
  v >>= 32;
  (void)v;
}
#define NICOGPU_BORDER_COLOR_B_BITS 32
#define NICOGPU_BORDER_COLOR_B_DW 2
#define NICOGPU_BORDER_COLOR_B_SHIFT 0
#define NICOGPU_BORDER_COLOR_B_MASK 0xffffffffu
static inline uint64_t nicogpu_border_color_get_b(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[2] >> 0) & 0xffffffffu) << 0;
  return v;
}
static inline void nicogpu_border_color_set_b(uint32_t *d, uint64_t v)
{
  d[2] = (d[2] & ~0xffffffffu) | (uint32_t)((v & 0xffffffffu) << 0);
  v >>= 32;
  (void)v;
}
#define NICOGPU_BORDER_COLOR_A_BITS 32
#define NICOGPU_BORDER_COLOR_A_DW 3
#define NICOGPU_BORDER_COLOR_A_SHIFT 0
#define NICOGPU_BORDER_COLOR_A_MASK 0xffffffffu
static inline uint64_t nicogpu_border_color_get_a(const uint32_t *d)
{
  uint64_t v = 0;
  v |= (uint64_t)((d[3] >> 0) & 0xffffffffu) << 0;
  return v;
}
static inline void nicogpu_border_color_set_a(uint32_t *d, uint64_t v)
{
  d[3] = (d[3] & ~0xffffffffu) | (uint32_t)((v & 0xffffffffu) << 0);
  v >>= 32;
  (void)v;
}
static inline bool nicogpu_border_color_is_zero(const uint32_t *d)
{
  uint32_t o = 0;
  for (int i = 0; i < 4; ++i) o |= d[i];
  return o == 0;
}
static inline bool nicogpu_border_color_reserved_ok(const uint32_t *d)
{
  for (int i = 0; i < 4; ++i)
    if (d[i] & nicogpu_border_color_reserved_mask[i]) return false;
  return true;
}

struct nicogpu_data_format_info {
  const char *name;
  uint8_t valid, bpe, block, nchan, linear_only, optional;
  uint8_t chan_bits[4], chan_lsb[4];
  uint8_t num_mask; /* bit n = enum nicogpu_num_format n allowed */
};
static const struct nicogpu_data_format_info nicogpu_data_formats[64] = {
  {"invalid", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"r8", 1, 8, 1, 1, 0, 0, {8, 0, 0, 0}, {0, 0, 0, 0}, 0xbf},
  {"r8g8", 1, 16, 1, 2, 0, 0, {8, 8, 0, 0}, {0, 8, 0, 0}, 0xbf},
  {"r8g8b8", 1, 24, 1, 3, 1, 0, {8, 8, 8, 0}, {0, 8, 16, 0}, 0xbf},
  {"r8g8b8a8", 1, 32, 1, 4, 0, 0, {8, 8, 8, 8}, {0, 8, 16, 24}, 0xbf},
  {"r16", 1, 16, 1, 1, 0, 0, {16, 0, 0, 0}, {0, 0, 0, 0}, 0x7f},
  {"r16g16", 1, 32, 1, 2, 0, 0, {16, 16, 0, 0}, {0, 16, 0, 0}, 0x7f},
  {"r16g16b16", 1, 48, 1, 3, 1, 0, {16, 16, 16, 0}, {0, 16, 32, 0}, 0x7f},
  {"r16g16b16a16", 1, 64, 1, 4, 0, 0, {16, 16, 16, 16}, {0, 16, 32, 48}, 0x7f},
  {"r32", 1, 32, 1, 1, 0, 0, {32, 0, 0, 0}, {0, 0, 0, 0}, 0x70},
  {"r32g32", 1, 64, 1, 2, 0, 0, {32, 32, 0, 0}, {0, 32, 0, 0}, 0x70},
  {"r32g32b32", 1, 96, 1, 3, 1, 0, {32, 32, 32, 0}, {0, 32, 64, 0}, 0x70},
  {"r32g32b32a32", 1, 128, 1, 4, 0, 0, {32, 32, 32, 32}, {0, 32, 64, 96}, 0x70},
  {"r4g4", 1, 8, 1, 2, 0, 0, {4, 4, 0, 0}, {0, 4, 0, 0}, 0x01},
  {"r4g4b4a4", 1, 16, 1, 4, 0, 0, {4, 4, 4, 4}, {0, 4, 8, 12}, 0x01},
  {"r5g6b5", 1, 16, 1, 3, 0, 0, {5, 6, 5, 0}, {0, 5, 11, 0}, 0x01},
  {"r5g5b5a1", 1, 16, 1, 4, 0, 0, {5, 5, 5, 1}, {0, 5, 10, 15}, 0x01},
  {"r1g5b5a5", 1, 16, 1, 4, 0, 0, {1, 5, 5, 5}, {0, 1, 6, 11}, 0x01},
  {"r10g10b10a2", 1, 32, 1, 4, 0, 0, {10, 10, 10, 2}, {0, 10, 20, 30}, 0x3f},
  {"r11g11b10", 1, 32, 1, 3, 0, 0, {11, 11, 10, 0}, {0, 11, 22, 0}, 0x40},
  {"r9g9b9e5", 1, 32, 1, 3, 0, 0, {9, 9, 9, 0}, {0, 9, 18, 0}, 0x40},
  {"d24x8", 1, 32, 1, 1, 0, 0, {24, 0, 0, 0}, {0, 0, 0, 0}, 0x01},
  {"x24s8", 1, 32, 1, 1, 0, 0, {8, 0, 0, 0}, {24, 0, 0, 0}, 0x10},
  {"r64", 1, 64, 1, 1, 0, 1, {64, 0, 0, 0}, {0, 0, 0, 0}, 0x30},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"bc1_rgb", 1, 64, 4, 3, 0, 0, {8, 8, 8, 0}, {0, 8, 16, 0}, 0x81},
  {"bc1_rgba", 1, 64, 4, 4, 0, 0, {8, 8, 8, 8}, {0, 8, 16, 24}, 0x81},
  {"bc2", 1, 128, 4, 4, 0, 0, {8, 8, 8, 8}, {0, 8, 16, 24}, 0x81},
  {"bc3", 1, 128, 4, 4, 0, 0, {8, 8, 8, 8}, {0, 8, 16, 24}, 0x81},
  {"bc4", 1, 64, 4, 1, 0, 0, {8, 0, 0, 0}, {0, 0, 0, 0}, 0x03},
  {"bc5", 1, 128, 4, 2, 0, 0, {8, 8, 0, 0}, {0, 8, 0, 0}, 0x03},
  {"bc6h_uf16", 1, 128, 4, 3, 0, 0, {16, 16, 16, 0}, {0, 16, 32, 0}, 0x40},
  {"bc6h_sf16", 1, 128, 4, 3, 0, 0, {16, 16, 16, 0}, {0, 16, 32, 0}, 0x40},
  {"bc7", 1, 128, 4, 4, 0, 0, {8, 8, 8, 8}, {0, 8, 16, 24}, 0x81},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
  {"", 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
};

/* order -> logical component (0=R 1=G 2=B 3=A) written by channel c0..c3; 0xff = none */
static const uint8_t nicogpu_order_map[8][4] = {
  {0, 1, 2, 3},
  {2, 1, 0, 3},
  {3, 2, 1, 0},
  {3, 0, 1, 2},
  {1, 0, 0xff, 0xff},
  {3, 0xff, 0xff, 0xff},
  {0xff, 0xff, 0xff, 0xff},
  {0xff, 0xff, 0xff, 0xff},
};

/* element bytes = 1 << index */
static const uint16_t nicogpu_layout_tile_2d[7][2] = {
  {256, 256},
  {256, 128},
  {128, 128},
  {128, 64},
  {64, 64},
  {64, 32},
  {32, 32},
};
/* element bytes = 1 << index */
static const uint16_t nicogpu_layout_micro_2d[7][2] = {
  {16, 16},
  {16, 8},
  {8, 8},
  {8, 4},
  {4, 4},
  {4, 2},
  {2, 2},
};
/* element bytes = 1 << index */
static const uint16_t nicogpu_layout_tile_3d[7][3] = {
  {64, 32, 32},
  {32, 32, 32},
  {32, 32, 16},
  {32, 16, 16},
  {16, 16, 16},
  {0, 0, 0},
  {0, 0, 0},
};
/* element bytes = 1 << index */
static const uint16_t nicogpu_layout_micro_3d[7][3] = {
  {8, 8, 4},
  {8, 4, 4},
  {4, 4, 4},
  {4, 4, 2},
  {4, 2, 2},
  {0, 0, 0},
  {0, 0, 0},
};
#define NICOGPU_LAYOUT_TILE_BYTES 65536u
#define NICOGPU_LAYOUT_MICRO_BYTES 256u
#define NICOGPU_LAYOUT_BASE_ALIGN_T64K 65536u
#define NICOGPU_LAYOUT_BASE_ALIGN_LINEAR 256u
#define NICOGPU_RESIDENCY_NONRESIDENT (1u << 0)
#define NICOGPU_DESC_MEM_BUFFER_SIZE 16u
#define NICOGPU_DESC_MEM_BUFFER_ALIGN 16u
#define NICOGPU_DESC_MEM_SAMPLER_SIZE 16u
#define NICOGPU_DESC_MEM_SAMPLER_ALIGN 16u
#define NICOGPU_DESC_MEM_IMAGE_SIZE 32u
#define NICOGPU_DESC_MEM_IMAGE_ALIGN 32u
#define NICOGPU_DESC_MEM_COMBINED_IMAGE_SAMPLER_SIZE 64u
#define NICOGPU_DESC_MEM_COMBINED_IMAGE_SAMPLER_ALIGN 32u
#define NICOGPU_DESC_MEM_MUTABLE_SIZE 64u
#define NICOGPU_DESC_MEM_MUTABLE_ALIGN 32u

#endif /* NICOGPU_DESC_H */
