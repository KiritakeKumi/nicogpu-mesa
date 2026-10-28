/* SPDX-License-Identifier: MIT */
/* ngv_layout.h -- image memory layout (the descriptor spec section 6), C port of descgen.level_layout.
 *
 * Header only, plain C99, no Vulkan / Mesa types: the driver (ngv_image.c) and src/nicogpu/tests/ngv_layout_test.c
 * (which checks this file against the golden vectors descgen generates from the Python reference) both include it.
 *
 *   ngv_layout_compute(): sizes, level offsets, layer stride and alignment of one resource;
 *   elements are texels (uncompressed) or compressed blocks; eb = bytes per element * samples.
 */
#ifndef NGV_LAYOUT_H
#define NGV_LAYOUT_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "nicogpu_desc.h"

#define NGV_LAYOUT_MAX_LEVELS 16

enum ngv_res_type { NGV_RES_1D = 0, NGV_RES_2D = 1, NGV_RES_3D = 2 }; /* == enum nicogpu_res_type */

struct ngv_layout_in {
   enum ngv_res_type res_type;
   bool linear;               /* tiling = linear (levels must be 1, samples 1) */
   uint32_t elem_bytes;       /* bytes per texel / compressed block */
   uint32_t block;            /* block edge in texels (1, or 4 for BC) */
   uint32_t samples_log2;     /* 0..2 */
   uint32_t width, height, depth; /* texels of level 0 (height = 1 for 1D, depth = 1 unless 3D) */
   uint32_t levels;           /* 1..16 */
   uint32_t layers;           /* array layers (1 for 3D) */
};

struct ngv_level {
   uint64_t offset;           /* byte offset of the level inside one layer */
   uint64_t size;             /* bytes of the level */
   uint32_t ew, eh, ed;       /* extent in elements */
   bool tail;                 /* inside the mip tail (micro-tile granularity) */
};

struct ngv_layout {
   struct ngv_level level[NGV_LAYOUT_MAX_LEVELS];
   uint32_t levels;
   uint32_t eb;               /* element bytes including samples */
   uint32_t tile[3], micro[3];/* t64k shapes in elements */
   uint64_t layer_size;       /* bytes of one layer (== layer stride) */
   uint64_t total;            /* bytes of the whole resource */
   uint32_t base_align;       /* required base address alignment */
   uint32_t pitch;            /* linear: row pitch in bytes (multiple of 16) */
   uint64_t lin_stride;       /* linear: bytes between array layers (3D: between slices), multiple of 256; descriptor layer_stride * 256 */
};

static inline uint64_t
ngv_layout_cdiv(uint64_t a, uint64_t b)
{
   return (a + b - 1) / b;
}

static inline uint32_t
ngv_layout_max1(uint32_t v)
{
   return v ? v : 1;
}

/* false: the combination cannot be described (element size not a power of two in 1..64, 3D with more than 16 B per
 * element, linear with several levels / samples, more than 16 levels) */
static inline bool
ngv_layout_compute(const struct ngv_layout_in *in, struct ngv_layout *out)
{
   memset(out, 0, sizeof(*out));
   if (in->levels < 1 || in->levels > NGV_LAYOUT_MAX_LEVELS || in->layers < 1 || in->samples_log2 > 2 ||
       in->elem_bytes == 0 || in->block == 0)
      return false;
   const uint32_t eb = in->elem_bytes << in->samples_log2;
   out->eb = eb;
   out->levels = in->levels;
   const uint32_t w = ngv_layout_max1(in->width);
   const uint32_t h = in->res_type == NGV_RES_1D ? 1 : ngv_layout_max1(in->height);
   const uint32_t d = in->res_type == NGV_RES_3D ? ngv_layout_max1(in->depth) : 1;

   if (in->linear) {
      if (in->levels != 1 || in->samples_log2 != 0 || in->block != 1)
         return false;
      const uint64_t row = (uint64_t)w * in->elem_bytes;
      const uint64_t pitch = (row + 15) & ~(uint64_t)15;
      if (pitch / 16 > 0xffffu)
         return false;
      out->pitch = (uint32_t)pitch;
      out->level[0] = (struct ngv_level){.offset = 0, .size = pitch * h * d, .ew = w, .eh = h, .ed = d, .tail = false};
      /* a 3D image steps through its slices with layer_stride (6.6): every slice starts on a 256 B boundary */
      out->lin_stride = (pitch * h + 255) & ~(uint64_t)255;
      out->layer_size = out->lin_stride * d;
      out->total = out->layer_size * in->layers;
      out->base_align = NICOGPU_LAYOUT_BASE_ALIGN_LINEAR;
      out->tile[0] = out->tile[1] = out->tile[2] = 0;
      return true;
   }

   if (eb == 0 || (eb & (eb - 1)) != 0 || eb > 64)
      return false;
   unsigned idx = 0;
   while ((1u << idx) < eb)
      idx++;
   if (in->res_type == NGV_RES_1D) {
      out->tile[0] = NICOGPU_LAYOUT_TILE_BYTES / eb;
      out->micro[0] = NICOGPU_LAYOUT_MICRO_BYTES / eb;
      out->tile[1] = out->tile[2] = out->micro[1] = out->micro[2] = 1;
   } else if (in->res_type == NGV_RES_2D) {
      out->tile[0] = nicogpu_layout_tile_2d[idx][0];
      out->tile[1] = nicogpu_layout_tile_2d[idx][1];
      out->micro[0] = nicogpu_layout_micro_2d[idx][0];
      out->micro[1] = nicogpu_layout_micro_2d[idx][1];
      out->tile[2] = out->micro[2] = 1;
   } else {
      if (!nicogpu_layout_tile_3d[idx][0])
         return false;
      for (unsigned k = 0; k < 3; k++) {
         out->tile[k] = nicogpu_layout_tile_3d[idx][k];
         out->micro[k] = nicogpu_layout_micro_3d[idx][k];
      }
   }

   uint64_t off = 0;
   bool tail = false;
   for (uint32_t l = 0; l < in->levels; l++) {
      const uint32_t lw = ngv_layout_max1(w >> l);
      const uint32_t lh = in->res_type == NGV_RES_1D ? 1 : ngv_layout_max1(h >> l);
      const uint32_t ld = in->res_type == NGV_RES_3D ? ngv_layout_max1(d >> l) : 1;
      const uint32_t ew = (uint32_t)ngv_layout_cdiv(lw, in->block);
      const uint32_t eh = (uint32_t)ngv_layout_cdiv(lh, in->block);
      const uint32_t ed = ld;
      if (!tail && (ew < out->tile[0] || eh < out->tile[1] || ed < out->tile[2]))
         tail = true;
      uint64_t size;
      if (tail)
         size = ngv_layout_cdiv(ew, out->micro[0]) * ngv_layout_cdiv(eh, out->micro[1]) *
                ngv_layout_cdiv(ed, out->micro[2]) * NICOGPU_LAYOUT_MICRO_BYTES;
      else
         size = ngv_layout_cdiv(ew, out->tile[0]) * ngv_layout_cdiv(eh, out->tile[1]) *
                ngv_layout_cdiv(ed, out->tile[2]) * NICOGPU_LAYOUT_TILE_BYTES;
      out->level[l] = (struct ngv_level){.offset = off, .size = size, .ew = ew, .eh = eh, .ed = ed, .tail = tail};
      off += size;
   }
   const uint64_t gran = out->level[0].tail ? NICOGPU_LAYOUT_MICRO_BYTES : NICOGPU_LAYOUT_TILE_BYTES;
   out->layer_size = ngv_layout_cdiv(off, gran) * gran;
   out->total = out->layer_size * in->layers;
   out->base_align = out->level[0].tail ? NICOGPU_LAYOUT_BASE_ALIGN_LINEAR : NICOGPU_LAYOUT_BASE_ALIGN_T64K;
   return true;
}

#endif /* NGV_LAYOUT_H */
