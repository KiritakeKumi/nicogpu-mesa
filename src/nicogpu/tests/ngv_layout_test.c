/* SPDX-License-Identifier: MIT */
/* ngv_layout_test.c -- host test of ngv_layout.h against the descgen golden vectors.
 *
 *   cc -std=c11 -Wall -Wextra -Werror -I<vulkan dir> -I<dir with nicogpu_desc.h and nicogpu_desc_vectors.h> \
 *      ngv_layout_test.c && ./a.out
 *
 * nicogpu_level_vectors: for every t64k resource of descgen.layout_cases the Python reference level_layout() gives each level's
 * offset, the mip-tail flag and the layer size; ngv_layout_compute() must agree.  Directed cases cover what the vectors do
 * not: linear layouts, multi-layer totals, base alignment and the "cannot be described" refusals.
 * Exit status 0 = all checks passed.
 */
#include <stdio.h>
#include <stdlib.h>

#include "ngv_layout.h"
#include "nicogpu_desc_vectors.h"

static int failures, checks;

#define CHECK(cond, ...)                      \
   do {                                       \
      checks++;                               \
      if (!(cond)) {                          \
         failures++;                          \
         printf("FAIL %s:%d: ", __FILE__, __LINE__); \
         printf(__VA_ARGS__);                 \
         printf("\n");                        \
      }                                       \
   } while (0)

/* resource geometry from an image descriptor, the way the driver derives it from the VkImageCreateInfo */
static void
in_from_desc(const uint32_t *d, struct ngv_layout_in *in)
{
   const struct nicogpu_data_format_info *df = &nicogpu_data_formats[nicogpu_image_get_data_format(d)];
   memset(in, 0, sizeof(*in));
   in->res_type = (enum ngv_res_type)nicogpu_image_get_res_type(d);
   in->linear = nicogpu_image_get_tiling(d) == NICOGPU_TILING_LINEAR;
   in->elem_bytes = df->bpe / 8u;
   in->block = df->block;
   in->samples_log2 = (uint32_t)nicogpu_image_get_samples_log2(d);
   in->width = (uint32_t)nicogpu_image_get_width_m1(d) + 1;
   in->height = (uint32_t)nicogpu_image_get_height_m1(d) + 1;
   in->depth = in->res_type == NGV_RES_3D ? (uint32_t)nicogpu_image_get_depth_m1(d) + 1 : 1;
   in->levels = (uint32_t)nicogpu_image_get_levels_m1(d) + 1;
   in->layers = 1;
}

static void
test_vectors(void)
{
   const size_t n = sizeof(nicogpu_level_vectors) / sizeof(nicogpu_level_vectors[0]);
   CHECK(n > 40, "too few level vectors (%zu)", n);
   for (size_t i = 0; i < n; i++) {
      const struct nicogpu_level_vector *v = &nicogpu_level_vectors[i];
      struct ngv_layout_in in;
      struct ngv_layout lay;
      in_from_desc(v->desc, &in);
      CHECK(ngv_layout_compute(&in, &lay), "vector %zu: compute refused", i);
      CHECK(v->level < lay.levels, "vector %zu: level %u out of range", i, v->level);
      if (v->level >= lay.levels)
         continue;
      CHECK(lay.level[v->level].offset == v->offset, "vector %zu level %u: offset %llx != %llx", i, v->level,
            (unsigned long long)lay.level[v->level].offset, (unsigned long long)v->offset);
      CHECK((uint32_t)lay.level[v->level].tail == v->tail, "vector %zu level %u: tail %d != %u", i, v->level,
            lay.level[v->level].tail, v->tail);
      CHECK(lay.layer_size == v->layer_size, "vector %zu: layer size %llx != %llx", i,
            (unsigned long long)lay.layer_size, (unsigned long long)v->layer_size);
   }
}

static struct ngv_layout_in
mk(enum ngv_res_type rt, uint32_t eb, uint32_t w, uint32_t h, uint32_t d, uint32_t levels, uint32_t layers)
{
   struct ngv_layout_in in;
   memset(&in, 0, sizeof(in));
   in.res_type = rt;
   in.elem_bytes = eb;
   in.block = 1;
   in.width = w;
   in.height = h;
   in.depth = d;
   in.levels = levels;
   in.layers = layers;
   return in;
}

static void
test_directed(void)
{
   struct ngv_layout lay;
   /* 2D RGBA8, one 128x128 tile: 64 KiB, one layer; level 0 not in the tail -> 64 KiB alignment */
   struct ngv_layout_in in = mk(NGV_RES_2D, 4, 128, 128, 1, 1, 1);
   CHECK(ngv_layout_compute(&in, &lay), "128x128");
   CHECK(lay.layer_size == 65536 && lay.total == 65536 && lay.base_align == 65536 && !lay.level[0].tail, "128x128 sizes");
   /* 16 layers multiply the total only */
   in.layers = 16;
   CHECK(ngv_layout_compute(&in, &lay) && lay.total == 16u * 65536u && lay.layer_size == 65536, "16 layers");
   /* an image smaller than a tile is all mip tail: 256 B granularity and alignment (1x1 RGBA8 = one micro-tile) */
   in = mk(NGV_RES_2D, 4, 1, 1, 1, 1, 1);
   CHECK(ngv_layout_compute(&in, &lay) && lay.layer_size == 256 && lay.base_align == 256 && lay.level[0].tail, "1x1");
   /* 16x16 RGBA8 with a full chain: every level in the tail, 4 levels of micro-tiles (4 + 1 + 1 + 1 + 1 = ... see below) */
   in = mk(NGV_RES_2D, 4, 16, 16, 1, 5, 1);
   CHECK(ngv_layout_compute(&in, &lay), "16x16 chain");
   CHECK(lay.level[0].size == 4 * 256 && lay.level[1].offset == 4 * 256 && lay.level[4].tail, "16x16 chain levels");
   /* multisample: 4x RGBA8 is a 16-byte element */
   in = mk(NGV_RES_2D, 4, 64, 64, 1, 1, 1);
   in.samples_log2 = 2;
   CHECK(ngv_layout_compute(&in, &lay) && lay.eb == 16 && lay.tile[0] == 64 && lay.tile[1] == 64, "4x MS tile shape");
   /* 1D: (65536 / eb) x 1 tiles */
   in = mk(NGV_RES_1D, 4, 70000, 1, 1, 1, 1);
   CHECK(ngv_layout_compute(&in, &lay) && lay.tile[0] == 16384 && lay.level[0].size == 5u * 65536u, "1D 70000 r32");
   /* 3D: tile 32x32x16 for 4-byte elements */
   in = mk(NGV_RES_3D, 4, 64, 64, 32, 1, 1);
   CHECK(ngv_layout_compute(&in, &lay) && lay.tile[2] == 16 && lay.level[0].size == 2u * 2u * 2u * 65536u, "3D tiles");
   /* linear: pitch = row rounded up to 16 B, layer = rows * pitch rounded up to 256 B */
   in = mk(NGV_RES_2D, 4, 100, 50, 1, 1, 3);
   in.linear = true;
   CHECK(ngv_layout_compute(&in, &lay) && lay.pitch == 400 && lay.layer_size == 20224 && lay.total == 3u * 20224u &&
            lay.base_align == 256,
         "linear 100x50");
   in = mk(NGV_RES_2D, 1, 5, 7, 1, 1, 1);
   in.linear = true;
   CHECK(ngv_layout_compute(&in, &lay) && lay.pitch == 16 && lay.layer_size == 256, "linear 5x7 r8");
 /* linear 3D: every slice starts on a 256 B boundary and layer_stride is the slice stride (the descriptor spec section 6.6) */
   in = mk(NGV_RES_3D, 4, 10, 3, 5, 1, 1);
   in.linear = true;
   CHECK(ngv_layout_compute(&in, &lay) && lay.pitch == 48 && lay.lin_stride == 256 && lay.layer_size == 5u * 256u,
         "linear 3D slice stride");
   in = mk(NGV_RES_2D, 4, 100, 50, 1, 1, 3);
   in.linear = true;
   CHECK(ngv_layout_compute(&in, &lay) && lay.lin_stride == 20224, "linear 2D layer stride");
   /* refusals */
   in = mk(NGV_RES_2D, 4, 64, 64, 1, 2, 1);
   in.linear = true;
   CHECK(!ngv_layout_compute(&in, &lay), "linear with 2 levels must be refused");
   in = mk(NGV_RES_2D, 4, 64, 64, 1, 1, 1);
   in.linear = true;
   in.samples_log2 = 1;
   CHECK(!ngv_layout_compute(&in, &lay), "linear multisample must be refused");
   in = mk(NGV_RES_2D, 3, 64, 64, 1, 1, 1);
   CHECK(!ngv_layout_compute(&in, &lay), "3-byte element in t64k must be refused");
   in = mk(NGV_RES_3D, 32, 8, 8, 8, 1, 1);
   CHECK(!ngv_layout_compute(&in, &lay), "3D with 32-byte elements must be refused");
   in = mk(NGV_RES_2D, 4, 64, 64, 1, 17, 1);
   CHECK(!ngv_layout_compute(&in, &lay), "17 levels must be refused");
   in = mk(NGV_RES_2D, 4, 64, 64, 1, 1, 0);
   CHECK(!ngv_layout_compute(&in, &lay), "0 layers must be refused");
   /* the descriptor's linear pitch field is 16 bits of 16-byte units: 40000 x 32 B = 80000 units cannot be described */
   in = mk(NGV_RES_2D, 32, 40000, 2, 1, 1, 1);
   in.linear = true;
   CHECK(!ngv_layout_compute(&in, &lay), "linear pitch beyond 16 bits of 16 B units must be refused");
   in = mk(NGV_RES_2D, 32, 2000, 2, 1, 1, 1);
   in.linear = true;
   CHECK(ngv_layout_compute(&in, &lay) && lay.pitch == 64000, "linear pitch 64000 B is the 16-bit field's range");
   /* t64k elements are 1..64 B */
   in = mk(NGV_RES_2D, 128, 64, 64, 1, 1, 1);
   CHECK(!ngv_layout_compute(&in, &lay), "128-byte element must be refused");
   /* linear images are never block compressed */
   in = mk(NGV_RES_2D, 8, 64, 64, 1, 1, 1);
   in.linear = true;
   in.block = 4;
   CHECK(!ngv_layout_compute(&in, &lay), "linear block-compressed must be refused");
   /* a compressed block format: BC1 (8-byte blocks of 4x4), 256x256 texels = 64x64 blocks = one 8-byte tile of 128x64 blocks */
   in = mk(NGV_RES_2D, 8, 256, 256, 1, 1, 1);
   in.block = 4;
   CHECK(ngv_layout_compute(&in, &lay) && lay.level[0].ew == 64 && lay.level[0].eh == 64, "BC1 block extents");
}

int
main(void)
{
   test_vectors();
   test_directed();
   printf("ngv_layout_test: %d checks, %d failures\n", checks, failures);
   return failures ? 1 : 0;
}
