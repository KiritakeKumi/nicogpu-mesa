/* SPDX-License-Identifier: MIT */
/* ngv_xo.h -- header check of an ngcomp code object (".xo", container "NGCO" / legacy "XGCO"), ABI 0.1, 0.2 and 0.3
 * (the ISA ABI spec section 7, ngcomp/emit.py). Pure, header-only, no Mesa / HAL dependency, so
 * src/nicogpu/tests/ngv_xo_test.c builds it on any host:
 * gcc -std=c11 -Wall -Wextra -Werror -I src/nicogpu/vulkan src/nicogpu/tests/ngv_xo_test.c \
 *       -o ngv_xo_test && ./ngv_xo_test
 *
 * Header (little endian, 32 bytes): 0x00 magic, 0x04 container version u16, 0x06 flags u16, 0x08 meta_off, 0x0C meta_len,
 * 0x10 code_off, 0x14 code_len, 0x18 abi code (1 = ABI 0.1, 2 = ABI 0.2), 0x1C word:
 *   ABI 0.1: reserved, must be 0;
 *   ABI 0.2, 0.3: stage_mask (nonzero), bit k set iff some segment has stage k (NGV_XO_STAGE_*); bits >= 5 are unknown
 *            and rejected.
 * ABI 0.3 ( R-A) is the version of an object that holds a `routine` segment or relocation records; everything
 * else is still written as 0.2.  A consumer that only runs whole objects (compute / graphics pipelines: ngv_xo_check)
 * accepts 0.1 and 0.2 and refuses 0.3 like any version it does not know; the shader linker (ngv_link.h) is the one that
 * understands routines and relocations and opens objects with ngv_xo_check_ex(allow_abi3 = true).  Every consumer
 * rejects: other version codes, unknown stage bits / codes, a segment table that disagrees with the header mask, a
 * segment kind the version does not define (0.2: prologue / body / epilogue; 0.3 adds routine), relocation records in a
 * 0.2 object, and for ABI 0.1 a nonzero reserved word (ABI section 7: unknown versions and nonzero reserved fields are
 * rejected). */
#ifndef NGV_XO_H
#define NGV_XO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NGV_XO_ABI_0_1 1u
#define NGV_XO_ABI_0_2 2u
#define NGV_XO_ABI_0_3 3u

/* stage codes == bit numbers of the header stage_mask (ABI_STAGES in ngcomp/emit.py) */
#define NGV_XO_STAGE_COMPUTE 0u    /* compute dispatch, u30:u31 = 0 (the only stage of an ABI 0.1 object) */
#define NGV_XO_STAGE_GEOMETRY 1u   /* DRAW-launched geometry dispatch, u30:u31 = draw parameter record VA */
#define NGV_XO_STAGE_FRAGMENT 2u   /* fragment warp (v0.1.7 entry layout, epilogue v35 / v36) */
#define NGV_XO_STAGE_TILE_BEGIN 3u /* tile begin program */
#define NGV_XO_STAGE_TILE_END 4u   /* tile end program */
#define NGV_XO_NUM_STAGES 5u
#define NGV_XO_STAGE_MASK_ALL ((1u << NGV_XO_NUM_STAGES) - 1u)

struct ngv_xo_info {
   uint32_t abi;        /* NGV_XO_ABI_* */
   uint32_t stage_mask; /* 0.1: always COMPUTE only; 0.2: the header word */
   uint32_t meta_off, meta_len, code_off, code_len;
};

/* Flat scan of the JSON metadata for the next occurrence of `pat` (the literal text `"key":`, length pl) at or after
 * *pos.  On a hit *val points past the pattern and any spaces / newlines, *pos is moved one past the hit (so a caller
 * that skips a malformed value just asks again), and true is returned.  Start with *pos = 0. */
static inline bool
ngv_xo_find_key(const char *js, size_t len, const char *pat, size_t pl, size_t *pos, const char **val)
{
   for (size_t i = *pos; i + pl < len; i++) {
      if (memcmp(js + i, pat, pl))
         continue;
      const char *p = js + i + pl;
      while (p < js + len && (*p == ' ' || *p == '\n'))
         p++;
      *pos = i + 1;
      *val = p;
      return true;
   }
   *pos = len;
   return false;
}

/* Largest / first integer value of `"key": <n>` in the (flat-scanned) JSON metadata; false when the key is absent. */
static inline bool
ngv_xo_json_int(const char *js, size_t len, const char *key, bool want_max, uint64_t *out)
{
   char pat[48];
   snprintf(pat, sizeof(pat), "\"%s\":", key);
   size_t pl = strlen(pat);
   bool any = false;
   uint64_t best = 0;
   size_t pos = 0;
   const char *p;
   while (ngv_xo_find_key(js, len, pat, pl, &pos, &p)) {
      char *end;
      uint64_t v = strtoull(p, &end, 10);
      if (end == p)
         continue;
      if (!any || (want_max && v > best))
         best = v;
      any = true;
      if (!want_max)
         break;
   }
   *out = best;
   return any;
}

/* The OR of (1 << stage) over the "stage": <n> fields of the segment table; *bad = a stage code >= NGV_XO_NUM_STAGES. */
static inline uint32_t
ngv_xo_table_stage_mask(const char *js, size_t len, bool *bad)
{
   static const char pat[] = "\"stage\":";
   const size_t pl = sizeof(pat) - 1;
   uint32_t mask = 0;
   *bad = false;
   size_t pos = 0;
   const char *p;
   while (ngv_xo_find_key(js, len, pat, pl, &pos, &p)) {
      char *end;
      unsigned long long v = strtoull(p, &end, 10);
      if (end == p)
         continue;
      if (v >= NGV_XO_NUM_STAGES)
         *bad = true;
      else
         mask |= 1u << v;
   }
   return mask;
}

/* Segment kinds in the segment table (`"segment": "<kind>"`): bit 0 prologue, 1 body, 2 epilogue, 3 routine; bit 31 = a
 * kind this version does not know (or a malformed value). */
#define NGV_XO_KIND_PROLOGUE (1u << 0)
#define NGV_XO_KIND_BODY (1u << 1)
#define NGV_XO_KIND_EPILOGUE (1u << 2)
#define NGV_XO_KIND_ROUTINE (1u << 3)
#define NGV_XO_KIND_UNKNOWN (1u << 31)

static inline uint32_t
ngv_xo_table_kinds(const char *js, size_t len)
{
   static const char pat[] = "\"segment\":";
   const size_t pl = sizeof(pat) - 1;
   static const struct {
      const char *name;
      uint32_t bit;
   } kinds[] = {{"prologue", NGV_XO_KIND_PROLOGUE},
                {"body", NGV_XO_KIND_BODY},
                {"epilogue", NGV_XO_KIND_EPILOGUE},
                {"routine", NGV_XO_KIND_ROUTINE}};
   uint32_t mask = 0;
   const char *e = js + len;
   size_t pos = 0;
   const char *p;
   while (ngv_xo_find_key(js, len, pat, pl, &pos, &p)) {
      uint32_t bit = NGV_XO_KIND_UNKNOWN;
      if (p < e && *p == '"') {
         p++;
         for (unsigned k = 0; k < sizeof(kinds) / sizeof(kinds[0]); k++) {
            size_t n = strlen(kinds[k].name);
            if ((size_t)(e - p) > n && !memcmp(p, kinds[k].name, n) && p[n] == '"')
               bit = kinds[k].bit;
         }
      }
      mask |= bit;
   }
   return mask;
}

/* true when the metadata carries at least one relocation record (`"relocs": [ {...` -- an empty list does not count) */
static inline bool
ngv_xo_has_relocs(const char *js, size_t len)
{
   static const char pat[] = "\"relocs\":";
   const size_t pl = sizeof(pat) - 1;
   const char *e = js + len;
   size_t pos = 0;
   const char *p;
   while (ngv_xo_find_key(js, len, pat, pl, &pos, &p)) {
      if (p < e && *p == '[') {
         p++;
         while (p < e && (*p == ' ' || *p == '\n'))
            p++;
         if (p < e && *p != ']')
            return true;
      } else {
         return true; /* not a list: malformed, treat as present */
      }
   }
   return false;
}

/* Validate the container header and the ABI-version dependent parts.  true = acceptable (info filled in).  ABI 0.3 is
 * accepted only with allow_abi3 (the shader linker); everyone else refuses it as an unknown version. */
static inline bool
ngv_xo_check_ex(const uint8_t *buf, size_t sz, struct ngv_xo_info *info, bool allow_abi3)
{
   if (sz <= 32)
      return false;
   uint32_t h[8];
   memcpy(h, buf, sizeof(h));
   if (memcmp(buf, "NGCO", 4) && memcmp(buf, "XGCO", 4))
      return false;
   if ((uint64_t)h[2] + h[3] > (uint64_t)sz || (uint64_t)h[4] + h[5] > (uint64_t)sz || !h[5])
      return false;
   info->abi = h[6];
   info->meta_off = h[2];
   info->meta_len = h[3];
   info->code_off = h[4];
   info->code_len = h[5];
   if (h[6] == NGV_XO_ABI_0_1) {
      if (h[7] != 0) /* reserved word */
         return false;
      info->stage_mask = 1u << NGV_XO_STAGE_COMPUTE;
      return true;
   }
   if (h[6] != NGV_XO_ABI_0_2 && !(h[6] == NGV_XO_ABI_0_3 && allow_abi3))
      return false;
   if (!h[7] || (h[7] & ~NGV_XO_STAGE_MASK_ALL))
      return false;
   bool bad;
   uint32_t table = ngv_xo_table_stage_mask((const char *)buf + h[2], h[3], &bad);
   /* the writer emits "stage" for every segment, so the table mask must equal the header word exactly */
   if (bad || table != h[7])
      return false;
   uint32_t kinds = ngv_xo_table_kinds((const char *)buf + h[2], h[3]);
   uint32_t known = NGV_XO_KIND_PROLOGUE | NGV_XO_KIND_BODY | NGV_XO_KIND_EPILOGUE |
                    (h[6] == NGV_XO_ABI_0_3 ? NGV_XO_KIND_ROUTINE : 0u);
   if (kinds & ~known)
      return false; /* a segment kind this version does not define */
   if (h[6] == NGV_XO_ABI_0_2 && ngv_xo_has_relocs((const char *)buf + h[2], h[3]))
      return false; /* relocation records need ABI 0.3 */
   info->stage_mask = h[7];
   return true;
}

static inline bool
ngv_xo_check(const uint8_t *buf, size_t sz, struct ngv_xo_info *info)
{
   return ngv_xo_check_ex(buf, sz, info, false);
}

/* A compute pipeline runs only objects whose segments are all compute (every ABI 0.1 object is). */
static inline bool
ngv_xo_is_compute(uint32_t stage_mask)
{
   return stage_mask == (1u << NGV_XO_STAGE_COMPUTE);
}

#endif /* NGV_XO_H */
