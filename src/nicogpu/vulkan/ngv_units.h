/* SPDX-License-Identifier: MIT */
/* ngv_units.h -- the ngcomp `--units` string of a device. Pure, header-only, no Mesa / HAL
 * dependency, so src/nicogpu/tests/ngv_units_test.c builds it on any host.
 *
 * Policy (ngv_units_resolve):
 *   1. $NGV_NGCOMP_UNITS (any non-empty string) wins: "all", "none" or a list of sfu,xlane,fp64,f16cvt;
 *   2. else the RM told us (HAL info_flags NICOGPU_INFO_F_SP_UNITS, from RM CAPS[26:24] with CAPS[27] UNITS_VALID, and
 * CAPS[29] F16CVT from on): the list of the units that are set, "none" for an empty set;
 *   3. else (an RM older than the unit bits, no RM regmap, an old kernel, the stub): the legacy default "sfu" -- what every
 *      shipped rm_sp (b4 .. b11) has.  A clear NICOGPU_INFO_F_SP_UNITS means "unknown", never "none".
 * The result is passed to `driver.py --units` and hashed into the shader-cache key. */
#ifndef NGV_UNITS_H
#define NGV_UNITS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define NGV_SPU_SFU (1u << 0) /* == NICOGPU_SP_UNIT_SFU (static-asserted in ngv_ngcomp.c) */
#define NGV_SPU_XLANE (1u << 1)
#define NGV_SPU_FP64 (1u << 2)
#define NGV_SPU_F16CVT (1u << 3) /* : VALU binary16 <-> binary32 conversions (RM CAPS[29]); ngcomp "f16cvt" */
#define NGV_SPU_ALL (NGV_SPU_SFU | NGV_SPU_XLANE | NGV_SPU_FP64 | NGV_SPU_F16CVT)
#define NGV_LEGACY_UNITS "sfu" /* rm_sp EN_SFU=1, EN_XLANE=0, EN_FP64=0 (b4 .. b11, which report no unit bits) */
#define NGV_UNITS_BUFSZ 24     /* "none" / "sfu,xlane,fp64,f16cvt" + NUL */

/* "none" or the comma list of the set units in the order sfu, xlane, fp64, f16cvt.  Unknown bits are ignored. */
static inline const char *
ngv_units_string(uint32_t sp_units, char buf[NGV_UNITS_BUFSZ])
{
   static const struct {
      uint32_t bit;
      const char *name;
   } units[] = {{NGV_SPU_SFU, "sfu"}, {NGV_SPU_XLANE, "xlane"}, {NGV_SPU_FP64, "fp64"}, {NGV_SPU_F16CVT, "f16cvt"}};
   size_t n = 0;
   buf[0] = 0;
   for (unsigned i = 0; i < sizeof(units) / sizeof(units[0]); i++) {
      if (!(sp_units & units[i].bit))
         continue;
      size_t l = strlen(units[i].name);
      if (n)
         buf[n++] = ',';
      memcpy(buf + n, units[i].name, l + 1);
      n += l;
   }
   if (!n)
      strcpy(buf, "none");
   return buf;
}

/* `env` = $NGV_NGCOMP_UNITS (may be NULL), `reported` = the device told us its units, `sp_units` = NICOGPU_SP_UNIT_* (only
 * looked at when `reported`).  Returns env or buf; *source gets "env" / "device" / "default" (for logs and tests). */
static inline const char *
ngv_units_resolve(const char *env, bool reported, uint32_t sp_units, char buf[NGV_UNITS_BUFSZ], const char **source)
{
   const char *dummy;
   if (!source)
      source = &dummy;
   if (env && *env) {
      *source = "env";
      return env;
   }
   if (reported) {
      *source = "device";
      return ngv_units_string(sp_units, buf);
   }
   *source = "default";
   return NGV_LEGACY_UNITS;
}

/* ---- : unit set -> reported subgroup operations (section 2) ----------------------------------- */

/* The bits of a units string as ngv_units_resolve returns it ("all", "none" or a list of sfu,xlane,fp64,f16cvt; case-insensitive,
 * separators ',' '+' ' ' like ngcomp/units.py parse). Unknown names are ignored here (ngcomp rejects them, so the
 * pipeline fails anyway). */
static inline uint32_t
ngv_units_parse(const char *s)
{
   static const struct {
      uint32_t bit;
      const char *name;
   } units[] = {{NGV_SPU_SFU, "sfu"}, {NGV_SPU_XLANE, "xlane"}, {NGV_SPU_FP64, "fp64"}, {NGV_SPU_F16CVT, "f16cvt"}};
   uint32_t bits = 0;
   char w[8];
   size_t n = 0;
   for (;; s++) {
      const char c = *s;
      if (c && c != ',' && c != '+' && c != ' ') {
         if (n < sizeof(w) - 1)
            w[n++] = (char)(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
         else
            n = sizeof(w); /* too long: no unit name */
         continue;
      }
      if (n && n < sizeof(w)) {
         w[n] = 0;
         if (!strcmp(w, "all"))
            bits |= NGV_SPU_ALL;
         for (unsigned i = 0; i < sizeof(units) / sizeof(units[0]); i++)
            if (!strcmp(w, units[i].name))
               bits |= units[i].bit;
      }
      n = 0;
      if (!c)
         return bits;
   }
}

/* VkSubgroupFeatureFlagBits values (checked against vulkan_core.h in ngv_device.c) */
#define NGV_SG_BASIC 0x01u
#define NGV_SG_VOTE 0x02u
#define NGV_SG_ARITHMETIC 0x04u
#define NGV_SG_BALLOT 0x08u
#define NGV_SG_SHUFFLE 0x10u
#define NGV_SG_SHUFFLE_RELATIVE 0x20u
#define NGV_SG_CLUSTERED 0x40u
#define NGV_SG_QUAD 0x80u

/* subgroupSupportedOperations for a unit set (PROVISIONAL):
 *   - BASIC / VOTE / BALLOT always: elect / vote / ballot / masks are EXEC + SALU code in ngcomp, and the broadcasts
 *     (read_first / read_invocation, vote_ieq's read_first) have an XLANE-free bit-serial form (~190 instructions);
 *   - SHUFFLE / SHUFFLE_RELATIVE / QUAD only with XLANE (shfl.* / quad.swizzle); ngcomp could run them bit-serially
 *     too, but a shuffle would cost ~190 instructions instead of one, and a shader that picks its shuffle path because the
 *     feature is reported would run far slower than its fallback;
 * - ARITHMETIC / CLUSTERED : the driver's nir_lower_subgroups lower_reduce turns reduce / scans into shuffles
 * behind a warp-uniform if (ngcomp/subgroup_selftest.py runs that shape on the model), so only with XLANE, like
 *     the shuffles they are built from.  dEQP-VK.subgroups.arithmetic / clustered pass on the model, hence on by default;
 *     `arith` = false ($NGV_SUBGROUP_ARITH=0, see ngv_subgroup_arith_enabled) leaves them out. */
static inline uint32_t
ngv_subgroup_ops(uint32_t sp_units, bool arith)
{
   uint32_t ops = NGV_SG_BASIC | NGV_SG_VOTE | NGV_SG_BALLOT;
   if (sp_units & NGV_SPU_XLANE) {
      ops |= NGV_SG_SHUFFLE | NGV_SG_SHUFFLE_RELATIVE | NGV_SG_QUAD;
      if (arith)
         ops |= NGV_SG_ARITHMETIC | NGV_SG_CLUSTERED;
   }
   return ops;
}

/* $NGV_SUBGROUP_ARITH: unset / empty = on (default), "0" = off, anything else = on. */
static inline bool
ngv_subgroup_arith_enabled(const char *env)
{
   return !(env && env[0] == '0' && env[1] == 0);
}

/* Does the resolved unit string `units` (ngv_units_resolve) contain the unit `name`? Same reading as ngcomp/
 * units.py parse(): "all" = everything, "none" = nothing, else a list separated by ',' '+' or ' ', case-insensitive.
 * the driver keeps packHalf2x16 / unpackHalf2x16 for ngcomp's hardware conversions iff this says "f16cvt". */
static inline bool
ngv_units_has(const char *units, const char *name)
{
   const size_t nl = strlen(name);
   const char *p = units ? units : "";
   while (*p) {
      while (*p == ',' || *p == '+' || *p == ' ')
         p++;
      const char *s = p;
      while (*p && *p != ',' && *p != '+' && *p != ' ')
         p++;
      const size_t l = (size_t)(p - s);
      bool all = l == 3, eq = l == nl;
      for (size_t i = 0; i < l; i++) {
         const char c = (s[i] >= 'A' && s[i] <= 'Z') ? (char)(s[i] - 'A' + 'a') : s[i];
         all = all && c == "all"[i];
         eq = eq && c == name[i];
      }
      if (all || eq)
         return true;
   }
   return false;
}

#endif /* NGV_UNITS_H */
