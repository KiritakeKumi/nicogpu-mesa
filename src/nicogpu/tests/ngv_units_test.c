/* SPDX-License-Identifier: MIT */
/* ngv_units_test.c -- host test of src/nicogpu/vulkan/ngv_units.h : the ngcomp `--units` string a
 * device gets.  No Vulkan, no Mesa, no device:
 * gcc -std=c11 -Wall -Wextra -Werror -I src/nicogpu/vulkan src/nicogpu/tests/ngv_units_test.c \
 *       -o ngv_units_test && ./ngv_units_test
 * The strings are the ones ngcomp/units.py parse accepts (checked by hand in : none, sfu, sfu,xlane,...). */
#include <stdio.h>

#include "ngv_units.h"

static int fails, checks;
#define CHECK(c) do { checks++; if (!(c)) { fails++; fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CHECK_STR(a, b) do { const char *_a = (a), *_b = (b); checks++; if (strcmp(_a, _b)) { fails++; \
   fprintf(stderr, "%s:%d: \"%s\" != \"%s\" (%s)\n", __FILE__, __LINE__, _a, _b, #a); } } while (0)

int
main(void)
{
   char buf[NGV_UNITS_BUFSZ];
   const char *src;

   /* the unit bit values are the ABI (RM CAPS[26:24] + CAPS[29], kernel NICOGPU_SP_UNIT_*, HAL NICOGPU_SP_UNIT_*) */
   CHECK(NGV_SPU_SFU == 1 && NGV_SPU_XLANE == 2 && NGV_SPU_FP64 == 4 && NGV_SPU_F16CVT == 8);

   CHECK_STR(ngv_units_string(0, buf), "none");
   CHECK_STR(ngv_units_string(NGV_SPU_SFU, buf), "sfu");
   CHECK_STR(ngv_units_string(NGV_SPU_XLANE, buf), "xlane");
   CHECK_STR(ngv_units_string(NGV_SPU_FP64, buf), "fp64");
   CHECK_STR(ngv_units_string(NGV_SPU_SFU | NGV_SPU_XLANE, buf), "sfu,xlane");
   CHECK_STR(ngv_units_string(NGV_SPU_SFU | NGV_SPU_FP64, buf), "sfu,fp64");
   CHECK_STR(ngv_units_string(NGV_SPU_XLANE | NGV_SPU_FP64, buf), "xlane,fp64");
 CHECK_STR(ngv_units_string(NGV_SPU_SFU | NGV_SPU_F16CVT, buf), "sfu,f16cvt"); /* rm_sp from on */
   CHECK_STR(ngv_units_string(NGV_SPU_ALL, buf), "sfu,xlane,fp64,f16cvt");
   CHECK_STR(ngv_units_string(0xFFFFFFF0u, buf), "none"); /* unknown bits are ignored */
   CHECK_STR(ngv_units_string(0xFFFFFFFFu, buf), "sfu,xlane,fp64,f16cvt");
   CHECK(strlen("sfu,xlane,fp64,f16cvt") + 1 <= NGV_UNITS_BUFSZ);

 /* : unit membership of a resolved string (the driver's switch for the hardware half conversions) */
   CHECK(ngv_units_has("sfu,f16cvt", "f16cvt") && ngv_units_has("all", "f16cvt") && ngv_units_has("ALL", "f16cvt"));
   CHECK(ngv_units_has("f16cvt", "f16cvt") && ngv_units_has(" sfu + F16CVT ", "f16cvt") && ngv_units_has("sfu,,f16cvt,", "f16cvt"));
   CHECK(!ngv_units_has("sfu", "f16cvt") && !ngv_units_has("none", "f16cvt") && !ngv_units_has("", "f16cvt"));
   CHECK(!ngv_units_has(NULL, "f16cvt") && !ngv_units_has("f16cvtx", "f16cvt") && !ngv_units_has("f16", "f16cvt"));
   CHECK(!ngv_units_has(NGV_LEGACY_UNITS, "f16cvt")); /* RMs without unit bits: software half packing */
   CHECK(ngv_units_has(ngv_units_resolve(NULL, true, NGV_SPU_SFU | NGV_SPU_F16CVT, buf, NULL), "f16cvt"));
   CHECK(!ngv_units_has(ngv_units_resolve("sfu", true, NGV_SPU_SFU | NGV_SPU_F16CVT, buf, NULL), "f16cvt"));

   /* policy: env > device > legacy default */
   CHECK_STR(ngv_units_resolve(NULL, false, 0, buf, &src), "sfu");
   CHECK_STR(src, "default");
   CHECK_STR(ngv_units_resolve("", false, 0, buf, &src), "sfu");
   CHECK_STR(src, "default");
   CHECK_STR(ngv_units_resolve(NULL, false, NGV_SPU_ALL, buf, &src), "sfu"); /* unreported: sp_units is not looked at */
   CHECK_STR(ngv_units_resolve(NULL, true, NGV_SPU_SFU, buf, &src), "sfu");
   CHECK_STR(src, "device");
   CHECK_STR(ngv_units_resolve(NULL, true, 0, buf, &src), "none"); /* reported and empty: really no optional unit */
   CHECK_STR(src, "device");
   CHECK_STR(ngv_units_resolve(NULL, true, NGV_SPU_ALL, buf, &src), "sfu,xlane,fp64,f16cvt");
   CHECK_STR(ngv_units_resolve("all", true, NGV_SPU_SFU, buf, &src), "all"); /* env wins over the device */
   CHECK_STR(src, "env");
   CHECK_STR(ngv_units_resolve("sfu,xlane", false, 0, buf, &src), "sfu,xlane");
   CHECK_STR(ngv_units_resolve(NULL, false, 0, buf, NULL), "sfu"); /* source is optional */

 /* : units string -> bits (what ngv_device.c reports subgroup operations from) */
   CHECK(ngv_units_parse("all") == NGV_SPU_ALL);
   CHECK(ngv_units_parse("none") == 0);
   CHECK(ngv_units_parse("") == 0);
   CHECK(ngv_units_parse("sfu") == NGV_SPU_SFU);
   CHECK(ngv_units_parse("sfu,xlane") == (NGV_SPU_SFU | NGV_SPU_XLANE));
   CHECK(ngv_units_parse(" SFU + XLane ") == (NGV_SPU_SFU | NGV_SPU_XLANE));
   CHECK(ngv_units_parse("xlane fp64") == (NGV_SPU_XLANE | NGV_SPU_FP64));
   CHECK(ngv_units_parse("sfu,xlanes,bogus,fp64") == (NGV_SPU_SFU | NGV_SPU_FP64)); /* unknown names ignored */
   CHECK(ngv_units_parse("averyveryverylongname,xlane") == NGV_SPU_XLANE);
   for (uint32_t u = 0; u <= NGV_SPU_ALL; u++) /* round trip with the string the device report turns into */
      CHECK(ngv_units_parse(ngv_units_string(u, buf)) == u);
   CHECK(ngv_units_parse(ngv_units_resolve(NULL, false, 0, buf, NULL)) == NGV_SPU_SFU);

 /* : subgroup operations by unit set (values = VkSubgroupFeatureFlagBits) */
   CHECK(NGV_SG_BASIC == 1 && NGV_SG_VOTE == 2 && NGV_SG_ARITHMETIC == 4 && NGV_SG_BALLOT == 8 && NGV_SG_SHUFFLE == 16 &&
         NGV_SG_SHUFFLE_RELATIVE == 32 && NGV_SG_CLUSTERED == 64 && NGV_SG_QUAD == 128);
   const uint32_t bvb = NGV_SG_BASIC | NGV_SG_VOTE | NGV_SG_BALLOT;
   const uint32_t xl = bvb | NGV_SG_SHUFFLE | NGV_SG_SHUFFLE_RELATIVE | NGV_SG_QUAD;
   const uint32_t ac = NGV_SG_ARITHMETIC | NGV_SG_CLUSTERED;
   CHECK(ngv_subgroup_ops(0, true) == bvb);
   CHECK(ngv_subgroup_ops(NGV_SPU_SFU, true) == bvb); /* the shipping rm_sp: arithmetic needs the shuffles, so no XLANE -> no */
   CHECK(ngv_subgroup_ops(NGV_SPU_SFU, false) == bvb);
   CHECK(ngv_subgroup_ops(NGV_SPU_SFU | NGV_SPU_XLANE, true) == (xl | ac));
   CHECK(ngv_subgroup_ops(NGV_SPU_ALL, true) == (xl | ac)); /* the model (hal): 0xff */
   CHECK(ngv_subgroup_ops(NGV_SPU_ALL, true) == 0xffu);
   CHECK(ngv_subgroup_ops(NGV_SPU_ALL, false) == xl); /* $NGV_SUBGROUP_ARITH=0 */
   CHECK(ngv_subgroup_arith_enabled(NULL) && ngv_subgroup_arith_enabled("") && ngv_subgroup_arith_enabled("1") &&
         !ngv_subgroup_arith_enabled("0") && ngv_subgroup_arith_enabled("00"));

   printf("ngv_units_test: %d checks, %d failed\n", checks, fails);
   return fails ? 1 : 0;
}
