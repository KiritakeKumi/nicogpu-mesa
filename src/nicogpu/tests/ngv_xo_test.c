/* SPDX-License-Identifier: MIT */
/* ngv_xo_test.c -- host test of src/nicogpu/vulkan/ngv_xo.h: the code-object header check of the
 * consumer (ABI 0.1, 0.2 and 0.3).  No Vulkan, no Mesa, no device:
 * gcc -std=c11 -Wall -Wextra -Werror -I src/nicogpu/vulkan src/nicogpu/tests/ngv_xo_test.c \
 *       -o ngv_xo_test && ./ngv_xo_test [--ok obj.xo ...] [--reject obj.xo ...]
 * With --ok / --reject the listed files (written by ngcomp, see the internal CI script) must be accepted / refused
 * as well.  Covered: a 0.1 object (also the legacy "XGCO" magic) loads, a 0.2 object loads, an unknown stage bit /
 * stage code / ABI version code, a nonzero 0.1 reserved word, a header mask that disagrees with the segment table and an
 * empty 0.2 mask are refused, ABI 0.3 (routine segments, relocations) is refused by ngv_xo_check and accepted by
 * ngv_xo_check_ex(allow_abi3), a 0.2 object may not carry a routine segment or relocation records, 0.3 does not know other
 * segment kinds, and a compute pipeline only takes compute objects. */
#include <stdio.h>

#include "ngv_xo.h"

static int fails, checks;
#define CHECK(c) do { checks++; if (!(c)) { fails++; fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); } } while (0)

/* header + metadata + 8 bytes of code, the way ngcomp/emit.py lays an object out */
static size_t
make_obj(uint8_t *out, const char *magic, uint32_t abi, uint32_t word, const char *meta)
{
   uint32_t h[8];
   uint32_t ml = (uint32_t)strlen(meta);
   memcpy(h, magic, 4);
   h[1] = 0x00010001u; /* container version 1 (u16), flags 0 (u16) */
   h[2] = 32;
   h[3] = ml;
   h[4] = 32 + ml;
   h[5] = 8;
   h[6] = abi;
   h[7] = word;
   memcpy(out, h, 32);
   memcpy(out + 32, meta, ml);
   memset(out + 32 + ml, 0, 8);
   return 32 + ml + 8;
}

#define SEG(stage) "{\"entry\": 0, \"name\": \"s\", \"scratch\": 0, \"segment\": \"body\", \"stage\": " #stage ", \"vgprs\": 4}"
#define META_V1 "{\"abi_version\": \"0.1\", \"entry\": 0, \"segments\": [{\"entry\": 0, \"name\": \"s\", \"scratch\": 16, \"vgprs\": 4}]}"
#define META_V2(segs) "{\"abi_version\": \"0.2\", \"entry\": 0, \"segments\": [" segs "]}"

static bool
accept(uint32_t abi, uint32_t word, const char *meta, const char *magic, struct ngv_xo_info *xi)
{
   uint8_t buf[1024];
   size_t n = make_obj(buf, magic, abi, word, meta);
   return ngv_xo_check(buf, n, xi);
}

static bool
accept3(uint32_t abi, uint32_t word, const char *meta, struct ngv_xo_info *xi)
{
   uint8_t buf[1024];
   size_t n = make_obj(buf, "NGCO", abi, word, meta);
   return ngv_xo_check_ex(buf, n, xi, true);
}

#define KSEG(kind) "{\"entry\": 0, \"name\": \"s\", \"scratch\": 0, \"segment\": \"" kind "\", \"stage\": 2, \"vgprs\": 4}"
#define META_V(ver, segs, extra) "{\"abi_version\": \"" ver "\", \"entry\": 0, " extra "\"segments\": [" segs "]}"
#define RELOC_ONE "\"relocs\": [{\"kind\": \"br19\", \"offset\": 0, \"symbol\": \"r\"}], "

static bool
accept_file(const char *path, struct ngv_xo_info *xi)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      return false;
   static uint8_t buf[1 << 20];
   size_t n = fread(buf, 1, sizeof(buf), f);
   fclose(f);
   return ngv_xo_check(buf, n, xi);
}

int
main(int argc, char **argv)
{
   struct ngv_xo_info xi;

   /* ABI 0.1: reserved word 0; the stage mask is compute only; legacy magic as well */
   CHECK(accept(1, 0, META_V1, "NGCO", &xi) && xi.abi == NGV_XO_ABI_0_1 && ngv_xo_is_compute(xi.stage_mask));
   CHECK(xi.meta_off == 32 && xi.code_len == 8 && xi.code_off == 32 + xi.meta_len);
   CHECK(accept(1, 0, META_V1, "XGCO", &xi) && ngv_xo_is_compute(xi.stage_mask));
   CHECK(!accept(1, 1, META_V1, "NGCO", &xi));          /* nonzero reserved word */
   CHECK(!accept(1, 0x80000000u, META_V1, "NGCO", &xi));

   /* ABI 0.2: the mask equals the segment table */
   CHECK(accept(2, 1, META_V2(SEG(0)), "NGCO", &xi) && xi.abi == NGV_XO_ABI_0_2 && ngv_xo_is_compute(xi.stage_mask));
   CHECK(accept(2, 4, META_V2(SEG(2)), "NGCO", &xi) && xi.stage_mask == 4 && !ngv_xo_is_compute(xi.stage_mask));
   CHECK(accept(2, 0x1e, META_V2(SEG(1) "," SEG(2) "," SEG(3) "," SEG(4)), "NGCO", &xi) && xi.stage_mask == 0x1e);
   CHECK(accept(2, 0x1f, META_V2(SEG(0) "," SEG(1) "," SEG(2) "," SEG(3) "," SEG(4)), "NGCO", &xi) && xi.stage_mask == 0x1f);
   CHECK(accept(2, 8, META_V2(SEG(3) "," SEG(3)), "NGCO", &xi) && xi.stage_mask == 8);

   /* ABI 0.3 (R-A): routine segments and relocations.  The default check refuses it; the linker's check accepts it */
   CHECK(!accept(3, 4, META_V("0.3", KSEG("routine"), ""), "NGCO", &xi));
   CHECK(accept3(3, 4, META_V("0.3", KSEG("routine"), ""), &xi) && xi.abi == NGV_XO_ABI_0_3 && xi.stage_mask == 4);
   CHECK(accept3(3, 4, META_V("0.3", KSEG("epilogue") "," KSEG("routine"), RELOC_ONE), &xi));
   CHECK(accept3(2, 4, META_V("0.2", KSEG("epilogue"), "\"relocs\": [], "), &xi));         /* an empty list is not a relocation */
   CHECK(!accept3(2, 4, META_V("0.2", KSEG("routine"), ""), &xi));                          /* routine needs 0.3 */
   CHECK(!accept3(2, 4, META_V("0.2", KSEG("epilogue"), RELOC_ONE), &xi));                   /* relocations need 0.3 */
   CHECK(!accept(2, 4, META_V("0.2", KSEG("routine"), ""), "NGCO", &xi));
   CHECK(!accept3(3, 4, META_V("0.3", KSEG("widget"), ""), &xi));                           /* unknown segment kind */
   CHECK(!accept3(3, 4, META_V("0.3", KSEG("body") "," KSEG("widget"), ""), &xi));
   CHECK(!accept3(2, 4, META_V("0.2", KSEG("widget"), ""), &xi));
   CHECK(!accept3(4, 4, META_V("0.3", KSEG("routine"), ""), &xi));                          /* 4 is unknown even for the linker */
   CHECK(!accept3(3, 1, META_V("0.3", KSEG("routine"), ""), &xi));                          /* mask != table */
   CHECK(accept(2, 4, META_V("0.2", KSEG("prologue") "," KSEG("body") "," KSEG("epilogue"), ""), "NGCO", &xi));

   /* refused: unknown ABI version codes, unknown stage bits and codes, inconsistent tables, an empty mask */
   CHECK(!accept(0, 0, META_V1, "NGCO", &xi));
   CHECK(!accept(3, 1, META_V2(SEG(0)), "NGCO", &xi));
   CHECK(!accept(0x10002u, 1, META_V2(SEG(0)), "NGCO", &xi));
   CHECK(!accept(2, 0x20 | 1, META_V2(SEG(0)), "NGCO", &xi));      /* unknown stage bit 5 */
   CHECK(!accept(2, 0x80000000u | 1, META_V2(SEG(0)), "NGCO", &xi));
   CHECK(!accept(2, 1, META_V2(SEG(9)), "NGCO", &xi));            /* unknown stage code in the table */
   CHECK(!accept(2, 1, META_V2(SEG(5)), "NGCO", &xi));
   CHECK(!accept(2, 0x20, META_V2(SEG(5)), "NGCO", &xi));
   CHECK(!accept(2, 1, META_V2(SEG(2)), "NGCO", &xi));            /* header mask != table */
   CHECK(!accept(2, 5, META_V2(SEG(0)), "NGCO", &xi));
   CHECK(!accept(2, 0, META_V2(SEG(0)), "NGCO", &xi));            /* empty mask */
   CHECK(!accept(2, 1, META_V1, "NGCO", &xi));                    /* a 0.2 header needs the stage table */
   CHECK(!accept(1, 0, META_V1, "BADM", &xi));
   {
      uint8_t buf[64] = {0};
      CHECK(!ngv_xo_check(buf, sizeof(buf), &xi));
      CHECK(!ngv_xo_check(buf, 16, &xi));
   }

   /* the integer scanner the loader uses (first value for entry, max over segments for vgprs / scratch) */
   {
      uint64_t v;
      const char *m = "{\"segments\": [{\"entry\": 64, \"vgprs\": 4, \"scratch\": 16}, {\"entry\": 128, \"vgprs\": 9, \"scratch\": 0}]}";
      CHECK(ngv_xo_json_int(m, strlen(m), "entry", false, &v) && v == 64);
      CHECK(ngv_xo_json_int(m, strlen(m), "vgprs", true, &v) && v == 9);
      CHECK(ngv_xo_json_int(m, strlen(m), "scratch", true, &v) && v == 16);
      CHECK(!ngv_xo_json_int(m, strlen(m), "stage", true, &v));
      /* "stage_name" is not "stage" */
      const char *m2 = "{\"stage_name\": \"fragment\"}";
      CHECK(!ngv_xo_json_int(m2, strlen(m2), "stage", true, &v));
   }

 /* real objects written by ngcomp */
   int mode = 0;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--ok")) {
         mode = 1;
      } else if (!strcmp(argv[i], "--reject")) {
         mode = 2;
      } else if (mode) {
         bool a = accept_file(argv[i], &xi);
         checks++;
         if (a != (mode == 1)) {
            fails++;
            fprintf(stderr, "%s: expected %s, got %s\n", argv[i], mode == 1 ? "accept" : "reject", a ? "accept" : "reject");
         }
      }
   }

   printf("ngv_xo_test: %d checks, %d failures\n", checks, fails);
   return fails ? 1 : 0;
}
