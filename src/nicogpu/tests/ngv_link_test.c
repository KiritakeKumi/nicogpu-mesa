/* SPDX-License-Identifier: MIT */
/* ngv_link_test.c -- host test of ngv_link.h : the C linker against the Python reference linker.
 *
 *   gcc -std=c11 -Wall -Wextra -Werror -I src/nicogpu/vulkan test/ngv_link_test.c -o ngv_link_test
 *   ./ngv_link_test <dir with manifest.txt>        (model/tests/data/ngcomp_link, written by
 * python ngcomp/seglink_selftest.py)
 *
 * Every positive case of the manifest must give exactly the image, VREGS, SCRATCH, flags and SET_FRAG_SHADER dwords the
 * Python linker produced; every negative case must give the same error code.  On top of that: the draw-time entry points
 * (variant choice from the draw state, link cache, no recompilation when only the state changes), the canonical variant
 * key strings, and damaged metadata (no crash, no image). */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ngv_link.h"

static int g_checks, g_fails;

#define CHECK(cond, ...)                                  \
   do {                                                   \
      g_checks++;                                         \
      if (!(cond)) {                                      \
         g_fails++;                                       \
         printf("FAIL %s:%d: ", __FILE__, __LINE__);      \
         printf(__VA_ARGS__);                             \
         printf("\n");                                    \
      }                                                   \
   } while (0)

#define MAX_OBJ 128
#define MAX_CASE 128
#define MAX_VAR 64

struct tobj {
   char logical[48], file[96], seg[64];
   uint8_t *data;
   size_t size;
   struct ngv_link_obj obj;
   bool open;
};

struct tcase {
   char name[64];
   int chain[3], nchain, lib[4], nlib;
   bool ok;
   char image[96];
   unsigned vregs, scratch, ordered, facts, sample, inz, inw;
   uint32_t pkt[8];
   int err;
   char msg[64];
};

struct tvar {
   char kind[16], key[300], logical[48];
};

static struct tobj g_obj[MAX_OBJ];
static int g_nobj;
static struct tcase g_case[MAX_CASE];
static int g_ncase;
static struct tvar g_var[MAX_VAR];
static int g_nvar;
static char g_dir[512];

static uint8_t *
read_file(const char *name, size_t *size)
{
   char path[1024];
   snprintf(path, sizeof(path), "%s/%s", g_dir, name);
   FILE *f = fopen(path, "rb");
   if (!f)
      return NULL;
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint8_t *buf = (uint8_t *)malloc((size_t)n + 1);
   if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
      fclose(f);
      free(buf);
      return NULL;
   }
   fclose(f);
   *size = (size_t)n;
   return buf;
}

static int
find_obj(const char *logical)
{
   for (int i = 0; i < g_nobj; i++)
      if (!strcmp(g_obj[i].logical, logical))
         return i;
   return -1;
}

static const char *const err_names[] = {"OK",     "OBJECT",     "ABI",   "STAGE",  "ORDER", "NEEDS", "ENDPGM", "DRAIN",
                                        "OPEN",   "INTERLOCK",  "UNRESOLVED", "RELOC", "ROUTINE", "VREGS", "ZS_MODE", "ALIGN",
                                        "UNSUPPORTED", "PRESERVE", "SCRATCH", "ZS_NOTEST", "CONVENTION", "EXEC"};

static int
err_code(const char *name)
{
   for (int i = 0; i < (int)(sizeof(err_names) / sizeof(err_names[0])); i++)
      if (!strcmp(err_names[i], name))
         return i;
   return -1;
}

static bool
load_manifest(void)
{
   size_t sz = 0;
   uint8_t *m = read_file("manifest.txt", &sz);
   if (!m)
      return false;
   m[sz] = 0;
   struct tcase *cur = NULL;
   char *next = (char *)m;
   while (*next) {
      char *line = next, *tok[24];
      int nt = 0;
      char *nl = strchr(line, '\n');
      if (nl) {
         *nl = 0;
         next = nl + 1;
      } else {
         next = line + strlen(line);
      }
      if (line[0] == '#')
         continue;
      for (char *t = line; *t && nt < 24;) {
         while (*t == ' ' || *t == '\r')
            *t++ = 0;
         if (!*t)
            break;
         tok[nt++] = t;
         while (*t && *t != ' ' && *t != '\r')
            t++;
      }
      if (!nt)
         continue;
      if (!strcmp(tok[0], "obj") && nt == 4 && g_nobj < MAX_OBJ) {
         struct tobj *o = &g_obj[g_nobj++];
         snprintf(o->logical, sizeof(o->logical), "%s", tok[1]);
         snprintf(o->file, sizeof(o->file), "%s", tok[2]);
         snprintf(o->seg, sizeof(o->seg), "%s", tok[3]);
      } else if (!strcmp(tok[0], "case") && g_ncase < MAX_CASE) {
         cur = &g_case[g_ncase++];
         memset(cur, 0, sizeof(*cur));
         snprintf(cur->name, sizeof(cur->name), "%s", tok[1]);
      } else if (cur && !strcmp(tok[0], "chain")) {
         for (int i = 1; i < nt; i++)
            cur->chain[cur->nchain++] = find_obj(tok[i]);
      } else if (cur && !strcmp(tok[0], "lib")) {
         for (int i = 1; i < nt; i++)
            cur->lib[cur->nlib++] = find_obj(tok[i]);
      } else if (cur && !strcmp(tok[0], "err")) {
         cur->ok = false;
         cur->err = err_code(tok[1]);
         if (nt > 2 && !strncmp(tok[2], "msg=", 4)) {
            /* the reason the case must give (words joined by '_'): the error code alone is shared by neighbouring checks */
            snprintf(cur->msg, sizeof(cur->msg), "%s", tok[2] + 4);
            for (char *mc = cur->msg; *mc; mc++)
               if (*mc == '_')
                  *mc = ' ';
         }
      } else if (cur && !strcmp(tok[0], "ok")) {
         cur->ok = true;
         for (int i = 1; i < nt; i++) {
            char *eq = strchr(tok[i], '=');
            if (!eq)
               continue;
            *eq++ = 0;
            if (!strcmp(tok[i], "image"))
               snprintf(cur->image, sizeof(cur->image), "%s", eq);
            else if (!strcmp(tok[i], "vregs"))
               cur->vregs = (unsigned)atoi(eq);
            else if (!strcmp(tok[i], "scratch"))
               cur->scratch = (unsigned)atoi(eq);
            else if (!strcmp(tok[i], "ordered"))
               cur->ordered = (unsigned)atoi(eq);
            else if (!strcmp(tok[i], "facts"))
               cur->facts = (unsigned)atoi(eq);
            else if (!strcmp(tok[i], "sample"))
               cur->sample = (unsigned)atoi(eq);
            else if (!strcmp(tok[i], "inz"))
               cur->inz = (unsigned)atoi(eq);
            else if (!strcmp(tok[i], "inw"))
               cur->inw = (unsigned)atoi(eq);
            else if (!strcmp(tok[i], "pkt")) {
               char *p = eq;
               for (int d = 0; d < 8; d++) {
                  cur->pkt[d] = (uint32_t)strtoul(p, &p, 16);
                  if (*p == ',')
                     p++;
               }
            }
         }
      } else if (!strcmp(tok[0], "variant") && nt == 4 && g_nvar < MAX_VAR) {
         struct tvar *v = &g_var[g_nvar++];
         snprintf(v->kind, sizeof(v->kind), "%s", tok[1]);
         snprintf(v->key, sizeof(v->key), "%s", tok[2]);
         snprintf(v->logical, sizeof(v->logical), "%s", tok[3]);
      }
   }
   free(m);
   return true;
}

static struct ngv_link_obj *
get_obj(int i)
{
   struct tobj *o = &g_obj[i];
   if (!o->open) {
      struct ngv_link_error e;
      o->data = read_file(o->file, &o->size);
      if (!o->data)
         return NULL;
      if (ngv_link_obj_open(&o->obj, o->data, o->size, &e)) {
         /* some negative-case objects are damaged on purpose in other tests; the manifest ones are all well formed */
         printf("cannot open %s: %s\n", o->file, e.msg);
         return NULL;
      }
      o->open = true;
   }
   return &o->obj;
}

static void
run_case(const struct tcase *c)
{
   struct ngv_link_seg_ref chain[3];
   const struct ngv_link_obj *libs[4];
   struct ngv_link_result res;
   struct ngv_link_error err;
   memset(&err, 0, sizeof(err));
   for (int i = 0; i < c->nchain; i++) {
      chain[i].obj = get_obj(c->chain[i]);
      chain[i].name = g_obj[c->chain[i]].seg;
   }
   for (int i = 0; i < c->nlib; i++)
      libs[i] = get_obj(c->lib[i]);
   int rc = ngv_link(chain, (unsigned)c->nchain, libs, (unsigned)c->nlib, NULL, &res, &err);
   if (!c->ok) {
      CHECK(rc == c->err, "%s: wanted error %s, got %d (%s)", c->name, err_names[c->err], rc, rc ? err.msg : "linked");
      CHECK(!c->msg[0] || (rc && strstr(err.msg, c->msg)), "%s: the message should say '%s', it says '%s'", c->name, c->msg, rc ? err.msg : "linked");
      CHECK(rc == 0 || !res.image, "%s: image after error", c->name);
      return;
   }
   CHECK(rc == 0, "%s: link failed: %s", c->name, err.msg);
   if (rc)
      return;
   size_t isz = 0;
   uint8_t *exp = read_file(c->image, &isz);
   CHECK(exp && isz == res.size && !memcmp(exp, res.image, isz), "%s: image differs from the Python linker (%zu vs %zu bytes)",
         c->name, res.size, exp ? isz : 0);
   free(exp);
   CHECK(res.vregs == c->vregs && res.scratch == c->scratch, "%s: vregs %u/%u scratch %u/%u", c->name, res.vregs, c->vregs,
         res.scratch, c->scratch);
   CHECK((unsigned)res.ordered == c->ordered && res.facts == c->facts &&
            (unsigned)res.sample_shading == c->sample && (unsigned)res.input_z == c->inz && (unsigned)res.input_w == c->inw,
         "%s: flags", c->name);
   uint32_t pkt[8];
   rc = ngv_link_frag_shader_packet(&res, 0x123450, (res.facts & NGV_LINK_FACT_ZS_TEST) != 0, 0, false, false, NULL, pkt, &err);
   CHECK(rc == 0 && !memcmp(pkt, c->pkt, sizeof(pkt)), "%s: SET_FRAG_SHADER dwords differ", c->name);
   ngv_link_result_free(&res);
}

/* --- variants served from the manifest ------------------------------------------------------------------------- */
struct vserver {
   unsigned asked;
};

static int
serve(void *user, const char *kind, const char *key, uint8_t **data, size_t *size)
{
   struct vserver *vs = (struct vserver *)user;
   vs->asked++;
   for (int i = 0; i < g_nvar; i++)
      if (!strcmp(g_var[i].kind, kind) && !strcmp(g_var[i].key, key)) {
         int o = find_obj(g_var[i].logical);
         size_t sz;
         uint8_t *d = read_file(g_obj[o].file, &sz);
         if (!d)
            return 1;
         *data = d;
         *size = sz;
         return 0;
      }
   return 1;
}

static const struct tcase *
case_named(const char *name)
{
   for (int i = 0; i < g_ncase; i++)
      if (!strcmp(g_case[i].name, name))
         return &g_case[i];
   return NULL;
}

static bool
same_image(const struct ngv_link_result *r, const char *casename)
{
   const struct tcase *c = case_named(casename);
   size_t sz = 0;
   uint8_t *e = c ? read_file(c->image, &sz) : NULL;
   bool ok = e && sz == r->size && !memcmp(e, r->image, sz);
   free(e);
   return ok;
}

static void
test_draws(void)
{
   struct ngv_link_ctx ctx;
   struct vserver vs = {0};
   struct ngv_link_error err;
   int body = find_obj("body_flat"), pack = find_obj("lib_pack");
   const struct ngv_link_obj *bo = get_obj(body);
   ngv_link_ctx_init(&ctx, serve, &vs);
   ctx.libs[0] = get_obj(pack);
   ctx.nlibs = 1;
   struct ngv_link_fs_state opaque = {1, 0, false, false, false, false, false, false, false};
   struct ngv_link_fs_state at_bl = {1, 0, true, true, false, false, false, false, false};
   struct ngv_link_fs_state bl = {1, 0, true, false, false, false, false, false, false};
   struct ngv_link_fs_state at_bl4 = {4, 0, true, true, false, false, false, false, false};
   const struct ngv_link_fs_state *seq[] = {&opaque, &at_bl, &bl, &at_bl, &opaque, &bl, &at_bl4};
   const char *want[] = {"flat_ep_opaque", "flat_ep_at_bl", "flat_ep_bl", "flat_ep_at_bl", "flat_ep_opaque", "flat_ep_bl",
                         "flat_ep_at_bl4"};
   for (unsigned i = 0; i < sizeof(seq) / sizeof(seq[0]); i++) {
      const struct ngv_link_result *r = ngv_link_fs_for_draw(&ctx, bo, NULL, NULL, NULL, seq[i], &err);
      CHECK(r != NULL, "draw %u: %s", i, r ? "" : err.msg);
      if (r)
         CHECK(same_image(r, want[i]), "draw %u: image is not the Python one for %s", i, want[i]);
   }
   /* 7 draws, 4 distinct states: 4 epilogues compiled (asked of the compiler), 4 links, 3 link-cache hits, body untouched */
   CHECK(vs.asked == 4 && ctx.variants.compiles == 4 && ctx.variants.hits == 3, "variants: asked %u compiles %u hits %u",
         vs.asked, ctx.variants.compiles, ctx.variants.hits);
   CHECK(ctx.cache.links == 4 && ctx.cache.hits == 3, "cache: links %u hits %u", ctx.cache.links, ctx.cache.hits);
   /* a flush (code heap recycled) relinks but still does not recompile */
   ngv_link_cache_flush(&ctx.cache);
   const struct ngv_link_result *r = ngv_link_fs_for_draw(&ctx, bo, NULL, NULL, NULL, &bl, &err);
   CHECK(r && same_image(r, "flat_ep_bl") && ctx.cache.links == 5 && vs.asked == 4, "flush: relinked without recompiling");
   /* unsupported state: no library epilogue, the driver must compile a whole shader */
   struct ngv_link_fs_state bad = {3, 0, false, false, false, false, false, false, false};
   r = ngv_link_fs_for_draw(&ctx, bo, NULL, NULL, NULL, &bad, &err);
   CHECK(!r && err.code == NGV_LINK_E_UNSUPPORTED, "3 samples is unsupported");
   struct ngv_link_fs_state unknown = {1, 0, true, true, true, true, true, false, false};
   r = ngv_link_fs_for_draw(&ctx, bo, NULL, NULL, NULL, &unknown, &err);
   CHECK(!r && err.code == NGV_LINK_E_UNSUPPORTED, "a state without a variant object is refused (%d)", err.code);
   /* variant key strings: C text == Python text for every fs_ep variant of the manifest */
   struct {
      struct ngv_link_fs_state st;
      const char *logical;
   } states[] = {
      {{1, 0, false, false, false, false, false, false, false}, "ep_opaque"}, {{1, 0, true, true, false, false, false, false, false}, "ep_at_bl"},
      {{1, 0, true, false, false, false, false, false, false}, "ep_bl"},      {{1, 0, false, true, false, false, false, false, false}, "ep_at"},
      {{1, 0, false, false, true, false, false, false, false}, "ep_dither"},  {{4, 0, false, false, false, false, false, false, false}, "ep_op4"},
      {{4, 0, true, false, false, false, false, false, false}, "ep_bl4"},     {{4, 0, true, true, false, false, false, false, false}, "ep_at_bl4"},
      {{4, 0, true, true, true, false, false, false, false}, "ep_dbl4"},      {{4, 0, false, false, false, true, false, false, false}, "ep_smask4"},
      {{1, 0, false, false, false, false, true, false, false}, "ep_capture"},
      /* R-E LATE ZS variants: zs_late -> zs.test epilogue; zs.test.z when the body exports depth */
      {{1, 0, false, false, false, false, false, true, false}, "ep_zs_s"},
      {{1, 0, false, true, false, false, false, true, false}, "ep_zs_s_at"},
      {{1, 0, false, true, false, false, false, true, true}, "ep_zs_z_at"},
      {{1, 0, true, false, false, false, false, true, false}, "ep_zs_s_bl"},
      {{4, 0, true, true, false, false, false, true, false}, "ep_zs_s_at_bl4"},
      {{2, 0, false, false, false, false, false, true, true}, "ep_zs_z_op2"},
   };
   for (unsigned i = 0; i < sizeof(states) / sizeof(states[0]); i++) {
      struct ngv_link_ep_key k;
      char ks[200];
      bool found = false;
      struct ngv_link_fs_state s = states[i].st;
      CHECK(ngv_link_ep_key_from_state(&s, &k, &err) == 0, "key from state %u", i);
      ngv_link_ep_key_string(&k, ks, sizeof(ks));
      for (int v = 0; v < g_nvar; v++)
         if (!strcmp(g_var[v].kind, "fs_ep") && !strcmp(g_var[v].key, ks) && !strcmp(g_var[v].logical, states[i].logical))
            found = true;
      CHECK(found, "variant key text '%s' is not the Python text of %s", ks, states[i].logical);
   }
   {
      struct ngv_link_vattr a[2] = {{0, 0, 0, "r32g32b32a32_sfloat"}, {1, 1, 0, "r8g8b8a8_unorm"}};
      char ks[300];
      bool found = false;
      ngv_link_vs_key_string(a, 2, ks, sizeof(ks));
      for (int v = 0; v < g_nvar; v++)
         if (!strcmp(g_var[v].kind, "vs_pro") && !strcmp(g_var[v].key, ks))
            found = true;
      CHECK(found, "vs key text '%s'", ks);
      const struct ngv_link_obj *vb = get_obj(find_obj("vs_body_ids_1"));
      r = ngv_link_vs_for_draw(&ctx, vb, NULL, a, 2, &err);
      CHECK(r && same_image(r, "vs_ids"), "vs draw image");
   }
   ngv_link_ctx_finish(&ctx);
}

static void
test_packets(void)
{
   struct ngv_link_result r;
   struct ngv_link_error err;
   uint32_t pkt[8];
   int body = find_obj("body_flat"), ep = find_obj("ep_at"), pack = find_obj("lib_pack");
   struct ngv_link_seg_ref chain[2] = {{get_obj(body), g_obj[body].seg}, {get_obj(ep), g_obj[ep].seg}};
   const struct ngv_link_obj *libs[1] = {get_obj(pack)};
   CHECK(ngv_link(chain, 2, libs, 1, "fragment", &r, &err) == 0, "link at: %s", err.msg);
   {
      /* R-E: the linker reports facts, the draw state decides EARLY / LATE (GFX.md section 4.4) */
      struct ngv_link_zs_state dw = {true, false, false, false}, none = {false, false, false, false};
      struct ngv_link_zs_state a2c = {true, false, false, true}, a2c_nowrite = {false, false, false, true};
      CHECK(r.facts == NGV_LINK_FACT_DISCARD, "alpha test reports the discard fact only (%u)", r.facts);
      CHECK(ngv_link_zs_late_needed(&r, &dw) && !ngv_link_zs_late_needed(&r, &none), "discard needs LATE only with a depth write");
      CHECK(ngv_link_zs_late_needed(&r, &a2c) && !ngv_link_zs_late_needed(&r, &a2c_nowrite), "alpha-to-coverage needs a ZS write");
      {
         /* alpha-to-coverage alone (no discard / sample-mask segment) is a coverage fact of the draw state */
         int b2 = find_obj("body_flat"), e2 = find_obj("ep_opaque");
         struct ngv_link_seg_ref ch2[2] = {{get_obj(b2), g_obj[b2].seg}, {get_obj(e2), g_obj[e2].seg}};
         struct ngv_link_result r0;
         CHECK(ngv_link(ch2, 2, libs, 1, "fragment", &r0, &err) == 0, "link opaque: %s", err.msg);
         CHECK(r0.facts == 0 && ngv_link_zs_late_needed(&r0, &a2c) && !ngv_link_zs_late_needed(&r0, &a2c_nowrite) &&
                  !ngv_link_zs_late_needed(&r0, &dw),
               "alpha-to-coverage forces LATE for a shader without coverage facts when ZS is written (%u)", r0.facts);
      }
      CHECK(ngv_link_frag_shader_packet(&r, 0x10010, false, 0, false, false, &dw, pkt, &err) == NGV_LINK_E_ZS_MODE, "early ZS refused");
      CHECK(ngv_link_frag_shader_packet(&r, 0x10010, false, 0, false, false, &none, pkt, &err) == 0, "early ZS fine without writes");
      CHECK(ngv_link_frag_shader_packet(&r, 0x10010, true, 0, false, false, &dw, pkt, &err) == NGV_LINK_E_ZS_NOTEST,
            "LATE without a zs.test segment refused");
      CHECK(ngv_link_frag_shader_packet(&r, 0x10010, true, 0, false, true, &dw, pkt, &err) == 0 && (pkt[7] & 3) == 3,
            "NULL_FS skips the ZS checks");
   }
   {
      /* a store is an API-visible side effect, an exported depth always needs LATE (unless EarlyFragmentTests) */
      struct ngv_link_zs_state none = {false, false, false, false}, early = {false, false, true, false};
      struct ngv_link_result r2;
      int bs = find_obj("body_stores"), ep0 = find_obj("ep_opaque"), bz = find_obj("body_zdepth"), ez = find_obj("ep_zs_z_at");
      struct ngv_link_seg_ref c1[2] = {{get_obj(bs), g_obj[bs].seg}, {get_obj(ep0), g_obj[ep0].seg}};
      struct ngv_link_seg_ref c2[2] = {{get_obj(bz), g_obj[bz].seg}, {get_obj(ez), g_obj[ez].seg}};
      CHECK(ngv_link(c1, 2, libs, 1, "fragment", &r2, &err) == 0, "link stores: %s", err.msg);
      CHECK(r2.facts == NGV_LINK_FACT_SIDE_EFFECT && ngv_link_zs_late_needed(&r2, &none) && !ngv_link_zs_late_needed(&r2, &early),
            "store: side-effect fact, LATE unless EarlyFragmentTests (%u)", r2.facts);
      ngv_link_result_free(&r2);
      CHECK(ngv_link(c2, 2, libs, 1, "fragment", &r2, &err) == 0, "link zdepth: %s", err.msg);
      CHECK((r2.facts & NGV_LINK_FACT_EXPORTS_DEPTH) && ngv_link_zs_late_needed(&r2, &none) && !ngv_link_zs_late_needed(&r2, &early),
            "depth export: LATE unless EarlyFragmentTests (%u)", r2.facts);
      ngv_link_result_free(&r2);
   }
   CHECK(ngv_link_frag_shader_packet(&r, 0x10008, false, 0, false, false, NULL, pkt, &err) == NGV_LINK_E_ALIGN, "unaligned image");
   CHECK(ngv_link_frag_shader_packet(&r, (uint64_t)1 << 48, false, 0, false, false, NULL, pkt, &err) == NGV_LINK_E_ALIGN, "49-bit image");
   CHECK(ngv_link_frag_shader_packet(&r, 0x7fff00123450ull, false, 0, true, false, NULL, pkt, &err) == 0 && pkt[0] == 0x123000 &&
            pkt[1] == 0x7fff && pkt[2] == 0x450 && (pkt[7] & 0x1f) == (4u | 16u),
         "48-bit split and flags");
   ngv_link_result_free(&r);
   {
      /* a chain with a zs.test epilogue may be LATE; the packet carries ZS_LATE and ORDERED */
      int ep2 = find_obj("ep_zs_s_at");
      struct ngv_link_seg_ref ch2[2] = {{get_obj(body), g_obj[body].seg}, {get_obj(ep2), g_obj[ep2].seg}};
      struct ngv_link_zs_state dw = {true, false, false, false};
      CHECK(ngv_link(ch2, 2, libs, 1, "fragment", &r, &err) == 0, "link zs epilogue: %s", err.msg);
      CHECK((r.facts & NGV_LINK_FACT_ZS_TEST) && (r.facts & NGV_LINK_FACT_DISCARD), "zs.test epilogue facts %u", r.facts);
      CHECK(ngv_link_frag_shader_packet(&r, 0x10010, true, 0, false, false, &dw, pkt, &err) == 0 && (pkt[7] & 6) == 6,
            "LATE ZS packet");
      CHECK(ngv_link_frag_shader_packet(&r, 0x10010, false, 0, false, false, &dw, pkt, &err) == NGV_LINK_E_ZS_MODE,
            "the same chain refuses EARLY with depth writes");
      ngv_link_result_free(&r);
   }
   CHECK(ngv_link(chain, 2, libs, 1, "geometry", &r, &err) == NGV_LINK_E_STAGE, "wanted stage pinned");
}

/* damaged metadata must never crash or yield an image */
static void
test_fuzz(void)
{
   int ep = find_obj("ep_bl"), body = find_obj("body_flat"), pack = find_obj("lib_pack");
   const struct ngv_link_obj *bo = get_obj(body), *po = get_obj(pack);
   struct tobj *e = &g_obj[ep];
   size_t sz = 0;
   uint8_t *orig = read_file(e->file, &sz);
   uint8_t *mut = (uint8_t *)malloc(sz);
   uint32_t meta_off, meta_len;
   unsigned linked = 0, refused = 0;
   if (!orig || !mut) {
      CHECK(0, "fuzz: no memory / file");
      return;
   }
   memcpy(&meta_off, orig + 8, 4);
   memcpy(&meta_len, orig + 12, 4);
   static const uint8_t pats[] = {'}', '"', '7', 0x00, 'x'};
   for (uint32_t i = 0; i < meta_len; i++)
      for (unsigned p = 0; p < sizeof(pats); p++) {
         struct ngv_link_obj o;
         struct ngv_link_error er;
         memcpy(mut, orig, sz);
         mut[meta_off + i] = pats[p];
         if (ngv_link_obj_open(&o, mut, sz, &er)) {
            refused++;
            continue;
         }
         struct ngv_link_seg_ref chain[2] = {{bo, g_obj[body].seg}, {&o, o.first_seg}};
         const struct ngv_link_obj *libs[1] = {po};
         struct ngv_link_result r;
         if (ngv_link(chain, 2, libs, 1, NULL, &r, &er) == 0) {
            linked++;
            CHECK(r.image && r.size % 16 == 0 && r.size <= 4096, "fuzz: sane image when it links");
            ngv_link_result_free(&r);
         } else {
            refused++;
            CHECK(!r.image, "fuzz: image after an error");
         }
         ngv_link_obj_close(&o);
      }
   /* truncations of the whole file */
   for (size_t n = 0; n < sz; n += 7) {
      struct ngv_link_obj o;
      struct ngv_link_error er;
      memcpy(mut, orig, n);
      if (ngv_link_obj_open(&o, mut, n, &er) == 0)
         ngv_link_obj_close(&o);
   }
   CHECK(refused > 0, "fuzz: some damage is refused (%u refused, %u still link)", refused, linked);
   free(orig);
   free(mut);
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      printf("usage: ngv_link_test <ngcomp_link data dir>\n");
      return 2;
   }
   snprintf(g_dir, sizeof(g_dir), "%s", argv[1]);
   if (!load_manifest() || !g_ncase) {
      printf("ngv_link_test: cannot read %s/manifest.txt\n", g_dir);
      return 2;
   }
   for (int i = 0; i < g_ncase; i++)
      run_case(&g_case[i]);
   test_draws();
   test_packets();
   test_fuzz();
   printf("ngv_link_test: %d cases, %d checks, %d failures\n", g_ncase, g_checks, g_fails);
   return g_fails ? 1 : 0;
}
