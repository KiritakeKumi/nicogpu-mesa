/* SPDX-License-Identifier: MIT */
/* ngv_link.h -- prologue / body / epilogue linking of ngcomp code objects at draw time.
 *
 * The driver joins separately compiled code segments (ABI 0.2, the ISA ABI spec section 5) by concatenation, with no
 * recompilation: segments are laid back to back on 16-byte boundaries, the previous one falls through into the next,
 * calls into shared library routines are the only patched places (the branch field of a `jal`, described by the
 * object's relocation records), and the register interface of every segment (ngcomp/segiface.py) is checked
 * before anything is emitted. This header is the C implementation of ngcomp/seglink.py; the two must agree on
 * every image and every error code (model/tests/test_shader_link.cpp and test/ngv_link_test.c run both on the same
 * objects).
 *
 * Pure, header-only, no Mesa / HAL dependency (needs ngv_xo.h); written in the C / C++ common subset so the host model
 * tests (C++) and the driver (C11) use the same file:
 * gcc -std=c11 -Wall -Wextra -Werror -I src/nicogpu/vulkan src/nicogpu/tests/ngv_link_test.c
 *
 * Pieces:
 *   ngv_link_obj_open / _close     parse one .xo (container header + JSON metadata)
 *   ngv_link()                     link a chain [prologue] body [epilogue] (+ the routines it calls) -> code image
 *   ngv_link_frag_shader_packet()  the eight SET_FRAG_SHADER payload dwords for a linked image
 *   ngv_link_cache                 chain -> linked image (a draw with unchanged state costs a lookup)
 *   ngv_link_variants              (kind, key) -> compiled variant object; the compiler is called only on a miss
 *   ngv_link_ctx + *_for_draw      the draw-time entry points: pick variants from the draw state, link (cached) */
#ifndef NGV_LINK_H
#define NGV_LINK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ngv_xo.h"

/* error codes: identical to CODES in ngcomp/seglink.py */
#define NGV_LINK_E_OK 0
#define NGV_LINK_E_OBJECT 1      /* malformed object / missing metadata */
#define NGV_LINK_E_ABI 2         /* not an ABI 0.2 object / segment without a stage */
#define NGV_LINK_E_STAGE 3       /* segments of different stages in one chain */
#define NGV_LINK_E_ORDER 4       /* not [prologue] body [epilogue] */
#define NGV_LINK_E_NEEDS 5       /* a segment reads a register nobody provides */
#define NGV_LINK_E_ENDPGM 6      /* a non-final segment can end the warp */
#define NGV_LINK_E_DRAIN 7       /* a non-final segment does not end with a full waitcnt */
#define NGV_LINK_E_OPEN 8        /* non-final segment does not fall through / final one does */
#define NGV_LINK_E_INTERLOCK 9   /* interlock section closed before a later segment needs it */
#define NGV_LINK_E_UNRESOLVED 10 /* call to an unknown routine */
#define NGV_LINK_E_RELOC 11      /* relocation not on the right branch / out of range */
#define NGV_LINK_E_ROUTINE 12    /* routine registers disagree with the call site */
#define NGV_LINK_E_VREGS 13      /* more than 128 vector registers */
#define NGV_LINK_E_ZS_MODE 14    /* ZS_LATE required but the draw asks for EARLY */
#define NGV_LINK_E_ALIGN 15      /* address alignment */
#define NGV_LINK_E_UNSUPPORTED 16 /* draw state with no library epilogue (the driver compiles a whole shader) */
#define NGV_LINK_E_PRESERVE 17   /* a segment writes a register it promised to preserve / a prologue writes an entry register /
                                  * "provides" lists a register that is not written on every path (R-B1) */
#define NGV_LINK_E_SCRATCH 18    /* callee scratch frame without / inside / misaligned relative to the caller's frame (R-A) */
#define NGV_LINK_E_ZS_NOTEST 19  /* ZS_LATE requested but no segment runs zs.test (R-E) */
#define NGV_LINK_E_CONVENTION 20 /* a geometry prologue does not provide VertexIndex / InstanceIndex / DrawID in v1 / v2 / v3 (R-B2) */
#define NGV_LINK_E_EXEC 21       /* a geometry segment may reopen lanes the dispatch switched off (R-B2) */

/* Root table / graphics block (GDB) layout of the library shaders (R-C, section 3.4; layout version 1).
 *   root table (u0:u1)   +0x000 .. +0x0ff  push constants (256 bytes, the Vulkan minimum maxPushConstantsSize)
 *                        +0x100            u64 VA of the GDB
 *                        total 0x108 bytes
 *   GDB                  +0x00 alpha reference (f32), +0x04 / +0x08 / +0x0c pass masks for a < ref / a == ref / a > ref,
 * +0x40 + 16 * b vertex binding b {va_lo, va_hi, stride, size} (ngcomp/fslib.py GDB_*)
 * H5 (mandatory driver rule): the fragment stage runs after END_PASS, long after the draw that wrote the root table and the
 * GDB was recorded, and DRAW only snapshots the SET_FRAG_USER_REGS values (the root table VA), never the memory behind them.
 * So a root table and the GDB it points to are IMMUTABLE once a DRAW has captured them: every change of GDB data (alpha
 * reference, blend constants, ...) allocates a NEW GDB and a NEW root table (copy of the push constants + the new GDB
 * pointer) and re-emits SET_FRAG_USER_REGS (u0:u1 = the new root table VA) before the next DRAW.  Rewriting either in
 * place changes the pixels of every earlier draw that shares it (model/tests/test_shader_link.cpp shows that). */
#define NGV_LINK_ROOT_PUSH_BYTES 0x100u
#define NGV_LINK_ROOT_GDB_PTR_OFF 0x100u
#define NGV_LINK_ROOT_TABLE_BYTES 0x108u
#define NGV_LINK_ROOT_LAYOUT_VERSION 1u

#define NGV_LINK_SEG_ALIGN 16u
#define NGV_LINK_MAX_PARTS 72u
#define NGV_LINK_NOP_WORD 0x4eu /* emit.nop_word: padding between pieces (ngcomp/emit.py) */

struct ngv_link_error {
   int code;
   char msg[240];
};

static inline int
ngv_link_fail(struct ngv_link_error *e, int code, const char *fmt, const char *a, const char *b)
{
   if (e) {
      e->code = code;
      snprintf(e->msg, sizeof(e->msg), fmt, a ? a : "", b ? b : "");
   }
   return code;
}

/* ------------------------------------------------------------------------------------------------------------------
 * register sets: v0..v127 in w[0..1], u0..u63 in w[2]
 */
struct ngv_link_regs {
   uint64_t w[3];
};

static inline void
ngv_regs_add(struct ngv_link_regs *r, char cls, unsigned n)
{
   if (cls == 'v')
      r->w[(n >> 6) & 1] |= 1ull << (n & 63);
   else
      r->w[2] |= 1ull << (n & 63);
}

static inline bool
ngv_regs_has(const struct ngv_link_regs *r, char cls, unsigned n)
{
   return cls == 'v' ? ((r->w[(n >> 6) & 1] >> (n & 63)) & 1) : ((r->w[2] >> (n & 63)) & 1);
}

static inline bool
ngv_regs_empty(const struct ngv_link_regs *r)
{
   return !(r->w[0] | r->w[1] | r->w[2]);
}

static inline struct ngv_link_regs
ngv_regs_minus(struct ngv_link_regs a, struct ngv_link_regs b)
{
   a.w[0] &= ~b.w[0];
   a.w[1] &= ~b.w[1];
   a.w[2] &= ~b.w[2];
   return a;
}

static inline struct ngv_link_regs
ngv_regs_or(struct ngv_link_regs a, struct ngv_link_regs b)
{
   a.w[0] |= b.w[0];
   a.w[1] |= b.w[1];
   a.w[2] |= b.w[2];
   return a;
}

static inline void
ngv_regs_fmt(const struct ngv_link_regs *r, char *buf, size_t n)
{
   size_t o = 0;
   const char cls[2] = {'v', 'u'};
   buf[0] = 0;
   for (int c = 0; c < 2; c++) {
      unsigned lim = c ? 64 : 128;
      for (unsigned i = 0; i < lim && o + 12 < n;) {
         if (!ngv_regs_has(r, cls[c], i)) {
            i++;
            continue;
         }
         unsigned j = i;
         while (j + 1 < lim && ngv_regs_has(r, cls[c], j + 1))
            j++;
         if (j == i)
            o += (size_t)snprintf(buf + o, n - o, "%s%c%u", o ? " " : "", cls[c], i);
         else
            o += (size_t)snprintf(buf + o, n - o, "%s%c%u:%u", o ? " " : "", cls[c], i, j - i + 1);
         i = j + 1;
      }
   }
   if (!o)
      snprintf(buf, n, "-");
}

/* "v0:4 v35 u0:2" -> set.  false on a bad token / a register outside the file. */
static inline bool
ngv_regs_parse(const char *s, struct ngv_link_regs *r)
{
   memset(r, 0, sizeof(*r));
   while (*s) {
      while (*s == ' ' || *s == ',')
         s++;
      if (!*s)
         break;
      char cls = *s++;
      if (cls != 'v' && cls != 'u')
         return false;
      char *end;
      unsigned long n = strtoul(s, &end, 10);
      if (end == s)
         return false;
      s = end;
      unsigned long cnt = 1;
      if (*s == ':') {
         cnt = strtoul(s + 1, &end, 10);
         if (end == s + 1)
            return false;
         s = end;
      }
      if (*s && *s != ' ' && *s != ',')
         return false;
      if (cnt < 1 || n + cnt > (cls == 'v' ? 128ul : 64ul))
         return false;
      for (unsigned long k = 0; k < cnt; k++)
         ngv_regs_add(r, cls, (unsigned)(n + k));
   }
   return true;
}

/* ABI.md section 3: caller-saved v0-v31, u16-u31; u60 driver scratch and u62 return address (a call writes it) */
static inline struct ngv_link_regs
ngv_link_caller_saved(void)
{
   struct ngv_link_regs r;
   memset(&r, 0, sizeof(r));
   for (unsigned i = 0; i < 32; i++)
      ngv_regs_add(&r, 'v', i);
   for (unsigned i = 16; i < 32; i++)
      ngv_regs_add(&r, 'u', i);
   ngv_regs_add(&r, 'u', 60);
   ngv_regs_add(&r, 'u', 62);
   return r;
}

/* ------------------------------------------------------------------------------------------------------------------
 * minimal JSON reader (the metadata is json.dumps output of ngcomp/emit.py: objects, arrays, plain strings
 * without escapes, integers, booleans).  Nodes are kept in a flat pool and addressed by index (-1 = none).
 */
enum { NGV_J_NULL, NGV_J_BOOL, NGV_J_NUM, NGV_J_STR, NGV_J_ARR, NGV_J_OBJ };

struct ngv_jnode {
   int32_t next, child;
   uint32_t koff, klen; /* member name (inside an object) */
   uint32_t soff, slen; /* string value */
   int64_t num;         /* number, or bool */
   uint8_t type;
};

struct ngv_json {
   const char *js;
   size_t len, pos;
   struct ngv_jnode *n;
   uint32_t count, cap;
   bool err;
};

static inline int32_t
ngv_json_new(struct ngv_json *j, uint8_t type)
{
   if (j->count == j->cap) {
      uint32_t cap = j->cap ? j->cap * 2 : 256;
      struct ngv_jnode *p = (struct ngv_jnode *)realloc(j->n, (size_t)cap * sizeof(*p));
      if (!p) {
         j->err = true;
         return -1;
      }
      j->n = p;
      j->cap = cap;
   }
   struct ngv_jnode *nd = &j->n[j->count];
   memset(nd, 0, sizeof(*nd));
   nd->type = type;
   nd->next = nd->child = -1;
   return (int32_t)j->count++;
}

static inline void
ngv_json_ws(struct ngv_json *j)
{
   while (j->pos < j->len && (j->js[j->pos] == ' ' || j->js[j->pos] == '\n' || j->js[j->pos] == '\r' || j->js[j->pos] == '\t'))
      j->pos++;
}

static inline bool
ngv_json_str(struct ngv_json *j, uint32_t *off, uint32_t *len)
{
   if (j->pos >= j->len || j->js[j->pos] != '"')
      return false;
   size_t s = ++j->pos;
   while (j->pos < j->len && j->js[j->pos] != '"') {
      if (j->js[j->pos] == '\\')
         return false; /* the writer never escapes; refuse rather than misread */
      j->pos++;
   }
   if (j->pos >= j->len)
      return false;
   *off = (uint32_t)s;
   *len = (uint32_t)(j->pos - s);
   j->pos++;
   return true;
}

static inline int32_t
ngv_json_value(struct ngv_json *j, int depth)
{
   if (depth > 12 || j->err)
      return -1;
   ngv_json_ws(j);
   if (j->pos >= j->len)
      return -1;
   char c = j->js[j->pos];
   if (c == '{' || c == '[') {
      const bool obj = c == '{';
      const char close = obj ? '}' : ']';
      int32_t me = ngv_json_new(j, (uint8_t)(obj ? NGV_J_OBJ : NGV_J_ARR));
      int32_t tail = -1;
      if (me < 0)
         return -1;
      j->pos++;
      ngv_json_ws(j);
      if (j->pos < j->len && j->js[j->pos] == close) {
         j->pos++;
         return me;
      }
      for (;;) {
         uint32_t ko = 0, kl = 0;
         ngv_json_ws(j);
         if (obj) {
            if (!ngv_json_str(j, &ko, &kl))
               return -1;
            ngv_json_ws(j);
            if (j->pos >= j->len || j->js[j->pos] != ':')
               return -1;
            j->pos++;
         }
         int32_t ch = ngv_json_value(j, depth + 1);
         if (ch < 0)
            return -1;
         j->n[ch].koff = ko;
         j->n[ch].klen = kl;
         if (tail < 0)
            j->n[me].child = ch;
         else
            j->n[tail].next = ch;
         tail = ch;
         ngv_json_ws(j);
         if (j->pos >= j->len)
            return -1;
         if (j->js[j->pos] == ',') {
            j->pos++;
            continue;
         }
         if (j->js[j->pos] == close) {
            j->pos++;
            return me;
         }
         return -1;
      }
   }
   if (c == '"') {
      int32_t me = ngv_json_new(j, NGV_J_STR);
      uint32_t o, l;
      if (me < 0 || !ngv_json_str(j, &o, &l))
         return -1;
      j->n[me].soff = o;
      j->n[me].slen = l;
      return me;
   }
   if (c == '-' || (c >= '0' && c <= '9')) {
      int32_t me = ngv_json_new(j, NGV_J_NUM);
      bool neg = false;
      int64_t v = 0;
      if (me < 0)
         return -1;
      if (c == '-') {
         neg = true;
         j->pos++;
      }
      size_t s = j->pos;
      while (j->pos < j->len && j->js[j->pos] >= '0' && j->js[j->pos] <= '9') {
         v = v * 10 + (j->js[j->pos] - '0');
         if (v > (int64_t)1 << 52)
            return -1;
         j->pos++;
      }
      if (j->pos == s || (j->pos < j->len && (j->js[j->pos] == '.' || j->js[j->pos] == 'e' || j->js[j->pos] == 'E')))
         return -1;
      j->n[me].num = neg ? -v : v;
      return me;
   }
   if (j->len - j->pos >= 4 && !memcmp(j->js + j->pos, "true", 4)) {
      int32_t me = ngv_json_new(j, NGV_J_BOOL);
      if (me < 0)
         return -1;
      j->n[me].num = 1;
      j->pos += 4;
      return me;
   }
   if (j->len - j->pos >= 5 && !memcmp(j->js + j->pos, "false", 5)) {
      int32_t me = ngv_json_new(j, NGV_J_BOOL);
      if (me < 0)
         return -1;
      j->pos += 5;
      return me;
   }
   if (j->len - j->pos >= 4 && !memcmp(j->js + j->pos, "null", 4)) {
      int32_t me = ngv_json_new(j, NGV_J_NULL);
      if (me < 0)
         return -1;
      j->pos += 4;
      return me;
   }
   return -1;
}

static inline int32_t
ngv_json_get(const struct ngv_json *j, int32_t obj, const char *key)
{
   if (obj < 0 || j->n[obj].type != NGV_J_OBJ)
      return -1;
   size_t kl = strlen(key);
   for (int32_t c = j->n[obj].child; c >= 0; c = j->n[c].next)
      if (j->n[c].klen == kl && !memcmp(j->js + j->n[c].koff, key, kl))
         return c;
   return -1;
}

static inline bool
ngv_json_streq(const struct ngv_json *j, int32_t node, const char *s)
{
   return node >= 0 && j->n[node].type == NGV_J_STR && j->n[node].slen == strlen(s) &&
          !memcmp(j->js + j->n[node].soff, s, j->n[node].slen);
}

/* copy a string node into buf (always terminated, false when truncated / not a string) */
static inline bool
ngv_json_copy(const struct ngv_json *j, int32_t node, char *buf, size_t n)
{
   if (node < 0 || j->n[node].type != NGV_J_STR || j->n[node].slen >= n) {
      if (n)
         buf[0] = 0;
      return false;
   }
   memcpy(buf, j->js + j->n[node].soff, j->n[node].slen);
   buf[j->n[node].slen] = 0;
   return true;
}

static inline bool
ngv_json_uint(const struct ngv_json *j, int32_t obj, const char *key, uint32_t *out)
{
   int32_t v = ngv_json_get(j, obj, key);
   if (v < 0 || j->n[v].type != NGV_J_NUM || j->n[v].num < 0 || j->n[v].num > 0x7fffffff)
      return false;
   *out = (uint32_t)j->n[v].num;
   return true;
}

/* ------------------------------------------------------------------------------------------------------------------
 * objects and segments
 */
struct ngv_link_obj {
   const uint8_t *data; /* caller keeps the bytes alive while the object is in use */
   size_t size;
   struct ngv_xo_info info;
   const uint8_t *code;
   struct ngv_json js;
   int32_t segs, relocs;
   uint64_t id; /* FNV-1a of the whole file: the cache identity */
   char first_seg[64];
};

static inline void
ngv_link_obj_close(struct ngv_link_obj *o)
{
   free(o->js.n);
   memset(o, 0, sizeof(*o));
}

static inline int
ngv_link_obj_open(struct ngv_link_obj *o, const uint8_t *data, size_t size, struct ngv_link_error *err)
{
   memset(o, 0, sizeof(*o));
   if (!ngv_xo_check_ex(data, size, &o->info, true))
      return ngv_link_fail(err, NGV_LINK_E_OBJECT, "not a valid NGCO object (header check failed)", NULL, NULL);
   o->data = data;
   o->size = size;
   o->code = data + o->info.code_off;
   o->js.js = (const char *)data + o->info.meta_off;
   o->js.len = o->info.meta_len;
   int32_t root = ngv_json_value(&o->js, 0);
   if (root < 0 || o->js.n[root].type != NGV_J_OBJ) {
      ngv_link_obj_close(o);
      return ngv_link_fail(err, NGV_LINK_E_OBJECT, "object metadata is not valid JSON", NULL, NULL);
   }
   o->segs = ngv_json_get(&o->js, root, "segments");
   o->relocs = ngv_json_get(&o->js, root, "relocs");
   if (o->segs < 0 || o->js.n[o->segs].type != NGV_J_ARR) {
      ngv_link_obj_close(o);
      return ngv_link_fail(err, NGV_LINK_E_OBJECT, "object has no segment table", NULL, NULL);
   }
   uint64_t h = 1469598103934665603ull;
   for (size_t i = 0; i < size; i++)
      h = (h ^ data[i]) * 1099511628211ull;
   o->id = h;
   if (o->js.n[o->segs].child >= 0)
      ngv_json_copy(&o->js, ngv_json_get(&o->js, o->js.n[o->segs].child, "name"), o->first_seg, sizeof(o->first_seg));
   return NGV_LINK_E_OK;
}

enum ngv_link_kind { NGV_LK_PROLOGUE, NGV_LK_BODY, NGV_LK_EPILOGUE, NGV_LK_ROUTINE, NGV_LK_BAD };
enum ngv_link_stage { NGV_LS_COMPUTE, NGV_LS_GEOMETRY, NGV_LS_FRAGMENT, NGV_LS_TILE_BEGIN, NGV_LS_TILE_END, NGV_LS_BAD, NGV_LS_NONE };

/* facts the linker reports about the chain (bit numbers shared with seglink.py FACT_* and the manifest, R-E) */
#define NGV_LINK_FACT_DISCARD 1u
#define NGV_LINK_FACT_SMASK 2u
#define NGV_LINK_FACT_SIDE_EFFECT 4u
#define NGV_LINK_FACT_EXPORTS_DEPTH 8u
#define NGV_LINK_FACT_EXPORTS_STENCIL 16u
#define NGV_LINK_FACT_ZS_TEST 32u

#define NGV_TR_TILE_ACCESS 0x01u
#define NGV_TR_ZS_TEST 0x02u
#define NGV_TR_INTERLOCK_BEGIN 0x04u
#define NGV_TR_INTERLOCK_END 0x08u
#define NGV_TR_DISCARD 0x10u
#define NGV_TR_ENDPGM 0x20u
#define NGV_TR_PER_SAMPLE 0x40u
#define NGV_TR_CALLS 0x80u
#define NGV_TR_SMASK 0x100u
#define NGV_TR_SIDE_EFFECT 0x200u
#define NGV_TR_EXEC_WIDEN 0x400u
#define NGV_TR_NAMES 11u

/* longest segment name (a copy of it travels in ngv_link_part) */
#define NGV_LINK_NAME_MAX 64

struct ngv_link_seg {
   const struct ngv_link_obj *obj;
   char name[NGV_LINK_NAME_MAX];
   uint32_t offset, length, entry, vgprs, scratch;
   uint32_t code_end; /* the segment's image runs to the next segment's offset (so its padding travels with it) */
   int stage;
   int kind;
   bool has_link, open, drained;
   uint32_t traits;
   struct ngv_link_regs reads, writes, must_writes, provides, preserves;
};

static inline bool
ngv_link_regs_json(const struct ngv_json *j, int32_t node, struct ngv_link_regs *r)
{
   static const char cls[2] = {'v', 'u'};
   memset(r, 0, sizeof(*r));
   if (node < 0 || j->n[node].type != NGV_J_OBJ)
      return false;
   for (int c = 0; c < 2; c++) {
      char key[2] = {cls[c], 0};
      int32_t arr = ngv_json_get(j, node, key);
      if (arr < 0 || j->n[arr].type != NGV_J_ARR)
         return false;
      for (int32_t p = j->n[arr].child; p >= 0; p = j->n[p].next) {
         int32_t lo = j->n[p].child, hi = lo >= 0 ? j->n[lo].next : -1;
         if (j->n[p].type != NGV_J_ARR || lo < 0 || hi < 0 || j->n[lo].type != NGV_J_NUM || j->n[hi].type != NGV_J_NUM)
            return false;
         int64_t a = j->n[lo].num, b = j->n[hi].num;
         if (a < 0 || b < a || b >= (c ? 64 : 128))
            return false;
         for (int64_t k = a; k <= b; k++)
            ngv_regs_add(r, cls[c], (unsigned)k);
      }
   }
   return true;
}

static inline int
ngv_link_seg_load(const struct ngv_link_obj *o, const char *name, struct ngv_link_seg *s, struct ngv_link_error *err)
{
   const struct ngv_json *j = &o->js;
   int32_t node = -1;
   memset(s, 0, sizeof(*s));
   for (int32_t c = j->n[o->segs].child; c >= 0; c = j->n[c].next)
      if (ngv_json_streq(j, ngv_json_get(j, c, "name"), name)) {
         node = c;
         break;
      }
   if (node < 0)
      return ngv_link_fail(err, NGV_LINK_E_OBJECT, "object has no segment '%s'", name, NULL);
   s->obj = o;
   ngv_json_copy(j, ngv_json_get(j, node, "name"), s->name, sizeof(s->name));
   if (!ngv_json_uint(j, node, "offset", &s->offset) || !ngv_json_uint(j, node, "length", &s->length) ||
       !ngv_json_uint(j, node, "entry", &s->entry))
      return ngv_link_fail(err, NGV_LINK_E_OBJECT, "segment '%s' has no offset / length / entry", name, NULL);
   if (!ngv_json_uint(j, node, "vgprs", &s->vgprs))
      s->vgprs = 0;
   if (!ngv_json_uint(j, node, "scratch", &s->scratch))
      s->scratch = 0;
   if (s->offset % 4 || s->length % 4 || (uint64_t)s->offset + s->length > o->info.code_len)
      return ngv_link_fail(err, NGV_LINK_E_OBJECT, "segment '%s' is not word aligned / runs past the code image", name, NULL);
   s->code_end = o->info.code_len;
   for (int32_t c = j->n[o->segs].child; c >= 0; c = j->n[c].next) {
      uint32_t off;
      if (ngv_json_uint(j, c, "offset", &off) && off > s->offset && off < s->code_end)
         s->code_end = off;
   }
   /* stage */
   int32_t st = ngv_json_get(j, node, "stage_name");
   if (st < 0)
      s->stage = NGV_LS_NONE;
   else if (ngv_json_streq(j, st, "compute"))
      s->stage = NGV_LS_COMPUTE;
   else if (ngv_json_streq(j, st, "geometry"))
      s->stage = NGV_LS_GEOMETRY;
   else if (ngv_json_streq(j, st, "fragment"))
      s->stage = NGV_LS_FRAGMENT;
   else if (ngv_json_streq(j, st, "tile_begin"))
      s->stage = NGV_LS_TILE_BEGIN;
   else if (ngv_json_streq(j, st, "tile_end"))
      s->stage = NGV_LS_TILE_END;
   else
      s->stage = NGV_LS_BAD;
   /* kind */
   int32_t kd = ngv_json_get(j, node, "segment");
   if (kd < 0 || ngv_json_streq(j, kd, "body"))
      s->kind = NGV_LK_BODY;
   else if (ngv_json_streq(j, kd, "prologue"))
      s->kind = NGV_LK_PROLOGUE;
   else if (ngv_json_streq(j, kd, "epilogue"))
      s->kind = NGV_LK_EPILOGUE;
   else if (ngv_json_streq(j, kd, "routine"))
      s->kind = NGV_LK_ROUTINE;
   else
      s->kind = NGV_LK_BAD;
   /* link metadata */
   int32_t lk = ngv_json_get(j, node, "link");
   if (lk >= 0 && j->n[lk].type == NGV_J_OBJ) {
      s->has_link = true;
      if (!ngv_link_regs_json(j, ngv_json_get(j, lk, "reads"), &s->reads) ||
          !ngv_link_regs_json(j, ngv_json_get(j, lk, "writes"), &s->writes) ||
          !ngv_link_regs_json(j, ngv_json_get(j, lk, "must_writes"), &s->must_writes) ||
          !ngv_link_regs_json(j, ngv_json_get(j, lk, "provides"), &s->provides))
         return ngv_link_fail(err, NGV_LINK_E_OBJECT, "segment '%s': malformed link register sets", name, NULL);
      int32_t b = ngv_json_get(j, lk, "open");
      s->open = b >= 0 && j->n[b].type == NGV_J_BOOL && j->n[b].num;
      b = ngv_json_get(j, lk, "drained");
      s->drained = b >= 0 && j->n[b].type == NGV_J_BOOL && j->n[b].num;
      int32_t tr = ngv_json_get(j, lk, "traits");
      static const char *const names[NGV_TR_NAMES] = {"tile_access", "zs_test", "interlock_begin", "interlock_end", "discard",
                                                      "endpgm",      "per_sample", "calls",           "smask",          "side_effect",
                                                      "exec_widen"};
      /* "preserves" is optional (a segment compiled before R-B1 has none) */
      if (ngv_json_get(j, lk, "preserves") >= 0 &&
          !ngv_link_regs_json(j, ngv_json_get(j, lk, "preserves"), &s->preserves))
         return ngv_link_fail(err, NGV_LINK_E_OBJECT, "segment '%s': malformed link register sets", name, NULL);
      if (tr >= 0 && j->n[tr].type == NGV_J_ARR)
         for (int32_t t = j->n[tr].child; t >= 0; t = j->n[t].next)
            for (unsigned k = 0; k < NGV_TR_NAMES; k++)
               if (ngv_json_streq(j, t, names[k]))
                  s->traits |= 1u << k;
   }
   return NGV_LINK_E_OK;
}

/* ------------------------------------------------------------------------------------------------------------------
 * link
 */
struct ngv_link_seg_ref {
   const struct ngv_link_obj *obj;
   const char *name;
};

struct ngv_link_part {
   int kind; /* NGV_LK_* */
   char name[NGV_LINK_NAME_MAX]; /* same size as ngv_link_seg::name: a part name is a copy, never truncated */
   uint32_t offset, length;
};

struct ngv_link_result {
   uint8_t *image;
   size_t size;
   uint32_t entry; /* offset of the first instruction: always 0 */
   struct ngv_link_part parts[NGV_LINK_MAX_PARTS];
   unsigned nparts;
   uint32_t vregs, scratch;
   bool ordered, sample_shading, input_z, input_w;
   uint32_t facts; /* NGV_LINK_FACT_* */
   uint32_t traits;
   int stage;
};

static inline void
ngv_link_result_free(struct ngv_link_result *r)
{
   free(r->image);
   memset(r, 0, sizeof(*r));
}

struct ngv_link_site {
   uint32_t place;   /* image offset of the branch */
   unsigned routine; /* index into the placed pieces */
   int64_t addend;
   int kind; /* 0 br19, 1 br22, 2 br17 */
};

struct ngv_link_piece {
   struct ngv_link_seg seg;
   uint32_t base;
};

static inline bool
ngv_link_reloc_kind(const struct ngv_json *j, int32_t rl, int *kind, unsigned *lsb, unsigned *width, uint32_t *opcode)
{
   int32_t k = ngv_json_get(j, rl, "kind");
   if (ngv_json_streq(j, k, "br19")) {
      *kind = 0, *lsb = 13, *width = 19, *opcode = 0x48;
   } else if (ngv_json_streq(j, k, "br22")) {
      *kind = 1, *lsb = 10, *width = 22, *opcode = 0x46;
   } else if (ngv_json_streq(j, k, "br17")) {
      *kind = 2, *lsb = 15, *width = 17, *opcode = 0x44;
   } else {
      return false;
   }
   return true;
}

static inline int
ngv_link_check_routine(const struct ngv_json *j, int32_t rl, const struct ngv_link_seg *r, const char *sym,
                       struct ngv_link_error *err)
{
   char abuf[96] = "", rbuf[96] = "", txt[160];
   struct ngv_link_regs args, rets, reads, stray, miss, cs = ngv_link_caller_saved();
   if (!r->has_link)
      return ngv_link_fail(err, NGV_LINK_E_OBJECT, "routine '%s' has no link interface metadata", sym, NULL);
   ngv_json_copy(j, ngv_json_get(j, rl, "args"), abuf, sizeof(abuf));
   ngv_json_copy(j, ngv_json_get(j, rl, "rets"), rbuf, sizeof(rbuf));
   if (!ngv_regs_parse(abuf, &args) || !ngv_regs_parse(rbuf, &rets))
      return ngv_link_fail(err, NGV_LINK_E_OBJECT, "bad args / rets list in a call-site record for '%s'", sym, NULL);
   reads = r->reads;
   reads.w[2] &= ~(1ull << 62);
   miss = ngv_regs_minus(reads, args);
   if (!ngv_regs_empty(&miss)) {
      ngv_regs_fmt(&miss, txt, sizeof(txt));
      return ngv_link_fail(err, NGV_LINK_E_ROUTINE, "routine %s reads registers the call site does not pass", sym, txt);
   }
   stray = ngv_regs_minus(ngv_regs_minus(r->writes, rets), cs);
   if (!ngv_regs_empty(&stray)) {
      ngv_regs_fmt(&stray, txt, sizeof(txt));
      return ngv_link_fail(err, NGV_LINK_E_ROUTINE, "routine %s writes registers that are neither returned nor caller-saved",
                           sym, txt);
   }
   miss = ngv_regs_minus(ngv_regs_minus(rets, r->must_writes), args);
   if (!ngv_regs_empty(&miss))
      return ngv_link_fail(err, NGV_LINK_E_ROUTINE, "routine %s does not write its returned registers on every path", sym, NULL);
   return NGV_LINK_E_OK;
}

static inline int
ngv_link_patch(uint8_t *image, size_t size, uint32_t place, int64_t target, int kind, const char *sym,
               struct ngv_link_error *err)
{
   static const unsigned lsb_of[3] = {13, 10, 15}, width_of[3] = {19, 22, 17};
   static const uint32_t opc_of[3] = {0x48, 0x46, 0x44};
   unsigned lsb = lsb_of[kind], width = width_of[kind];
   uint32_t word;
   if ((uint64_t)place + 4 > size)
      return ngv_link_fail(err, NGV_LINK_E_RELOC, "relocation to %s lies outside the image", sym, NULL);
   memcpy(&word, image + place, 4);
   if ((word & 0x7f) != opc_of[kind])
      return ngv_link_fail(err, NGV_LINK_E_RELOC, "relocation to %s: the instruction is not a branch of this kind", sym, NULL);
   uint32_t mask = (uint32_t)(((1ull << width) - 1) << lsb);
   if (word & mask)
      return ngv_link_fail(err, NGV_LINK_E_RELOC, "relocation to %s: branch field is not zero (already linked?)", sym, NULL);
   int64_t dist = target - (int64_t)place;
   if (dist % 4)
      return ngv_link_fail(err, NGV_LINK_E_RELOC, "relocation to %s: unaligned branch distance", sym, NULL);
   int64_t q = dist / 4;
   if (q < -((int64_t)1 << (width - 1)) || q >= ((int64_t)1 << (width - 1)))
      return ngv_link_fail(err, NGV_LINK_E_RELOC, "relocation to %s: branch distance does not fit the field", sym, NULL);
   word |= ((uint32_t)q & (uint32_t)((1ull << width) - 1)) << lsb;
   memcpy(image + place, &word, 4);
   return NGV_LINK_E_OK;
}

/* entry registers (ABI.md section 2 + ERRATA v0.1.7 F1) */
static inline struct ngv_link_regs
ngv_link_entry_state(int stage, bool input_z, bool input_w)
{
   struct ngv_link_regs r;
   memset(&r, 0, sizeof(r));
   r.w[2] = ~0ull;
   if (stage == NGV_LS_COMPUTE || stage == NGV_LS_GEOMETRY) {
      ngv_regs_add(&r, 'v', 0);
   } else {
      for (unsigned i = 0; i < 4; i++)
         ngv_regs_add(&r, 'v', i);
      if (stage == NGV_LS_FRAGMENT) {
         if (input_z)
            ngv_regs_add(&r, 'v', 4);
         if (input_w)
            ngv_regs_add(&r, 'v', 5);
      }
   }
   return r;
}

/* Registers a PROLOGUE of this stage may not write (segiface.protected_entry): u0-u30 (ABI.md v0.3) and the entry vector registers */
static inline struct ngv_link_regs
ngv_link_protected_entry(int stage)
{
   struct ngv_link_regs r;
   unsigned nv = 1;
   memset(&r, 0, sizeof(r));
   for (unsigned i = 0; i < 31; i++)
      ngv_regs_add(&r, 'u', i);
   if (stage == NGV_LS_FRAGMENT)
      nv = 6;
   else if (stage == NGV_LS_TILE_BEGIN || stage == NGV_LS_TILE_END)
      nv = 4;
   for (unsigned i = 0; i < nv; i++)
      ngv_regs_add(&r, 'v', i);
   return r;
}

#define NGV_LINK_NO_FRAME (-1)

struct ngv_link_edge {
   unsigned caller, callee;
   int64_t frame; /* call-site scratch frame offset, NGV_LINK_NO_FRAME when the record has none */
};

/* total(p) = max(own(p), max over calls p -> q of frame + total(q)); a call whose callee tree uses scratch must give a frame
 * offset >= the caller's own (16-aligned) frame and 16-byte aligned (seglink.py _scratch_total).  state: 0 new, 1 open, 2 done */
static inline int
ngv_link_scratch_visit(unsigned p, const uint32_t *own, const struct ngv_link_edge *edges, unsigned nedges, uint8_t *state,
                       uint64_t *total, const char *const *names, struct ngv_link_error *err)
{
   uint64_t t = own[p];
   if (state[p] == 2)
      return NGV_LINK_E_OK;
   if (state[p] == 1)
      return ngv_link_fail(err, NGV_LINK_E_ROUTINE, "routine call cycle through '%s' (no finite scratch frame)", names[p], NULL);
   state[p] = 1;
   for (unsigned e = 0; e < nedges; e++) {
      if (edges[e].caller != p)
         continue;
      const unsigned q = edges[e].callee;
      int rc = ngv_link_scratch_visit(q, own, edges, nedges, state, total, names, err);
      if (rc)
         return rc;
      if (total[q]) {
         if (edges[e].frame == NGV_LINK_NO_FRAME)
            return ngv_link_fail(err, NGV_LINK_E_SCRATCH,
                                 "call from '%s': the callee needs scratch, the call site gives no frame offset", names[p], NULL);
         if (edges[e].frame < (int64_t)own[p] || edges[e].frame % 16)
            return ngv_link_fail(err, NGV_LINK_E_SCRATCH,
                                 "call from '%s': frame offset overlaps the caller's frame or is not 16-byte aligned", names[p], NULL);
         if ((uint64_t)edges[e].frame + total[q] > t)
            t = (uint64_t)edges[e].frame + total[q];
      }
   }
   state[p] = 2;
   total[p] = t;
   return NGV_LINK_E_OK;
}

static inline int
ngv_link_grow(uint8_t **buf, size_t *cap, size_t need)
{
   if (need <= *cap)
      return 0;
   size_t nc = *cap ? *cap : 256;
   while (nc < need)
      nc *= 2;
   uint8_t *p = (uint8_t *)realloc(*buf, nc);
   if (!p)
      return 1;
   *buf = p;
   *cap = nc;
   return 0;
}

/* Link `chain` (execution order) into one image.  `libs`: objects searched for routines after the chain's own.
 * `want_stage` (a stage name or NULL) pins the stage of the chain.  On success *res owns the image (ngv_link_result_free). */
static inline int
ngv_link(const struct ngv_link_seg_ref *chain, unsigned n, const struct ngv_link_obj *const *libs, unsigned nlibs,
         const char *want_stage, struct ngv_link_result *res, struct ngv_link_error *err)
{
   struct ngv_link_piece *pieces = NULL;
   struct ngv_link_site *sites = NULL;
   struct ngv_link_edge *edges = NULL;
   uint8_t *image = NULL;
   size_t isz = 0, icap = 0;
   unsigned npieces = 0, nsites = 0, nedges = 0, nchain = n, i, k;
   int rc = NGV_LINK_E_OK;
   struct ngv_link_regs avail;
   char txt[160];
   static const int ok_seq[4][3] = {{NGV_LK_BODY, -1, -1},
                                    {NGV_LK_PROLOGUE, NGV_LK_BODY, -1},
                                    {NGV_LK_BODY, NGV_LK_EPILOGUE, -1},
                                    {NGV_LK_PROLOGUE, NGV_LK_BODY, NGV_LK_EPILOGUE}};
   static const char *const stage_names[5] = {"compute", "geometry", "fragment", "tile_begin", "tile_end"};

   memset(res, 0, sizeof(*res));
   if (n == 0)
      return ngv_link_fail(err, NGV_LINK_E_ORDER, "empty chain", NULL, NULL);
   if (n > 3)
      return ngv_link_fail(err, NGV_LINK_E_ORDER, "more than three segments in one chain", NULL, NULL);
   pieces = (struct ngv_link_piece *)calloc(NGV_LINK_MAX_PARTS, sizeof(*pieces));
   sites = (struct ngv_link_site *)calloc(256, sizeof(*sites));
   edges = (struct ngv_link_edge *)calloc(256, sizeof(*edges));
   if (!pieces || !sites || !edges) {
      free(pieces);
      free(sites);
      free(edges);
      return ngv_link_fail(err, NGV_LINK_E_OBJECT, "out of memory", NULL, NULL);
   }
#define NGV_LINK_FAIL(code, fmt, a, b)           \
   do {                                          \
      rc = ngv_link_fail(err, code, fmt, a, b);  \
      goto out;                                  \
   } while (0)

   /* 1. load and sanity-check the chain */
   for (i = 0; i < n; i++) {
      if (chain[i].obj->info.abi != NGV_XO_ABI_0_2 && chain[i].obj->info.abi != NGV_XO_ABI_0_3)
         NGV_LINK_FAIL(NGV_LINK_E_ABI, "object of segment '%s' is not ABI 0.2 / 0.3: linking needs ABI 0.2 / 0.3 segments", chain[i].name, NULL);
      rc = ngv_link_seg_load(chain[i].obj, chain[i].name, &pieces[i].seg, err);
      if (rc)
         goto out;
      if (pieces[i].seg.stage == NGV_LS_NONE)
         NGV_LINK_FAIL(NGV_LINK_E_ABI, "segment '%s' has no stage", chain[i].name, NULL);
   }
   if (want_stage) {
      const char *have = pieces[0].seg.stage < 5 ? stage_names[pieces[0].seg.stage] : "?";
      if (strcmp(have, want_stage))
         NGV_LINK_FAIL(NGV_LINK_E_STAGE, "chain stage '%s', wanted '%s'", have, want_stage);
   }
   for (i = 0; i < n; i++)
      if (pieces[i].seg.stage != pieces[0].seg.stage)
         NGV_LINK_FAIL(NGV_LINK_E_STAGE, "segment '%s' belongs to another stage than the rest of the chain", pieces[i].seg.name, NULL);
   {
      bool seq_ok = false;
      for (k = 0; k < 4 && !seq_ok; k++) {
         unsigned len = 0;
         while (len < 3 && ok_seq[k][len] >= 0)
            len++;
         if (len != n)
            continue;
         seq_ok = true;
         for (i = 0; i < n; i++)
            if (pieces[i].seg.kind != ok_seq[k][i])
               seq_ok = false;
      }
      if (!seq_ok)
         NGV_LINK_FAIL(NGV_LINK_E_ORDER, "segment kinds are not [prologue] body [epilogue]", NULL, NULL);
   }
   res->stage = pieces[0].seg.stage;
   for (i = 0; i < n; i++) {
      const struct ngv_link_seg *s = &pieces[i].seg;
      if (!s->has_link)
         NGV_LINK_FAIL(NGV_LINK_E_OBJECT, "segment '%s' has no link interface metadata", s->name, NULL);
      if (i + 1 != n) {
         if (!s->open)
            NGV_LINK_FAIL(NGV_LINK_E_OPEN, "segment '%s' is followed by another segment but does not fall through", s->name, NULL);
         if (!s->drained)
            NGV_LINK_FAIL(NGV_LINK_E_DRAIN, "segment '%s' does not end with a full waitcnt", s->name, NULL);
         if (s->traits & NGV_TR_ENDPGM)
            NGV_LINK_FAIL(NGV_LINK_E_ENDPGM, "segment '%s' can end the warp; the following segments would not run", s->name, NULL);
      } else if (s->open) {
         NGV_LINK_FAIL(NGV_LINK_E_OPEN, "the last segment '%s' falls through past the end of the code", s->name, NULL);
      }
      res->traits |= s->traits;
   }
   if (res->stage == NGV_LS_BAD)
      NGV_LINK_FAIL(NGV_LINK_E_STAGE, "unknown stage", NULL, NULL);
   /* 2. R-B1: preserves / provides / the prologue's protected registers; R-B2: geometry conventions and EXEC */
   {
      const struct ngv_link_regs prot = ngv_link_protected_entry(res->stage);
      for (i = 0; i < n; i++) {
         const struct ngv_link_seg *s = &pieces[i].seg;
         struct ngv_link_regs bad;
         bad = ngv_regs_minus(s->preserves, ngv_regs_minus(s->preserves, s->writes)); /* preserves & writes */
         if (!ngv_regs_empty(&bad)) {
            ngv_regs_fmt(&bad, txt, sizeof(txt));
            NGV_LINK_FAIL(NGV_LINK_E_PRESERVE, "segment '%s' promises to preserve registers it writes: %s", s->name, txt);
         }
         bad = ngv_regs_minus(s->preserves, ngv_regs_minus(s->preserves, s->provides)); /* preserves & provides */
         if (!ngv_regs_empty(&bad)) {
            ngv_regs_fmt(&bad, txt, sizeof(txt));
            NGV_LINK_FAIL(NGV_LINK_E_PRESERVE, "segment '%s' both preserves and provides: %s", s->name, txt);
         }
         if (i + 1 != n) {
            bad = ngv_regs_minus(s->provides, s->must_writes);
            if (!ngv_regs_empty(&bad)) {
               ngv_regs_fmt(&bad, txt, sizeof(txt));
               NGV_LINK_FAIL(NGV_LINK_E_PRESERVE, "segment '%s' provides registers it does not write on every path: %s", s->name, txt);
            }
         }
         if (s->kind == NGV_LK_PROLOGUE) {
            bad = ngv_regs_minus(s->writes, ngv_regs_minus(s->writes, prot)); /* writes & prot */
            if (!ngv_regs_empty(&bad)) {
               ngv_regs_fmt(&bad, txt, sizeof(txt));
               NGV_LINK_FAIL(NGV_LINK_E_PRESERVE, "prologue '%s' writes entry registers: %s", s->name, txt);
            }
            if (res->stage == NGV_LS_GEOMETRY &&
                !(ngv_regs_has(&s->provides, 'v', 1) && ngv_regs_has(&s->provides, 'v', 2) && ngv_regs_has(&s->provides, 'v', 3)))
               NGV_LINK_FAIL(NGV_LINK_E_CONVENTION,
                             "geometry prologue '%s' must provide VertexIndex / InstanceIndex / DrawID in v1 / v2 / v3", s->name, NULL);
         }
      }
      if (res->stage == NGV_LS_GEOMETRY)
         for (i = 0; i < n; i++)
            if (pieces[i].seg.traits & NGV_TR_EXEC_WIDEN)
               NGV_LINK_FAIL(NGV_LINK_E_EXEC, "segment '%s' may reopen lanes the dispatch (or the prologue) switched off in EXEC",
                             pieces[i].seg.name, NULL);
   }
   /* 3. register availability */
   for (i = 0; i < n; i++) {
      if (ngv_regs_has(&pieces[i].seg.reads, 'v', 4) && res->stage == NGV_LS_FRAGMENT)
         res->input_z = true;
      if (ngv_regs_has(&pieces[i].seg.reads, 'v', 5) && res->stage == NGV_LS_FRAGMENT)
         res->input_w = true;
   }
   avail = ngv_link_entry_state(res->stage, res->input_z, res->input_w);
   for (i = 0; i < n; i++) {
      const struct ngv_link_seg *s = &pieces[i].seg;
      struct ngv_link_regs miss = ngv_regs_minus(s->reads, avail);
      if (!ngv_regs_empty(&miss)) {
         ngv_regs_fmt(&miss, txt, sizeof(txt));
         NGV_LINK_FAIL(NGV_LINK_E_NEEDS, "segment '%s' reads registers nothing provides: %s", s->name, txt);
      }
      avail = ngv_regs_or(ngv_regs_minus(avail, s->writes), s->provides);
   }
   /* 4. interlock critical section, one zs.test per warp */
   for (i = 0; i < n; i++)
      if (pieces[i].seg.traits & NGV_TR_INTERLOCK_END)
         for (k = i + 1; k < n; k++)
            if (pieces[k].seg.traits & (NGV_TR_TILE_ACCESS | NGV_TR_ZS_TEST | NGV_TR_INTERLOCK_BEGIN))
               NGV_LINK_FAIL(NGV_LINK_E_INTERLOCK, "segment '%s' closes the interlock section before a later segment needs it",
                             pieces[i].seg.name, NULL);
   {
      unsigned nzs = 0;
      for (i = 0; i < n; i++)
         if (pieces[i].seg.traits & NGV_TR_ZS_TEST)
            nzs++;
      if (nzs > 1)
         NGV_LINK_FAIL(NGV_LINK_E_ZS_MODE, "more than one segment runs zs.test (a warp may execute it once)", NULL, NULL);
   }
   res->ordered = (res->traits & (NGV_TR_TILE_ACCESS | NGV_TR_ZS_TEST | NGV_TR_INTERLOCK_BEGIN | NGV_TR_INTERLOCK_END)) != 0;
   {
      struct ngv_link_regs prov;
      memset(&prov, 0, sizeof(prov));
      for (i = 0; i < n; i++)
         prov = ngv_regs_or(prov, pieces[i].seg.provides);
      res->facts = ((res->traits & NGV_TR_DISCARD) ? NGV_LINK_FACT_DISCARD : 0u) |
                   ((res->traits & NGV_TR_SMASK) ? NGV_LINK_FACT_SMASK : 0u) |
                   ((res->traits & NGV_TR_SIDE_EFFECT) ? NGV_LINK_FACT_SIDE_EFFECT : 0u) |
                   (ngv_regs_has(&prov, 'v', 32) ? NGV_LINK_FACT_EXPORTS_DEPTH : 0u) |
                   (ngv_regs_has(&prov, 'v', 33) ? NGV_LINK_FACT_EXPORTS_STENCIL : 0u) |
                   ((res->traits & NGV_TR_ZS_TEST) ? NGV_LINK_FACT_ZS_TEST : 0u);
   }
   res->sample_shading = (res->traits & NGV_TR_PER_SAMPLE) != 0;

   /* 4. layout: the chain, then the routines it calls (transitively) */
#define NGV_LINK_PLACE(idx, pkind)                                                                          \
   do {                                                                                                     \
      struct ngv_link_piece *pc_ = &pieces[idx];                                                            \
      uint32_t len_ = pc_->seg.code_end - pc_->seg.offset;                                                  \
      while (isz % NGV_LINK_SEG_ALIGN) {                                                                    \
         if (isz + 4 > icap && ngv_link_grow(&image, &icap, isz + 4)) NGV_LINK_FAIL(NGV_LINK_E_OBJECT, "out of memory", NULL, NULL); \
         { uint32_t nop_ = NGV_LINK_NOP_WORD; memcpy(image + isz, &nop_, 4); }                              \
         isz += 4;                                                                                          \
      }                                                                                                     \
      if (isz + len_ > icap && ngv_link_grow(&image, &icap, isz + len_)) NGV_LINK_FAIL(NGV_LINK_E_OBJECT, "out of memory", NULL, NULL); \
      memcpy(image + isz, pc_->seg.obj->code + pc_->seg.offset, len_);                                      \
      pc_->base = (uint32_t)isz;                                                                            \
      isz += len_;                                                                                          \
      if (res->nparts < NGV_LINK_MAX_PARTS) {                                                               \
         struct ngv_link_part *pt_ = &res->parts[res->nparts++];                                            \
         pt_->kind = (pkind);                                                                               \
         snprintf(pt_->name, sizeof(pt_->name), "%s", pc_->seg.name);                                       \
         pt_->offset = pc_->base;                                                                           \
         pt_->length = len_;                                                                                \
      }                                                                                                     \
      if (pc_->seg.vgprs > res->vregs) res->vregs = pc_->seg.vgprs;                                         \
   } while (0)

   npieces = n;
   for (i = 0; i < n; i++)
      NGV_LINK_PLACE(i, pieces[i].seg.kind);
   for (k = 0; k < npieces; k++) {
      const struct ngv_link_seg *s = &pieces[k].seg;
      const struct ngv_json *j = &s->obj->js;
      int32_t rl;
      uint32_t lo = s->offset, hi = s->code_end;
      if (s->obj->relocs < 0)
         continue;
      for (rl = j->n[s->obj->relocs].child; rl >= 0; rl = j->n[rl].next) {
         uint32_t off;
         char sym[64];
         int kind;
         unsigned lsb, width;
         uint32_t opc;
         unsigned r, found = 0;
         bool have = false;
         struct ngv_link_seg rs;
         const struct ngv_link_obj *robj = NULL;
         int64_t addend = 0;
         int32_t ad;
         memset(&rs, 0, sizeof(rs));
         if (!ngv_json_uint(j, rl, "offset", &off))
            NGV_LINK_FAIL(NGV_LINK_E_OBJECT, "relocation record without an offset", NULL, NULL);
         if (off < lo || off >= hi)
            continue;
         if (!ngv_json_copy(j, ngv_json_get(j, rl, "symbol"), sym, sizeof(sym)))
            NGV_LINK_FAIL(NGV_LINK_E_OBJECT, "relocation record without a symbol", NULL, NULL);
         if (!ngv_link_reloc_kind(j, rl, &kind, &lsb, &width, &opc))
            NGV_LINK_FAIL(NGV_LINK_E_OBJECT, "relocation to '%s' has an unknown kind", sym, NULL);
         ad = ngv_json_get(j, rl, "addend");
         if (ad >= 0 && j->n[ad].type == NGV_J_NUM)
            addend = j->n[ad].num;
         /* routine lookup: the chain's own objects, then the libraries */
         for (r = 0; r < nchain + nlibs && !have; r++) {
            const struct ngv_link_obj *cand = r < nchain ? chain[r].obj : libs[r - nchain];
            struct ngv_link_error ignore;
            if (ngv_link_seg_load(cand, sym, &rs, &ignore) == NGV_LINK_E_OK && rs.kind == NGV_LK_ROUTINE) {
               have = true;
               robj = cand;
            }
         }
         if (!have)
            NGV_LINK_FAIL(NGV_LINK_E_UNRESOLVED, "undefined symbol '%s'", sym, NULL);
         if (robj->info.abi != NGV_XO_ABI_0_3)
            NGV_LINK_FAIL(NGV_LINK_E_ABI, "routine '%s' is not an ABI 0.3 object", sym, NULL);
         for (r = 0; r < npieces; r++)
            if (pieces[r].seg.obj->id == robj->id && !strcmp(pieces[r].seg.name, sym)) {
               found = r + 1;
               break;
            }
         if (!found) {
            rc = ngv_link_check_routine(j, rl, &rs, sym, err);
            if (rc)
               goto out;
            if (npieces >= NGV_LINK_MAX_PARTS - 1)
               NGV_LINK_FAIL(NGV_LINK_E_RELOC, "routine nesting too deep", NULL, NULL);
            pieces[npieces].seg = rs;
            found = npieces + 1;
            npieces++;
            NGV_LINK_PLACE(found - 1, NGV_LK_ROUTINE);
         }
         if (nsites >= 256)
            NGV_LINK_FAIL(NGV_LINK_E_RELOC, "too many call sites", NULL, NULL);
         sites[nsites].place = pieces[k].base + (off - s->offset);
         sites[nsites].routine = found - 1;
         sites[nsites].addend = addend;
         sites[nsites].kind = kind;
         nsites++;
         {
            int32_t fr = ngv_json_get(j, rl, "frame");
            edges[nedges].caller = k;
            edges[nedges].callee = found - 1;
            edges[nedges].frame = (fr >= 0 && j->n[fr].type == NGV_J_NUM) ? j->n[fr].num : NGV_LINK_NO_FRAME;
            nedges++;
         }
      }
   }
   while (isz % NGV_LINK_SEG_ALIGN) {
      if (isz + 4 > icap && ngv_link_grow(&image, &icap, isz + 4))
         NGV_LINK_FAIL(NGV_LINK_E_OBJECT, "out of memory", NULL, NULL);
      {
         uint32_t nop = NGV_LINK_NOP_WORD;
         memcpy(image + isz, &nop, 4);
      }
      isz += 4;
   }
   for (i = 0; i < nsites; i++) {
      const struct ngv_link_piece *rp = &pieces[sites[i].routine];
      int64_t target = (int64_t)rp->base + sites[i].addend + ((int64_t)rp->seg.entry - (int64_t)rp->seg.offset);
      rc = ngv_link_patch(image, isz, sites[i].place, target, sites[i].kind, rp->seg.name, err);
      if (rc)
         goto out;
   }
   if (res->vregs > 128)
      NGV_LINK_FAIL(NGV_LINK_E_VREGS, "needs more than 128 vector registers", NULL, NULL);
   if (res->input_w && res->vregs < 6)
      res->vregs = 6;
   else if (res->input_z && res->vregs < 5)
      res->vregs = 5;
   if (res->vregs < 1)
      res->vregs = 1;
   /* R-A: a callee's scratch frame lives while the caller's does, so the frames add up along every call path (a routine
    * call cycle has no finite frame: refused); the chain's segments run one after the other and share */
   {
      uint32_t own[NGV_LINK_MAX_PARTS];
      uint64_t total[NGV_LINK_MAX_PARTS];
      uint8_t state[NGV_LINK_MAX_PARTS];
      const char *names[NGV_LINK_MAX_PARTS];
      uint64_t best = 0;
      memset(total, 0, sizeof(total));
      memset(state, 0, sizeof(state));
      for (i = 0; i < npieces; i++) {
         own[i] = (pieces[i].seg.scratch + 15) / 16 * 16;
         names[i] = pieces[i].seg.name;
      }
      for (i = 0; i < npieces; i++) {
         rc = ngv_link_scratch_visit(i, own, edges, nedges, state, total, names, err);
         if (rc)
            goto out;
      }
      for (i = 0; i < nchain; i++)
         if (total[i] > best)
            best = total[i];
      res->scratch = (uint32_t)((best + 15) / 16 * 16);
   }
   res->image = image;
   res->size = isz;
   image = NULL;
   res->entry = 0;
out:
   free(image);
   free(pieces);
   free(sites);
   free(edges);
   if (rc)
      memset(res, 0, sizeof(*res));
   return rc;
#undef NGV_LINK_PLACE
#undef NGV_LINK_FAIL
}

/* The draw state GFX.md section 4.4 needs for the EARLY / LATE decision (R-E: the linker reports facts, the driver decides). */
struct ngv_link_zs_state {
   bool depth_write, stencil_write, early_fragment_tests, alpha_to_coverage;
};

static inline bool
ngv_link_zs_late_needed(const struct ngv_link_result *r, const struct ngv_link_zs_state *z)
{
   const bool exports = (r->facts & (NGV_LINK_FACT_EXPORTS_DEPTH | NGV_LINK_FACT_EXPORTS_STENCIL)) != 0;
   const bool coverage = (r->facts & (NGV_LINK_FACT_DISCARD | NGV_LINK_FACT_SMASK)) != 0 || z->alpha_to_coverage;
   const bool side = (r->facts & NGV_LINK_FACT_SIDE_EFFECT) != 0;
   return (exports && !z->early_fragment_tests) || (coverage && (z->depth_write || z->stencil_write)) ||
          (side && !z->early_fragment_tests);
}

/* SET_FRAG_SHADER payload (the graphics packet table): CODE_BASE_LO/HI, ENTRY, RSRC, SCRATCH_BYTES, SCRATCH_VA_LO/HI, FLAGS.
 * The linked image lives at `image_va` (16-byte aligned, 48-bit): CODE_BASE is its 4 KiB page, ENTRY the offset in it. */
static inline int
ngv_link_frag_shader_packet(const struct ngv_link_result *r, uint64_t image_va, bool zs_late, uint64_t scratch_va,
                            bool occ_at_end, bool null_fs, const struct ngv_link_zs_state *zs, uint32_t out[8],
                            struct ngv_link_error *err)
{
   if (image_va % 16)
      return ngv_link_fail(err, NGV_LINK_E_ALIGN, "image address is not 16-byte aligned", NULL, NULL);
   if (image_va >> 48)
      return ngv_link_fail(err, NGV_LINK_E_ALIGN, "image address is beyond 48 bits", NULL, NULL);
   if (r->scratch && ((scratch_va % 256) || (scratch_va >> 48)))
      return ngv_link_fail(err, NGV_LINK_E_ALIGN, "scratch address is not 256-byte aligned / beyond 48 bits", NULL, NULL);
   if (!r->scratch)
      scratch_va = 0;
   if (!null_fs) {
      if (zs && !zs_late && ngv_link_zs_late_needed(r, zs))
         return ngv_link_fail(err, NGV_LINK_E_ZS_MODE, "GFX.md section 4.4 requires ZS_LATE for this shader and draw state", NULL, NULL);
      if (zs_late && !(r->facts & NGV_LINK_FACT_ZS_TEST))
         return ngv_link_fail(err, NGV_LINK_E_ZS_NOTEST, "ZS_LATE needs a segment that runs zs.test", NULL, NULL);
   }
   if (r->vregs > 128)
      return ngv_link_fail(err, NGV_LINK_E_VREGS, "VREGS field is 1..128", NULL, NULL);
   uint64_t code_base = image_va & ~0xfffull;
   out[0] = (uint32_t)code_base;
   out[1] = (uint32_t)((code_base >> 32) & 0xffff);
   out[2] = (uint32_t)(image_va & 0xfff);
   out[3] = (r->vregs & 0xffu) | (r->input_z ? 0x100u : 0) | (r->input_w ? 0x200u : 0);
   out[4] = r->scratch;
   out[5] = (uint32_t)scratch_va;
   out[6] = (uint32_t)((scratch_va >> 32) & 0xffff);
   out[7] = (null_fs ? 1u : 0) | (zs_late ? 2u : 0) | (r->ordered ? 4u : 0) | (r->sample_shading ? 8u : 0) |
            (occ_at_end ? 16u : 0);
   return NGV_LINK_E_OK;
}

/* ------------------------------------------------------------------------------------------------------------------
 * cache: linked images by chain identity
 */
#define NGV_LINK_CACHE_MAX 64u

struct ngv_link_cache_entry {
   char key[512];
   struct ngv_link_result res;
   bool used;
};

struct ngv_link_cache {
   struct ngv_link_cache_entry e[NGV_LINK_CACHE_MAX];
   unsigned next, links, hits;
};

static inline void
ngv_link_cache_init(struct ngv_link_cache *c)
{
   memset(c, 0, sizeof(*c));
}

static inline void
ngv_link_cache_finish(struct ngv_link_cache *c)
{
   for (unsigned i = 0; i < NGV_LINK_CACHE_MAX; i++)
      if (c->e[i].used)
         ngv_link_result_free(&c->e[i].res);
   memset(c, 0, sizeof(*c));
}

/* Drop every entry (e.g. after the code heap was recycled).  Results returned earlier become invalid. */
static inline void
ngv_link_cache_flush(struct ngv_link_cache *c)
{
   unsigned links = c->links, hits = c->hits;
   ngv_link_cache_finish(c);
   c->links = links;
   c->hits = hits;
}

/* The returned result stays valid until the entry is evicted (the 65th distinct chain) or the cache is flushed. */
static inline const struct ngv_link_result *
ngv_link_cache_get(struct ngv_link_cache *c, const struct ngv_link_seg_ref *chain, unsigned n,
                   const struct ngv_link_obj *const *libs, unsigned nlibs, struct ngv_link_error *err)
{
   char key[512];
   size_t o = 0;
   unsigned i;
   for (i = 0; i < n && o < sizeof(key) - 100; i++)
      o += (size_t)snprintf(key + o, sizeof(key) - o, "%016llx/%s;", (unsigned long long)chain[i].obj->id, chain[i].name);
   for (i = 0; i < nlibs && o < sizeof(key) - 40; i++)
      o += (size_t)snprintf(key + o, sizeof(key) - o, "L%016llx;", (unsigned long long)libs[i]->id);
   if (o >= sizeof(key) - 40) {
      ngv_link_fail(err, NGV_LINK_E_ORDER, "link key too long", NULL, NULL);
      return NULL;
   }
   for (i = 0; i < NGV_LINK_CACHE_MAX; i++)
      if (c->e[i].used && !strcmp(c->e[i].key, key)) {
         c->hits++;
         return &c->e[i].res;
      }
   struct ngv_link_cache_entry *slot = NULL;
   for (i = 0; i < NGV_LINK_CACHE_MAX; i++)
      if (!c->e[i].used) {
         slot = &c->e[i];
         break;
      }
   if (!slot) {
      slot = &c->e[c->next++ % NGV_LINK_CACHE_MAX];
      ngv_link_result_free(&slot->res);
      slot->used = false;
   }
   if (ngv_link(chain, n, libs, nlibs, NULL, &slot->res, err))
      return NULL;
   memcpy(slot->key, key, o + 1);
   slot->used = true;
   c->links++;
   return &slot->res;
}

/* ------------------------------------------------------------------------------------------------------------------
 * variants: library prologue / epilogue objects by (kind, key); the compiler runs only on a miss
 */
#define NGV_LINK_VARIANT_MAX 64u

/* Compile the variant `kind`:`key`; on success returns 0 with a malloc'd .xo image the cache takes over. */
typedef int (*ngv_link_compile_fn)(void *user, const char *kind, const char *key, uint8_t **data, size_t *size);

struct ngv_link_variant {
   char id[320]; /* "kind:key" */
   uint8_t *data;
   struct ngv_link_obj obj;
   bool used;
};

struct ngv_link_variants {
   ngv_link_compile_fn compile;
   void *user;
   struct ngv_link_variant e[NGV_LINK_VARIANT_MAX];
   unsigned compiles, hits;
};

static inline void
ngv_link_variants_init(struct ngv_link_variants *v, ngv_link_compile_fn fn, void *user)
{
   memset(v, 0, sizeof(*v));
   v->compile = fn;
   v->user = user;
}

static inline void
ngv_link_variants_finish(struct ngv_link_variants *v)
{
   for (unsigned i = 0; i < NGV_LINK_VARIANT_MAX; i++)
      if (v->e[i].used) {
         ngv_link_obj_close(&v->e[i].obj);
         free(v->e[i].data);
      }
   memset(v, 0, sizeof(*v));
}

static inline const struct ngv_link_obj *
ngv_link_variant_get(struct ngv_link_variants *v, const char *kind, const char *key, struct ngv_link_error *err)
{
   char id[320];
   unsigned i;
   if (snprintf(id, sizeof(id), "%s:%s", kind, key) >= (int)sizeof(id)) {
      ngv_link_fail(err, NGV_LINK_E_UNSUPPORTED, "variant key too long", NULL, NULL);
      return NULL;
   }
   for (i = 0; i < NGV_LINK_VARIANT_MAX; i++)
      if (v->e[i].used && !strcmp(v->e[i].id, id)) {
         v->hits++;
         return &v->e[i].obj;
      }
   for (i = 0; i < NGV_LINK_VARIANT_MAX && v->e[i].used; i++)
      ;
   if (i == NGV_LINK_VARIANT_MAX) {
      ngv_link_fail(err, NGV_LINK_E_UNSUPPORTED, "variant table full", NULL, NULL);
      return NULL;
   }
   uint8_t *data = NULL;
   size_t size = 0;
   if (v->compile(v->user, kind, key, &data, &size)) {
      ngv_link_fail(err, NGV_LINK_E_UNSUPPORTED, "no library variant '%s'", id, NULL);
      return NULL;
   }
   v->compiles++;
   if (ngv_link_obj_open(&v->e[i].obj, data, size, err)) {
      free(data);
      return NULL;
   }
   v->e[i].data = data;
   memcpy(v->e[i].id, id, strlen(id) + 1);
   v->e[i].used = true;
   return &v->e[i].obj;
}

/* ------------------------------------------------------------------------------------------------------------------
 * draw-time selection. The canonical key strings are shared with ngcomp/fslib.py (key_string).
 */
struct ngv_link_ep_key {
   uint32_t toff;    /* byte offset of render target 0 inside the tile record (dword aligned, < 128) */
   uint32_t samples; /* 1, 2 or 4 */
   bool alpha_test, blend, dither, smask, call_pack, capture;
   uint32_t zs; /* 0 none, 1 zs.test.s (raster depth, stencil reference from u30 by facing), 2 zs.test.z (shader depth v32) */
};

/* The state of one draw that the fragment epilogue depends on (dynamic state included). */
struct ngv_link_fs_state {
   uint32_t samples;
   uint32_t rt0_tile_offset;
   bool rt0_blend_src_alpha; /* colour blend enabled with SRC_ALPHA / ONE_MINUS_SRC_ALPHA over RGBA8_UNORM */
   bool alpha_test;
   bool dither;
   bool writes_sample_mask;
   bool capture; /* test probe */
   /* R-E: the driver decided ZS_LATE for this draw (zs_late_needed) -> the epilogue runs zs.test itself; the variant depends
    * on whether the body exports depth (v32: zs.test.z) or not (zs.test.s) */
   bool zs_late;
   bool zs_exports_depth;
};

static inline int
ngv_link_ep_key_from_state(const struct ngv_link_fs_state *s, struct ngv_link_ep_key *k, struct ngv_link_error *err)
{
   if (!(s->samples == 1 || s->samples == 2 || s->samples == 4) || (s->rt0_tile_offset % 4) || s->rt0_tile_offset > 124)
      return ngv_link_fail(err, NGV_LINK_E_UNSUPPORTED, "draw state has no library epilogue (samples / tile offset)", NULL, NULL);
   k->toff = s->rt0_tile_offset;
   k->samples = s->samples;
   k->alpha_test = s->alpha_test;
   k->blend = s->rt0_blend_src_alpha;
   k->dither = s->dither;
   k->smask = s->writes_sample_mask;
   k->call_pack = true;
   k->capture = s->capture;
   k->zs = s->zs_late ? (s->zs_exports_depth ? 2u : 1u) : 0u;
   return NGV_LINK_E_OK;
}

static inline void
ngv_link_ep_key_string(const struct ngv_link_ep_key *k, char *buf, size_t n)
{
   snprintf(buf, n, "toff=%u,samples=%u,alpha_test=%d,blend=%d,dither=%d,smask=%d,call_pack=%d,capture=%d,zs=%u", k->toff,
            k->samples, k->alpha_test, k->blend, k->dither, k->smask, k->call_pack, k->capture, k->zs);
}

struct ngv_link_vattr {
   uint32_t loc, binding, offset;
   const char *fmt; /* Vulkan format name in lower case: r32g32b32a32_sfloat, r8g8b8a8_unorm ... */
};

static inline void
ngv_link_vs_key_string(const struct ngv_link_vattr *a, unsigned n, char *buf, size_t sz)
{
   size_t o = 0;
   buf[0] = 0;
   for (unsigned i = 0; i < n && o < sz; i++)
      o += (size_t)snprintf(buf + o, sz - o, "%sloc=%u,binding=%u,offset=%u,fmt=%s", i ? ";" : "", a[i].loc, a[i].binding,
                            a[i].offset, a[i].fmt);
}

struct ngv_link_ctx {
   struct ngv_link_variants variants;
   struct ngv_link_cache cache;
   const struct ngv_link_obj *libs[4];
   unsigned nlibs;
};

static inline void
ngv_link_ctx_init(struct ngv_link_ctx *c, ngv_link_compile_fn fn, void *user)
{
   memset(c, 0, sizeof(*c));
   ngv_link_variants_init(&c->variants, fn, user);
   ngv_link_cache_init(&c->cache);
}

static inline void
ngv_link_ctx_finish(struct ngv_link_ctx *c)
{
   ngv_link_cache_finish(&c->cache);
   ngv_link_variants_finish(&c->variants);
   memset(c, 0, sizeof(*c));
}

/* Fragment shader of one draw: [FS prologue variant] + body + epilogue variant chosen from `st`.  The body is the compiled
 * application shader (its segment is looked up by name, NULL = the first one).  Nothing is recompiled when only the
 * state changes: a different epilogue key links the same body again (or hits the cache). */
static inline const struct ngv_link_result *
ngv_link_fs_for_draw(struct ngv_link_ctx *c, const struct ngv_link_obj *body, const char *body_seg, const struct ngv_link_obj *prologue,
                     const char *prologue_seg, const struct ngv_link_fs_state *st, struct ngv_link_error *err)
{
   struct ngv_link_seg_ref chain[3];
   struct ngv_link_ep_key k;
   char ks[200];
   unsigned n = 0;
   const struct ngv_link_obj *ep;
   if (ngv_link_ep_key_from_state(st, &k, err))
      return NULL;
   ngv_link_ep_key_string(&k, ks, sizeof(ks));
   ep = ngv_link_variant_get(&c->variants, "fs_ep", ks, err);
   if (!ep)
      return NULL;
   if (prologue) {
      chain[n].obj = prologue;
      chain[n++].name = prologue_seg ? prologue_seg : prologue->first_seg;
   }
   chain[n].obj = body;
   chain[n++].name = body_seg ? body_seg : body->first_seg;
   chain[n].obj = ep;
   chain[n++].name = ep->first_seg;
   return ngv_link_cache_get(&c->cache, chain, n, c->libs, c->nlibs, err);
}

/* R-E: the driver decides ZS_LATE (GFX.md section 4.4), the linker only reports facts -- but the epilogue variant depends on
 * the decision (a LATE draw needs the epilogue that runs zs.test), and the decision depends on the facts of the chain
 * (discard of the alpha-test epilogue, the body's depth export, side effects).  This helper does the two steps:
 *   1. link with the epilogue of `st` as given (st->zs_late is ignored);
 *   2. if ngv_link_zs_late_needed(facts, zs) -> take the variant that runs zs.test (zs.test.z when the body exports depth,
 *      zs.test.s otherwise) and link again.
 * *late is set to the decision; pass it to ngv_link_frag_shader_packet() together with `zs`.  A draw without ZS work
 * (depth / stencil off) never becomes LATE because of an alpha test: the hardware then needs no test at all. */
static inline const struct ngv_link_result *
ngv_link_fs_for_draw_zs(struct ngv_link_ctx *c, const struct ngv_link_obj *body, const char *body_seg, const struct ngv_link_obj *prologue,
                        const char *prologue_seg, const struct ngv_link_fs_state *st, const struct ngv_link_zs_state *zs, bool *late,
                        struct ngv_link_error *err)
{
   struct ngv_link_fs_state s = *st;
   const struct ngv_link_result *r;
   s.zs_late = false;
   s.zs_exports_depth = false;
   *late = false;
   r = ngv_link_fs_for_draw(c, body, body_seg, prologue, prologue_seg, &s, err);
   if (!r || !ngv_link_zs_late_needed(r, zs))
      return r;
   s.zs_late = true;
   s.zs_exports_depth = (r->facts & NGV_LINK_FACT_EXPORTS_DEPTH) != 0;
   *late = true;
   return ngv_link_fs_for_draw(c, body, body_seg, prologue, prologue_seg, &s, err);
}

/* Vertex (geometry) shader of one draw: the attribute-fetch prologue for the pipeline's vertex input state + body. */
static inline const struct ngv_link_result *
ngv_link_vs_for_draw(struct ngv_link_ctx *c, const struct ngv_link_obj *body, const char *body_seg, const struct ngv_link_vattr *attrs,
                     unsigned nattrs, struct ngv_link_error *err)
{
   struct ngv_link_seg_ref chain[2];
   char ks[300];
   const struct ngv_link_obj *pro;
   ngv_link_vs_key_string(attrs, nattrs, ks, sizeof(ks));
   pro = ngv_link_variant_get(&c->variants, "vs_pro", ks, err);
   if (!pro)
      return NULL;
   chain[0].obj = pro;
   chain[0].name = pro->first_seg;
   chain[1].obj = body;
   chain[1].name = body_seg ? body_seg : body->first_seg;
   return ngv_link_cache_get(&c->cache, chain, 2, c->libs, c->nlibs, err);
}

#endif /* NGV_LINK_H */
