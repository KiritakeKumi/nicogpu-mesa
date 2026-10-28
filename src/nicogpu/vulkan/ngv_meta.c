/* SPDX-License-Identifier: MIT */
/* ngv_meta.c -- compute "meta" kernels for the image transfer commands.
 *
 * Images are in the T64K / linear hardware layout (the descriptor spec section 6, ngv_layout.h); the CP COPY engine moves
 * byte ranges only (command-set spec: MODE 1/2 tiled copies are not in CP v0.1), so vkCmdCopyBufferToImage / ImageToBuffer /
 * CopyImage / ClearColorImage / ClearDepthStencilImage run as small compute kernels that go through the texture unit's own
 * img.load / img.store, i.e. through the one definition of the layout the hardware (and the model) implements:
 *
 *   B2I   buffer -> image        I2B   image -> buffer        I2I   image -> image
 *   CLEAR_TYPED  img.store of a VkClearColorValue through the image's own format (the hardware does the num_format
 *                encoding: unorm / snorm / srgb / packed ...)      CLEAR_RAW   a raw element (depth / stencil)
 *
 * Copies are bit exact: both sides are described by a RAW view of one mip level -- an unsigned-integer descriptor with the
 * element size of the format (r8 / r16 / r32 / r32g32 / r32g32b32a32), dimensions in elements (texels, or 4x4 blocks of a
 * compressed format) and the level's offset in base_va (the descriptor spec section 4.5: a single-level uncompressed view of a
 * compressed or packed image addresses the same bytes).  No format conversion can happen, whatever the numeric format of
 * the image (NaN payloads, snorm -128, sRGB, packed 10/10/10/2, ...).
 *
 * One thread = one element.  Workgroup 8x8x1, z of the grid = slice (array layer or 3D z).  The kernels are generated as
 * ngcomp NIR-subset text (ngcomp/nir.py) and compiled on first use with the same compiler / cache as application
 * shaders; they use push constants for their parameters (u10..) and a descriptor block built in the command buffer's
 * upload area (u2:u3 = set 0), and they do not use the application's pipeline / descriptor state: the CP shader / user
 * register shadow of the command buffer is invalidated after each dispatch, so the next application dispatch re-emits
 * both.
 *
 * Limits (an over-limit region puts the command buffer into an error state, VK_ERROR_FEATURE_NOT_PRESENT): buffer row /
 * slice strides must fit 32 bits; multisampled images are not copied (no way to write them yet).  PROVISIONAL.
 */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "ngv_cmd_stream.h"
#include "ngv_private.h"

#include "util/format_rgb9e5.h"
#include "vk_format.h"

#define META_WG_X 8
#define META_WG_Y 8

enum meta_kind { MK_B2I = 1, MK_I2B, MK_I2I, MK_CLEAR_TYPED, MK_CLEAR_RAW };
enum meta_dim { MD_1DA = 0, MD_2DA = 1, MD_3D = 2 };

struct meta_key {
   uint8_t kind;
   uint8_t dim_a;  /* the (first) image */
   uint8_t dim_b;  /* I2I: the destination image */
   uint8_t eb;     /* element bytes of the raw view (1, 2, 4, 8, 16) */
};

#define META_CACHE_MAX 192

struct ngv_meta_entry {
   uint32_t key; /* 0 = free */
   struct ngv_pipeline p;
};

struct ngv_meta_cache {
   struct ngv_meta_entry e[META_CACHE_MAX];
};

static uint32_t
key_pack(const struct meta_key *k)
{
   unsigned ebl = 0;
   while ((1u << ebl) < k->eb)
      ebl++;
   return 0x80000000u | ((uint32_t)k->kind << 16) | ((uint32_t)k->dim_a << 12) | ((uint32_t)k->dim_b << 8) | ebl;
}

/* ---- text generation ---------------------------------------------------------------------------------------- */

struct tb {
   char *s;
   size_t n, cap;
   unsigned id;
   bool oom;
};

static void
tb_add(struct tb *t, const char *fmt, ...)
{
   char line[512];
   va_list ap;
   va_start(ap, fmt);
   int n = vsnprintf(line, sizeof(line), fmt, ap);
   va_end(ap);
   if (n < 0 || (size_t)n >= sizeof(line)) {
      t->oom = true;
      return;
   }
   if (t->n + (size_t)n + 1 > t->cap) {
      size_t cap = t->cap ? t->cap * 2 : 4096;
      while (cap < t->n + (size_t)n + 1)
         cap *= 2;
      char *ns = realloc(t->s, cap);
      if (!ns) {
         t->oom = true;
         return;
      }
      t->s = ns;
      t->cap = cap;
   }
   memcpy(t->s + t->n, line, (size_t)n + 1);
   t->n += (size_t)n;
}

/* a value reference printed as text ("s12" or "s12.1") */
struct ref {
   char s[24];
};

static struct ref
R(unsigned id)
{
   struct ref r;
   snprintf(r.s, sizeof(r.s), "s%u", id);
   return r;
}

static struct ref
RC(unsigned id, unsigned c)
{
   struct ref r;
   snprintf(r.s, sizeof(r.s), "s%u.%u", id, c);
   return r;
}

static unsigned
def_const(struct tb *t, uint32_t v)
{
   unsigned d = t->id++;
   tb_add(t, "  s%u:32 = load_const 0x%x\n", d, v);
   return d;
}

static unsigned
def_push(struct tb *t, unsigned zero, unsigned bits, unsigned off)
{
   unsigned d = t->id++;
   tb_add(t, "  s%u:%u = load_push_constant s%u {base=%u}\n", d, bits, zero, off);
   return d;
}

static unsigned
def_alu(struct tb *t, const char *op, unsigned bits, struct ref a, struct ref b)
{
   unsigned d = t->id++;
   tb_add(t, "  s%u:%u = %s %s %s\n", d, bits, op, a.s, b.s);
   return d;
}

static unsigned
def_un(struct tb *t, const char *op, unsigned bits, struct ref a)
{
   unsigned d = t->id++;
   tb_add(t, "  s%u:%u = %s %s\n", d, bits, op, a.s);
   return d;
}

static const unsigned dim_code[3] = {4 /* 1darray */, 5 /* 2darray */, 2 /* 3d */};

/* the coordinate scalars of an image access: (x, layer) for 1D arrays, (x, y, layer-or-z) otherwise */
static unsigned
coords(struct tb *t, unsigned dim, struct ref x, struct ref y, struct ref z, struct ref out[3])
{
   (void)t;
   unsigned n = 0;
   out[n++] = x;
   if (dim != MD_1DA)
      out[n++] = y;
   out[n++] = z;
   return n;
}

static void
tex_line(struct tb *t, int dst, const char *op, unsigned dim, unsigned dmask, unsigned img, const struct ref *data,
         unsigned ndata, const struct ref *co, unsigned nco)
{
   if (dst >= 0) {
      unsigned ncomp = (unsigned)__builtin_popcount(dmask);
      if (ncomp > 1)
         tb_add(t, "  s%d:32x%u = tex s%u", dst, ncomp, img);
      else
         tb_add(t, "  s%d:32 = tex s%u", dst, img);
   } else {
      tb_add(t, "  tex s%u", img);
   }
   for (unsigned i = 0; i < ndata; i++)
      tb_add(t, " %s", data[i].s);
   for (unsigned i = 0; i < nco; i++)
      tb_add(t, " %s", co[i].s);
   tb_add(t, " {op=%s, dim=%u, dmask=%u, toff=0, voff=0, rsd=0, ndata=%u}\n", op, dim_code[dim], dmask, ndata);
}

static unsigned
def_image_desc(struct tb *t, unsigned binding)
{
   unsigned d = t->id++;
   tb_add(t, "  s%u:32x8 = load_image_desc {set=0, binding=%u}\n", d, binding);
   return d;
}

/* dwords of a raw element */
static unsigned
eb_ncomp(unsigned eb)
{
   return eb <= 4 ? 1 : eb / 4;
}

/* load one element at `addr` (64-bit) from global memory into 32-bit scalars; returns the refs in comp[] */
static void
gen_load_elem(struct tb *t, unsigned eb, unsigned addr, struct ref comp[4])
{
   if (eb < 4) {
      unsigned v = t->id++;
      tb_add(t, "  s%u:%u = load_global s%u {off=0}\n", v, eb * 8, addr);
      comp[0] = R(def_un(t, "u2u32", 32, R(v)));
      return;
   }
   for (unsigned c = 0; c < eb / 4; c++) {
      unsigned v = t->id++;
      tb_add(t, "  s%u:32 = load_global s%u {off=%u}\n", v, addr, 4 * c);
      comp[c] = R(v);
   }
}

static void
gen_store_elem(struct tb *t, unsigned eb, unsigned addr, const struct ref comp[4])
{
   if (eb < 4) {
      unsigned v = def_un(t, eb == 1 ? "u2u8" : "u2u16", eb * 8, comp[0]);
      tb_add(t, "  store_global s%u s%u {off=0}\n", v, addr);
      return;
   }
   for (unsigned c = 0; c < eb / 4; c++)
      tb_add(t, "  store_global %s s%u {off=%u}\n", comp[c].s, addr, 4 * c);
}

static char *
gen_text(const struct meta_key *k)
{
   struct tb t = {0};
   const char *names[] = {"", "b2i", "i2b", "i2i", "clear", "clearraw"};
   const unsigned eb = k->eb;
   const unsigned nc = eb_ncomp(eb);
   const unsigned dmask = (1u << nc) - 1;
   tb_add(&t, "shader meta_%s_%u_%u_%u local_size=%d,%d,1\nb0:\n", names[k->kind], k->dim_a, k->dim_b, eb, META_WG_X,
          META_WG_Y);
   const unsigned zero = def_const(&t, 0);
   switch (k->kind) {
   case MK_B2I:
   case MK_I2B: {
      const bool b2i = k->kind == MK_B2I;
      const unsigned bufva = def_push(&t, zero, 64, 0);
      const unsigned row = def_push(&t, zero, 32, 8), slice = def_push(&t, zero, 32, 12);
      const unsigned w = def_push(&t, zero, 32, 16), h = def_push(&t, zero, 32, 20);
      const unsigned ox = def_push(&t, zero, 32, 24), oy = def_push(&t, zero, 32, 28), oz = def_push(&t, zero, 32, 32);
      const unsigned gid = t.id++, wid = t.id++;
      tb_add(&t, "  s%u:32x3 = load_global_invocation_id\n", gid);
      tb_add(&t, "  s%u:32x3 = load_workgroup_id\n", wid);
      const unsigned c0 = def_alu(&t, "ult", 1, RC(gid, 0), R(w)), c1 = def_alu(&t, "ult", 1, RC(gid, 1), R(h));
      const unsigned c = def_alu(&t, "iand", 1, R(c0), R(c1));
      tb_add(&t, "  br s%u b1 b2\nb1:\n", c);
      const unsigned img = def_image_desc(&t, 0);
      const unsigned ebc = def_const(&t, eb);
      const unsigned o0 = def_alu(&t, "imul", 32, RC(wid, 2), R(slice));
      const unsigned o1 = def_alu(&t, "imul", 32, RC(gid, 1), R(row));
      const unsigned o2 = def_alu(&t, "imul", 32, RC(gid, 0), R(ebc));
      const unsigned o3 = def_alu(&t, "iadd", 32, R(o0), R(o1));
      const unsigned o4 = def_alu(&t, "iadd", 32, R(o3), R(o2));
      const unsigned o5 = def_un(&t, "u2u64", 64, R(o4));
      const unsigned addr = def_alu(&t, "iadd", 64, R(bufva), R(o5));
      const unsigned cx = def_alu(&t, "iadd", 32, RC(gid, 0), R(ox)), cy = def_alu(&t, "iadd", 32, RC(gid, 1), R(oy));
      const unsigned cz = def_alu(&t, "iadd", 32, RC(wid, 2), R(oz));
      struct ref co[3], data[4];
      const unsigned nco = coords(&t, k->dim_a, R(cx), R(cy), R(cz), co);
      if (b2i) {
         gen_load_elem(&t, eb, addr, data);
         tex_line(&t, -1, "img_store", k->dim_a, dmask, img, data, nc, co, nco);
      } else {
         const unsigned r = t.id++;
         tex_line(&t, (int)r, "img_load", k->dim_a, dmask, img, NULL, 0, co, nco);
         for (unsigned i = 0; i < nc; i++)
            data[i] = nc > 1 ? RC(r, i) : R(r);
         gen_store_elem(&t, eb, addr, data);
      }
      break;
   }
   case MK_I2I: {
      const unsigned w = def_push(&t, zero, 32, 0), h = def_push(&t, zero, 32, 4);
      const unsigned sx = def_push(&t, zero, 32, 8), sy = def_push(&t, zero, 32, 12), sz = def_push(&t, zero, 32, 16);
      const unsigned dx = def_push(&t, zero, 32, 20), dy = def_push(&t, zero, 32, 24), dz = def_push(&t, zero, 32, 28);
      const unsigned gid = t.id++, wid = t.id++;
      tb_add(&t, "  s%u:32x3 = load_global_invocation_id\n", gid);
      tb_add(&t, "  s%u:32x3 = load_workgroup_id\n", wid);
      const unsigned c0 = def_alu(&t, "ult", 1, RC(gid, 0), R(w)), c1 = def_alu(&t, "ult", 1, RC(gid, 1), R(h));
      const unsigned c = def_alu(&t, "iand", 1, R(c0), R(c1));
      tb_add(&t, "  br s%u b1 b2\nb1:\n", c);
      const unsigned simg = def_image_desc(&t, 0), dimg = def_image_desc(&t, 2);
      const unsigned a0 = def_alu(&t, "iadd", 32, RC(gid, 0), R(sx)), a1 = def_alu(&t, "iadd", 32, RC(gid, 1), R(sy));
      const unsigned a2 = def_alu(&t, "iadd", 32, RC(wid, 2), R(sz));
      const unsigned b0 = def_alu(&t, "iadd", 32, RC(gid, 0), R(dx)), b1 = def_alu(&t, "iadd", 32, RC(gid, 1), R(dy));
      const unsigned b2 = def_alu(&t, "iadd", 32, RC(wid, 2), R(dz));
      struct ref sco[3], dco[3], data[4];
      const unsigned ns = coords(&t, k->dim_a, R(a0), R(a1), R(a2), sco);
      const unsigned nd = coords(&t, k->dim_b, R(b0), R(b1), R(b2), dco);
      const unsigned r = t.id++;
      tex_line(&t, (int)r, "img_load", k->dim_a, dmask, simg, NULL, 0, sco, ns);
      for (unsigned i = 0; i < nc; i++)
         data[i] = nc > 1 ? RC(r, i) : R(r);
      tex_line(&t, -1, "img_store", k->dim_b, dmask, dimg, data, nc, dco, nd);
      break;
   }
   case MK_CLEAR_TYPED:
   case MK_CLEAR_RAW: {
      const bool typed = k->kind == MK_CLEAR_TYPED;
      const unsigned w = def_push(&t, zero, 32, 0), h = def_push(&t, zero, 32, 4);
      const unsigned ox = def_push(&t, zero, 32, 8), oy = def_push(&t, zero, 32, 12), oz = def_push(&t, zero, 32, 16);
      const unsigned ncd = typed ? 4 : 1;
      unsigned col[4];
      for (unsigned i = 0; i < ncd; i++)
         col[i] = def_push(&t, zero, 32, 20 + 4 * i);
      const unsigned gid = t.id++, wid = t.id++;
      tb_add(&t, "  s%u:32x3 = load_global_invocation_id\n", gid);
      tb_add(&t, "  s%u:32x3 = load_workgroup_id\n", wid);
      const unsigned c0 = def_alu(&t, "ult", 1, RC(gid, 0), R(w)), c1 = def_alu(&t, "ult", 1, RC(gid, 1), R(h));
      const unsigned c = def_alu(&t, "iand", 1, R(c0), R(c1));
      tb_add(&t, "  br s%u b1 b2\nb1:\n", c);
      const unsigned img = def_image_desc(&t, 0);
      const unsigned cx = def_alu(&t, "iadd", 32, RC(gid, 0), R(ox)), cy = def_alu(&t, "iadd", 32, RC(gid, 1), R(oy));
      const unsigned cz = def_alu(&t, "iadd", 32, RC(wid, 2), R(oz));
      struct ref co[3], data[4];
      const unsigned nco = coords(&t, k->dim_a, R(cx), R(cy), R(cz), co);
      for (unsigned i = 0; i < ncd; i++)
         data[i] = R(col[i]);
      tex_line(&t, -1, "img_store", k->dim_a, typed ? 15u : 1u, img, data, ncd, co, nco);
      break;
   }
   default:
      free(t.s);
      return NULL;
   }
   tb_add(&t, "  jump b2\nb2:\n  return\n");
   if (t.oom) {
      free(t.s);
      return NULL;
   }
   return t.s;
}

/* $NGV_META_DUMP_DIR: write every generated kernel text there (<name>.nir) before it is compiled -- the quickest way to
 * look at one, or to feed it to ngcomp/driver.py by hand (ngcomp/tex_selftest.py checks the same texts) */
static void
meta_dump(const char *text)
{
   const char *dir = getenv("NGV_META_DUMP_DIR");
   if (!dir || !*dir)
      return;
   char name[96] = "meta";
   if (!strncmp(text, "shader ", 7)) {
      size_t i = 0;
      for (const char *s = text + 7; *s && *s != ' ' && i + 1 < sizeof(name); s++)
         name[i++] = *s;
      name[i] = 0;
   }
   char path[512];
   snprintf(path, sizeof(path), "%s/%s.nir", dir, name);
   FILE *f = fopen(path, "w");
   if (f) {
      fputs(text, f);
      fclose(f);
   }
}

/* ---- pipelines ---------------------------------------------------------------------------------------------- */

static VkResult
meta_get(struct ngv_device *dev, const struct meta_key *k, const struct ngv_pipeline **out)
{
   const uint32_t key = key_pack(k);
   VkResult res = VK_SUCCESS;
   simple_mtx_lock(&dev->meta_lock);
   if (!dev->meta) {
      dev->meta = calloc(1, sizeof(*dev->meta));
      if (!dev->meta) {
         simple_mtx_unlock(&dev->meta_lock);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
   }
   struct ngv_meta_entry *slot = NULL;
   for (unsigned i = 0; i < META_CACHE_MAX; i++) {
      if (dev->meta->e[i].key == key) {
         *out = &dev->meta->e[i].p;
         simple_mtx_unlock(&dev->meta_lock);
         return VK_SUCCESS;
      }
      if (!slot && !dev->meta->e[i].key)
         slot = &dev->meta->e[i];
   }
   if (!slot) {
      simple_mtx_unlock(&dev->meta_lock);
      return VK_ERROR_TOO_MANY_OBJECTS;
   }
   char *text = gen_text(k);
   if (!text) {
      simple_mtx_unlock(&dev->meta_lock);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   meta_dump(text);
   const uint32_t sp = dev->pdev->info.sp_per_core ? dev->pdev->info.sp_per_core : 1;
   const uint32_t warps = (META_WG_X * META_WG_Y + 31) / 32;
   struct ngv_shader_object so = {0}; /* ngv_ngcomp_compile fills only what the .xo holds: shared_bytes must be 0 here */
   res = ngv_ngcomp_compile(dev, text, (warps + (sp < 2 ? sp : 2) - 1) / (sp < 2 ? sp : 2), &so, NULL, NULL);
   free(text);
   if (res != VK_SUCCESS && getenv("NGV_META_DEBUG"))
      fprintf(stderr, "ngv_meta: kernel %08x did not compile (VkResult %d)\n", key, (int)res);
   if (res == VK_SUCCESS && so.scratch_bytes) {
      if (getenv("NGV_META_DEBUG"))
         fprintf(stderr, "ngv_meta: kernel %08x needs scratch %u / shared %u bytes\n", key, so.scratch_bytes, so.shared_bytes);
      free(so.code);
      res = VK_ERROR_UNKNOWN;
   }
   if (res == VK_SUCCESS) {
      struct ngv_bo code;
      res = ngv_bo_create(dev, align(so.code_size, NICOGPU_PAGE_SIZE), NGV_BO_CODE, &code);
      if (res == VK_SUCCESS) {
         memcpy(code.map, so.code, so.code_size);
         free(so.code);
         __sync_synchronize();
         memset(slot, 0, sizeof(*slot));
         slot->p.code = code;
         slot->p.entry = so.entry;
         slot->p.vregs = so.vregs;
         slot->p.shared_bytes = so.shared_bytes;
         slot->p.wg[0] = META_WG_X;
         slot->p.wg[1] = META_WG_Y;
         slot->p.wg[2] = 1;
         slot->p.base_push_offset = -1;
         slot->p.numwg_push_offset = -1;
         slot->key = key;
         *out = &slot->p;
      } else {
         free(so.code);
      }
   }
   simple_mtx_unlock(&dev->meta_lock);
   return res;
}

void
ngv_meta_finish(struct ngv_device *dev)
{
   if (!dev->meta)
      return;
   for (unsigned i = 0; i < META_CACHE_MAX; i++)
      if (dev->meta->e[i].key)
         ngv_bo_destroy(dev, &dev->meta->e[i].p.code);
   free(dev->meta);
   dev->meta = NULL;
}

/* ---- dispatch ------------------------------------------------------------------------------------------------ */

/* One internal dispatch: SET_SHADER + user registers (u2:u3 = the descriptor block, u10.. = the parameters) + DISPATCH.
 * The command buffer's shader / user-register shadow is invalidated: the application's next dispatch re-emits both. */
static void
meta_dispatch(struct ngv_cmd_buffer *cmd, const struct ngv_pipeline *p, uint64_t set_va, const uint32_t *push,
              unsigned npush, uint32_t gx, uint32_t gy, uint32_t gz)
{
   uint32_t u[NGV_NUM_UREGS] = {0};
   u[2] = (uint32_t)set_va;
   u[3] = (uint32_t)(set_va >> 32);
   memcpy(&u[10], push, npush * 4);
   uint32_t *dw = ngv_cmd_reserve(cmd, NGV_IB_RESERVE_DW);
   if (!dw)
      return;
   uint32_t n = ngv_emit_set_shader(dw, p->code.va, p->entry, p->vregs, p->shared_bytes, 0, 0, p->wg[0], p->wg[1], p->wg[2]);
   n += ngv_emit_set_user_regs(dw + n, 0, u, NGV_NUM_UREGS);
   n += ngv_emit_dispatch(dw + n, gx, gy, gz);
   ngv_cmd_commit(cmd, n);
   cmd->emitted_pipeline = NULL;
   cmd->u_valid = false;
}

/* Element size the kernels move: 3-component formats (3, 6, 12 B) are moved as 3 elements of a third of the size. */
static unsigned
raw_elem(const struct ngv_vkfmt *f)
{
   return f->elem % 3 == 0 ? f->elem / 3 : f->elem;
}

static unsigned
dim_of(const struct ngv_image *img)
{
   switch (img->vk.image_type) {
   case VK_IMAGE_TYPE_1D:
      return MD_1DA;
   case VK_IMAGE_TYPE_3D:
      return MD_3D;
   default:
      return MD_2DA;
   }
}

static bool
meta_fail_at(struct ngv_cmd_buffer *cmd, VkResult r, int line)
{
   if (getenv("NGV_META_DEBUG"))
      fprintf(stderr, "ngv_meta: command buffer error %d at ngv_meta.c:%d\n", (int)r, line);
   vk_command_buffer_set_error(&cmd->vk, r);
   return false;
}
#define meta_fail(cmd, r) meta_fail_at(cmd, r, __LINE__)

/* The pipeline for `k`, or an error state. */
static const struct ngv_pipeline *
meta_pipeline(struct ngv_cmd_buffer *cmd, const struct meta_key *k)
{
   const struct ngv_pipeline *p = NULL;
   struct ngv_device *dev = container_of(cmd->vk.base.device, struct ngv_device, vk);
   VkResult r = meta_get(dev, k, &p);
   if (r != VK_SUCCESS) {
      meta_fail(cmd, r == VK_ERROR_OUT_OF_HOST_MEMORY ? r : VK_ERROR_FEATURE_NOT_PRESENT);
      return NULL;
   }
   return p;
}

static void
grid_of(uint32_t w, uint32_t h, uint32_t slices, uint32_t *gx, uint32_t *gy, uint32_t *gz)
{
   *gx = (w + META_WG_X - 1) / META_WG_X;
   *gy = (h + META_WG_Y - 1) / META_WG_Y;
   *gz = slices;
}

/* ---- buffer <-> image --------------------------------------------------------------------------------------- */

void
ngv_meta_buffer_image(struct ngv_cmd_buffer *cmd, struct ngv_buffer *buf, struct ngv_image *img, uint32_t count,
                      const VkBufferImageCopy2 *regions, bool to_image)
{
   const struct ngv_vkfmt *f = img->fmt;
   for (uint32_t i = 0; i < count; i++) {
      const VkBufferImageCopy2 *r = &regions[i];
      const VkImageSubresourceLayers *sub = &r->imageSubresource;
      /* hardware-layout images have exactly one aspect (combined depth/stencil formats are not in the table) */
      const VkImageAspectFlags all = vk_format_aspects(img->vk.format);
      if (img->vk.samples != VK_SAMPLE_COUNT_1_BIT || sub->aspectMask != all) {
         meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      const struct vk_image_buffer_layout bl = vk_image_buffer_copy_layout(&img->vk, r);
      VkExtent3D ext = vk_image_extent_to_elements(&img->vk, r->imageExtent);
      VkOffset3D off = vk_image_offset_to_elements(&img->vk, r->imageOffset);
      const unsigned mult = ngv_format_raw_mult(f);
      ext.width *= mult;
      off.x *= mult;
      const uint32_t layers = vk_image_subresource_layer_count(&img->vk, sub);
      const uint32_t slices = MAX2(ext.depth, layers);
      if (!ext.width || !ext.height || !slices)
         continue;
      if (bl.row_stride_B > UINT32_MAX || bl.image_stride_B > UINT32_MAX ||
          bl.image_stride_B * (uint64_t)slices > (uint64_t)UINT32_MAX + 1) {
         meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      struct meta_key k = {to_image ? MK_B2I : MK_I2B, (uint8_t)dim_of(img), 0, (uint8_t)raw_elem(f)};
      const struct ngv_pipeline *p = meta_pipeline(cmd, &k);
      if (!p)
         return;
      uint64_t set_va;
      void *ptr;
      if (!ngv_cmd_upload(cmd, 64, 64, &set_va, &ptr))
         return;
      memset(ptr, 0, 64);
      if (!ngv_image_raw_desc(ptr, img, sub->mipLevel)) {
         meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      const uint64_t va = buf->vk.device_address + r->bufferOffset;
      const uint32_t slice0 = img->vk.image_type == VK_IMAGE_TYPE_3D ? (uint32_t)off.z : sub->baseArrayLayer;
      const uint32_t push[9] = {(uint32_t)va,  (uint32_t)(va >> 32), (uint32_t)bl.row_stride_B, (uint32_t)bl.image_stride_B,
                                ext.width,     ext.height,           (uint32_t)off.x,             (uint32_t)off.y, slice0};
      /* parameter order of the kernels: bufva(0..1) row(2) slice(3) w(4) h(5) ox(6) oy(7) oz(8) */
      uint32_t gx, gy, gz;
      grid_of(ext.width, ext.height, slices, &gx, &gy, &gz);
      meta_dispatch(cmd, p, set_va, push, 9, gx, gy, gz);
   }
}

/* ---- image -> image ----------------------------------------------------------------------------------------- */

void
ngv_meta_image_image(struct ngv_cmd_buffer *cmd, struct ngv_image *src, struct ngv_image *dst, uint32_t count,
                     const VkImageCopy2 *regions)
{
   const struct ngv_vkfmt *sf = src->fmt, *df = dst->fmt;
   /* compressed <-> uncompressed copies (equal bytes per texel block) are element-for-element copies of the raw views */
   if (sf->elem != df->elem || src->vk.samples != VK_SAMPLE_COUNT_1_BIT ||
       dst->vk.samples != VK_SAMPLE_COUNT_1_BIT) {
      meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   for (uint32_t i = 0; i < count; i++) {
      const VkImageCopy2 *r = &regions[i];
      if (r->srcSubresource.aspectMask != vk_format_aspects(src->vk.format) ||
          r->dstSubresource.aspectMask != vk_format_aspects(dst->vk.format)) {
         meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      VkExtent3D ext = vk_image_extent_to_elements(&src->vk, r->extent);
      VkOffset3D so = vk_image_offset_to_elements(&src->vk, r->srcOffset);
      VkOffset3D dof = vk_image_offset_to_elements(&dst->vk, r->dstOffset);
      const unsigned mult = ngv_format_raw_mult(sf);
      ext.width *= mult;
      so.x *= mult;
      dof.x *= mult;
      const uint32_t layers = vk_image_subresource_layer_count(&src->vk, &r->srcSubresource);
      const uint32_t slices = MAX2(ext.depth, layers);
      if (!ext.width || !ext.height || !slices)
         continue;
      struct meta_key k = {MK_I2I, (uint8_t)dim_of(src), (uint8_t)dim_of(dst), (uint8_t)raw_elem(sf)};
      const struct ngv_pipeline *p = meta_pipeline(cmd, &k);
      if (!p)
         return;
      uint64_t set_va;
      void *ptr;
      if (!ngv_cmd_upload(cmd, 64, 64, &set_va, &ptr))
         return;
      memset(ptr, 0, 64);
      if (!ngv_image_raw_desc(ptr, src, r->srcSubresource.mipLevel) ||
          !ngv_image_raw_desc((uint32_t *)ptr + 8, dst, r->dstSubresource.mipLevel)) {
         meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      const uint32_t sz = src->vk.image_type == VK_IMAGE_TYPE_3D ? (uint32_t)so.z : r->srcSubresource.baseArrayLayer;
      const uint32_t dz = dst->vk.image_type == VK_IMAGE_TYPE_3D ? (uint32_t)dof.z : r->dstSubresource.baseArrayLayer;
      const uint32_t push[8] = {ext.width, ext.height, (uint32_t)so.x, (uint32_t)so.y, sz, (uint32_t)dof.x, (uint32_t)dof.y, dz};
      uint32_t gx, gy, gz;
      grid_of(ext.width, ext.height, slices, &gx, &gy, &gz);
      meta_dispatch(cmd, p, set_va, push, 8, gx, gy, gz);
   }
}

/* ---- clears ------------------------------------------------------------------------------------------------- */

static void
clear_dispatch(struct ngv_cmd_buffer *cmd, struct ngv_image *img, bool typed, const VkImageSubresourceRange *range,
               const uint32_t color[4])
{
   const struct ngv_vkfmt *f = img->fmt;
   const uint32_t nlev = range->levelCount == VK_REMAINING_MIP_LEVELS ? img->vk.mip_levels - range->baseMipLevel
                                                                       : range->levelCount;
   const uint32_t nlay = range->layerCount == VK_REMAINING_ARRAY_LAYERS ? img->vk.array_layers - range->baseArrayLayer
                                                                         : range->layerCount;
   for (uint32_t l = range->baseMipLevel; l < range->baseMipLevel + nlev; l++) {
      const struct ngv_level *lv = &img->lay.level[l];
      const uint32_t slices = img->vk.image_type == VK_IMAGE_TYPE_3D ? lv->ed : nlay;
      struct meta_key k = {typed ? MK_CLEAR_TYPED : MK_CLEAR_RAW, (uint8_t)dim_of(img), 0, typed ? 4 : f->elem};
      if (!typed && f->elem > 4) {
         meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      const struct ngv_pipeline *p = meta_pipeline(cmd, &k);
      if (!p)
         return;
      uint64_t set_va;
      void *ptr;
      if (!ngv_cmd_upload(cmd, 64, 64, &set_va, &ptr))
         return;
      memset(ptr, 0, 64);
      const bool ok = typed ? ngv_image_typed_desc(ptr, img, l) : ngv_image_raw_desc(ptr, img, l);
      if (!ok) {
         meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      const uint32_t z0 = img->vk.image_type == VK_IMAGE_TYPE_3D ? 0 : range->baseArrayLayer;
      const uint32_t push[9] = {lv->ew, lv->eh, 0, 0, z0, color[0], color[1], color[2], color[3]};
      uint32_t gx, gy, gz;
      grid_of(lv->ew, lv->eh, slices, &gx, &gy, &gz);
      meta_dispatch(cmd, p, set_va, push, typed ? 9 : 6, gx, gy, gz);
   }
}

VKAPI_ATTR void VKAPI_CALL
ngv_CmdClearColorImage(VkCommandBuffer commandBuffer, VkImage _image, VkImageLayout imageLayout,
                       const VkClearColorValue *pColor, uint32_t rangeCount, const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_image, img, _image);
   if (!img->fmt || img->vk.samples != VK_SAMPLE_COUNT_1_BIT) {
      meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (img->vk.format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) {
      /* not storable (img.store drops r9g9b9e5): pack the colour here (float -> shared-exponent, round to nearest like Mesa's
       * float3_to_rgb9e5) and clear the 32-bit elements raw */
      const uint32_t color[4] = {float3_to_rgb9e5(pColor->float32), 0, 0, 0};
      for (uint32_t i = 0; i < rangeCount; i++)
         clear_dispatch(cmd, img, false, &pRanges[i], color);
      return;
   }
   if (!(img->fmt->flags & NGV_FMT_STORABLE)) {
      meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   for (uint32_t i = 0; i < rangeCount; i++)
      clear_dispatch(cmd, img, true, &pRanges[i], pColor->uint32);
}

/* Depth / stencil clear value as the raw element: D16 / X8_D24 / D32 and S8 are single-aspect formats here. */
VKAPI_ATTR void VKAPI_CALL
ngv_CmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage _image, VkImageLayout imageLayout,
                              const VkClearDepthStencilValue *pDepthStencil, uint32_t rangeCount,
                              const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(ngv_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(ngv_image, img, _image);
   if (!img->fmt || !(img->fmt->flags & (NGV_FMT_DEPTH | NGV_FMT_STENCIL)) || img->vk.samples != VK_SAMPLE_COUNT_1_BIT) {
      meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint32_t raw;
   const float d = pDepthStencil->depth < 0.0f ? 0.0f : pDepthStencil->depth > 1.0f ? 1.0f : pDepthStencil->depth;
   switch (img->vk.format) {
   case VK_FORMAT_D16_UNORM:
      raw = (uint32_t)(d * 65535.0f + 0.5f);
      break;
   case VK_FORMAT_X8_D24_UNORM_PACK32:
      raw = (uint32_t)((double)d * 16777215.0 + 0.5);
      break;
   case VK_FORMAT_D32_SFLOAT:
      memcpy(&raw, &pDepthStencil->depth, 4);
      break;
   case VK_FORMAT_S8_UINT:
      raw = pDepthStencil->stencil & 0xff;
      break;
   default:
      meta_fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   const uint32_t color[4] = {raw, 0, 0, 0};
   for (uint32_t i = 0; i < rangeCount; i++)
      clear_dispatch(cmd, img, false, &pRanges[i], color);
}
