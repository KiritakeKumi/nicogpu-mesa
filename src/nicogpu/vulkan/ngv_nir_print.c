/* SPDX-License-Identifier: MIT */
/* ngv_nir_print.c -- NIR -> the ngcomp NIR-subset text (ngcomp/nir.py, section 2).
 *
 * The text format was designed as a mechanical image of nir_print: this file is that mechanical printer.  It runs on
 * a shader that ngv_shader_to_nir (ngv_nir.c) has lowered: scalar ALU, scalar load_const / phi, no derefs, buffer
 * accesses as load_ssbo / store_ssbo / load_ubo whose index resolves to constants {set, slot}.
 *
 * What it does beyond one-to-one printing:
 *   - structured control flow is already a CFG in NIR (block successors): `br` / `jump` / `return` come from them;
 *   - every load_ssbo / store_ssbo / load_ubo gets a `load_buffer_desc {set=S, binding=SLOT}` line in front of it
 *     (ngcomp's load_buffer_desc is the fused vulkan_resource_index + load_vulkan_descriptor; one per distinct
 *     descriptor per block);
 *   - stores with a partial write mask are split into one store per written component;
 *   - ssa_undef becomes `load_const 0`;
 * - an ALU instruction with nir_alu_instr::exact set (SPIR-V NoContraction) carries `{exact=1}`;
 *   - ssbo_atomic / global_atomic / shared_atomic (+ _swap) print with a symbolic {atomic_op=...}; load_scratch /
 * store_scratch print as they are and the header carries scratch=<nir->scratch_size>;
 *   - subgroup intrinsics (elect, vote_*, ballot, read_*_invocation, shuffle*, quad_*, subgroup masks / id / size) print
 * one-to-one (print_subgroup).
 * Anything outside the subset makes the printer fail with a message naming the instruction (never silently dropped).
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"

#include "ngv_private.h"

#define MAX_DESC_CACHE 32

enum desc_kind { DK_BUFFER, DK_IMAGE, DK_SAMPLER };

struct pr {
   FILE *f;
   nir_function_impl *impl;
   uint32_t next_id;
   bool failed;
   char err[256];
   struct {
      uint32_t set, slot, id;
      enum desc_kind kind;
   } dc[MAX_DESC_CACHE];
   unsigned ndc;
   /* texture / image results are printed compacted to the components that are read (dmask): dm[ssa index] is that mask
    * (0 = the def is printed as it is) */
   uint8_t *dm;
   uint32_t ndm;
};

static void
pfail(struct pr *p, const char *fmt, ...)
{
   if (p->failed)
      return;
   p->failed = true;
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(p->err, sizeof(p->err), fmt, ap);
   va_end(ap);
}

static unsigned
def_mask(const struct pr *p, const nir_def *d)
{
   return d->index < p->ndm ? p->dm[d->index] : 0;
}

static void
psrc(struct pr *p, const nir_def *d, int comp)
{
   fprintf(p->f, " s%u", d->index);
   const unsigned m = def_mask(p, d);
   if (m) {
      /* component `comp` of the NIR vector is component popcount(m & below) of the compacted result */
      const unsigned full = (1u << d->num_components) - 1;
      if (comp < 0 ? m != full : !(m & (1u << comp))) {
         pfail(p, "s%u: a component of a texture result that its dmask does not cover is read", d->index);
         return;
      }
      if (comp >= 0 && util_bitcount(m) > 1)
         fprintf(p->f, ".%d", util_bitcount(m & ((1u << comp) - 1)));
      return;
   }
   if (d->num_components > 1 && comp >= 0)
      fprintf(p->f, ".%d", comp);
}

static void
pdst(struct pr *p, const nir_def *d)
{
   fprintf(p->f, "  s%u:%u", d->index, d->bit_size);
   if (d->num_components > 1)
      fprintf(p->f, "x%u", d->num_components);
   fprintf(p->f, " =");
}

static const char *
access_str(unsigned access, char *buf, size_t n)
{
   static const struct {
      unsigned bit;
      const char *name;
   } names[] = {{ACCESS_COHERENT, "coherent"},         {ACCESS_VOLATILE, "volatile"},
                {ACCESS_NON_TEMPORAL, "non_temporal"}, {ACCESS_NON_WRITEABLE, "non_writable"},
                {ACCESS_CAN_REORDER, "can_reorder"}};
   buf[0] = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(names); i++) {
      if (!(access & names[i].bit))
         continue;
      if (buf[0])
         strncat(buf, "|", n - strlen(buf) - 1);
      strncat(buf, names[i].name, n - strlen(buf) - 1);
   }
   return buf;
}

/* ", access=a|b" or "" */
static void
paccess(struct pr *p, unsigned access)
{
   char buf[96];
   if (access_str(access, buf, sizeof(buf))[0])
      fprintf(p->f, ", access=%s", buf);
}

static uint32_t
desc_emit(struct pr *p, enum desc_kind kind, uint32_t set, uint32_t slot)
{
   for (unsigned i = 0; i < p->ndc; i++)
      if (p->dc[i].set == set && p->dc[i].slot == slot && p->dc[i].kind == kind)
         return p->dc[i].id;
   uint32_t id = p->next_id++;
   static const char *const op[] = {"load_buffer_desc", "load_image_desc", "load_sampler_desc"};
   static const unsigned ndw[] = {4, 8, 4};
   fprintf(p->f, "  s%u:32x%u = %s {set=%u, binding=%u}\n", id, ndw[kind], op[kind], set, slot);
   if (p->ndc < MAX_DESC_CACHE) {
      p->dc[p->ndc].set = set;
      p->dc[p->ndc].slot = slot;
      p->dc[p->ndc].id = id;
      p->dc[p->ndc].kind = kind;
      p->ndc++;
   }
   return id;
}

/* Descriptor: load_vulkan_descriptor result after ngv_nir_lower_pipeline_layout == {set, slot, 0}; the constants are
 * visible through vec / mov chains (nir_scalar_resolved). */
static uint32_t
desc_for(struct pr *p, nir_def *idx)
{
   if (idx->num_components < 2) {
      pfail(p, "buffer index is not a {set, slot} pair");
      return 0;
   }
   nir_scalar a = nir_scalar_resolved(idx, 0), b = nir_scalar_resolved(idx, 1);
   if (!nir_scalar_is_const(a) || !nir_scalar_is_const(b)) {
      pfail(p, "non-constant descriptor index (waterfall loop for dynamically indexed descriptor arrays is a gap)");
      return 0;
   }
   return desc_emit(p, DK_BUFFER, (uint32_t)nir_scalar_as_uint(a), (uint32_t)nir_scalar_as_uint(b));
}

/* Image / sampler handle (ngv_nir.c, res_handle): (set << 24) | slot as a compile-time constant. */
static uint32_t
handle_desc(struct pr *p, const nir_src *src, enum desc_kind kind)
{
   nir_scalar h = nir_scalar_resolved(src->ssa, 0);
   if (src->ssa->num_components != 1 || !nir_scalar_is_const(h)) {
      pfail(p, "non-constant image or sampler index (waterfall loop for dynamically indexed descriptor arrays is a gap)");
      return 0;
   }
   const uint32_t v = (uint32_t)nir_scalar_as_uint(h);
   return desc_emit(p, kind, v >> 24, v & 0xffffffu);
}

static void
psym_mask(FILE *f, const char *key, const char *const *names, const unsigned *bits, unsigned n, unsigned value)
{
   bool any = false;
   for (unsigned i = 0; i < n; i++) {
      if (value & bits[i]) {
         fprintf(f, "%s%s", any ? "|" : "", names[i]);
         any = true;
      }
   }
}

static const char *
scope_str(struct pr *p, unsigned s)
{
   switch (s) {
   case SCOPE_NONE: return "none";
   case SCOPE_INVOCATION: return "invocation";
   case SCOPE_SUBGROUP: return "subgroup";
   case SCOPE_WORKGROUP: return "workgroup";
   case SCOPE_QUEUE_FAMILY: return "queue_family";
   case SCOPE_DEVICE: return "device";
   default:
      pfail(p, "barrier scope %u is not supported", s);
      return "none";
   }
}

static void
print_barrier(struct pr *p, nir_intrinsic_instr *intr)
{
   static const char *const sem_names[] = {"acquire", "release", "make_available", "make_visible"};
   static const unsigned sem_bits[] = {NIR_MEMORY_ACQUIRE, NIR_MEMORY_RELEASE, NIR_MEMORY_MAKE_AVAILABLE,
                                       NIR_MEMORY_MAKE_VISIBLE};
   static const char *const mode_names[] = {"shared", "global", "image"};
   static const unsigned mode_bits[] = {nir_var_mem_shared, nir_var_mem_ssbo | nir_var_mem_global, nir_var_image};
   unsigned sem = nir_intrinsic_memory_semantics(intr), modes = nir_intrinsic_memory_modes(intr);
   fprintf(p->f, "  barrier {exec_scope=%s, mem_scope=%s", scope_str(p, nir_intrinsic_execution_scope(intr)),
           scope_str(p, nir_intrinsic_memory_scope(intr)));
   if (sem & (NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE | NIR_MEMORY_MAKE_AVAILABLE | NIR_MEMORY_MAKE_VISIBLE)) {
      fprintf(p->f, ", semantics=");
      psym_mask(p->f, "semantics", sem_names, sem_bits, ARRAY_SIZE(sem_names), sem);
   }
   if (modes & (nir_var_mem_shared | nir_var_mem_ssbo | nir_var_mem_global | nir_var_image)) {
      fprintf(p->f, ", modes=");
      psym_mask(p->f, "modes", mode_names, mode_bits, ARRAY_SIZE(mode_names), modes);
   }
   fprintf(p->f, "}\n");
}

/* Stores: one line for a full write mask, else one per written component (the isel stores whole vectors).  The
 * constant byte offset is {base=K} for store_shared (ngcomp's shared forms read `base`, like shared_atomic) and
 * {off=K} for the others (: shared printed `off`, which isel ignored -- harmless while base was always 0, but
 * a partial-mask store_shared would have put every component at the first one's address). */
static void
print_store(struct pr *p, const char *op, nir_def *data, unsigned mask, const uint32_t *pre_ids,
            nir_def *const *tail, unsigned ntail, int base, unsigned access)
{
   const char *key = !strcmp(op, "store_shared") ? "base" : "off";
   const unsigned n = data->num_components, bytes = data->bit_size / 8;
   const unsigned full = (1u << n) - 1;
   mask &= full;
   if (!mask)
      return;
   for (unsigned c = 0; c < n; c++) {
      if (mask == full && c)
         break;
      if (mask != full && !(mask & (1u << c)))
         continue;
      fprintf(p->f, "  %s", op);
      psrc(p, data, mask == full ? -1 : (int)c);
      for (unsigned i = 0; pre_ids && i < 1; i++)
         fprintf(p->f, " s%u", pre_ids[i]);
      for (unsigned i = 0; i < ntail; i++)
         psrc(p, tail[i], -1);
      fprintf(p->f, " {%s=%d", key, base + (mask == full ? 0 : (int)(c * bytes)));
      paccess(p, access);
      fprintf(p->f, "}\n");
   }
}

/* nir_atomic_op -> the symbol ngcomp/isel_table.py ATOMIC_OPS knows (nir.py "Atomics / scratch"); NULL = none */
static const char *
atomic_op_str(nir_atomic_op op)
{
   switch (op) {
   case nir_atomic_op_iadd: return "iadd";
   case nir_atomic_op_imin: return "imin";
   case nir_atomic_op_umin: return "umin";
   case nir_atomic_op_imax: return "imax";
   case nir_atomic_op_umax: return "umax";
   case nir_atomic_op_iand: return "iand";
   case nir_atomic_op_ior: return "ior";
   case nir_atomic_op_ixor: return "ixor";
   case nir_atomic_op_xchg: return "xchg";
   case nir_atomic_op_cmpxchg: return "cmpxchg";
   case nir_atomic_op_fadd: return "fadd";
   case nir_atomic_op_fmin: return "fmin";
   case nir_atomic_op_fmax: return "fmax";
   case nir_atomic_op_inc_wrap: return "inc_wrap";
   case nir_atomic_op_dec_wrap: return "dec_wrap";
   default: return NULL; /* fcmpxchg, ... */
   }
}

/* Atomics.  Sources: the address operands (ssbo: descriptor index + offset; global: 64-bit address; shared: offset),
 * then one data source (two for the _swap forms: compare, new value).  The _swap forms are always cmpxchg in ngcomp
 * (no atomic_op attribute); a float compare-exchange has no instruction. */
static void
print_atomic(struct pr *p, nir_intrinsic_instr *intr, bool swap)
{
   const nir_atomic_op op = nir_intrinsic_atomic_op(intr);
   const char *opname = atomic_op_str(op);
   const char *name = nir_intrinsic_infos[intr->intrinsic].name;
   if (!opname || (swap && op != nir_atomic_op_cmpxchg)) {
      pfail(p, "%s: atomic op %d is not supported", name, (int)op);
      return;
   }
   const bool ssbo = intr->intrinsic == nir_intrinsic_ssbo_atomic || intr->intrinsic == nir_intrinsic_ssbo_atomic_swap;
   const bool shared = intr->intrinsic == nir_intrinsic_shared_atomic ||
                       intr->intrinsic == nir_intrinsic_shared_atomic_swap;
   const unsigned nsrc = (ssbo ? 2 : 1) + (swap ? 2 : 1);
   /* the load_buffer_desc line must come before the atomic's own line */
   const uint32_t desc = ssbo ? desc_for(p, intr->src[0].ssa) : 0;
   pdst(p, &intr->def);
   fprintf(p->f, " %s", name);
   if (ssbo)
      fprintf(p->f, " s%u", desc);
   for (unsigned i = ssbo ? 1 : 0; i < nsrc; i++)
      psrc(p, intr->src[i].ssa, -1);
   /* shared_atomic carries a constant byte offset in BASE (-> {base=K}); the ssbo / global forms have none ({off=0}) */
   const char *key = shared ? "base" : "off";
   const int imm = shared ? (int)nir_intrinsic_base(intr) : 0;
   if (swap)
      fprintf(p->f, " {%s=%d}\n", key, imm);
   else
      fprintf(p->f, " {atomic_op=%s, %s=%d}\n", opname, key, imm);
}

/* : subgroup intrinsics print one-to-one (name, every source whole, the reduction op / cluster size of the 
 * scan / reduce family as {reduction_op=..., cluster_size=N}); ngcomp/isel_subgroup.py selects them. What reaches
 * here is the output of ngv_nir_lower_subgroups (ngv_nir.c): scalar, 32-bit ballots. */
static void
print_subgroup(struct pr *p, nir_intrinsic_instr *intr)
{
   const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
   if (!info->has_dest) {
      pfail(p, "%s: subgroup intrinsic without a result", info->name);
      return;
   }
   pdst(p, &intr->def);
   fprintf(p->f, " %s", info->name);
   for (unsigned i = 0; i < info->num_srcs; i++)
      psrc(p, intr->src[i].ssa, -1);
   const bool red = nir_intrinsic_has_reduction_op(intr), clu = nir_intrinsic_has_cluster_size(intr);
   if (red || clu) {
      fprintf(p->f, " {");
      if (red)
         fprintf(p->f, "reduction_op=%s", nir_op_infos[nir_intrinsic_reduction_op(intr)].name);
      if (clu)
         fprintf(p->f, "%scluster_size=%u", red ? ", " : "", nir_intrinsic_cluster_size(intr));
      fprintf(p->f, "}");
   }
   fprintf(p->f, "\n");
}

/* ---- textures and images -------------------------------------------------------------------------------- */

/* The register stream of one tex.* / img.* instruction (errata notes T1, v0.1.8), as 32-bit scalar operands:
 * either a component of a NIR def or a freshly emitted constant. */
#define MAX_TEX_OPS 24
struct opnds {
   struct {
      const nir_def *d;
      int comp;
      int raw; /* >= 0: the id of a scalar emitted by us */
   } o[MAX_TEX_OPS];
   unsigned n;
};

static void
op_def(struct pr *p, struct opnds *os, const nir_def *d, unsigned comp)
{
   if (os->n >= MAX_TEX_OPS) {
      pfail(p, "tex: too many operands");
      return;
   }
   if (d->bit_size != 32) {
      pfail(p, "tex: a %u-bit operand (only the 32-bit stream exists; a16 / d16 are a gap)", d->bit_size);
      return;
   }
   os->o[os->n].d = d;
   os->o[os->n].comp = d->num_components > 1 ? (int)comp : -1;
   os->o[os->n].raw = -1;
   os->n++;
}

static void
op_vec(struct pr *p, struct opnds *os, const nir_def *d, unsigned count)
{
   for (unsigned c = 0; c < count; c++)
      op_def(p, os, d, c);
}

static void
op_const(struct pr *p, struct opnds *os, uint32_t v)
{
   if (os->n >= MAX_TEX_OPS) {
      pfail(p, "tex: too many operands");
      return;
   }
   const uint32_t id = p->next_id++;
   fprintf(p->f, "  s%u:32 = load_const 0x%x\n", id, v);
   os->o[os->n].d = NULL;
   os->o[os->n].comp = -1;
   os->o[os->n].raw = (int)id;
   os->n++;
}

static void
op_print(struct pr *p, const struct opnds *os)
{
   for (unsigned i = 0; i < os->n; i++) {
      if (os->o[i].raw >= 0)
         fprintf(p->f, " s%d", os->o[i].raw);
      else
         psrc(p, os->o[i].d, os->o[i].comp);
   }
}

/* the ISA dim code (= descriptor view_type) of a sampler / image dimension */
static int
dim_code(struct pr *p, enum glsl_sampler_dim dim, bool array, bool storage)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D: return array ? 4 : 0;
   case GLSL_SAMPLER_DIM_2D: return array ? 5 : 1;
   case GLSL_SAMPLER_DIM_3D: return 2;
   /* storage cube views are 2D arrays in the descriptor (ngv_image.c): (x, y, face) / (x, y, cube * 6 + face) */
   case GLSL_SAMPLER_DIM_CUBE: return storage ? 5 : (array ? 6 : 3);
   case GLSL_SAMPLER_DIM_MS: return array ? 8 : 7;
   case GLSL_SAMPLER_DIM_BUF: return 9;
   default:
      pfail(p, "unsupported image dimension %d (rect / external / subpass)", (int)dim);
      return -1;
   }
}

/* components of a NIR sampler coordinate, as the ISA wants them (cube: direction vector + layer; arrays: layer last) */
static unsigned
coord_comps(enum glsl_sampler_dim dim, bool array)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D:
   case GLSL_SAMPLER_DIM_BUF:
      return 1 + array;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_MS:
      return 2 + array;
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
      return 3 + array;
   default:
      return 0;
   }
}

/* spatial dimensions that get texel offsets */
static unsigned
offset_comps(enum glsl_sampler_dim dim)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D: return 1;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_MS: return 2;
   case GLSL_SAMPLER_DIM_3D: return 3;
   default: return 0;
   }
}

static void
tex_emit(struct pr *p, const char *op, int dim, unsigned dmask, unsigned toff, bool voff, unsigned ndata,
         uint32_t img_desc, int smp_desc, const struct opnds *os, const nir_def *dst, unsigned dst_regs)
{
   if (dst) {
      fprintf(p->f, "  s%u:32", dst->index);
      if (dst_regs > 1)
         fprintf(p->f, "x%u", dst_regs);
      fprintf(p->f, " = tex s%u", img_desc);
   } else {
      fprintf(p->f, "  tex s%u", img_desc);
   }
   if (smp_desc >= 0)
      fprintf(p->f, " s%d", smp_desc);
   op_print(p, os);
   fprintf(p->f, " {op=%s, dim=%d, dmask=%u, toff=%u, voff=%u, rsd=0, ndata=%u}\n", op, dim, dmask, toff, voff ? 1u : 0u,
           ndata);
}

static void
print_tex(struct pr *p, nir_tex_instr *tex)
{
   const enum glsl_sampler_dim sdim = tex->sampler_dim;
   const int dim = dim_code(p, sdim, tex->is_array, false);
   if (dim < 0)
      return;
   if (tex->def.bit_size != 32) {
      pfail(p, "tex: %u-bit result (16-bit sampled types are a gap)", tex->def.bit_size);
      return;
   }
   if (nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0 || nir_tex_instr_src_index(tex, nir_tex_src_texture_deref) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_plane) >= 0 || tex->is_sparse) {
      pfail(p, "tex: min_lod / sparse residency / YCbCr plane operands are not supported");
      return;
   }
   const int ti = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   const int si = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   if (ti < 0) {
      pfail(p, "tex: no texture handle (an unlowered deref?)");
      return;
   }
   const int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int bi = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   const int li = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   const int cmpi = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
   const int dxi = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
   const int dyi = nir_tex_instr_src_index(tex, nir_tex_src_ddy);
   const int oi = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   const int msi = nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
   const unsigned ncoord = coord_comps(sdim, tex->is_array);
   const bool cmp = cmpi >= 0;

   const char *base = NULL;
   bool fetch = false, has_smp = true, lz = false;
   unsigned dmask = 0;
   unsigned nregs = 0;
   const unsigned read = nir_def_components_read(&tex->def);
   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
      base = bi >= 0 ? (cmp ? "tex_sample_c_b" : "tex_sample_b") : (cmp ? "tex_sample_c" : "tex_sample");
      break;
   case nir_texop_txl: {
      if (li >= 0) {
         nir_scalar l = nir_scalar_resolved(tex->src[li].src.ssa, 0);
         lz = nir_scalar_is_const(l) && nir_scalar_as_float(l) == 0.0;
      }
      if (li < 0) {
         pfail(p, "txl without a lod operand");
         return;
      }
      base = lz ? (cmp ? "tex_sample_c_lz" : "tex_sample_lz") : (cmp ? "tex_sample_c_l" : "tex_sample_l");
      break;
   }
   case nir_texop_txd:
      base = cmp ? "tex_sample_c_d" : "tex_sample_d";
      break;
   case nir_texop_tg4:
      base = cmp ? "tex_gather4_c" : "tex_gather4";
      break;
   case nir_texop_txf:
   case nir_texop_txf_ms:
      base = "tex_fetch";
      fetch = true;
      has_smp = false;
      break;
   case nir_texop_txs:
      base = "tex_query_size";
      has_smp = false;
      break;
   case nir_texop_query_levels:
      base = "tex_query_levels";
      has_smp = false;
      break;
   case nir_texop_texture_samples:
      base = "tex_query_samples";
      has_smp = false;
      break;
   case nir_texop_lod:
      base = "tex_query_lod";
      break;
   default:
      pfail(p, "unsupported NIR texture op %d", (int)tex->op);
      return;
   }
   if (has_smp && si < 0) {
      pfail(p, "tex: %s needs a sampler", base);
      return;
   }

   struct opnds os = {.n = 0};
   unsigned toff = 0;
   bool voff = false;
   const bool is_query = tex->op == nir_texop_txs || tex->op == nir_texop_query_levels ||
                         tex->op == nir_texop_texture_samples;

   if (tex->op == nir_texop_tg4) {
      dmask = cmp ? 1u : 1u << tex->component;
      nregs = 4;
   } else if (tex->op == nir_texop_query_levels || tex->op == nir_texop_texture_samples) {
      dmask = 1;
      nregs = 1;
   } else {
      dmask = read & 0xf;
      if (!dmask)
         dmask = 1;
      nregs = util_bitcount(dmask);
   }
   if (tex->def.num_components > 4) {
      pfail(p, "tex: result wider than 4 components");
      return;
   }

   if (tex->op == nir_texop_txs) {
      /* the level is read by the instruction whatever the view */
      if (li >= 0)
         op_def(p, &os, tex->src[li].src.ssa, 0);
      else
         op_const(p, &os, 0);
   } else if (!is_query) {
      if (ci < 0) {
         pfail(p, "tex: no coordinate");
         return;
      }
      op_vec(p, &os, tex->src[ci].src.ssa, ncoord);
      if (fetch) {
         if (sdim != GLSL_SAMPLER_DIM_MS && sdim != GLSL_SAMPLER_DIM_BUF) {
            if (li >= 0)
               op_def(p, &os, tex->src[li].src.ssa, 0);
            else
               op_const(p, &os, 0);
         }
         if (sdim == GLSL_SAMPLER_DIM_MS) {
            if (msi >= 0)
               op_def(p, &os, tex->src[msi].src.ssa, 0);
            else
               op_const(p, &os, 0);
         }
      } else {
         if (tex->op == nir_texop_txl && !lz)
            op_def(p, &os, tex->src[li].src.ssa, 0);
         if (bi >= 0 && (tex->op == nir_texop_tex || tex->op == nir_texop_txb))
            op_def(p, &os, tex->src[bi].src.ssa, 0);
         if (tex->op == nir_texop_txd) {
            if (dxi < 0 || dyi < 0) {
               pfail(p, "txd without derivatives");
               return;
            }
            const unsigned ng = ncoord - (tex->is_array ? 1 : 0);
            op_vec(p, &os, tex->src[dxi].src.ssa, ng);
            op_vec(p, &os, tex->src[dyi].src.ssa, ng);
         }
         if (cmp)
            op_def(p, &os, tex->src[cmpi].src.ssa, 0);
      }
      if (oi >= 0 && tex->op != nir_texop_lod) {
         const unsigned no = offset_comps(sdim);
         nir_def *off = tex->src[oi].src.ssa;
         bool all_const = no > 0;
         int v[3] = {0, 0, 0};
         for (unsigned c = 0; c < no && all_const; c++) {
            nir_scalar sc = nir_scalar_resolved(off, c);
            if (!nir_scalar_is_const(sc)) {
               all_const = false;
               break;
            }
            v[c] = (int)(int32_t)nir_scalar_as_int(sc);
            if (v[c] < -8 || v[c] > 7)
               all_const = false;
         }
         if (no > 0) {
            if (all_const) {
               toff = (unsigned)(v[0] & 15) | ((unsigned)(v[1] & 15) << 4) | ((unsigned)(v[2] & 15) << 8);
            } else {
               voff = true;
               op_vec(p, &os, off, no);
            }
         }
      }
   }
   if (p->failed)
      return;

   const uint32_t img = handle_desc(p, &tex->src[ti].src, DK_IMAGE);
   int smp = -1;
   if (has_smp)
      smp = (int)handle_desc(p, &tex->src[si].src, DK_SAMPLER);
   if (p->failed)
      return;
   /* the result: a gather always has 4 registers, others popcount(dmask); NIR vector components not in dmask must not be
    * read (dmask is built from what is read, so they are not) */
   if (tex->op != nir_texop_tg4 && p->dm && tex->def.index < p->ndm && tex->def.num_components > 1)
      p->dm[tex->def.index] = (uint8_t)dmask;
   if (tex->op == nir_texop_tg4 && tex->def.num_components != 4) {
      pfail(p, "tg4 result is not a vec4");
      return;
   }
   tex_emit(p, base, dim, dmask, toff, voff, 0, img, smp, &os, &tex->def, nregs);
}

static const char *
img_atomic_name(struct pr *p, nir_intrinsic_instr *intr, bool rtn, bool swap)
{
   static char buf[40];
   const char *n = NULL;
   switch (nir_intrinsic_atomic_op(intr)) {
   case nir_atomic_op_iadd: n = "add_u32"; break;
   case nir_atomic_op_imin: n = "min_i32"; break;
   case nir_atomic_op_imax: n = "max_i32"; break;
   case nir_atomic_op_umin: n = "min_u32"; break;
   case nir_atomic_op_umax: n = "max_u32"; break;
   case nir_atomic_op_iand: n = "and_b32"; break;
   case nir_atomic_op_ior: n = "or_b32"; break;
   case nir_atomic_op_ixor: n = "xor_b32"; break;
   case nir_atomic_op_xchg: n = "xchg_b32"; break;
   case nir_atomic_op_cmpxchg: n = "cmpxchg_b32"; break;
   case nir_atomic_op_inc_wrap: n = "inc_u32"; break;
   case nir_atomic_op_dec_wrap: n = "dec_u32"; break;
   case nir_atomic_op_fadd: n = "fadd_f32"; break;
   default:
      pfail(p, "image atomic op %d is not supported", (int)nir_intrinsic_atomic_op(intr));
      return NULL;
   }
   (void)swap;
   snprintf(buf, sizeof(buf), "img_atom_%s%s", n, rtn ? "_rtn" : "");
   return buf;
}

/* bindless_image_{load,store,atomic,atomic_swap,size,samples}: handle in src[0]; img.* instructions (T4) */
static void
print_image(struct pr *p, nir_intrinsic_instr *intr)
{
   const enum glsl_sampler_dim sdim = nir_intrinsic_image_dim(intr);
   const bool arr = nir_intrinsic_image_array(intr);
   if (sdim == GLSL_SAMPLER_DIM_MS) {
      pfail(p, "storage images with multiple samples are not supported");
      return;
   }
   const int dim = dim_code(p, sdim, arr, true);
   if (dim < 0)
      return;
   const uint32_t img = handle_desc(p, &intr->src[0], DK_IMAGE);
   if (p->failed)
      return;
   struct opnds os = {.n = 0};
   /* image coordinates: cube images are 2D arrays of faces (x, y, face) */
   unsigned nc = sdim == GLSL_SAMPLER_DIM_CUBE ? 3 : coord_comps(sdim, arr);
   switch (intr->intrinsic) {
   case nir_intrinsic_bindless_image_load: {
      if (intr->def.bit_size != 32) {
         pfail(p, "image load: %u-bit result", intr->def.bit_size);
         return;
      }
      op_vec(p, &os, intr->src[1].ssa, nc);
      unsigned dmask = nir_def_components_read(&intr->def) & 0xf;
      if (!dmask)
         dmask = 1;
      if (p->dm && intr->def.index < p->ndm && intr->def.num_components > 1)
         p->dm[intr->def.index] = (uint8_t)dmask;
      tex_emit(p, "img_load", dim, dmask, 0, false, 0, img, -1, &os, &intr->def, util_bitcount(dmask));
      break;
   }
   case nir_intrinsic_bindless_image_store: {
      const nir_def *data = intr->src[3].ssa;
      if (data->bit_size != 32) {
         pfail(p, "image store: %u-bit data", data->bit_size);
         return;
      }
      /* data registers first (rd), R G B A, missing components 0 */
      for (unsigned c = 0; c < 4; c++) {
         if (c < data->num_components)
            op_def(p, &os, data, c);
         else
            op_const(p, &os, 0);
      }
      op_vec(p, &os, intr->src[1].ssa, nc);
      tex_emit(p, "img_store", dim, 15, 0, false, 4, img, -1, &os, NULL, 0);
      break;
   }
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap: {
      const bool swap = intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap;
      const bool rtn = !nir_def_is_unused(&intr->def);
      const char *name = img_atomic_name(p, intr, rtn, swap);
      if (!name)
         return;
      if (intr->def.bit_size != 32 || intr->src[3].ssa->bit_size != 32) {
         pfail(p, "image atomic: only the 32-bit forms exist");
         return;
      }
      op_def(p, &os, intr->src[3].ssa, 0);
      if (swap)
         op_def(p, &os, intr->src[4].ssa, 0);
      op_vec(p, &os, intr->src[1].ssa, nc);
      /* name is a static buffer: copy before tex_emit prints other things */
      char opn[48];
      snprintf(opn, sizeof(opn), "%s", name);
      tex_emit(p, opn, dim, swap ? 3 : 1, 0, false, swap ? 2 : 1, img, -1, &os, rtn ? &intr->def : NULL, rtn ? 1 : 0);
      break;
   }
   case nir_intrinsic_bindless_image_size: {
      op_def(p, &os, intr->src[1].ssa, 0);
      unsigned dmask = nir_def_components_read(&intr->def) & 0xf;
      if (!dmask)
         dmask = 1;
      if (p->dm && intr->def.index < p->ndm && intr->def.num_components > 1)
         p->dm[intr->def.index] = (uint8_t)dmask;
      tex_emit(p, "tex_query_size", dim, dmask, 0, false, 0, img, -1, &os, &intr->def, util_bitcount(dmask));
      break;
   }
   case nir_intrinsic_bindless_image_samples:
      tex_emit(p, "tex_query_samples", dim, 1, 0, false, 0, img, -1, &os, &intr->def, 1);
      break;
   default:
      pfail(p, "unsupported image intrinsic");
   }
}

static void
print_intrinsic(struct pr *p, nir_intrinsic_instr *intr)
{
   const nir_def *d = nir_intrinsic_infos[intr->intrinsic].has_dest ? &intr->def : NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_elect:
   case nir_intrinsic_first_invocation:
   case nir_intrinsic_load_subgroup_size:
   case nir_intrinsic_load_num_subgroups:
   case nir_intrinsic_load_subgroup_id:
   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
   case nir_intrinsic_as_uniform:
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_ballot:
   case nir_intrinsic_ballot_relaxed:
   case nir_intrinsic_inverse_ballot:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
      print_subgroup(p, intr);
      break;
   case nir_intrinsic_load_push_constant:
      pdst(p, d);
      fprintf(p->f, " load_push_constant");
      psrc(p, intr->src[0].ssa, -1);
      fprintf(p->f, " {base=%u}\n", nir_intrinsic_base(intr));
      break;
   case nir_intrinsic_load_workgroup_id:
   case nir_intrinsic_load_local_invocation_id:
   case nir_intrinsic_load_global_invocation_id:
   case nir_intrinsic_load_subgroup_invocation:
      if (d->bit_size != 32) {
         pfail(p, "%s: only the 32-bit form exists", nir_intrinsic_infos[intr->intrinsic].name);
         return;
      }
      pdst(p, d);
      fprintf(p->f, " %s\n", nir_intrinsic_infos[intr->intrinsic].name);
      break;
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_ubo: {
      uint32_t desc = desc_for(p, intr->src[0].ssa);
      pdst(p, d);
      fprintf(p->f, " %s s%u", nir_intrinsic_infos[intr->intrinsic].name, desc);
      psrc(p, intr->src[1].ssa, -1);
      fprintf(p->f, " {off=0");
      if (intr->intrinsic == nir_intrinsic_load_ssbo)
         paccess(p, nir_intrinsic_access(intr));
      fprintf(p->f, "}\n");
      break;
   }
   case nir_intrinsic_get_ssbo_size: {
      /* OpArrayLength on an SSBO runtime array: byte size = buffer descriptor dw2 (ngcomp: one mov.vu) */
      uint32_t desc = desc_for(p, intr->src[0].ssa);
      pdst(p, d);
      fprintf(p->f, " get_ssbo_size s%u\n", desc);
      break;
   }
   case nir_intrinsic_store_ssbo: {
      uint32_t desc = desc_for(p, intr->src[1].ssa);
      print_store(p, "store_ssbo", intr->src[0].ssa, nir_intrinsic_write_mask(intr), &desc, &intr->src[2].ssa, 1, 0,
                  nir_intrinsic_access(intr));
      break;
   }
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
      pdst(p, d);
      fprintf(p->f, " load_global");
      psrc(p, intr->src[0].ssa, -1);
      fprintf(p->f, " {off=0");
      paccess(p, nir_intrinsic_access(intr));
      fprintf(p->f, "}\n");
      break;
   case nir_intrinsic_store_global:
      print_store(p, "store_global", intr->src[0].ssa, nir_intrinsic_write_mask(intr), NULL, &intr->src[1].ssa, 1, 0,
                  nir_intrinsic_access(intr));
      break;
   case nir_intrinsic_load_shared:
      pdst(p, d);
      fprintf(p->f, " load_shared");
      psrc(p, intr->src[0].ssa, -1);
      fprintf(p->f, " {base=%d}\n", nir_intrinsic_base(intr));
      break;
   case nir_intrinsic_store_shared:
      print_store(p, "store_shared", intr->src[0].ssa, nir_intrinsic_write_mask(intr), NULL, &intr->src[1].ssa, 1,
                  nir_intrinsic_base(intr), 0);
      break;
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_shared_atomic:
      print_atomic(p, intr, false);
      break;
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_shared_atomic_swap:
      print_atomic(p, intr, true);
      break;
   case nir_intrinsic_load_scratch:
      /* src0 = byte offset in the lane's scratch frame (ngcomp adds the frame base u63) */
      pdst(p, d);
      fprintf(p->f, " load_scratch");
      psrc(p, intr->src[0].ssa, -1);
      fprintf(p->f, " {off=0}\n");
      break;
   case nir_intrinsic_store_scratch:
      print_store(p, "store_scratch", intr->src[0].ssa, nir_intrinsic_write_mask(intr), NULL, &intr->src[1].ssa, 1, 0,
                  0);
      break;
   case nir_intrinsic_barrier:
      print_barrier(p, intr);
      break;
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_bindless_image_samples:
      print_image(p, intr);
      break;
   default:
      pfail(p, "unsupported NIR intrinsic '%s' (no ngcomp instruction selection for it)",
            nir_intrinsic_infos[intr->intrinsic].name);
   }
}

static void
print_alu(struct pr *p, nir_alu_instr *alu)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   const bool is_vec = alu->op == nir_op_vec2 || alu->op == nir_op_vec3 || alu->op == nir_op_vec4;
   if (alu->def.num_components != 1 && !is_vec) {
      pfail(p, "vector ALU op '%s' x%u survived scalarization", info->name, alu->def.num_components);
      return;
   }
   if (alu->op == nir_op_vec5 || alu->op == nir_op_vec8 || alu->op == nir_op_vec16) {
      pfail(p, "unsupported vector size in '%s'", info->name);
      return;
   }
   pdst(p, &alu->def);
   fprintf(p->f, " %s", info->name);
   for (unsigned i = 0; i < info->num_inputs; i++) {
      if (info->input_sizes[i] > 1) {
         pfail(p, "ALU op '%s' reads a vector source", info->name);
         return;
      }
      psrc(p, alu->src[i].src.ssa, alu->src[i].swizzle[0]);
   }
 /* : SPIR-V NoContraction (and float_controls2 without AllowContract) arrive as nir_alu_instr::exact; ngcomp
 * must not fuse such an fmul / fadd into ffma (ngcomp/isel.py fuse_fma, DESIGN.md section 13 / 15) */
   if (alu->exact)
      fprintf(p->f, " {exact=1}");
   fprintf(p->f, "\n");
}

static void
print_instr(struct pr *p, nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      print_alu(p, nir_instr_as_alu(instr));
      break;
   case nir_instr_type_load_const: {
      nir_load_const_instr *lc = nir_instr_as_load_const(instr);
      if (lc->def.num_components != 1) {
         pfail(p, "vector load_const survived scalarization");
         return;
      }
      pdst(p, &lc->def);
      uint64_t v = nir_const_value_as_uint(lc->value[0], lc->def.bit_size);
      if (lc->def.bit_size == 1)
         fprintf(p->f, " load_const %s\n", v ? "true" : "false");
      else
         fprintf(p->f, " load_const 0x%llx\n", (unsigned long long)v);
      break;
   }
   case nir_instr_type_undef: {
      nir_undef_instr *u = nir_instr_as_undef(instr);
      if (u->def.num_components != 1) {
         pfail(p, "vector ssa_undef");
         return;
      }
      pdst(p, &u->def);
      fprintf(p->f, u->def.bit_size == 1 ? " load_const false\n" : " load_const 0\n");
      break;
   }
   case nir_instr_type_phi: {
      nir_phi_instr *phi = nir_instr_as_phi(instr);
      if (phi->def.num_components != 1) {
         pfail(p, "vector phi survived scalarization");
         return;
      }
      pdst(p, &phi->def);
      fprintf(p->f, " phi");
      nir_foreach_phi_src (src, phi)
         fprintf(p->f, " b%u:s%u", src->pred->index, src->src.ssa->index);
      fprintf(p->f, "\n");
      break;
   }
   case nir_instr_type_intrinsic:
      print_intrinsic(p, nir_instr_as_intrinsic(instr));
      break;
   case nir_instr_type_tex:
      print_tex(p, nir_instr_as_tex(instr));
      break;
   case nir_instr_type_jump:
      break; /* the terminator is derived from the block successors */
   case nir_instr_type_deref:
      pfail(p, "a deref survived lowering (a variable neither explicit_io nor nir_lower_vars_to_scratch took)");
      break;
   default:
      pfail(p, "unsupported NIR instruction type %d", (int)instr->type);
   }
}

static void
print_terminator(struct pr *p, nir_block *blk)
{
   nir_block *end = p->impl->end_block;
   nir_instr *last = nir_block_last_instr(blk);
   if (last && last->type == nir_instr_type_jump) {
      switch (nir_instr_as_jump(last)->type) {
      case nir_jump_return:
      case nir_jump_halt:
         fprintf(p->f, "  return\n");
         return;
      case nir_jump_break:
      case nir_jump_continue:
         fprintf(p->f, "  jump b%u\n", blk->successors[0]->index);
         return;
      default:
         pfail(p, "unsupported jump instruction");
         return;
      }
   }
   if (blk->successors[1]) {
      nir_if *nif = nir_block_get_following_if(blk);
      if (!nif) {
         pfail(p, "block b%u has two successors but no following if", blk->index);
         return;
      }
      if (nif->condition.ssa->bit_size != 1 || nif->condition.ssa->num_components != 1) {
         pfail(p, "if condition is not a 1-bit scalar");
         return;
      }
      fprintf(p->f, "  br");
      psrc(p, nif->condition.ssa, -1);
      fprintf(p->f, " b%u b%u\n", blk->successors[0]->index, blk->successors[1]->index);
   } else if (!blk->successors[0] || blk->successors[0] == end) {
      fprintf(p->f, "  return\n");
   } else {
      fprintf(p->f, "  jump b%u\n", blk->successors[0]->index);
   }
}

/* Returns a malloc'd NUL-terminated text, or NULL with the reason in err. */
char *
ngv_nir_print_ngcomp(nir_shader *nir, char *err, size_t err_len)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   char *buf = NULL;
   size_t len = 0;
   struct pr p = {.impl = impl};
   p.f = open_memstream(&buf, &len);
   if (!p.f) {
      snprintf(err, err_len, "open_memstream failed");
      return NULL;
   }
   if (nir->info.workgroup_size_variable) {
      pfail(&p, "variable workgroup size is not supported");
   } else {
      nir_index_ssa_defs(impl);
      nir_index_blocks(impl);
      p.next_id = impl->ssa_alloc;
      p.ndm = impl->ssa_alloc;
      p.dm = calloc(impl->ssa_alloc ? impl->ssa_alloc : 1, 1);
      /* function names in the ngcomp text are plain identifiers */
      char name[64];
      const char *ep = impl->function->name ? impl->function->name : "main";
      size_t i = 0;
      for (; ep[i] && i < sizeof(name) - 1; i++)
         name[i] = (ep[i] >= '0' && ep[i] <= '9') || (ep[i] >= 'a' && ep[i] <= 'z') || (ep[i] >= 'A' && ep[i] <= 'Z')
                      ? ep[i]
                      : '_';
      name[i] = 0;
      if (!i || (name[0] >= '0' && name[0] <= '9'))
         snprintf(name, sizeof(name), "main");
      fprintf(p.f, "shader %s local_size=%u,%u,%u shared=%u scratch=%u\n", name,
              nir->info.workgroup_size[0], nir->info.workgroup_size[1], nir->info.workgroup_size[2],
              nir->info.shared_size, nir->scratch_size);
      nir_foreach_block (blk, impl) {
         if (blk == impl->end_block)
            continue;
         fprintf(p.f, "b%u:\n", blk->index);
         p.ndc = 0;
         nir_foreach_instr (instr, blk)
            print_instr(&p, instr);
         print_terminator(&p, blk);
      }
   }
   fclose(p.f);
   free(p.dm);
   if (p.failed) {
      snprintf(err, err_len, "%s", p.err);
      free(buf);
      return NULL;
   }
   return buf;
}
