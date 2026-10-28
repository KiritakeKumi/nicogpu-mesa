/* SPDX-License-Identifier: MIT */
/* ngv_nir.c -- SPIR-V -> NIR for the ngv compute path, and ngv_nir_lower_pipeline_layout (batch 3).
 *
 * The pass list follows lavapipe (lvp_shader_lower) up to the point where the pipeline layout is applied, then
 * scalarizes so that the result is inside the subset ngcomp/nir.py can read (ngv_nir_print.c prints it):
 *   - buffers: spirv_to_nir + nir_lower_explicit_io(vec2_index_32bit_offset) leave
 *     vulkan_resource_index(set, binding, array_index) / load_vulkan_descriptor feeding load_ssbo / store_ssbo /
 *     load_ubo;  ngv_nir_lower_pipeline_layout rewrites vulkan_resource_index to the constant-foldable vec3
 *     {set, SLOT, 0} where SLOT = byte offset of the binding in the set layout / 16 (+ array index * descriptor
 *     slots).  That is exactly the "binding" the ngcomp text calls `load_buffer_desc {set=S, binding=SLOT}`
 * (.md 6.2): the printer resolves the constants and emits the descriptor load.
 *   - push constants: load_push_constant with 32-bit offsets; the first 16 dwords live in u10..u25 (ngcomp).
 *   - atomics: nir_lower_explicit_io turns deref atomics into ssbo_atomic(_swap) / global_atomic(_swap) /
 *     shared_atomic(_swap), which ngv_nir_print.c prints and ngcomp selects (isel_atomic.py);
 *   - scratch: function-local variables that stay in memory go through nir_lower_vars_to_scratch to load_scratch /
 *     store_scratch (nir->scratch_size bytes per lane).
 *   - base workgroup id (vkCmdDispatchBase): load_base_workgroup_id becomes a load_push_constant of three dwords the
 * driver writes behind the layout's push range (.md 14.3).
 * - gl_NumWorkGroups (PROVISIONAL): load_num_workgroups becomes three 32-bit load_global from a 64-bit VA the
 *     driver writes into two push dwords behind the base-workgroup dwords (or behind the push range): the grid of a
 *     direct dispatch lives in the command buffer's upload area, the one of an indirect dispatch is read straight
 *     from the VkDispatchIndirectCommand (no CP packet can put memory contents into a user register).
 */
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_deref.h"
#include "compiler/spirv/nir_spirv.h"
#include "vk_nir.h"
#include "vk_pipeline.h"

#include "ngv_private.h"

/* NIR compiler options: everything ngcomp's isel table cannot do is lowered by NIR.  PROVISIONAL list -- extend as
 * dEQP finds gaps; an op outside the table fails the pipeline with a message naming it (isel's "known gaps"). */
const nir_shader_compiler_options ngv_nir_options = {
   .lower_fdiv = false,
   .lower_flrp16 = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_fpow = true,
   .lower_fmod = true,
   .lower_fsign = true,
   .lower_isign = true,
   .lower_scmp = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .lower_bitfield_extract = true,
   .lower_bitfield_insert = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_uadd_sat = true,
   .lower_usub_sat = true,
   .lower_iadd_sat = true,
   .lower_ifind_msb = true,
   .lower_find_lsb = true,
   .lower_ldexp = true,
   .lower_pack_half_2x16 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_64_2x32 = true,
   .lower_unpack_half_2x16 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_hadd = true,
   .lower_mul_32x16 = false,
   .lower_to_scalar = true,
   .lower_vector_cmp = true,
   .lower_int64_options = nir_lower_divmod64 | nir_lower_imul_high64 | nir_lower_minmax64 |
                          nir_lower_isign64 | nir_lower_icmp64 | nir_lower_iabs64 | nir_lower_ineg64 |
                          nir_lower_bit_count64 | nir_lower_extract64 | nir_lower_ufind_msb64 | nir_lower_conv64,
   .has_isub = true,
   .has_fsub = true,
   .max_unroll_iterations = 0,
   .support_16bit_alu = false,
   .scalarize_ddx = true,
 /* gl_DeviceIndex (SPIR-V DeviceGroup): every device group of this driver has exactly one device */
   .lower_device_index_to_zero = true,
};

/* nir_lower_phis_to_scalar's default filter only splits a phi when one of its sources is "scalarizable" (an ALU op, a
 * load_const,...); a phi whose sources are all texture / image results (: tex instructions define vec4s) would stay a
 * vector and the printer rejects it.  Every phi gets split: ngcomp has no vector registers for values. */
static uint8_t
ngv_phi_width(const nir_instr *instr, const void *data)
{
   return 1;
}

static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   uint32_t comp_size = glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length;
   *align = comp_size;
}

static bool
ngv_optimize_once(nir_shader *nir)
{
   bool progress = false;
   NIR_PASS(progress, nir, nir_split_var_copies);
   NIR_PASS(progress, nir, nir_lower_var_copies);
   NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
   NIR_PASS(progress, nir, nir_copy_prop);
   NIR_PASS(progress, nir, nir_opt_remove_phis);
   NIR_PASS(progress, nir, nir_opt_dce);
   NIR_PASS(progress, nir, nir_opt_dead_cf);
   NIR_PASS(progress, nir, nir_opt_cse);
   NIR_PASS(progress, nir, nir_opt_algebraic);
   NIR_PASS(progress, nir, nir_opt_constant_folding);
   NIR_PASS(progress, nir, nir_opt_undef);
   NIR_PASS(progress, nir, nir_opt_deref);
   NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
   NIR_PASS(progress, nir, nir_opt_if, nir_opt_if_optimize_phi_true_false);
   return progress;
}

static void
ngv_optimize(nir_shader *nir)
{
   while (ngv_optimize_once(nir))
      ;
}

/* ---- ngv_lower_half_pack ------------------------------------------------------------------------- */

/* packHalf2x16 / unpackHalf2x16 (ncnn's fp16 storage) in 32-bit integer and float arithmetic, for targets without the
 * VALU binary16 conversions (RMs before : units without "f16cvt"; ngcomp has no 16-bit types either way).
 * half -> float is exact; denormal halves go through u2f * 2^-24 so a flush-to-zero FPU still gets them right.
 * float -> half rounds to nearest even, NaN -> 0x7e00, overflow -> inf (the usual branch-free form, F. Giesen).
 * Differences to the hardware path (section 2.4): NaN payloads / signs and sNaN quieting only. */
static nir_def *
half_to_float(nir_builder *b, nir_def *h)
{
   nir_def *s = nir_ishl_imm(b, nir_iand_imm(b, h, 0x8000), 16);
   nir_def *e = nir_iand_imm(b, nir_ushr_imm(b, h, 10), 0x1f);
   nir_def *m = nir_iand_imm(b, h, 0x3ff);
   nir_def *m13 = nir_ishl_imm(b, m, 13);
   nir_def *norm = nir_ior(b, nir_ishl_imm(b, nir_iadd_imm(b, e, 112), 23), m13);
   nir_def *inf = nir_ior_imm(b, m13, 0x7f800000);
   nir_def *den = nir_fmul_imm(b, nir_u2f32(b, m), 1.0 / 16777216.0);
   nir_def *r = nir_bcsel(b, nir_ieq_imm(b, e, 0), den, nir_bcsel(b, nir_ieq_imm(b, e, 31), inf, norm));
   return nir_ior(b, r, s);
}

static nir_def *
float_to_half(nir_builder *b, nir_def *f)
{
   nir_def *sign = nir_iand_imm(b, f, 0x80000000u);
   nir_def *x = nir_iand_imm(b, f, 0x7fffffff);
   nir_def *o_inf = nir_bcsel(b, nir_ult(b, nir_imm_int(b, 0x7f800000), x), nir_imm_int(b, 0x7e00),
                              nir_imm_int(b, 0x7c00));
   /* |f| < 2^-14: the half is denormal; adding 0.5 puts its bits at the bottom of the mantissa, rounded (RTE) */
   nir_def *o_den = nir_iadd_imm(b, nir_fadd_imm(b, x, 0.5), -0x3f000000);
   nir_def *odd = nir_iand_imm(b, nir_ushr_imm(b, x, 13), 1);
   nir_def *o_norm = nir_ushr_imm(b, nir_iadd(b, nir_iadd_imm(b, x, (int32_t)0xc8000fffu), odd), 13);
   nir_def *o = nir_bcsel(b, nir_uge_imm(b, x, 0x47800000), o_inf,
                          nir_bcsel(b, nir_ult_imm(b, x, 0x38800000), o_den, o_norm));
   return nir_ior(b, o, nir_ushr_imm(b, sign, 16));
}

static bool
half_pack_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   switch (nir_instr_as_alu(instr)->op) {
   case nir_op_unpack_half_2x16_split_x:
   case nir_op_unpack_half_2x16_split_y:
   case nir_op_pack_half_2x16_split:
      return true;
   default:
      return false;
   }
}

static nir_def *
half_pack_lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *s0 = nir_ssa_for_alu_src(b, alu, 0);
   switch (alu->op) {
   case nir_op_unpack_half_2x16_split_x:
      return half_to_float(b, nir_iand_imm(b, s0, 0xffff));
   case nir_op_unpack_half_2x16_split_y:
      return half_to_float(b, nir_ushr_imm(b, s0, 16));
   default: { /* pack_half_2x16_split */
 /* : a single f32 -> f16 conversion is pack_half_2x16_split(x, 0): skip the half that is known to be 0 */
      nir_def *s1 = nir_ssa_for_alu_src(b, alu, 1);
      nir_scalar c1 = nir_scalar_resolved(s1, 0);
      if (nir_scalar_is_const(c1) && nir_scalar_as_uint(c1) == 0)
         return float_to_half(b, s0);
      return nir_ior(b, float_to_half(b, s0), nir_ishl_imm(b, float_to_half(b, s1), 16));
   }
   }
}

static bool
ngv_lower_half_pack(nir_shader *nir)
{
   return nir_shader_lower_instructions(nir, half_pack_filter, half_pack_lower, NULL);
}

/* ---- 16-bit types --------------------------------------------------------------- */

/* : the 8-bit types (shaderInt8, 8-bit storage) take the same path as the 16-bit ones: zero-extended in a 32-bit
 * register (ld.*.u8, u2u8 = zext.b), every ALU op redone in 32 bits, scalar ld/st.*.u8 / b8. */
static inline bool
ngv_small_bits(unsigned bit_size)
{
   return bit_size == 8 || bit_size == 16;
}

/* shaderInt16 / shaderFloat16 / 16-bit storage.  The ISA has binary16 and int16 ALU instructions (H format), but the
 * RTL VALU does not decode them (ng_valu_dec skips them, section 1.3) and ngcomp has no 16-bit ALU rules, so
 * the only path that runs on the model *and* on the board is promotion to 32 bits:
 *   - a 16-bit value lives zero-extended in a 32-bit register (ld.*.u16, u2u16 = zext.h); isel keeps mov / vec / bcsel / phi /
 *     load_const / u2u* / i2i* / b2i16 / pack_32_2x16_split / unpack_32_2x16_split_x|y on 16-bit values and nothing else;
 *   - every other ALU op with a 16-bit operand or result is rebuilt by nir_lower_bit_size in 32 bits (sign / zero extension
 *     per the operand's NIR type, result narrowed again);
 *   - float16 <-> float32 is the pair of binary16 conversions the hardware (units "f16cvt") or ngv_lower_half_pack (integer
 *     code, any RM) executes: f2f32(h) = unpack_half_2x16_split_x(u2u32 h), f2f16(f) = u2u16(pack_half_2x16_split(f, 0)).
 *     A float16 op is therefore the f32 op with its result rounded to nearest even and its operands exact: the same value as a
 *     binary16 operation for + - * / and sqrt (an f32 significand holds the exact result or enough bits for a single correct
 * rounding), double rounded for fma and the transcendentals (section 3);
 *   - 16-bit memory accesses are split into scalar ones (ld/st.*.u16 / b16) by nir_lower_mem_access_bit_sizes. */
static bool
ngv_alu_keeps_16bit(nir_op op)
{
   switch (op) {
   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_vec5:
   case nir_op_vec8:
   case nir_op_vec16:
   case nir_op_bcsel:
   case nir_op_b32csel:
   case nir_op_u2u8:
   case nir_op_u2u16:
   case nir_op_u2u32:
   case nir_op_u2u64:
   case nir_op_i2i8:
   case nir_op_i2i16:
   case nir_op_i2i32:
   case nir_op_i2i64:
   case nir_op_b2i8:
   case nir_op_b2i16:
   case nir_op_pack_32_2x16_split:
   case nir_op_unpack_32_2x16_split_x:
   case nir_op_unpack_32_2x16_split_y:
   case nir_op_pack_half_2x16_split:
   case nir_op_pack_half_2x16_rtz_split:
   case nir_op_unpack_half_2x16_split_x:
   case nir_op_unpack_half_2x16_split_y:
      return true;
   default:
      return false;
   }
}

/* nir_lower_bit_size callback: the ALU ops whose result type follows their sources (osz == 0) or is a bool (comparisons,
 * i2b1 / f2b1) are redone in 32 bits.  The conversions with a fixed result type are ngv_lower_16bit_conv's. */
static unsigned
ngv_bit_size_cb(const nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_alu)
      return 0;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (ngv_alu_keeps_16bit(alu->op))
      return 0;
   const nir_op_info *info = &nir_op_infos[alu->op];
   bool has16 = ngv_small_bits(alu->def.bit_size);
   for (unsigned i = 0; i < info->num_inputs; i++)
      has16 |= ngv_small_bits(nir_src_bit_size(alu->src[i].src));
   if (!has16)
      return 0;
   const unsigned osz = nir_alu_type_get_type_size(info->output_type);
   return (osz == 0 || osz == 1) ? 32 : 0;
}

static nir_def *
ngv_f16_to_f32(nir_builder *b, nir_def *h)
{
   return nir_unpack_half_2x16_split_x(b, nir_u2u32(b, h));
}

static nir_def *
ngv_f32_to_f16(nir_builder *b, nir_def *f, bool rtz)
{
   nir_def *z = nir_imm_float(b, 0.0f);
   return nir_u2u16(b, rtz ? nir_pack_half_2x16_rtz_split(b, f, z) : nir_pack_half_2x16_split(b, f, z));
}

/* ALU instructions that still touch a float16 (or convert an int16 to float) after nir_lower_bit_size: the conversions. */
static bool
conv16_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (ngv_alu_keeps_16bit(alu->op))
      return false;
   const nir_op_info *info = &nir_op_infos[alu->op];
   for (unsigned i = 0; i < info->num_inputs; i++) {
      if (nir_alu_type_get_type_size(info->input_types[i]) != 0 || !ngv_small_bits(nir_src_bit_size(alu->src[i].src)))
         continue;
      if (nir_alu_type_get_base_type(info->input_types[i]) == nir_type_float)
         return true;
      /* an int16 operand of a float result needs its sign / zero extension; bit_count / ufind_msb / ... read the
       * zero-extended register as it is */
      if (nir_alu_type_get_base_type(info->output_type) == nir_type_float)
         return true;
   }
   if (ngv_small_bits(alu->def.bit_size) && nir_alu_type_get_type_size(info->output_type) == alu->def.bit_size)
      return nir_alu_type_get_base_type(info->output_type) == nir_type_float || alu->op == nir_op_f2i16 ||
             alu->op == nir_op_f2u16 || alu->op == nir_op_f2i8 || alu->op == nir_op_f2u8;
   return false;
}

static nir_def *
conv16_lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_def *srcs[NIR_ALU_MAX_INPUTS] = {NULL};
   for (unsigned i = 0; i < info->num_inputs; i++) {
      srcs[i] = nir_ssa_for_alu_src(b, alu, i);
      if (nir_alu_type_get_type_size(info->input_types[i]) != 0 || !ngv_small_bits(srcs[i]->bit_size))
         continue;
      switch (nir_alu_type_get_base_type(info->input_types[i])) {
      case nir_type_float:
         srcs[i] = ngv_f16_to_f32(b, srcs[i]);
         break;
      case nir_type_int:
         if (nir_alu_type_get_base_type(info->output_type) == nir_type_float)
            srcs[i] = nir_i2i32(b, srcs[i]);
         break;
      case nir_type_uint:
         if (nir_alu_type_get_base_type(info->output_type) == nir_type_float)
            srcs[i] = nir_u2u32(b, srcs[i]);
         break;
      default:
         break;
      }
   }
   switch (alu->op) {
   case nir_op_f2f16:
   case nir_op_f2f16_rtne:
   case nir_op_f2fmp:
      return ngv_f32_to_f16(b, srcs[0], false);
   case nir_op_f2f16_rtz:
      return ngv_f32_to_f16(b, srcs[0], true);
   case nir_op_i2f16:
   case nir_op_i2fmp:
      return ngv_f32_to_f16(b, nir_i2f32(b, srcs[0]), false);
   case nir_op_u2f16:
      return ngv_f32_to_f16(b, nir_u2f32(b, srcs[0]), false);
   case nir_op_b2f16:
      return ngv_f32_to_f16(b, nir_b2f32(b, srcs[0]), false);
   case nir_op_f2i16:
      return nir_u2u16(b, nir_f2i32(b, srcs[0]));
   case nir_op_f2u16:
      return nir_u2u16(b, nir_f2u32(b, srcs[0]));
   case nir_op_f2i8:
      return nir_u2u8(b, nir_f2i32(b, srcs[0]));
   case nir_op_f2u8:
      return nir_u2u8(b, nir_f2u32(b, srcs[0]));
   case nir_op_f2f32:
      return srcs[0]; /* the operand was a float16 (conv16_filter), already unpacked */
   default:
      return nir_build_alu_src_arr(b, alu->op, srcs);
   }
}

static nir_mem_access_size_align
ngv_mem_access_cb(nir_intrinsic_op intrin, uint8_t bytes, uint8_t bit_size, uint32_t align_mul, uint32_t align_offset,
                  bool offset_is_const, enum gl_access_qualifier access, const void *cb_data)
{
   /* 16-bit accesses: one ld.*.u16 / st.*.b16 per component (a vector of 16-bit values would be read as packed dwords);
    * everything else is left as it is (the requested size and a minimal alignment: nothing is split) */
   if (bit_size == 16)
      return (nir_mem_access_size_align){.num_components = 1, .bit_size = 16, .align = 2};
 if (bit_size == 8) /* : 8-bit accesses are scalar ld.*.u8 / st.*.b8 too */
      return (nir_mem_access_size_align){.num_components = 1, .bit_size = 8, .align = 1};
 if (bit_size == 64) /* (shaderInt64): 64-bit vectors are read / written as scalar 64-bit accesses */
      return (nir_mem_access_size_align){.num_components = 1, .bit_size = 64, .align = 4};
   return (nir_mem_access_size_align){.num_components = bytes / (bit_size / 8), .bit_size = bit_size, .align = 1};
}

static bool
def16_cb(nir_def *def, void *data)
{
   if (ngv_small_bits(def->bit_size) || def->bit_size == 64) {
      *(bool *)data = true;
      return false;
   }
   return true;
}

/* Does the shader have any 16-bit value?  The passes below only run then, so every other shader compiles as before. */
static bool
ngv_shader_uses_16bit(nir_shader *nir)
{
   bool found = false;
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            nir_foreach_def(instr, def16_cb, &found);
            if (found)
               return true;
         }
      }
   }
   return false;
}

/* Memory accesses first (they create 16-bit vec / mov), then the ALU. */
static bool
ngv_nir_lower_16bit_mem(nir_shader *nir)
{
   if (!ngv_shader_uses_16bit(nir))
      return false;
   const nir_lower_mem_access_bit_sizes_options opts = {
      .callback = ngv_mem_access_cb,
      .modes = nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_mem_global | nir_var_mem_constant | nir_var_mem_shared |
               nir_var_shader_temp | nir_var_function_temp,
   };
   return nir_lower_mem_access_bit_sizes(nir, &opts);
}

static bool
ngv_nir_lower_16bit_alu(nir_shader *nir)
{
   bool progress = false;
   /* GLSL.std.450 FMix on float16 arrives as flrp (the options flag only keeps the optimizer from making new ones) */
   NIR_PASS(progress, nir, nir_lower_flrp, 16, false);
   NIR_PASS(progress, nir, nir_lower_pack); /* pack_32_2x16 / unpack_32_2x16 / pack_64_4x16 -> the _split forms isel has */
   NIR_PASS(progress, nir, nir_lower_bit_size, ngv_bit_size_cb, NULL);
   NIR_PASS(progress, nir, nir_shader_lower_instructions, conv16_filter, conv16_lower, NULL);
   return progress;
}

/* ---- ngv_nir_lower_pipeline_layout ---------------------------------------------------------------------- */

struct layout_ctx {
   const struct vk_pipeline_layout *layout;
   int32_t base_push_offset; /* -1: pipeline was not created with DISPATCH_BASE */
   int32_t numwg_push_offset; /* -1: no room for the num-workgroups pointer in the first 16 push dwords */
   bool uses_num_workgroups;
   const char *error;
};

static const struct ngv_binding_layout *
find_binding(const struct vk_pipeline_layout *layout, uint32_t set, uint32_t binding)
{
   if (set >= layout->set_count || !layout->set_layouts[set])
      return NULL;
   const struct ngv_descriptor_set_layout *sl =
      container_of(layout->set_layouts[set], struct ngv_descriptor_set_layout, vk);
   for (uint32_t i = 0; i < sl->binding_count; i++)
      if (sl->bindings[i].binding == binding)
         return &sl->bindings[i];
   return NULL;
}

static bool
lower_layout_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct layout_ctx *ctx = data;
   switch (intr->intrinsic) {
   case nir_intrinsic_vulkan_resource_index: {
      b->cursor = nir_before_instr(&intr->instr);
      const uint32_t set = nir_intrinsic_desc_set(intr), binding = nir_intrinsic_binding(intr);
      const struct ngv_binding_layout *bl = find_binding(ctx->layout, set, binding);
      if (!bl) {
         ctx->error = "shader uses a descriptor binding that is not in the pipeline layout";
         return false;
      }
      /* slot = byte offset / 16; one array element = stride / 16 slots (buffers: 1) */
      nir_def *idx = nir_iadd_imm(b, nir_imul_imm(b, intr->src[0].ssa, bl->stride / 16), bl->offset / 16);
      nir_def_rewrite_uses(&intr->def, nir_vec3(b, nir_imm_int(b, set), idx, nir_imm_int(b, 0)));
      return true;
   }
   case nir_intrinsic_vulkan_resource_reindex: {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *old = intr->src[0].ssa, *delta = intr->src[1].ssa;
      /* the reindex delta counts array elements; PROVISIONAL: element == 1 slot (buffer descriptors only) */
      nir_def_rewrite_uses(&intr->def, nir_vec3(b, nir_channel(b, old, 0), nir_iadd(b, nir_channel(b, old, 1), delta),
                                                nir_channel(b, old, 2)));
      return true;
   }
   case nir_intrinsic_load_vulkan_descriptor:
      nir_def_rewrite_uses(&intr->def, intr->src[0].ssa);
      return true;
   case nir_intrinsic_load_base_workgroup_id: {
      b->cursor = nir_before_instr(&intr->instr);
      if (ctx->base_push_offset < 0) {
         /* not a DISPATCH_BASE pipeline: the base is 0 by VU */
         nir_def_rewrite_uses(&intr->def, nir_imm_zero(b, 3, 32));
      } else {
         nir_def *v = nir_load_push_constant(b, 3, 32, nir_imm_int(b, ctx->base_push_offset), .base = 0, .range = 12);
         nir_def_rewrite_uses(&intr->def, v);
      }
      return true;
   }
   case nir_intrinsic_load_num_workgroups: {
      if (ctx->numwg_push_offset < 0) {
         ctx->error = "gl_NumWorkGroups: the push range leaves no room for the driver's num-workgroups pointer in the "
                      "first 16 push dwords";
         return false;
      }
      if (intr->def.bit_size != 32) {
         ctx->error = "gl_NumWorkGroups: only the 32-bit form exists";
         return false;
      }
      b->cursor = nir_before_instr(&intr->instr);
      ctx->uses_num_workgroups = true;
      /* one scalar load per component: unused components become dead code, and no 96-bit access at the (only 4 B
       * aligned) VkDispatchIndirectCommand offset is ever issued */
      nir_def *va = nir_load_push_constant(b, 1, 64, nir_imm_int(b, ctx->numwg_push_offset), .base = 0, .range = 8);
      nir_def *c[3];
      for (unsigned k = 0; k < 3; k++) {
         c[k] = nir_load_global(b, k ? nir_iadd_imm(b, va, 4 * k) : va, 4, 1, 32);
         nir_intrinsic_set_access(nir_instr_as_intrinsic(c[k]->parent_instr),
                                  ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      }
      nir_def *v = nir_vec3(b, c[0], c[1], c[2]);
      if (intr->def.num_components < 3)
         v = nir_trim_vector(b, v, intr->def.num_components);
      nir_def_rewrite_uses(&intr->def, v);
      return true;
   }
   default:
      return false;
   }
}

/* Applies the pipeline layout to resource accesses and the driver's push dwords to the system values they back (base
 * workgroup id, num workgroups; *uses_num_workgroups tells whether the latter is read).  Returns false (and sets
 * *error) if the shader refers to a binding the layout does not have or needs push space that is not there. */
bool
ngv_nir_lower_pipeline_layout(nir_shader *nir, const struct vk_pipeline_layout *layout, int32_t base_push_offset,
                              int32_t numwg_push_offset, bool *uses_num_workgroups, const char **error)
{
   struct layout_ctx ctx = {
      .layout = layout,
      .base_push_offset = base_push_offset,
      .numwg_push_offset = numwg_push_offset,
   };
   nir_shader_intrinsics_pass(nir, lower_layout_instr, nir_metadata_control_flow, &ctx);
   if (ctx.error && error)
      *error = ctx.error;
   if (uses_num_workgroups)
      *uses_num_workgroups = ctx.uses_num_workgroups;
   return !ctx.error;
}

/* ---- textures and images ------------------------------------------------------------------------------- */

/* Image / sampler derefs become handles: texture_handle / sampler_handle sources of tex instructions and the handle source of
 * bindless_image_* intrinsics.  A handle is the 32-bit value (set << 24) | slot, slot = descriptor byte offset / 16
 * (docs: the printer turns it into `load_image_desc` / `load_sampler_desc`); an array element adds index * stride / 16.  The
 * sampler half of a COMBINED_IMAGE_SAMPLER binding is two slots (32 B) behind the image half.  A handle that is not a
 * compile-time constant (dynamic descriptor index) makes the printer fail: the waterfall loop is a gap. */
struct tex_ctx {
   const struct vk_pipeline_layout *layout;
   const char *error;
};

#define NGV_HANDLE_SET_SHIFT 24

static nir_def *
res_handle(nir_builder *b, struct tex_ctx *ctx, nir_deref_instr *deref, bool sampler_half)
{
   nir_deref_path path;
   nir_deref_path_init(&path, deref, NULL);
   nir_variable *var = path.path[0]->var;
   const struct ngv_binding_layout *bl = find_binding(ctx->layout, var->data.descriptor_set, var->data.binding);
   if (!bl) {
      ctx->error = "shader uses an image or sampler binding that is not in the pipeline layout";
      nir_deref_path_finish(&path);
      return NULL;
   }
   bool ok;
   if (sampler_half)
      ok = bl->type == VK_DESCRIPTOR_TYPE_SAMPLER || bl->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   else
      ok = bl->type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || bl->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
           bl->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
           bl->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER || bl->type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
   if (!ok) {
      ctx->error = "image or sampler variable bound to a descriptor of another type";
      nir_deref_path_finish(&path);
      return NULL;
   }
   nir_def *index = NULL;
   const struct glsl_type *t = var->type;
   for (nir_deref_instr **p = &path.path[1]; *p; p++) {
      if ((*p)->deref_type != nir_deref_type_array || !glsl_type_is_array(t)) {
         ctx->error = "unsupported deref of an image or sampler variable";
         nir_deref_path_finish(&path);
         return NULL;
      }
      const struct glsl_type *elem = glsl_get_array_element(t);
      const unsigned mult = glsl_type_is_array(elem) ? glsl_get_aoa_size(elem) : 1;
      nir_def *i = (*p)->arr.index.ssa;
      if (i->bit_size != 32)
         i = nir_u2u32(b, i);
      if (mult != 1)
         i = nir_imul_imm(b, i, mult);
      index = index ? nir_iadd(b, index, i) : i;
      t = elem;
   }
   nir_deref_path_finish(&path);
   const uint32_t base = (var->data.descriptor_set << NGV_HANDLE_SET_SHIFT) |
                         (bl->offset / 16 + (sampler_half && bl->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? 2 : 0));
   if (!index)
      return nir_imm_int(b, base);
   return nir_iadd_imm(b, nir_imul_imm(b, index, bl->stride / 16), base);
}

static bool
lower_tex_res_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct tex_ctx *ctx = data;
   if (ctx->error)
      return false;
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      const int ti = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      const int si = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
      if (ti < 0 && si < 0)
         return false;
      b->cursor = nir_before_instr(instr);
      nir_def *th = NULL, *sh = NULL;
      if (ti >= 0) {
         th = res_handle(b, ctx, nir_src_as_deref(tex->src[ti].src), false);
         if (!th)
            return false;
      }
      if (si >= 0) {
         sh = res_handle(b, ctx, nir_src_as_deref(tex->src[si].src), true);
         if (!sh)
            return false;
      }
      /* remove the later index first */
      if (si >= 0 && si > ti) {
         nir_tex_instr_remove_src(tex, si);
         if (ti >= 0)
            nir_tex_instr_remove_src(tex, ti);
      } else {
         if (ti >= 0)
            nir_tex_instr_remove_src(tex, ti);
         if (si >= 0)
            nir_tex_instr_remove_src(tex, si);
      }
      if (th)
         nir_tex_instr_add_src(tex, nir_tex_src_texture_handle, th);
      if (sh)
         nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, sh);
      return true;
   }
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples: {
      b->cursor = nir_before_instr(instr);
      nir_def *h = res_handle(b, ctx, nir_src_as_deref(intr->src[0]), false);
      if (!h)
         return false;
      nir_rewrite_image_intrinsic(intr, h, true);
      return true;
   }
   default:
      return false;
   }
}

static bool
ngv_nir_lower_tex_resources(nir_shader *nir, const struct vk_pipeline_layout *layout, const char **error)
{
   struct tex_ctx ctx = {.layout = layout};
   nir_shader_instructions_pass(nir, lower_tex_res_instr, nir_metadata_control_flow, &ctx);
   if (ctx.error && error)
      *error = ctx.error;
   return !ctx.error;
}

/* ---- subgroups ---------------------------------------------------------------- */

/* One subgroup = one warp (32 lanes, fixed).  ngcomp selects elect / vote_any / vote_all / vote_ieq (bool) / ballot /
 * inverse_ballot / read_first_invocation / read_invocation / shuffle (+ _xor / _up / _down) / quad_* / first_invocation /
 * as_uniform / load_subgroup_{size,id} / load_num_subgroups natively (ngcomp/isel_subgroup.py: EXEC + SALU code,
 * XLANE instructions only if the target has the unit, else a bit-serial cross-lane read); everything else is lowered here
 * onto those (PROVISIONAL option set):
 *   - ballots are one 32-bit word; ballot_bitfield_extract / bit_count / find_lsb / find_msb become 32-bit ALU;
 *   - the subgroup masks become shifts of load_subgroup_invocation;
 *   - vote_ieq / vote_feq (non-bool) become read_first + compare + vote_all; 64-bit shuffles split in two 32-bit ones;
 *   - bool shuffles / read_invocation go through ballot (lower_boolean_shuffle);
 *   - dynamic quad_broadcast and (clustered) rotate become shuffles;
 *   - reduce / inclusive_scan / exclusive_scan (ARITHMETIC / CLUSTERED) become the generic shuffle-based code of
 *     lower_reduce / lower_boolean_reduce, except (target with XLANE) the whole-warp scalar 32-bit integer reduce
 *     (iadd imin imax umin umax iand ior ixor) and iadd scans: those stay intrinsics, ngcomp selects xred.* / xscan.add.u32.
 * The pass emits intrinsics it lowers itself (lower_reduce -> load_subgroup_lt_mask, ballot_find_msb, ...), hence the
 * loop.  Runs before nir_lower_int64 (64-bit compares from lower_vote_ieq) and the scalarization passes. */
static const char ngv_sg_native; /* its address is filter_data when the target has XLANE */

static bool
ngv_subgroup_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   const bool has_xlane = data != NULL;
   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (!has_xlane)
      return true;
   switch (intr->intrinsic) {
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan: {
      const unsigned cl = intr->intrinsic == nir_intrinsic_reduce ? nir_intrinsic_cluster_size(intr) : 0;
      if (intr->def.bit_size != 32 || intr->def.num_components != 1 || (cl != 0 && cl < 32))
         return true;
      switch (nir_intrinsic_reduction_op(intr)) {
      case nir_op_iadd:
         return false;
      case nir_op_imin:
      case nir_op_imax:
      case nir_op_umin:
      case nir_op_umax:
      case nir_op_iand:
      case nir_op_ior:
      case nir_op_ixor:
         return intr->intrinsic != nir_intrinsic_reduce;
      default:
         return true;
      }
   }
   default:
      return true;
   }
}

static bool
ngv_nir_lower_subgroups(nir_shader *nir, bool has_xlane)
{
   const nir_lower_subgroups_options opts = {
      .filter = ngv_subgroup_filter,
      .filter_data = has_xlane ? &ngv_sg_native : NULL,
      .subgroup_size = 32, /* == nicogpu_device_info::warp_size of every Nicogpu device (ISA: 32-lane warps) */
      .ballot_bit_size = 32,
      .ballot_components = 1,
      .lower_to_scalar = true,
      .lower_vote_feq = true,
      .lower_vote_ieq = true,
      .lower_vote_bool_eq = false, /* ngcomp: (b & EXEC) == 0 || (b & EXEC) == EXEC */
      .lower_subgroup_masks = true,
      .lower_shuffle_to_32bit = true,
      .lower_quad_broadcast_dynamic = true,
      .lower_quad_vote = true,
      .lower_rotate_to_shuffle = true,
      .lower_rotate_clustered_to_shuffle = true,
      .lower_reduce = true,
      .lower_boolean_reduce = true,
      .lower_boolean_shuffle = true,
   };
   bool any = false;
   for (unsigned k = 0; k < 4; k++) {
      bool progress = false;
      NIR_PASS(progress, nir, nir_lower_subgroups, &opts);
      if (!progress)
         break;
      any = true;
   }
   return any;
}

/* ---- SPIR-V -> NIR -------------------------------------------------------------------------------------- */

VkResult
ngv_shader_to_nir(struct ngv_device *dev, VkPipelineCreateFlags2KHR pipeline_flags,
                  const VkPipelineShaderStageCreateInfo *sinfo, const struct vk_pipeline_layout *layout,
                  int32_t base_push_offset, int32_t numwg_push_offset, bool *uses_num_workgroups, nir_shader **out)
{
   const struct spirv_to_nir_options spirv_options = {
      .environment = NIR_SPIRV_VULKAN,
      .ubo_addr_format = nir_address_format_vec2_index_32bit_offset,
      .ssbo_addr_format = nir_address_format_vec2_index_32bit_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_logical,
      .shared_addr_format = nir_address_format_32bit_offset,
      .constant_addr_format = nir_address_format_64bit_global,
   };

   nir_shader *nir = NULL;
   VkResult r = vk_pipeline_shader_stage_to_nir(&dev->vk, pipeline_flags, sinfo, &spirv_options, &ngv_nir_options, NULL,
                                                &nir);
   if (r != VK_SUCCESS)
      return r;

   NIR_PASS(_, nir, nir_lower_system_values);
   const struct nir_lower_compute_system_values_options csv = {
      .has_base_workgroup_id = base_push_offset >= 0,
      .lower_local_invocation_index = true,   /* hardware gives only the packed local id (v0) */
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_uniform | nir_var_image, NULL);
   ngv_optimize(nir);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   NIR_PASS(_, nir, nir_split_var_copies);
   /* Private variables with constant initializers (lookup tables): turn the initializer into stores first */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_shader_temp | nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const, nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo | nir_var_mem_ssbo,
            nir_address_format_vec2_index_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global | nir_var_mem_constant, nir_address_format_64bit_global);
 /* VK_KHR_workgroup_memory_explicit_layout : for Block-decorated Workgroup variables the pass only assigns
    * driver_location (aliased blocks share one region) and sets shared_size; shared_var_info is not used then */
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared, shared_var_info);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
 /* / : workgroup memory is UNDEFINED at workgroup start in hardware (command-set spec section 2.2). A shader whose
    * Workgroup variables carry a null initializer (VK_KHR_zero_initialize_workgroup_memory) gets an explicit store
    * loop (or a single guarded store when one round covers it) + workgroup barrier here.  The pass reads
    * load_local_invocation_index, which this driver lowers to the packed local id (v0), so the compute system values
    * are lowered again afterwards.  The pass stores whole 16-byte chunks up to the 16-aligned size, so the shared
    * window (SHARED_BYTES = nir->info.shared_size, ngv_pipeline.c) grows to that size: a window that ends inside the
    * last chunk would trap (SharedBounds) on the zeroing store.  The forms the pass produces (loop with a divergent
 * break, vec4 store_shared, barrier) are checked on the model by ngcomp/isel_selftest.py kernel_zinit_*. */
   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      const unsigned chunk_size = 16;
      const unsigned shared_size = ALIGN_POT(nir->info.shared_size, chunk_size);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, shared_size, chunk_size);
      nir->info.shared_size = shared_size;
      NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);
   }
 /* : function-local variables that survived SSA conversion (dynamically indexed arrays,...) and Private
    * variables moved into main() by nir_lower_global_vars_to_local live in per-lane scratch: load_scratch /
    * store_scratch with a byte offset, nir->scratch_size = bytes per lane (the printer emits `scratch=BYTES`, ngcomp
    * adds its spill slots behind it, the driver backs the total with a BO per device: ngv_scratch_get).  The globals
    * moved into main() have not been through SSA conversion yet, so run the optimizer first or they would all end up in
    * scratch.  size_threshold 0: the pass leaves variables of at most that size alone for a later
    * nir_lower_indirect_derefs, which this driver does not run, so every remaining variable must go to scratch. */
   ngv_optimize(nir);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);
   NIR_PASS(_, nir, nir_lower_vars_to_scratch, nir_var_function_temp, 0, glsl_get_natural_size_align_bytes,
            glsl_get_natural_size_align_bytes);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_temp, NULL);

   const char *err = NULL;
   /* texture / image ops: what nir_lower_tex can rewrite onto the ISA's tex.* set (no projector, 4-offset gather, implicit
    * LOD where the stage has no derivatives), then derefs -> descriptor handles */
   {
      const nir_lower_tex_options tex_opts = {
         .lower_txp = ~0u,
         .lower_tg4_offsets = true,
         .lower_invalid_implicit_lod = true,
      };
      NIR_PASS(_, nir, nir_lower_tex, &tex_opts);
      if (!ngv_nir_lower_tex_resources(nir, layout, &err)) {
         ralloc_free(nir);
         return ngv_shader_error(dev, VK_ERROR_UNKNOWN, "%s", err);
      }
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_uniform | nir_var_image, NULL);
   }
   if (!ngv_nir_lower_pipeline_layout(nir, layout, base_push_offset, numwg_push_offset, uses_num_workgroups, &err)) {
      ralloc_free(nir);
      return ngv_shader_error(dev, VK_ERROR_UNKNOWN, "%s", err);
   }

 NIR_PASS(_, nir, ngv_nir_lower_16bit_mem); /* : scalar 16-bit loads / stores (before the vectors are scalarized) */
   NIR_PASS(_, nir, ngv_nir_lower_subgroups, ngv_target_has_xlane(dev->pdev));
   NIR_PASS(_, nir, nir_lower_int64);
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_phis_to_scalar, ngv_phi_width, NULL);
 /* : shaderInt16 / shaderFloat16: 16-bit ALU ops are promoted to 32 bits; float16 <-> float32 become
    * pack / unpack_half_2x16_split (below).  The optimizer can create new ones (lowered bitfield / sat ops, ...), so the
    * pass runs again after it. */
   NIR_PASS(_, nir, ngv_nir_lower_16bit_alu);
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   ngv_optimize(nir);
 /* (shaderInt64): the optimizer makes new 64-bit ops from the lowered ones (isub(0, x) -> ineg,...) */
   NIR_PASS(_, nir, nir_lower_int64);
   NIR_PASS(_, nir, ngv_nir_lower_16bit_alu);
   /* after the optimizer: nir_opt_algebraic is what splits packHalf2x16 / unpackHalf2x16 (lower_pack_half_2x16).
 * a target with the VALU binary16 conversions (units "f16cvt") keeps the split ops for ngcomp
    * (pkcvt.f16.f32 / fcvt.f32.f16.lo/hi); older RMs get the integer expansion (ngv_lower_half_pack). */
   if (!ngv_target_has_f16cvt(dev->pdev))
      NIR_PASS(_, nir, ngv_lower_half_pack);
   /* constant folding re-vectorizes constants (vec3 {set, slot, 0}): scalarize last, nothing may fold afterwards */
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   *out = nir;
   return VK_SUCCESS;
}
