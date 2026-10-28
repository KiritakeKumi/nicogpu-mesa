/* SPDX-License-Identifier: MIT */
/* ngv_cmd_stream.h -- encoders for the compute subset of the CP command
 * stream (command-set spec v0.2), built on the generated nicogpu_cmd.h.
 * the encoders used by ngv_cmd_buffer.c (batch 2); this was a
 * skeleton in batch 1.  Every encoder appends to a dword array and returns
 * the new length; the caller guarantees space (SET_SHADER = 12 dwords,
 * SET_USER_REGS = 2 + n, DISPATCH = 4, CACHE = 2, FENCE = 6). */
#ifndef NGV_CMD_STREAM_H
#define NGV_CMD_STREAM_H

#include <stdbool.h>
#include <stdint.h>

#include "nicogpu_cmd.h"

static inline uint32_t
ngv_emit_set_shader(uint32_t *dw, uint64_t code_base, uint32_t entry, uint32_t vregs, uint32_t shared_bytes,
                    uint32_t scratch_bytes, uint64_t scratch_va, uint32_t wg_x, uint32_t wg_y, uint32_t wg_z)
{
   uint32_t n = 0;
   dw[n++] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_SET_SHADER, NICOGPU_CMD_SET_SHADER_COUNT_MIN);
   dw[n++] = (uint32_t)code_base;       /* 4 KiB aligned */
   dw[n++] = (uint32_t)(code_base >> 32) & NICOGPU_CMD_SET_SHADER_CODE_BASE_HI_VA_MASK;
   dw[n++] = entry;                     /* 16 B aligned, relative to code_base */
   dw[n++] = vregs & NICOGPU_CMD_SET_SHADER_RSRC_VREGS_MASK;
   dw[n++] = shared_bytes;
   dw[n++] = scratch_bytes;
   dw[n++] = (uint32_t)scratch_va;      /* 256 B aligned */
   dw[n++] = (uint32_t)(scratch_va >> 32) & NICOGPU_CMD_SET_SHADER_SCRATCH_VA_HI_VA_MASK;
   dw[n++] = wg_x;
   dw[n++] = wg_y;
   dw[n++] = wg_z;
   return n;
}

/* u[start .. start+count-1] = values; start+count <= 26 (ABI v0.1 section 2). */
static inline uint32_t
ngv_emit_set_user_regs(uint32_t *dw, uint32_t start, const uint32_t *values, uint32_t count)
{
   uint32_t n = 0;
   dw[n++] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_SET_USER_REGS, 1 + count);
   dw[n++] = start & NICOGPU_CMD_SET_USER_REGS_CTRL_START_MASK;
   for (uint32_t i = 0; i < count; i++)
      dw[n++] = values[i];
   return n;
}

static inline uint32_t
ngv_emit_dispatch(uint32_t *dw, uint32_t x, uint32_t y, uint32_t z)
{
   dw[0] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_DISPATCH, NICOGPU_CMD_DISPATCH_COUNT_MIN);
   dw[1] = x;
   dw[2] = y;
   dw[3] = z;
   return 4;
}

static inline uint32_t
ngv_emit_cache(uint32_t *dw, uint32_t flags)
{
   dw[0] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_CACHE, NICOGPU_CMD_CACHE_COUNT_MIN);
   dw[1] = flags;
   return 2;
}

/* In-IB fence (no IRQ): queries / events / user-space timeline values. */
static inline uint32_t
ngv_emit_fence64(uint32_t *dw, uint64_t va, uint64_t value)
{
   dw[0] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_FENCE, NICOGPU_CMD_FENCE_COUNT_MIN);
   dw[1] = (uint32_t)va;
   dw[2] = (uint32_t)(va >> 32);
   dw[3] = (uint32_t)value;
   dw[4] = (uint32_t)(value >> 32);
   dw[5] = NICOGPU_CMD_FENCE_FLAGS_BIT64_MASK;
   return 6;
}

static inline uint32_t
ngv_emit_dispatch_indirect(uint32_t *dw, uint64_t args_va)
{
   dw[0] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_DISPATCH_INDIRECT, NICOGPU_CMD_DISPATCH_INDIRECT_COUNT_MIN);
   dw[1] = (uint32_t)args_va;
   dw[2] = (uint32_t)(args_va >> 32);
   return 3;
}

/* End-of-pipe 64-bit device timestamp to va (8 B aligned). */
static inline uint32_t
ngv_emit_timestamp(uint32_t *dw, uint64_t va)
{
   dw[0] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_TIMESTAMP, NICOGPU_CMD_TIMESTAMP_COUNT_MIN);
   dw[1] = (uint32_t)va;
   dw[2] = (uint32_t)(va >> 32);
   return 3;
}

/* LINEAR copy, byte granular, 8 dwords. */
static inline uint32_t
ngv_emit_copy(uint32_t *dw, uint64_t src, uint64_t dst, uint64_t size)
{
   dw[0] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_COPY, NICOGPU_CMD_COPY_COUNT_MIN);
   dw[1] = (uint32_t)src;
   dw[2] = (uint32_t)(src >> 32);
   dw[3] = (uint32_t)dst;
   dw[4] = (uint32_t)(dst >> 32);
   dw[5] = (uint32_t)size;
   dw[6] = (uint32_t)(size >> 32);
   dw[7] = 0;
   return 8;
}

/* dst and size 4 B aligned. */
static inline uint32_t
ngv_emit_fill(uint32_t *dw, uint64_t dst, uint64_t size, uint32_t value)
{
   dw[0] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_FILL, NICOGPU_CMD_FILL_COUNT_MIN);
   dw[1] = (uint32_t)dst;
   dw[2] = (uint32_t)(dst >> 32);
   dw[3] = (uint32_t)size;
   dw[4] = (uint32_t)(size >> 32);
   dw[5] = value;
   return 6;
}

/* 2 + k dwords; va 4 B aligned, k >= 1. */
static inline uint32_t
ngv_emit_write_data(uint32_t *dw, uint64_t va, const uint32_t *data, uint32_t k)
{
   dw[0] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_WRITE_DATA, 2 + k);
   dw[1] = (uint32_t)va;
   dw[2] = (uint32_t)(va >> 32);
   for (uint32_t i = 0; i < k; i++)
      dw[3 + i] = data[i];
   return 3 + k;
}

/* Wait until (*va & mask) FUNC (ref & mask); 32- or 64-bit. */
static inline uint32_t
ngv_emit_wait_mem(uint32_t *dw, uint64_t va, uint64_t ref, uint64_t mask, uint32_t func, bool bit64)
{
   dw[0] = NICOGPU_CMD_HEADER(NICOGPU_CMD_OP_WAIT_MEM, NICOGPU_CMD_WAIT_MEM_COUNT_MIN);
   dw[1] = (uint32_t)va;
   dw[2] = (uint32_t)(va >> 32);
   dw[3] = (uint32_t)ref;
   dw[4] = bit64 ? (uint32_t)(ref >> 32) : 0;
   dw[5] = (uint32_t)mask;
   dw[6] = bit64 ? (uint32_t)(mask >> 32) : 0;
   dw[7] = (func & NICOGPU_CMD_WAIT_MEM_CTRL_FUNC_MASK) | (bit64 ? NICOGPU_CMD_WAIT_MEM_CTRL_BIT64_MASK : 0);
   return 8;
}

#endif
