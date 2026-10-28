/* SPDX-License-Identifier: MIT */
/* ngv_stub_cp.c -- software command processor of the stub winsys (batch 2).
 *
 * The stub used to refuse every command buffer.  To smoke-test the whole recording path (descriptors, push
 * constants, dispatch, copy, fill, queries, events) on a machine with neither the model nor the board, it now
 * interprets the IB packets itself (command-set spec v0.2 semantics, serial execution -- "a legal implementation",
 * section 3.5).  Shader execution is NOT an ISA interpreter: DISPATCH looks at the code object, and only the
 * placeholder blob {"NGVPHKRN", kernel id} (ngv_pipeline.c) is runnable; the built-in kernel reads the descriptors
 * through the user registers exactly as a compiled shader would (set 0 base in u2:u3, buffer descriptors 16 B,
 * push constants in u10..), so the driver's descriptor / push-constant / root-register code is really exercised.
 * Anything else (real code objects, QUERY, SET_PREDICATE, nested INDIRECT, DRAW*) stops the CP with an error, never
 * a silent success.
 *
 * Buffer access follows the descriptor spec section 3.2: reads past `size` return 0, writes past `size` are dropped.
 */
#include <string.h>

#include "nicogpu_cmd.h"
#include "ngv_private.h" /* NGV_PH_MAGIC, struct ngv_ph_header */
#include "ngv_stub_cp.h"

/* u0..u25: SET_USER_REGS may write registers 0 .. COUNT_MAX-2 (COUNT_MAX includes the CTRL dword). */
#define U_REGS NICOGPU_CMD_LIMIT_USER_UREGS
_Static_assert(U_REGS == NICOGPU_CMD_SET_USER_REGS_COUNT_MAX - 1, "user register count derives from the command spec");

struct cp_state {
   bool have_shader;
   uint64_t code_base;
   uint32_t entry, wg[3];
   uint32_t u[U_REGS];
};

static int
fail(struct ngv_stub_cp *cp, uint32_t cp_error, int err)
{
   cp->cp_error = cp_error;
   return err;
}

static int
mem_read(struct ngv_stub_cp *cp, uint64_t va, void *dst, uint64_t n)
{
   uint64_t avail = 0;
   void *p = cp->xlate(cp->ctx, va, &avail);
   if (!p || avail < n)
      return NICOGPU_ERR_IO;
   memcpy(dst, p, n);
   return 0;
}

static int
mem_write(struct ngv_stub_cp *cp, uint64_t va, const void *src, uint64_t n)
{
   uint64_t avail = 0;
   void *p = cp->xlate(cp->ctx, va, &avail);
   if (!p || avail < n)
      return NICOGPU_ERR_IO;
   memcpy(p, src, n);
   return 0;
}

struct buf_desc {
   uint64_t base;
   uint32_t size;
};

static int
read_desc(struct ngv_stub_cp *cp, uint64_t va, struct buf_desc *d)
{
   uint32_t w[4];
   int r = mem_read(cp, va, w, 16);
   if (r)
      return r;
   d->base = (uint64_t)w[0] | ((uint64_t)(w[1] & 0xffffu) << 32);
   d->size = w[2];
   return 0;
}

/* built-in kernel 1: dst[i] = src[i] * scale + bias for i < src.length, one invocation per element.
 *   set 0 binding 0 (slot 0) = src (readonly), binding 1 (slot 1) = dst, push constants {scale, bias}. */
static int
kernel_scale_bias(struct ngv_stub_cp *cp, const struct cp_state *st, uint32_t gx, uint32_t gy, uint32_t gz,
                  uint32_t base_x)
{
   if (st->wg[1] != 1 || st->wg[2] != 1 || gy != 1 || gz != 1)
      return fail(cp, NICOGPU_CMD_CP_ERROR_UNSUPPORTED, NICOGPU_ERR_NOTSUP);
   uint64_t set0 = (uint64_t)st->u[2] | ((uint64_t)(st->u[3] & 0xffffu) << 32);
   struct buf_desc src, dst;
   int r = read_desc(cp, set0, &src);
   if (!r)
      r = read_desc(cp, set0 + 16, &dst);
   if (r)
      return fail(cp, NICOGPU_CMD_CP_ERROR_MEM_FAULT, r);
   uint32_t scale = st->u[10], bias = st->u[11];
   uint64_t first = (uint64_t)base_x * st->wg[0]; /* DISPATCH_BASE: workgroup ids start at base_x (ABI: the driver passes it) */
   uint64_t total = first + (uint64_t)gx * st->wg[0];
   uint64_t len = src.size / 4;
   uint64_t avail_s = 0, avail_d = 0;
   uint32_t *ps = src.size ? cp->xlate(cp->ctx, src.base, &avail_s) : NULL;
   uint32_t *pd = dst.size ? cp->xlate(cp->ctx, dst.base, &avail_d) : NULL;
   for (uint64_t i = first; i < total && i < len; i++) {
      if (!ps || (i + 1) * 4 > avail_s)
         return fail(cp, NICOGPU_CMD_CP_ERROR_SHADER_FAULT, NICOGPU_ERR_IO);
      uint32_t v = ps[i] * scale + bias;
      if ((i + 1) * 4 <= dst.size) { /* in bounds of the descriptor: else the write is dropped */
         if (!pd || (i + 1) * 4 > avail_d)
            return fail(cp, NICOGPU_CMD_CP_ERROR_SHADER_FAULT, NICOGPU_ERR_IO);
         pd[i] = v;
      }
   }
   return 0;
}

static int
do_dispatch(struct ngv_stub_cp *cp, const struct cp_state *st, uint32_t gx, uint32_t gy, uint32_t gz)
{
   if (!st->have_shader)
      return fail(cp, NICOGPU_CMD_CP_ERROR_NO_SHADER_STATE, NICOGPU_ERR_INVAL);
   struct ngv_ph_header h;
   int r = mem_read(cp, st->code_base + st->entry, &h, sizeof(h));
   if (r)
      return fail(cp, NICOGPU_CMD_CP_ERROR_MEM_FAULT, r);
   if (memcmp(h.magic, NGV_PH_MAGIC, 8))
      return fail(cp, NICOGPU_CMD_CP_ERROR_UNSUPPORTED, NICOGPU_ERR_NOTSUP); /* a real code object: the stub is not a GPU */
   if (!gx || !gy || !gz)
      return 0;
   switch (h.kernel) {
   case NGV_PH_KERNEL_SCALE_BIAS:
      return kernel_scale_bias(cp, st, gx, gy, gz, 0);
   case NGV_PH_KERNEL_SCALE_BIAS_BASE:
      return kernel_scale_bias(cp, st, gx, gy, gz, st->u[12]);
   default:
      return fail(cp, NICOGPU_CMD_CP_ERROR_UNSUPPORTED, NICOGPU_ERR_NOTSUP);
   }
}

static uint64_t
va64(uint32_t lo, uint32_t hi)
{
   return (uint64_t)lo | ((uint64_t)hi << 32);
}

int
ngv_stub_cp_exec_ib(struct ngv_stub_cp *cp, uint64_t ib_va, uint32_t size_dw)
{
   cp->cp_error = 0;
   uint64_t avail = 0;
   const uint32_t *ib = cp->xlate(cp->ctx, ib_va, &avail);
   if (!ib || avail < (uint64_t)size_dw * 4)
      return fail(cp, NICOGPU_CMD_CP_ERROR_MEM_FAULT, NICOGPU_ERR_IO);

   struct cp_state st;
   memset(&st, 0, sizeof(st));
   for (uint32_t pos = 0; pos < size_dw;) {
      uint32_t hdr = ib[pos];
      uint32_t op = hdr & NICOGPU_CMD_HDR_OPCODE_MASK, count = hdr >> NICOGPU_CMD_HDR_COUNT_SHIFT;
      if ((hdr & NICOGPU_CMD_HDR_RSVD_MASK) || (hdr & NICOGPU_CMD_HDR_PRED_MASK))
         return fail(cp, NICOGPU_CMD_CP_ERROR_BAD_PACKET, NICOGPU_ERR_INVAL); /* no predication in the stub */
      if (pos + 1 + count > size_dw)
         return fail(cp, NICOGPU_CMD_CP_ERROR_BAD_PACKET, NICOGPU_ERR_INVAL);
      const uint32_t *p = &ib[pos + 1];
      int r = 0;
      switch (op) {
      case NICOGPU_CMD_OP_NOP:
         break;
      case NICOGPU_CMD_OP_SET_SHADER:
         if (count != NICOGPU_CMD_SET_SHADER_COUNT_MIN)
            return fail(cp, NICOGPU_CMD_CP_ERROR_BAD_PACKET, NICOGPU_ERR_INVAL);
         st.code_base = va64(p[0], p[1]);
         st.entry = p[2];
         st.wg[0] = p[8];
         st.wg[1] = p[9];
         st.wg[2] = p[10];
         st.have_shader = true;
         break;
      case NICOGPU_CMD_OP_SET_USER_REGS: {
         if (count < NICOGPU_CMD_SET_USER_REGS_COUNT_MIN)
            return fail(cp, NICOGPU_CMD_CP_ERROR_BAD_PACKET, NICOGPU_ERR_INVAL);
         uint32_t start = p[0] & NICOGPU_CMD_SET_USER_REGS_CTRL_START_MASK, k = count - 1;
         if (start + k > U_REGS)
            return fail(cp, NICOGPU_CMD_CP_ERROR_BAD_PACKET, NICOGPU_ERR_INVAL);
         memcpy(&st.u[start], &p[1], k * 4);
         break;
      }
      case NICOGPU_CMD_OP_DISPATCH:
         r = do_dispatch(cp, &st, p[0], p[1], p[2]);
         break;
      case NICOGPU_CMD_OP_DISPATCH_INDIRECT: {
         uint32_t g[3];
         r = mem_read(cp, va64(p[0], p[1]), g, sizeof(g));
         if (r)
            return fail(cp, NICOGPU_CMD_CP_ERROR_MEM_FAULT, r);
         r = do_dispatch(cp, &st, g[0], g[1], g[2]);
         break;
      }
      case NICOGPU_CMD_OP_CACHE:
         break; /* serial execution: everything is always idle and coherent */
      case NICOGPU_CMD_OP_COPY: {
         if (p[6] & NICOGPU_CMD_COPY_FLAGS_MODE_MASK)
            return fail(cp, NICOGPU_CMD_CP_ERROR_UNSUPPORTED, NICOGPU_ERR_NOTSUP);
         uint64_t src = va64(p[0], p[1]), dst = va64(p[2], p[3]), n = va64(p[4], p[5]);
         uint64_t as = 0, ad = 0;
         if (!n)
            break;
         void *ps = cp->xlate(cp->ctx, src, &as), *pd = cp->xlate(cp->ctx, dst, &ad);
         if (!ps || !pd || as < n || ad < n)
            return fail(cp, NICOGPU_CMD_CP_ERROR_MEM_FAULT, NICOGPU_ERR_IO);
         memmove(pd, ps, n);
         break;
      }
      case NICOGPU_CMD_OP_FILL: {
         uint64_t dst = va64(p[0], p[1]), n = va64(p[2], p[3]);
         uint64_t ad = 0;
         if ((dst & 3) || (n & 3))
            return fail(cp, NICOGPU_CMD_CP_ERROR_BAD_PACKET, NICOGPU_ERR_INVAL);
         if (!n)
            break;
         uint32_t *pd = cp->xlate(cp->ctx, dst, &ad);
         if (!pd || ad < n)
            return fail(cp, NICOGPU_CMD_CP_ERROR_MEM_FAULT, NICOGPU_ERR_IO);
         for (uint64_t i = 0; i < n / 4; i++)
            pd[i] = p[4];
         break;
      }
      case NICOGPU_CMD_OP_WRITE_DATA: {
         if (count < NICOGPU_CMD_WRITE_DATA_COUNT_MIN)
            return fail(cp, NICOGPU_CMD_CP_ERROR_BAD_PACKET, NICOGPU_ERR_INVAL);
         r = mem_write(cp, va64(p[0], p[1]), &p[2], (uint64_t)(count - 2) * 4);
         if (r)
            return fail(cp, NICOGPU_CMD_CP_ERROR_MEM_FAULT, r);
         break;
      }
      case NICOGPU_CMD_OP_TIMESTAMP: {
         cp->timestamp += 1000;
         uint64_t t = cp->timestamp;
         r = mem_write(cp, va64(p[0], p[1]), &t, 8);
         if (r)
            return fail(cp, NICOGPU_CMD_CP_ERROR_MEM_FAULT, r);
         break;
      }
      case NICOGPU_CMD_OP_FENCE: {
         uint64_t v = va64(p[2], p[3]);
         bool b64 = p[4] & NICOGPU_CMD_FENCE_FLAGS_BIT64_MASK;
         r = mem_write(cp, va64(p[0], p[1]), &v, b64 ? 8 : 4); /* little endian */
         if (r)
            return fail(cp, NICOGPU_CMD_CP_ERROR_MEM_FAULT, r);
         break;
      }
      case NICOGPU_CMD_OP_WAIT_MEM: {
         bool b64 = p[6] & NICOGPU_CMD_WAIT_MEM_CTRL_BIT64_MASK;
         uint32_t func = p[6] & NICOGPU_CMD_WAIT_MEM_CTRL_FUNC_MASK;
         uint64_t v = 0, ref = va64(p[2], p[3]), mask = va64(p[4], p[5]);
         r = mem_read(cp, va64(p[0], p[1]), &v, b64 ? 8 : 4);
         if (r)
            return fail(cp, NICOGPU_CMD_CP_ERROR_MEM_FAULT, r);
         v &= mask;
         ref &= mask;
         bool ok = func == NICOGPU_CMD_WAIT_FUNC_EQ   ? v == ref
                   : func == NICOGPU_CMD_WAIT_FUNC_NE ? v != ref
                   : func == NICOGPU_CMD_WAIT_FUNC_GE ? v >= ref
                   : func == NICOGPU_CMD_WAIT_FUNC_GT ? v > ref
                   : func == NICOGPU_CMD_WAIT_FUNC_LT ? v < ref
                                                   : v <= ref;
         /* Serial execution has no other agent that could satisfy it later: a wait that is not already true
          * would hang the real CP (watchdog), here it is reported. */
         if (!ok)
            return fail(cp, NICOGPU_CMD_CP_ERROR_WATCHDOG, NICOGPU_ERR_IO);
         break;
      }
      default:
         return fail(cp, NICOGPU_CMD_CP_ERROR_UNSUPPORTED, NICOGPU_ERR_NOTSUP);
      }
      if (r)
         return r;
      pos += 1 + count;
   }
   return 0;
}
