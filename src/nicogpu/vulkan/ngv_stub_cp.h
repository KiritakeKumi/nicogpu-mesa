/* SPDX-License-Identifier: MIT */
/* ngv_stub_cp.h -- software CP of the stub winsys (batch 2).  Internal to ngv_winsys_stub.c. */
#ifndef NGV_STUB_CP_H
#define NGV_STUB_CP_H

#include <stdint.h>

/* Translate [va, va+1) of the job's VM to a host pointer; *avail = bytes from there to the end of the mapping.
 * NULL = no mapping (the CP would raise MEM_FAULT). */
typedef void *(*ngv_stub_xlate_fn)(void *ctx, uint64_t va, uint64_t *avail);

struct ngv_stub_cp {
   ngv_stub_xlate_fn xlate;
   void *ctx;
   uint64_t timestamp; /* device clock, advanced on every TIMESTAMP */
   uint32_t cp_error;  /* NICOGPU_CMD_CP_ERROR_* of the last failed IB, 0 if none */
};

/* Executes one IB (what the kernel's INDIRECT would): returns 0, or a negative NICOGPU_ERR_* with cp_error set.
 * The CP state (shader, user registers) starts empty, like for every INDIRECT. */
int ngv_stub_cp_exec_ib(struct ngv_stub_cp *cp, uint64_t ib_va, uint32_t size_dw);

#endif
