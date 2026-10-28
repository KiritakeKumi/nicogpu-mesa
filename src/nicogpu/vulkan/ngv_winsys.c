/* SPDX-License-Identifier: MIT */
/* ngv winsys: backend selection and error mapping. */
#include <stdlib.h>
#include <string.h>

#include "ngv_winsys.h"

const struct ngv_winsys_ops *
ngv_winsys_select(void)
{
   const char *e = getenv("NGV_WINSYS");
   const char *want = (e && *e) ? e : "auto";

   if (!strcmp(want, "stub"))
      return &ngv_winsys_stub_ops;
   if (!strcmp(want, "drm"))
      return &ngv_winsys_drm_ops;
#ifdef NGV_WITH_HAL
   if (!strcmp(want, "hal"))
      return &ngv_winsys_hal_ops;
#endif
   if (!strcmp(want, "auto")) {
      /* drm first: a real device wins over the simulator.  The stub is never
       * picked automatically. */
      uint32_t n = 0;
      if (ngv_winsys_drm_ops.enumerate(NULL, 0, &n) == NICOGPU_OK && n > 0)
         return &ngv_winsys_drm_ops;
#ifdef NGV_WITH_HAL
      return &ngv_winsys_hal_ops;
#else
      return &ngv_winsys_drm_ops; /* enumerates 0 devices: an empty instance */
#endif
   }
   return NULL;
}

/* Generic mapping for calls whose only interesting failures are resource
 * exhaustion and device loss (allocation, creation, submission). */
VkResult
ngv_winsys_result(int err)
{
   switch (err) {
   case NICOGPU_OK:
      return VK_SUCCESS;
   case NICOGPU_ERR_NOMEM:
   case NICOGPU_ERR_NOSPC:
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   case NICOGPU_ERR_CANCELED:
   case NICOGPU_ERR_NODEV:
      return VK_ERROR_DEVICE_LOST;
   case NICOGPU_ERR_TIME:
      return VK_TIMEOUT;
   case NICOGPU_ERR_NOTSUP:
      return VK_ERROR_FEATURE_NOT_PRESENT;
   case NICOGPU_ERR_BUSY:
   case NICOGPU_ERR_IO:
   default:
      return VK_ERROR_UNKNOWN;
   }
}
