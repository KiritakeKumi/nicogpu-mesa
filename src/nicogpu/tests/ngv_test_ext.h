/* SPDX-License-Identifier: MIT */
/* ngv_test_ext.h -- shared helper of the ngv tests: the 1.2 / 1.3 functionality the tests use, through the extensions.
 *
 * The device is Vulkan 1.1 (NGV_API_VERSION); timeline semaphores, synchronization2 and host query reset are
 * exposed as VK_KHR_timeline_semaphore / VK_KHR_synchronization2 / VK_EXT_host_query_reset.  A 1.1 application must
 * enable them by name, chain the per-extension feature struct (not VkPhysicalDeviceVulkan12/13Features) and fetch the
 * entry points with vkGetDeviceProcAddr.
 *
 *   struct xt_dev x;
 *   if (xt_require(pd, XT_TIMELINE | XT_SYNC2))       -- fails with a message if the device does not advertise them
 *      return 1;
 *   xt_dci(&dci, &x, XT_TIMELINE | XT_SYNC2);         -- extension names + feature chain into the VkDeviceCreateInfo
 *   vkCreateDevice(...);  xt_load(dev);               -- then xt.QueueSubmit2(...), xt.WaitSemaphores(...), ... */
#ifndef NGV_TEST_EXT_H
#define NGV_TEST_EXT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

/* ---- helpers shared by the ngv tests (everything here is static inline / macros: unused ones cost nothing) ---- */

/* The usual test macros: a failed call / expectation prints FAIL file:line and makes the enclosing int function
 * return 1.  A test that needs a different failure action (ngv_shaders: CASE_FAIL, ngv_api_fixes: counting EXPECT)
 * defines XT_NO_CHECK_MACROS before including this header and keeps its own. */
#ifndef XT_NO_CHECK_MACROS
#define CHECK(x)                                                              \
   do {                                                                       \
      VkResult r_ = (x);                                                      \
      if (r_ != VK_SUCCESS) {                                                 \
         fprintf(stderr, "FAIL %s:%d %s = %d\n", __FILE__, __LINE__, #x, r_); \
         return 1;                                                            \
      }                                                                       \
   } while (0)
#define EXPECT(c)                                                    \
   do {                                                              \
      if (!(c)) {                                                    \
         fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
         return 1;                                                   \
      }                                                              \
   } while (0)
#endif

/* Whole file (a SPIR-V module) into a malloc'ed buffer; NULL if missing / short read.  *sz = size in bytes. */
static inline uint32_t *
xt_load_spv(const char *p, size_t *sz)
{
   FILE *f = fopen(p, "rb");
   if (!f)
      return NULL;
   fseek(f, 0, SEEK_END);
   *sz = (size_t)ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *d = malloc(*sz);
   if (fread(d, 1, *sz, f) != *sz) {
      free(d);
      d = NULL;
   }
   fclose(f);
   return d;
}

/* First memory type allowed by `bits` that has every `want` flag and none of the `avoid` flags, or -1. */
static inline int
xt_mem_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits, VkMemoryPropertyFlags want,
            VkMemoryPropertyFlags avoid)
{
   for (uint32_t t = 0; t < mp->memoryTypeCount; t++)
      if ((bits & (1u << t)) && (mp->memoryTypes[t].propertyFlags & want) == want &&
          !(mp->memoryTypes[t].propertyFlags & avoid))
         return (int)t;
   return -1;
}

enum { XT_TIMELINE = 1, XT_SYNC2 = 2, XT_HQR = 4 };

struct xt_dev {
   VkPhysicalDeviceTimelineSemaphoreFeaturesKHR ts;
   VkPhysicalDeviceSynchronization2FeaturesKHR s2;
   VkPhysicalDeviceHostQueryResetFeaturesEXT hqr;
   const char *names[3];
};

struct xt_fns {
   PFN_vkCmdPipelineBarrier2KHR CmdPipelineBarrier2;
   PFN_vkCmdWriteTimestamp2KHR CmdWriteTimestamp2;
   PFN_vkCmdSetEvent2KHR CmdSetEvent2;
   PFN_vkCmdWaitEvents2KHR CmdWaitEvents2;
   PFN_vkQueueSubmit2KHR QueueSubmit2;
   PFN_vkSignalSemaphoreKHR SignalSemaphore;
   PFN_vkGetSemaphoreCounterValueKHR GetSemaphoreCounterValue;
   PFN_vkWaitSemaphoresKHR WaitSemaphores;
   PFN_vkResetQueryPoolEXT ResetQueryPool;
};
static struct xt_fns xt __attribute__((unused));

static inline const char *
xt_name(unsigned bit)
{
   return bit == XT_TIMELINE ? VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
          : bit == XT_SYNC2  ? VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME
                             : VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME;
}

/* 0 = every extension of the mask is advertised; else a message on stderr and -1 */
static inline int
xt_require(VkPhysicalDevice pd, unsigned mask)
{
   uint32_t n = 0;
   vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
   VkExtensionProperties *p = calloc(n ? n : 1, sizeof(*p));
   if (!p)
      return -1;
   vkEnumerateDeviceExtensionProperties(pd, NULL, &n, p);
   int rc = 0;
   for (unsigned bit = 1; bit <= XT_HQR; bit <<= 1) {
      if (!(mask & bit))
         continue;
      int found = 0;
      for (uint32_t i = 0; i < n; i++)
         found |= !strcmp(p[i].extensionName, xt_name(bit));
      if (!found) {
         fprintf(stderr, "FAIL device extension %s is not advertised (the device is Vulkan 1.1)\n", xt_name(bit));
         rc = -1;
      }
   }
   free(p);
   return rc;
}

/* extension names + feature structs (all TRUE) of the mask into a VkDeviceCreateInfo; x must outlive vkCreateDevice */
static inline void
xt_dci(VkDeviceCreateInfo *dci, struct xt_dev *x, unsigned mask)
{
   memset(x, 0, sizeof(*x));
   x->ts.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
   x->ts.timelineSemaphore = VK_TRUE;
   x->s2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
   x->s2.synchronization2 = VK_TRUE;
   x->hqr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT;
   x->hqr.hostQueryReset = VK_TRUE;
   void *chain = NULL;
   uint32_t n = 0;
   if (mask & XT_TIMELINE) {
      x->ts.pNext = chain, chain = &x->ts, x->names[n++] = xt_name(XT_TIMELINE);
   }
   if (mask & XT_SYNC2) {
      x->s2.pNext = chain, chain = &x->s2, x->names[n++] = xt_name(XT_SYNC2);
   }
   if (mask & XT_HQR) {
      x->hqr.pNext = chain, chain = &x->hqr, x->names[n++] = xt_name(XT_HQR);
   }
   dci->pNext = chain;
   dci->enabledExtensionCount = n;
   dci->ppEnabledExtensionNames = x->names;
}

/* the entry points of the enabled extensions (an extension that is not enabled leaves its pointers NULL) */
static inline void
xt_load(VkDevice d)
{
#define XT_LOAD(f, name) xt.f = (PFN_vk##name)vkGetDeviceProcAddr(d, "vk" #name)
   XT_LOAD(CmdPipelineBarrier2, CmdPipelineBarrier2KHR);
   XT_LOAD(CmdWriteTimestamp2, CmdWriteTimestamp2KHR);
   XT_LOAD(CmdSetEvent2, CmdSetEvent2KHR);
   XT_LOAD(CmdWaitEvents2, CmdWaitEvents2KHR);
   XT_LOAD(QueueSubmit2, QueueSubmit2KHR);
   XT_LOAD(SignalSemaphore, SignalSemaphoreKHR);
   XT_LOAD(GetSemaphoreCounterValue, GetSemaphoreCounterValueKHR);
   XT_LOAD(WaitSemaphores, WaitSemaphoresKHR);
   XT_LOAD(ResetQueryPool, ResetQueryPoolEXT);
#undef XT_LOAD
}

#endif
