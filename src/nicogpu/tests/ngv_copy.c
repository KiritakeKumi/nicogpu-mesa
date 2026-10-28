/* SPDX-License-Identifier: MIT */
/* ngv_copy.c -- repro of the vkpeak "copy-h2d" path (: segfault, cause unknown).
 * Covers what ncnn/vkpeak does: every memory type allocated and mapped, host-visible staging buffers,
 * vkCmdCopyBuffer host -> "device" -> host for 4 B .. 64 MiB, plus the failure paths (allocation larger than
 * the device can give must return an error, never a NULL map / crash).
 * Build: gcc -O1 -o ngv_copy ngv_copy.c -lvulkan ; run with NGV_WINSYS=hal|stub|drm (VK_ICD_FILENAMES=ngv icd).
 * NGV_COPY_MAX=<bytes> caps the largest copy (default 64 MiB).  Exit status 0 = all checks passed. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "ngv_test_ext.h"

struct buf {
   VkBuffer b;
   VkDeviceMemory m;
   void *map;
   VkDeviceSize size;
};

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties mp;

/* ncnn-style type pick: required flags, then must-not-have flags */
static int
find_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid)
{
   return xt_mem_type(&mp, bits, want, avoid);
}

/* returns VkResult of the first failing step; on failure everything is released */
static VkResult
buf_create(struct buf *o, VkDeviceSize size, VkBufferUsageFlags usage, int type, int map)
{
   memset(o, 0, sizeof(*o));
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
   VkResult r = vkCreateBuffer(dev, &bci, NULL, &o->b);
   if (r != VK_SUCCESS)
      return r;
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, o->b, &req);
   if (type < 0 || !(req.memoryTypeBits & (1u << type))) {
      vkDestroyBuffer(dev, o->b, NULL);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                               .memoryTypeIndex = (uint32_t)type};
   r = vkAllocateMemory(dev, &mai, NULL, &o->m);
   if (r != VK_SUCCESS) {
      vkDestroyBuffer(dev, o->b, NULL);
      return r;
   }
   r = vkBindBufferMemory(dev, o->b, o->m, 0);
   if (r == VK_SUCCESS && map)
      r = vkMapMemory(dev, o->m, 0, VK_WHOLE_SIZE, 0, &o->map);
   if (r == VK_SUCCESS && map && !o->map)
      r = VK_ERROR_MEMORY_MAP_FAILED; /* "success" with a NULL pointer: the vkpeak crash if it ever happens */
   if (r != VK_SUCCESS) {
      vkFreeMemory(dev, o->m, NULL);
      vkDestroyBuffer(dev, o->b, NULL);
      return r;
   }
   o->size = size;
   return VK_SUCCESS;
}

static void
buf_destroy(struct buf *o)
{
   if (o->map)
      vkUnmapMemory(dev, o->m);
   vkDestroyBuffer(dev, o->b, NULL);
   vkFreeMemory(dev, o->m, NULL);
}

static int
submit_wait(VkQueue q, VkCommandBuffer cb, VkFence f)
{
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
   CHECK(vkQueueSubmit(q, 1, &si, f));
   CHECK(vkWaitForFences(dev, 1, &f, VK_TRUE, 60ull * 1000000000ull));
   CHECK(vkResetFences(dev, 1, &f));
   return 0;
}

int
main(void)
{
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice pd;
   VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
   EXPECT((er == VK_SUCCESS || er == VK_INCOMPLETE) && n >= 1);

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   printf("device %s: maxMemoryAllocationCount %u, maxStorageBufferRange %u, nonCoherentAtomSize %llu\n",
          props.deviceName, props.limits.maxMemoryAllocationCount, props.limits.maxStorageBufferRange,
          (unsigned long long)props.limits.nonCoherentAtomSize);
   VkPhysicalDeviceMaintenance3Properties m3 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
   VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m3};
   vkGetPhysicalDeviceProperties2(pd, &p2);
   printf("maxMemoryAllocationSize %llu\n", (unsigned long long)m3.maxMemoryAllocationSize);
   for (uint32_t t = 0; t < mp.memoryTypeCount; t++)
      printf("type %u flags 0x%x heap %u (size %llu)\n", t, mp.memoryTypes[t].propertyFlags, mp.memoryTypes[t].heapIndex,
             (unsigned long long)mp.memoryHeaps[mp.memoryTypes[t].heapIndex].size);

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
                                  .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue q;
   vkGetDeviceQueue(dev, 0, 0, &q);

   /* 1. every host-visible type: allocate, map twice (same pointer / offset), write, flush, invalidate, read */
   for (uint32_t t = 0; t < mp.memoryTypeCount; t++) {
      VkMemoryPropertyFlags f = mp.memoryTypes[t].propertyFlags;
      VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = 1 << 20,
                                  .memoryTypeIndex = t};
      VkDeviceMemory mem;
      CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
      if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
         unsigned char *p = NULL, *p2_ = NULL;
         CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&p));
         EXPECT(p != NULL);
         memset(p, 0x5a, 8192);
         VkMappedMemoryRange rg = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = mem, .offset = 0,
                                   .size = VK_WHOLE_SIZE};
         CHECK(vkFlushMappedMemoryRanges(dev, 1, &rg));
         CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &rg));
         EXPECT(p[0] == 0x5a && p[4095] == 0x5a);
         vkUnmapMemory(dev, mem);
         CHECK(vkMapMemory(dev, mem, 4096, 4096, 0, (void **)&p2_));
         EXPECT(p2_ != NULL && p2_[0] == 0x5a);
         vkUnmapMemory(dev, mem);
      }
      vkFreeMemory(dev, mem, NULL);
      printf("type %u: alloc/map OK\n", t);
   }

   /* 2. failure paths: an oversize allocation must fail cleanly (vkpeak/ncnn deref the NULL it would get back) */
   {
      int ht = find_type(~0u, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
      EXPECT(ht >= 0);
      VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                  .allocationSize = m3.maxMemoryAllocationSize, .memoryTypeIndex = (uint32_t)ht};
      VkDeviceMemory mem = VK_NULL_HANDLE;
      VkResult r = vkAllocateMemory(dev, &mai, NULL, &mem);
      printf("alloc maxMemoryAllocationSize (%llu): %d\n", (unsigned long long)mai.allocationSize, r);
      if (r == VK_SUCCESS) {
         void *p = NULL;
         CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &p));
         EXPECT(p != NULL);
         vkFreeMemory(dev, mem, NULL);
      } else {
         /* limit says it fits but the device refused: report, this is the suspected vkpeak trigger */
         fprintf(stderr, "NOTE: maxMemoryAllocationSize is not allocatable (r=%d): limit is too optimistic\n", r);
      }
   }

   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = 0};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));

   VkDeviceSize maxcopy = 64ull << 20;
   if (getenv("NGV_COPY_MAX"))
      maxcopy = strtoull(getenv("NGV_COPY_MAX"), NULL, 0);
   if (maxcopy > m3.maxMemoryAllocationSize)
      maxcopy = m3.maxMemoryAllocationSize;

   /* "device" side = a type without HOST_CACHED preference like ncnn's blob allocator (device local first),
    * staging side = host visible|coherent, cached if offered */
   int stage_t = find_type(~0u, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           0);
   int cached_t = find_type(~0u, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
   if (cached_t >= 0)
      stage_t = cached_t;
   int dev_t = find_type(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (dev_t < 0)
      dev_t = find_type(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); /* no-VRAM RM: the single host type */
   if (dev_t < 0)
      dev_t = stage_t;
   EXPECT(stage_t >= 0 && dev_t >= 0);
   printf("staging type %d, device type %d, max copy %llu\n", stage_t, dev_t, (unsigned long long)maxcopy);

   const VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
   int rc = 0;
   for (VkDeviceSize sz = 4; sz <= maxcopy; sz *= 4) {
      struct buf up = {0}, dv = {0}, down = {0};
      VkResult r = buf_create(&up, sz, usage, stage_t, 1);
      if (r == VK_SUCCESS)
         r = buf_create(&dv, sz, usage, dev_t, 0);
      if (r == VK_SUCCESS) {
         r = buf_create(&down, sz, usage, stage_t, 1);
         if (r != VK_SUCCESS)
            buf_destroy(&dv);
      }
      if (r != VK_SUCCESS) {
         if (up.size)
            buf_destroy(&up);
         fprintf(stderr, "size %llu: create failed (%d) -- device cannot hold this (not a crash)\n",
                 (unsigned long long)sz, r);
         rc = 2; /* soft failure: reported, but keep probing smaller/other paths already done */
         break;
      }
      uint32_t *w = up.map, *rd = down.map;
      VkDeviceSize nw = sz / 4;
      for (VkDeviceSize i = 0; i < nw; i++) {
         w[i] = (uint32_t)(i * 2654435761u) ^ 0xa5a5a5a5u;
         rd[i] = 0xdeadbeefu;
      }
      CHECK(vkResetCommandBuffer(cb, 0));
      VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      CHECK(vkBeginCommandBuffer(cb, &bi));
      VkBufferCopy rg = {.srcOffset = 0, .dstOffset = 0, .size = sz};
      vkCmdCopyBuffer(cb, up.b, dv.b, 1, &rg);
      VkBufferMemoryBarrier bar = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                   .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                   .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .buffer = dv.b, .size = VK_WHOLE_SIZE};
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 1, &bar, 0,
                           NULL);
      vkCmdCopyBuffer(cb, dv.b, down.b, 1, &rg);
      CHECK(vkEndCommandBuffer(cb));
      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      if (submit_wait(q, cb, fence))
         return 1;
      clock_gettime(CLOCK_MONOTONIC, &t1);
      const double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
      VkDeviceSize bad = 0;
      for (VkDeviceSize i = 0; i < nw; i++) {
         uint32_t want = (uint32_t)(i * 2654435761u) ^ 0xa5a5a5a5u;
         if (rd[i] != want) {
            if (bad < 4 || bad % 16384 == 0)
               fprintf(stderr, "  byte 0x%llx: got %08x want %08x (staging-in %08x)\n", (unsigned long long)i * 4,
                       rd[i], want, w[i]);
            bad++;
         }
      }
      printf("copy %10llu B round trip: %s  %.3f s, %.2f MB/s", (unsigned long long)sz, bad ? "MISMATCH" : "ok", dt,
             2.0 * sz / dt * 1e-6);
      if (bad)
         printf(" (%llu of %llu words)", (unsigned long long)bad, (unsigned long long)nw);
      printf("\n");
      EXPECT(bad == 0);
      buf_destroy(&down);
      buf_destroy(&dv);
      buf_destroy(&up);
   }

   /* 3. sub-range copies with non-8 offsets (4 B aligned), two regions in one call, FillBuffer, UpdateBuffer */
   {
      struct buf a, b;
      CHECK(buf_create(&a, 4096, usage, stage_t, 1));
      CHECK(buf_create(&b, 4096, usage, stage_t, 1));
      uint32_t *pa = a.map, *pb = b.map;
      for (int i = 0; i < 1024; i++) {
         pa[i] = 0x1000u + i;
         pb[i] = 0;
      }
      VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      CHECK(vkResetCommandBuffer(cb, 0));
      CHECK(vkBeginCommandBuffer(cb, &bi));
      VkBufferCopy rgs[2] = {{.srcOffset = 4, .dstOffset = 12, .size = 20}, {.srcOffset = 100, .dstOffset = 200, .size = 36}};
      vkCmdCopyBuffer(cb, a.b, b.b, 2, rgs);
      vkCmdFillBuffer(cb, b.b, 1024, 64, 0xcafef00du);
      uint32_t upd[3] = {1, 2, 3};
      vkCmdUpdateBuffer(cb, b.b, 2048, sizeof(upd), upd);
      vkCmdFillBuffer(cb, b.b, 3000, VK_WHOLE_SIZE, 7); /* WHOLE_SIZE from an unaligned-to-16 offset */
      CHECK(vkEndCommandBuffer(cb));
      if (submit_wait(q, cb, fence))
         return 1;
      EXPECT(pb[3] == 0x1001 && pb[7] == 0x1005 && pb[8] == 0);
      EXPECT(pb[50] == 0x1000u + 25 && pb[58] == 0x1000u + 33);
      EXPECT(pb[256] == 0xcafef00du && pb[271] == 0xcafef00du && pb[272] == 0);
      EXPECT(pb[512] == 1 && pb[514] == 3);
      EXPECT(pb[750] == 7 && pb[1023] == 7);
      buf_destroy(&a);
      buf_destroy(&b);
      printf("sub-range copy/fill/update: ok\n");
   }

   /* 4. many small allocations (ncnn's block allocator makes lots of them) */
   {
      enum { N = 256 };
      static VkDeviceMemory mems[N];
      uint32_t got = 0;
      for (; got < N; got++) {
         VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = 64 << 10,
                                     .memoryTypeIndex = (uint32_t)stage_t};
         if (vkAllocateMemory(dev, &mai, NULL, &mems[got]) != VK_SUCCESS)
            break;
      }
      printf("%u x 64 KiB allocations ok\n", got);
      EXPECT(got > 0);
      for (uint32_t i = 0; i < got; i++)
         vkFreeMemory(dev, mems[i], NULL);
   }

   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("ngv_copy: %s\n", rc ? "PARTIAL (see stderr)" : "OK");
   return rc;
}
