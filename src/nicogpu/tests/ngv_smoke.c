/* SPDX-License-Identifier: MIT */
/* ngv_smoke.c -- end-to-end smoke test of the ngv skeleton through the Vulkan
 * loader. Built and run by the internal build script with
 * NGV_WINSYS=stub: instance -> physical device -> device -> memory (all
 * types) -> buffer + device address -> fence -> empty submit -> wait.
 * Exit status 0 = all checks passed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "ngv_test_ext.h"

int
main(void)
{
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t n = 0;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
   EXPECT(n == 1);
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, &pd));

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   {
      /* derived from the device (stub 1024, real RM possibly fewer lanes x warps): range + power of two */
      uint32_t wgi = props.limits.maxComputeWorkGroupInvocations;
      EXPECT(wgi >= 128 && wgi <= 1024 && (wgi & (wgi - 1)) == 0);
   }
   EXPECT(props.limits.maxComputeSharedMemorySize == 65536);
   EXPECT(props.limits.minUniformBufferOffsetAlignment == 16);
   printf("device: %s, timestampPeriod %.1f ns\n", props.deviceName, props.limits.timestampPeriod);

   uint32_t qn = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
 EXPECT(qn == 2); /* universal + compute-only family, same capabilities */
   VkQueueFamilyProperties qps[2];
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qps);
   for (uint32_t f = 0; f < 2; f++)
      EXPECT((qps[f].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qps[f].queueFlags & VK_QUEUE_GRAPHICS_BIT));

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   /* stub: 3 heaps; drm with VRAM: VRAM + GTT; drm on a no-VRAM RM (rm_cp / rm_sp): one host-visible type */
   if (mp.memoryTypeCount == 1) {
      EXPECT(mp.memoryHeapCount == 1 && (mp.memoryTypes[0].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
   } else {
      EXPECT(mp.memoryTypeCount == 3 && mp.memoryHeapCount >= 2);
      EXPECT(mp.memoryTypes[0].propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      EXPECT(mp.memoryTypes[1].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      EXPECT(mp.memoryTypes[2].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   VkDevice dev;
   if (getenv("NGV_TEST_NOCP")) {
      /* device without a command processor (NGV_STUB_NO_CP=1): queue_create -> -ENODEV must read as "cannot
       * initialize", not as a lost device (batch 3) */
      VkResult r = vkCreateDevice(pd, &dci, NULL, &dev);
      EXPECT(r == VK_ERROR_INITIALIZATION_FAILED);
      vkDestroyInstance(inst, NULL);
      printf("ngv_smoke (no CP): OK\n");
      return 0;
   }
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue q;
   vkGetDeviceQueue(dev, 0, 0, &q);

   /* memory of every type; host-visible ones are mapped and written */
   for (uint32_t t = 0; t < mp.memoryTypeCount; t++) {
      VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = 1 << 20,
                                  .memoryTypeIndex = t};
      VkDeviceMemory mem;
      CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
      if (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
         void *p = NULL;
         CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &p));
         memset(p, 0xa5, 4096);
         EXPECT(((unsigned char *)p)[4095] == 0xa5);
         vkUnmapMemory(dev, mem);
      }
      if (t == 1) {
         VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
         VkBuffer buf;
         CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
         VkMemoryRequirements req;
         vkGetBufferMemoryRequirements(dev, buf, &req);
         EXPECT(req.size >= 4096 && req.alignment >= 4 && (req.memoryTypeBits & (1u << t)));
         CHECK(vkBindBufferMemory(dev, buf, mem, 0));
         vkDestroyBuffer(dev, buf, NULL);
      }
      vkFreeMemory(dev, mem, NULL);
   }

   /* fence: unsignaled -> submit -> signaled */
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   EXPECT(vkGetFenceStatus(dev, fence) == VK_NOT_READY);
   EXPECT(vkWaitForFences(dev, 1, &fence, VK_TRUE, 1000000) == VK_TIMEOUT);
   CHECK(vkQueueSubmit(q, 0, NULL, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 1000000000ull));
   CHECK(vkResetFences(dev, 1, &fence));
   EXPECT(vkGetFenceStatus(dev, fence) == VK_NOT_READY);
   vkDestroyFence(dev, fence, NULL);

   /* command pool / buffer lifecycle */
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
   vkFreeCommandBuffers(dev, pool, 1, &cb);
   vkDestroyCommandPool(dev, pool, NULL);

   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("ngv_smoke: OK\n");
   return 0;
}
