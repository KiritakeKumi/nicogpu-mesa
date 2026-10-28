/* SPDX-License-Identifier: MIT */
/* ngv_membw: shader load/store bandwidth on host (GTT) memory, the baseline.
 *
 *   ngv_membw <ngv_membw.spv> [MiB=4] [workgroups=16]
 *
 * Two host-visible storage buffers of MiB each; shaders/ngv_membw.comp runs a grid-stride loop with workgroups x 64
 * threads in three modes (copy, read, write, see the shader).  Prints the time and MB/s per mode (copy counts the
 * bytes read + written) and checks every word.  XGP_TIMEOUT_MS bounds each dispatch (default 600 s). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "ngv_test_ext.h"

#define CHK(x)                                                                                                         \
   do {                                                                                                                \
      VkResult r_ = (x);                                                                                               \
      if (r_ != VK_SUCCESS) {                                                                                          \
         fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #x, r_);                                               \
         exit(1);                                                                                                      \
      }                                                                                                                \
   } while (0)

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties mp;

static double
now(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return t.tv_sec + t.tv_nsec * 1e-9;
}

struct buf {
   VkBuffer b;
   VkDeviceMemory m;
   uint32_t *map;
};

static void
make_buf(struct buf *o, VkDeviceSize size)
{
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
   CHK(vkCreateBuffer(dev, &bci, NULL, &o->b));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, o->b, &req);
   int t = -1;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & 6) == 6) {
         t = (int)i;
         break;
      }
   if (t < 0) {
      fprintf(stderr, "no host-visible coherent memory\n");
      exit(1);
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = (uint32_t)t};
   CHK(vkAllocateMemory(dev, &mai, NULL, &o->m));
   CHK(vkBindBufferMemory(dev, o->b, o->m, 0));
   CHK(vkMapMemory(dev, o->m, 0, VK_WHOLE_SIZE, 0, (void **)&o->map));
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s <ngv_membw.spv> [MiB=4] [workgroups=16]\n", argv[0]);
      return 2;
   }
   const uint32_t mib = argc > 2 ? (uint32_t)atoi(argv[2]) : 4;
   const uint32_t groups = argc > 3 ? (uint32_t)atoi(argv[3]) : 16;
   const uint32_t nw = mib * (1u << 18), nthr = groups * 64;
   uint64_t timeout_ns = 600ull * 1000000000ull;
   const char *tm = getenv("XGP_TIMEOUT_MS");
   if (tm)
      timeout_ns = strtoull(tm, NULL, 0) * 1000000ull;

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice pd;
   VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || n < 1)
      return 1;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
   CHK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, 0, 0, &queue);
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
   VkCommandPool pool;
   CHK(vkCreateCommandPool(dev, &cpi, NULL, &pool));

   struct buf src, dst;
   make_buf(&src, (VkDeviceSize)nw * 4);
   make_buf(&dst, (VkDeviceSize)nw * 4);
   for (uint32_t i = 0; i < nw; i++)
      src.map[i] = i ^ 0x5a5a5a5au;

   VkDescriptorSetLayoutBinding b[2] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
   VkDescriptorSetLayoutCreateInfo dli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = b};
   VkDescriptorSetLayout dsl;
   CHK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl};
   VkPipelineLayout pl;
   CHK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
   VkDescriptorPoolCreateInfo dpi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
   VkDescriptorPool dp;
   CHK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
   VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                      .descriptorPool = dp,
                                      .descriptorSetCount = 1,
                                      .pSetLayouts = &dsl};
   VkDescriptorSet set;
   CHK(vkAllocateDescriptorSets(dev, &dai, &set));
   VkDescriptorBufferInfo dbi[2] = {{src.b, 0, VK_WHOLE_SIZE}, {dst.b, 0, VK_WHOLE_SIZE}};
   VkWriteDescriptorSet w[2] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[0]},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[1]}};
   vkUpdateDescriptorSets(dev, 2, w, 0, NULL);

   size_t sz;
   uint32_t *code = xt_load_spv(argv[1], &sz);
   if (!code) {
      fprintf(stderr, "cannot read %s\n", argv[1]);
      return 1;
   }
   VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = code};
   VkShaderModule sm;
   CHK(vkCreateShaderModule(dev, &smi, NULL, &sm));

   printf("ngv_membw: 2 x %u MiB host buffers, %u workgroups x 64 threads\n", mib, groups);
   static const char *mode_name[3] = {"copy", "read", "write"};
   int bad_total = 0;
   for (uint32_t mode = 0; mode < 3; mode++) {
      uint32_t spec[3] = {mode, nw, nthr};
      VkSpecializationMapEntry me[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
      VkSpecializationInfo spi = {3, me, sizeof(spec), spec};
      VkComputePipelineCreateInfo cpci = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                   .module = sm,
                   .pName = "main",
                   .pSpecializationInfo = &spi},
         .layout = pl};
      VkPipeline pipe;
      CHK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
      memset(dst.map, 0xAA, (size_t)nw * 4);
      VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = pool,
                                         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1};
      VkCommandBuffer cb;
      CHK(vkAllocateCommandBuffers(dev, &cai, &cb));
      VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CHK(vkBeginCommandBuffer(cb, &bi));
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
      vkCmdDispatch(cb, groups, 1, 1);
      VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                            .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL,
                           0, NULL);
      CHK(vkEndCommandBuffer(cb));
      VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      VkFence fence;
      CHK(vkCreateFence(dev, &fci, NULL, &fence));
      VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
      double t0 = now();
      CHK(vkQueueSubmit(queue, 1, &si, fence));
      VkResult r = vkWaitForFences(dev, 1, &fence, VK_TRUE, timeout_ns);
      double dt = now() - t0;
      if (r != VK_SUCCESS) {
         printf("  %-5s wait failed (%d) after %.3f s\n", mode_name[mode], r, dt);
         return 1;
      }
      uint32_t bad = 0, first = 0;
      if (mode == 0) {
         for (uint32_t i = 0; i < nw; i++)
            if (dst.map[i] != src.map[i] && !bad++)
               first = i;
      } else if (mode == 1) {
         for (uint32_t g = 0; g < nthr; g++) {
            uint32_t acc = 0;
            for (uint32_t i = g; i < nw; i += nthr)
               acc += src.map[i];
            if (dst.map[g] != acc && !bad++)
               first = g;
         }
      } else {
         for (uint32_t i = 0; i < nw; i++)
            if (dst.map[i] != i * 2654435761u && !bad++)
               first = i;
      }
      double bytes = (double)nw * 4 * (mode == 0 ? 2 : 1);
      printf("  %-5s %8.3f s  %9.3f MB/s  %s", mode_name[mode], dt, bytes / dt * 1e-6, bad ? "MISMATCH" : "ok");
      if (bad)
         printf(" (%u words, first [%u] = 0x%08x)", bad, first, dst.map[first]);
      printf("\n");
      bad_total += bad != 0;
      vkDestroyFence(dev, fence, NULL);
      vkFreeCommandBuffers(dev, pool, 1, &cb);
      vkDestroyPipeline(dev, pipe, NULL);
   }
   printf("ngv_membw: %s\n", bad_total ? "FAIL" : "OK");
   return bad_total ? 1 : 0;
}
