/* SPDX-License-Identifier: MIT */
/* ngv_peak: minimal vkpeak-style throughput probe (fp32/int32 mad chains, after nihui/vkpeak, MIT).
 *
 *   ngv_peak <dir with xgp_{fp32,int32}_{p1,p4}.spv> [repeat] [local_size_x]
 *
 * repeat / local_size_x must match what gen_shaders.py was run with (defaults 16 / 32; a warp is 32 wide, so use a multiple of 32).  The `loop` specialization
 * constant is swept (XGP_LOOPS=a,b,c overrides); the dispatch is XGP_GROUPS workgroups (default 4); XGP_TIMEOUT_MS
 * bounds each dispatch.  Prints GFLOP/s or GIOP/s (2 ops per mad) and checks the results against a host reference
 * (int32 exact, fp32 either mul+add or fma rounding). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "../ngv_test_ext.h"

static VkDevice dev;
static VkQueue queue;
static VkPhysicalDeviceMemoryProperties mp;
static VkCommandPool pool;
static uint64_t timeout_ns = 600ull * 1000000000ull;

#define CHK(x)                                                                                                         \
   do {                                                                                                                \
      VkResult r_ = (x);                                                                                               \
      if (r_ != VK_SUCCESS) {                                                                                          \
         fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #x, r_);                                               \
         exit(1);                                                                                                      \
      }                                                                                                                \
   } while (0)

static double
now(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return t.tv_sec + t.tv_nsec * 1e-9;
}

/* LocalSize x from the module's OpExecutionMode (opcode 16, mode 17), 0 if absent */
static uint32_t
spv_local_size_x(const uint32_t *d, size_t sz)
{
   for (size_t i = 5; i < sz / 4;) {
      uint32_t wc = d[i] >> 16;
      if (!wc)
         break;
      if ((d[i] & 0xffff) == 16 && wc >= 4 && d[i + 2] == 17)
         return d[i + 3];
      i += wc;
   }
   return 0;
}

static float
ref_f(uint32_t gx, uint32_t lx, int vec4, int loop, int rep, int fused)
{
   const float da[4] = {0, 1, 2, -3}, db[4] = {2, 3, 5, -7};
   float cs[4];
   int n = vec4 ? 4 : 1;
   for (int k = 0; k < n; k++) {
      float a = vec4 ? (float)gx + da[k] : (float)gx;
      float b = vec4 ? (float)lx + db[k] : (float)lx;
      float c = (float)gx;
      for (long i = 0; i < (long)loop * rep; i++)
         c = fused ? fmaf(a, c, b) : a * c + b;
      cs[k] = c;
   }
   return vec4 ? (cs[0] + cs[1]) + (cs[2] + cs[3]) : cs[0];
}

static int32_t
ref_i(uint32_t gx, uint32_t lx, int vec4, int loop, int rep)
{
   const int32_t da[4] = {0, 1, 2, -3}, db[4] = {2, 3, 5, -7};
   uint32_t res[4];
   int n = vec4 ? 4 : 1;
   for (int k = 0; k < n; k++) {
      uint32_t a = (uint32_t)(vec4 ? (int32_t)gx + da[k] : (int32_t)gx);
      uint32_t b = (uint32_t)(vec4 ? (int32_t)lx + db[k] : (int32_t)lx);
      uint32_t c = gx;
      for (long i = 0; i < (long)loop * rep; i++)
         c = a * c + b;
      res[k] = c;
   }
   if (!vec4)
      return (int32_t)res[0];
   return (int32_t)((res[0] + res[1]) + (res[2] + res[3]));
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s <spvdir> [repeat] [local_size_x]\n", argv[0]);
      return 2;
   }
   int rep = argc > 2 ? atoi(argv[2]) : 16;
   int lsx = argc > 3 ? atoi(argv[3]) : 32;
   const char *g = getenv("XGP_GROUPS");
   int groups = g ? atoi(g) : 4;
   const char *tm = getenv("XGP_TIMEOUT_MS");
   if (tm)
      timeout_ns = strtoull(tm, NULL, 0) * 1000000ull;
   int loops[32], nl = 0;
   const char *ls = getenv("XGP_LOOPS");
   if (!ls)
      ls = "1,2,4,8,16,32,64";
   for (char *s = strdup(ls), *t = strtok(s, ","); t && nl < 32; t = strtok(NULL, ","))
      loops[nl++] = atoi(t);

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice pd;
   VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || n < 1)
      return 1;
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s, maxWorkGroupInvocations %u, maxWorkGroupSize.x %u\n", props.deviceName,
          props.limits.maxComputeWorkGroupInvocations, props.limits.maxComputeWorkGroupSize[0]);
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
   CHK(vkCreateDevice(pd, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
   CHK(vkCreateCommandPool(dev, &cpi, NULL, &pool));

   uint32_t ninv = (uint32_t)groups * lsx;
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = ninv * 4,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
   VkBuffer buf;
   CHK(vkCreateBuffer(dev, &bci, NULL, &buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, buf, &req);
   int t = -1;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & 6) == 6) {
         t = i;
         break;
      }
   if (t < 0) {
      fprintf(stderr, "no host-visible coherent memory\n");
      return 1;
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = t};
   VkDeviceMemory mem;
   CHK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHK(vkBindBufferMemory(dev, buf, mem, 0));
   uint32_t *map;
   CHK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&map));

   VkDescriptorSetLayoutBinding b0 = {.binding = 0,
                                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      .descriptorCount = 1,
                                      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo dli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b0};
   VkDescriptorSetLayout dsl;
   CHK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl};
   VkPipelineLayout pl;
   CHK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
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
   VkDescriptorBufferInfo dbi = {buf, 0, VK_WHOLE_SIZE};
   VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = set,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .pBufferInfo = &dbi};
   vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);

   static const struct {
      const char *name;
      int fp, vec4;
   } k[] = {{"fp32_p1", 1, 0}, {"fp32_p4", 1, 1}, {"int32_p1", 0, 0}, {"int32_p4", 0, 1}};
   int bad = 0;
   const char *only = getenv("XGP_ONLY"); /* kernel name, e.g. fp32_p1 */
   for (unsigned ki = 0; ki < 4; ki++) {
      if (only && strcmp(only, k[ki].name))
         continue;
      char path[600];
      snprintf(path, sizeof(path), "%s/xgp_%s.spv", argv[1], k[ki].name);
      size_t sz;
      uint32_t *code = xt_load_spv(path, &sz);
      if (!code) {
         printf("%s: cannot read %s\n", k[ki].name, path);
         bad++;
         continue;
      }
      uint32_t spv_lsx = spv_local_size_x(code, sz);
      if (spv_lsx != (uint32_t)lsx) {
         printf("%s: %s has local_size_x %u, not %d (rerun gen_shaders.py with it)\n", k[ki].name, path, spv_lsx,
                lsx);
         free(code);
         bad++;
         continue;
      }
      VkShaderModuleCreateInfo smi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = code};
      VkShaderModule sm;
      CHK(vkCreateShaderModule(dev, &smi, NULL, &sm));
      printf("%s (%d groups x %d lanes, %d mad/iter, %s):\n", k[ki].name, groups, lsx, rep,
             k[ki].vec4 ? "vec4" : "scalar");
      for (int li = 0; li < nl; li++) {
         int loop = loops[li];
         VkSpecializationMapEntry me = {0, 0, 4};
         VkSpecializationInfo spi = {1, &me, 4, &loop};
         VkComputePipelineCreateInfo cpci = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                      .module = sm,
                      .pName = "main",
                      .pSpecializationInfo = &spi},
            .layout = pl};
         VkPipeline pipe;
         VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe);
         if (r != VK_SUCCESS) {
            printf("  loop %d: pipeline creation failed (%d)\n", loop, r);
            bad++;
            break;
         }
         memset(map, 0xAA, ninv * 4);
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
         vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                              NULL, 0, NULL);
         CHK(vkEndCommandBuffer(cb));
         VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
         VkFence fence;
         CHK(vkCreateFence(dev, &fci, NULL, &fence));
         VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
         double t0 = now();
         CHK(vkQueueSubmit(queue, 1, &si, fence));
         r = vkWaitForFences(dev, 1, &fence, VK_TRUE, timeout_ns);
         double dt = now() - t0;
         if (r != VK_SUCCESS) {
            printf("  loop %d: wait failed (%d) after %.3f s\n", loop, r, dt);
            return 1;
         }
         vkDestroyFence(dev, fence, NULL);
         vkFreeCommandBuffers(dev, pool, 1, &cb);
         vkDestroyPipeline(dev, pipe, NULL);
         unsigned mism = 0;
         for (uint32_t gx = 0; gx < ninv; gx++) {
            uint32_t lx = gx % lsx;
            if (k[ki].fp) {
               float got;
               memcpy(&got, &map[gx], 4);
               int ok = 0;
               for (int q = 0; q < 2; q++) {
                  float wv = ref_f(gx, lx, k[ki].vec4, loop, rep, q);
                  if (got == wv || (isnan(got) && isnan(wv)) || fabsf(got - wv) <= 1e-4f * fabsf(wv))
                     ok = 1;
               }
               if (!ok && mism++ < 4) {
                  float wv = ref_f(gx, lx, k[ki].vec4, loop, rep, 0);
                  uint32_t wb;
                  memcpy(&wb, &wv, 4);
                  printf("    [%u] got 0x%08x want 0x%08x\n", gx, map[gx], wb);
               }
            } else if ((int32_t)map[gx] != ref_i(gx, lx, k[ki].vec4, loop, rep) && mism++ < 4) {
               printf("    [%u] got 0x%08x want 0x%08x\n", gx, map[gx], (uint32_t)ref_i(gx, lx, k[ki].vec4, loop, rep));
            }
         }
         double ops = (double)ninv * loop * rep * 2 * (k[ki].vec4 ? 4 : 1);
         printf("  loop %-4d %8.3f s  %9.4f G%s/s  %s\n", loop, dt, ops / dt * 1e-9, k[ki].fp ? "FLOP" : "IOP",
                mism ? "MISMATCH" : "ok");
         if (mism) {
            printf("    %u/%u results wrong\n", mism, ninv);
            bad++;
         }
      }
      vkDestroyShaderModule(dev, sm, NULL);
   }
   printf("ngv_peak: %s\n", bad ? "FAIL" : "OK");
   return bad ? 1 : 0;
}
