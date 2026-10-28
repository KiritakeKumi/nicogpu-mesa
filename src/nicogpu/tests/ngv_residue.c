/* SPDX-License-Identifier: MIT */
/* ngv_residue: (section 8) -- does one owner's data survive in the SP's shared memory for the next owner?
 *
 *   ngv_residue <ngv_residue.spv> fill        # write the pattern into shared memory, exit
 *   ngv_residue <ngv_residue.spv> read        # read shared memory without writing it first, report what is there
 *   ngv_residue <ngv_residue.spv> both        # fill then read on the same device (same VM = same owner, control)
 *   ngv_residue <ngv_residue.spv> <what> XODIR  # the same for the vector register file (regs mode)
 *
 * shared mode -- shaders/ngv_residue.comp: MODE 0 fills the 2048-word shared array with 0xA5A50000 | i, MODE 1
 * copies it out unwritten.  One workgroup of 64 invocations (two warps), so both dispatches use shared window 0.
 * regs mode -- the pipelines come from XODIR/fill/main.xo and XODIR/read/main.xo (NGV_XO_DIR override; built from
 * shaders/ngv_residue_regs_{fill,read}.xir): fill sets v8..v31 of every lane to 0xC3000000 | r << 16 | lane, read
 * stores v8..v31 unwritten to o[lane * 32 + r - 8]; a word counts as pattern when its top byte is 0xC3.
 * Prints "residue <mode>: pattern P zero Z other O" over the checked words (2048 shared, 64 x 24 regs) and exits 0;
 * the verdict is the caller's (ngv_residue.sh): cross-process `read` after `fill` must give only zeros (the
 * owner-change scrub cleared it); `both` should give P > 0, which shows the test can see residue at all.
 * Exit 1 on a Vulkan error, 2 on usage. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "ngv_test_ext.h"

#define NW 2048u

#define CHK(x)                                                                                                         \
   do {                                                                                                                \
      VkResult r_ = (x);                                                                                               \
      if (r_ != VK_SUCCESS) {                                                                                          \
         fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #x, r_);                                               \
         exit(1);                                                                                                      \
      }                                                                                                                \
   } while (0)

static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkPipelineLayout pl;
static VkDescriptorSet set;
static VkShaderModule sm;
static const char *xodir;   /* regs mode */

static void
run(uint32_t mode)
{
   char d[512];
   if (xodir) {   /* the driver reads NGV_XO_DIR at pipeline creation */
      snprintf(d, sizeof(d), "%s/%s", xodir, mode ? "read" : "fill");
      setenv("NGV_XO_DIR", d, 1);
   }
   VkSpecializationMapEntry me = {0, 0, 4};
   VkSpecializationInfo spi = {1, &me, 4, &mode};
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
   vkCmdDispatch(cb, 1, 1, 1);
   CHK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
   CHK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHK(vkQueueWaitIdle(queue));
   vkFreeCommandBuffers(dev, pool, 1, &cb);
   vkDestroyPipeline(dev, pipe, NULL);
}

int
main(int argc, char **argv)
{
   if (argc < 3 || (strcmp(argv[2], "fill") && strcmp(argv[2], "read") && strcmp(argv[2], "both"))) {
      fprintf(stderr, "usage: %s <ngv_residue.spv> fill|read|both [XODIR]\n", argv[0]);
      return 2;
   }
   const char *what = argv[2];
   xodir = argc > 3 ? argv[3] : NULL;

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice pd;
   VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || n < 1)
      return 1;
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
   CHK(vkCreateDevice(pd, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   CHK(vkCreateCommandPool(dev, &cpi, NULL, &pool));

   /* output buffer, host-visible + coherent, pre-filled with a sentinel so unwritten words show up as "other" */
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = NW * 4,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
   VkBuffer buf;
   CHK(vkCreateBuffer(dev, &bci, NULL, &buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, buf, &req);
   int t = -1;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & 6) == 6) {
         t = (int)i;
         break;
      }
   if (t < 0) {
      fprintf(stderr, "no host-visible coherent memory\n");
      return 1;
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = (uint32_t)t};
   VkDeviceMemory mem;
   CHK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHK(vkBindBufferMemory(dev, buf, mem, 0));
   uint32_t *map;
   CHK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&map));

   VkDescriptorSetLayoutBinding b = {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo dli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b};
   VkDescriptorSetLayout dsl;
   CHK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl};
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
   CHK(vkAllocateDescriptorSets(dev, &dai, &set));
   VkDescriptorBufferInfo dbi = {buf, 0, VK_WHOLE_SIZE};
   VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
                             .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .pBufferInfo = &dbi};
   vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);

   size_t sz;
   uint32_t *code = xt_load_spv(argv[1], &sz);
   if (!code) {
      fprintf(stderr, "cannot read %s\n", argv[1]);
      return 1;
   }
   VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = code};
   CHK(vkCreateShaderModule(dev, &smi, NULL, &sm));

   if (strcmp(what, "read"))
      run(0);
   if (strcmp(what, "fill")) {
      for (uint32_t i = 0; i < NW; i++)
         map[i] = 0xDEADBEEFu;
      run(1);
      uint32_t pat = 0, zero = 0, other = 0;
      for (uint32_t i = 0; i < NW; i++) {
         if (xodir && (i & 31) >= 24)   /* regs mode: words 24..31 of each lane are not written */
            continue;
         if (xodir ? (map[i] >> 24) == 0xC3 : map[i] == (0xA5A50000u | i))
            pat++;
         else if (map[i] == 0)
            zero++;
         else
            other++;
      }
      printf("residue %s: pattern %u zero %u other %u\n", xodir ? "regs" : "shared", pat, zero, other);
   } else {
      printf("residue %s: filled\n", xodir ? "regs" : "shared");
   }

   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return 0;
}
