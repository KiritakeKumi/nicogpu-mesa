/* SPDX-License-Identifier: MIT */
/* ngv_compute.c -- compute smoke of the ngv driver through the Vulkan loader (batch 2).
 *
 * device -> buffers -> descriptor set (storage + dynamic storage) -> push constants -> compute pipeline ->
 * dispatch + dispatch-indirect + barrier + copy/fill/update -> timestamps -> events -> fence / timeline semaphores
 * -> result check; then DispatchBase and a secondary command buffer; then, on a fresh device, fault injection
 * (device lost).  Runs against the stub winsys' software CP
 * (NGV_WINSYS=stub); the shader is the placeholder kernel "ngv_ph_scale_bias" (see test/shaders/scale_bias.comp).
 *
 * NGV_TEST_DEGRADED=1: the stub is started with features masked (NGV_STUB_FEATURES_MASK, see ngv_winsys_stub.c)
 * and the expectations flip: no timestamps, and the sync features degrade (CPU_SIGNAL / WAIT_PENDING absent).
 * Exit status 0 = all checks passed. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "shaders/scale_bias.spv.h"
#include "shaders/scale_bias_real.spv.h"
#include "ngv_test_ext.h"


#define N 256 /* source elements: 4 workgroups of 64 */

/* Which scale/bias shader: the placeholder module (entry "ngv_ph_scale_bias", runs on the stub CP only) or the real
 * SPIR-V (entry "ngv_scale_bias", compiled NIR -> ngcomp, runs on the functional model: NGV_WINSYS=hal). */
static int use_real;
static const uint32_t *sb_code;
static size_t sb_size;
static const char *sb_entry;

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties mp;

struct buf {
   VkBuffer buf;
   VkDeviceMemory mem;
   void *map;
   VkDeviceSize size;
};

static int
make_buf(struct buf *b, VkDeviceSize size, VkBufferUsageFlags usage)
{
   b->size = size;
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
   CHECK(vkCreateBuffer(dev, &bci, NULL, &b->buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, b->buf, &req);
   uint32_t type = 0;
   for (uint32_t t = 0; t < mp.memoryTypeCount; t++)
      if ((req.memoryTypeBits & (1u << t)) && (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
         type = t;
         break;
      }
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                               .memoryTypeIndex = type};
   CHECK(vkAllocateMemory(dev, &mai, NULL, &b->mem));
   CHECK(vkBindBufferMemory(dev, b->buf, b->mem, 0));
   CHECK(vkMapMemory(dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map));
   memset(b->map, 0, size);
   return 0;
}

static void
barrier(VkCommandBuffer cb)
{
   VkMemoryBarrier2KHR mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
                          .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
                          .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT_KHR,
                          .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
                          .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT_KHR | VK_ACCESS_2_MEMORY_WRITE_BIT_KHR};
   VkDependencyInfoKHR di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
   xt.CmdPipelineBarrier2(cb, &di);
}

/* Fault injection on a fresh device: a dispatch with no descriptor set bound makes the (software) CP fault, the
 * device latches "lost" and the driver must report VK_ERROR_DEVICE_LOST instead of hanging.  Afterwards a second
 * submit (waiting on a timeline value that is never signaled) goes behind the fault, and the device is DESTROYED
 * with the loss still latched: ngv_queue_submit never fails the runtime's submit thread (it signals and returns
 * success, ngv_sync waits let PENDING through on a lost device), so vk_queue_stop_submit_thread drains cleanly
 * (before batch 3 the process had to leave the device alone, upstream asserts there). */
static int
fault_test(VkPhysicalDevice pd)
{
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
                                  .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   struct xt_dev xd;
   if (xt_require(pd, XT_TIMELINE))
      return 1;
   xt_dci(&dci, &xd, XT_TIMELINE);
   VkDevice d;
   CHECK(vkCreateDevice(pd, &dci, NULL, &d));
   VkQueue q;
   vkGetDeviceQueue(d, 0, 0, &q);
   VkDescriptorSetLayoutBinding b[2] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
   VkDescriptorSetLayoutCreateInfo dslci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 2, .pBindings = b};
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(d, &dslci, NULL, &dsl));
   VkPushConstantRange fpcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
   VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                      .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &fpcr};
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(d, &plci, NULL, &pl));
   VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = sb_size, .pCode = sb_code};
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(d, &smci, NULL, &sm));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm, .pName = sb_entry},
      .layout = pl};
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(d, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(d, &cai, &cb));
   VkCommandBufferBeginInfo cbbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdDispatch(cb, 1, 1, 1);
   CHECK(vkEndCommandBuffer(cb));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(d, &fci, NULL, &fence));
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
   VkResult r = vkQueueSubmit(q, 1, &si, fence);
   if (r == VK_SUCCESS)
      r = vkWaitForFences(d, 1, &fence, VK_TRUE, 5000000000ull);
   if (r != VK_ERROR_DEVICE_LOST) {
      fprintf(stderr, "FAIL fault injection: result %d, want VK_ERROR_DEVICE_LOST\n", r);
      return 1;
   }
   EXPECT(vkQueueWaitIdle(q) == VK_ERROR_DEVICE_LOST);

   /* behind the loss: a submit that waits for a timeline value nobody will signal, plus an ordinary one.  Both may
    * be accepted (SUCCESS) or refused (DEVICE_LOST); what matters is that nothing hangs and the device can go. */
   VkSemaphoreTypeCreateInfoKHR tci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR,
                                       .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR};
   VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tci};
   VkSemaphore tl;
   CHECK(vkCreateSemaphore(d, &sci, NULL, &tl));
   uint64_t wv = 7, sv = 8;
   VkTimelineSemaphoreSubmitInfoKHR tsi = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
                                        .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &wv,
                                        .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &sv};
   VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
   VkSubmitInfo si2 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &tsi, .waitSemaphoreCount = 1,
                       .pWaitSemaphores = &tl, .pWaitDstStageMask = &stage, .signalSemaphoreCount = 1,
                       .pSignalSemaphores = &tl};
   r = vkQueueSubmit(q, 1, &si2, VK_NULL_HANDLE);
   EXPECT(r == VK_SUCCESS || r == VK_ERROR_DEVICE_LOST);
   r = vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
   EXPECT(r == VK_SUCCESS || r == VK_ERROR_DEVICE_LOST);
   EXPECT(vkQueueWaitIdle(q) == VK_ERROR_DEVICE_LOST);

   vkDestroySemaphore(d, tl, NULL);
   vkDestroyFence(d, fence, NULL);
   vkFreeCommandBuffers(d, pool, 1, &cb);
   vkDestroyCommandPool(d, pool, NULL);
   vkDestroyPipeline(d, pipe, NULL);
   vkDestroyShaderModule(d, sm, NULL);
   vkDestroyPipelineLayout(d, pl, NULL);
   vkDestroyDescriptorSetLayout(d, dsl, NULL);
   vkDestroyDevice(d, NULL);
   return 0;
}

int
main(void)
{
   const int degraded = getenv("NGV_TEST_DEGRADED") && atoi(getenv("NGV_TEST_DEGRADED"));
   const char *ws = getenv("NGV_WINSYS");
   use_real = ws && !strcmp(ws, "hal");
   sb_code = use_real ? ngv_scale_bias_real_spv : ngv_scale_bias_spv;
   sb_size = use_real ? sizeof(ngv_scale_bias_real_spv) : sizeof(ngv_scale_bias_spv);
   sb_entry = use_real ? "ngv_scale_bias" : "ngv_ph_scale_bias";

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, &pd));
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

 /* queue family: timestamps follow the winsys feature bit (leftover 3) */
   uint32_t qn = 1;
   VkQueueFamilyProperties qp;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, &qp);
   EXPECT(qp.timestampValidBits == (degraded ? 0 : 64));
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   EXPECT(props.limits.timestampComputeAndGraphics == (degraded ? VK_FALSE : VK_TRUE));

   /* device with timeline semaphores + synchronization2 + host query reset */
   const unsigned want = XT_TIMELINE | XT_SYNC2 | XT_HQR;
   if (xt_require(pd, want))
      return 1;
   struct xt_dev xd;
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
                                  .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   xt_dci(&dci, &xd, want);
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   xt_load(dev);
   VkQueue q;
   vkGetDeviceQueue(dev, 0, 0, &q);

   /* buffers: src (N u32), dst (2N u32, second half written through the dynamic offset), args, scratch for
    * copy / fill / update, query results */
   struct buf src, dst, args, cpy, qres;
   const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
   if (make_buf(&src, N * 4, usage) || make_buf(&dst, 2 * N * 4, usage) || make_buf(&args, 64, usage) ||
       make_buf(&cpy, 4096, usage) || make_buf(&qres, 256, usage))
      return 1;
   uint32_t *psrc = src.map, *pdst = dst.map, *pargs = args.map, *pcpy = cpy.map;
   for (uint32_t i = 0; i < N; i++)
      psrc[i] = i;
   memset(pdst, 0xee, 2 * N * 4);
   pargs[0] = N / 64, pargs[1] = 1, pargs[2] = 1;

   /* descriptors: set 0 = { 0: storage buffer, 1: storage buffer dynamic } */
   VkDescriptorSetLayoutBinding bindings[2] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
   VkDescriptorSetLayoutCreateInfo dslci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 2, .pBindings = bindings};
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkDescriptorPoolSize psz[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1}};
   VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2,
                                      .poolSizeCount = 2, .pPoolSizes = psz,
                                      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT};
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                       .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl};
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
   VkDescriptorBufferInfo bi_src = {src.buf, 0, VK_WHOLE_SIZE};
   VkDescriptorBufferInfo bi_dst = {dst.buf, 0, N * 4}; /* a window of N elements; dynamic offset moves it */
   VkWriteDescriptorSet wr[2] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi_src},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, .pBufferInfo = &bi_dst}};
   vkUpdateDescriptorSets(dev, 2, wr, 0, NULL);
   /* a second set, freed again (pool free path) */
   VkDescriptorSet set2;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &set2));
   CHECK(vkFreeDescriptorSets(dev, dpool, 1, &set2));

   VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
   VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                      .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = sb_size, .pCode = sb_code};
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm, .pName = sb_entry},
      .layout = pl};
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   /* a shader with an unknown entry point (no placeholder, no .xo): creation must fail, not crash */
   cpci.stage.pName = "no_such_entry";
   VkPipeline bad = VK_NULL_HANDLE;
   EXPECT(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &bad) != VK_SUCCESS && bad == VK_NULL_HANDLE);

   /* timestamp pool */
   VkQueryPool qpool = VK_NULL_HANDLE;
   VkQueryPoolCreateInfo qpci = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                 .queryCount = 4};
   if (degraded) {
      EXPECT(vkCreateQueryPool(dev, &qpci, NULL, &qpool) != VK_SUCCESS);
      qpool = VK_NULL_HANDLE;
   } else {
      CHECK(vkCreateQueryPool(dev, &qpci, NULL, &qpool));
      xt.ResetQueryPool(dev, qpool, 0, 4);
   }

   VkEventCreateInfo eci = {.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
   VkEvent ev;
   CHECK(vkCreateEvent(dev, &eci, NULL, &ev));
   EXPECT(vkGetEventStatus(dev, ev) == VK_EVENT_RESET);

   /* ---- record ---- */
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
   VkCommandBufferBeginInfo cbbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   if (qpool) {
      vkCmdResetQueryPool(cb, qpool, 0, 4);
      xt.CmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR, qpool, 0);
   }
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   uint32_t dyn0 = 0, dyn1 = N * 4;
   uint32_t pc0[2] = {3, 7}, pc1[2] = {5, 1};
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 1, &dyn0);
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc0);
   vkCmdDispatch(cb, N / 64, 1, 1);                         /* dst[i] = 3 i + 7 */
   barrier(cb);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 1, &dyn1);
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc1);
   vkCmdDispatchIndirect(cb, args.buf, 0);                  /* dst[N + i] = 5 i + 1 */
   barrier(cb);
   /* transfer: fill, update, copy dst[0..15] -> cpy[64..], then event set/wait */
   vkCmdFillBuffer(cb, cpy.buf, 0, 64, 0xdeadbeef);
   uint32_t upd[4] = {11, 22, 33, 44};
   vkCmdUpdateBuffer(cb, cpy.buf, 64, 16, upd);
   barrier(cb);
   VkBufferCopy region = {0, 128, 64};
   vkCmdCopyBuffer(cb, dst.buf, cpy.buf, 1, &region); /* VK_KHR_copy_commands2 is not advertised (Vulkan 1.1) */
   VkDependencyInfoKHR none = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR};
   xt.CmdSetEvent2(cb, ev, &none);
   xt.CmdWaitEvents2(cb, 1, &ev, &none);
   if (qpool) {
      xt.CmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR, qpool, 1);
      vkCmdCopyQueryPoolResults(cb, qpool, 0, 2, qres.buf, 0, 16,
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT | VK_QUERY_RESULT_WAIT_BIT);
   }
   CHECK(vkEndCommandBuffer(cb));

   /* ---- submit: signals a fence and a timeline semaphore; a second submit waits for a host-signaled value ---- */
   VkSemaphoreTypeCreateInfoKHR stci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR,
                                     .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR, .initialValue = 0};
   VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &stci};
   VkSemaphore tl;
   CHECK(vkCreateSemaphore(dev, &sci, NULL, &tl));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));

   uint64_t wait_val = 1, sig_val = 2;
   VkTimelineSemaphoreSubmitInfoKHR tsi = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
                                        .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &wait_val,
                                        .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &sig_val};
   VkPipelineStageFlags wstage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &tsi, .waitSemaphoreCount = 1,
                      .pWaitSemaphores = &tl, .pWaitDstStageMask = &wstage, .commandBufferCount = 1,
                      .pCommandBuffers = &cb, .signalSemaphoreCount = 1, .pSignalSemaphores = &tl};
   /* wait-before-signal: the submit waits for timeline value 1 which the host signals only afterwards.  Needs the
    * runtime's submit thread, i.e. WAIT_PENDING; without it (degraded) signal first. */
   VkSemaphoreSignalInfoKHR ssi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR, .semaphore = tl, .value = 1};
   uint64_t v = 99;
   if (degraded)
      CHECK(xt.SignalSemaphore(dev, &ssi));
   CHECK(vkQueueSubmit(q, 1, &si, fence));
   if (!degraded) {
      EXPECT(vkGetFenceStatus(dev, fence) == VK_NOT_READY);
      CHECK(xt.GetSemaphoreCounterValue(dev, tl, &v));
      EXPECT(v == 0);
      CHECK(xt.SignalSemaphore(dev, &ssi));
   }
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
   VkSemaphoreWaitInfoKHR swi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR, .semaphoreCount = 1, .pSemaphores = &tl,
                              .pValues = &sig_val};
   CHECK(xt.WaitSemaphores(dev, &swi, 5000000000ull));

   /* ---- results ---- */
   for (uint32_t i = 0; i < N; i++) {
      if (pdst[i] != 3 * i + 7 || pdst[N + i] != 5 * i + 1) {
         fprintf(stderr, "FAIL result[%u]: %u %u (want %u %u)\n", i, pdst[i], pdst[N + i], 3 * i + 7, 5 * i + 1);
         return 1;
      }
   }
   for (int i = 0; i < 16; i++)
      EXPECT(pcpy[i] == 0xdeadbeefu);
   for (int i = 0; i < 4; i++)
      EXPECT(pcpy[16 + i] == upd[i]);
   for (uint32_t i = 0; i < 16; i++)
      EXPECT(pcpy[32 + i] == 3 * i + 7); /* copy of dst[0..15] at byte 128 */
   EXPECT(vkGetEventStatus(dev, ev) == VK_EVENT_SET);
   CHECK(vkResetEvent(dev, ev));
   EXPECT(vkGetEventStatus(dev, ev) == VK_EVENT_RESET);

   uint64_t host[4] = {0};
   if (qpool) {
      CHECK(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(host), host, 16,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
      EXPECT(host[0] && host[2] > host[0] && host[1] == 1 && host[3] == 1);
      uint64_t *gpu = qres.map; /* the same through CmdCopyQueryPoolResults */
      EXPECT(gpu[0] == host[0] && gpu[1] == 1 && gpu[2] == host[2] && gpu[3] == 1);
      uint64_t one = 1;
      EXPECT(vkGetQueryPoolResults(dev, qpool, 2, 1, 8, &one, 8, VK_QUERY_RESULT_64_BIT) == VK_NOT_READY);
   }

   /* ---- timeline: host-side signal / wait round trip, then a binary semaphore + fence reuse ---- */
   uint64_t v3 = 3;
   VkSemaphoreSignalInfoKHR ssi3 = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR, .semaphore = tl, .value = v3};
   CHECK(xt.SignalSemaphore(dev, &ssi3));
   CHECK(xt.GetSemaphoreCounterValue(dev, tl, &v));
   EXPECT(v == 3);
   CHECK(vkResetFences(dev, 1, &fence));
   VkSubmitInfo empty = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
   CHECK(vkQueueSubmit(q, 1, &empty, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
   CHECK(vkResetFences(dev, 1, &fence));

   /* reuse the command buffer (resubmit without re-recording) */
   memset(pdst, 0, 2 * N * 4);
   CHECK(vkQueueSubmit(q, 1, &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                                             .pCommandBuffers = &cb}, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
   EXPECT(pdst[N - 1] == 3 * (N - 1) + 7 && pdst[2 * N - 1] == 5 * (N - 1) + 1);
   CHECK(vkResetFences(dev, 1, &fence));

   /* ---- batch 3: DispatchBase (DISPATCH_BASE pipeline) and 32-bit CopyQueryPoolResults ---- */
   {
      VkPipeline pipe_base;
      cpci.stage.pName = sb_entry; /* the failure test above left a bogus name */
      cpci.flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
      CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe_base));
      cpci.flags = 0;
      VkCommandBuffer cb2;
      CHECK(vkAllocateCommandBuffers(dev, &cai, &cb2));
      CHECK(vkBeginCommandBuffer(cb2, &cbbi));
      uint32_t zero_dyn = 0, pc2[2] = {2, 9};
      vkCmdBindPipeline(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_base);
      vkCmdBindDescriptorSets(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 1, &zero_dyn);
      vkCmdPushConstants(cb2, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc2);
      vkCmdDispatchBase(cb2, 2, 0, 0, 2, 1, 1); /* workgroups 2 and 3 only: elements 128..255 */
      if (qpool) {
         barrier(cb2);
         vkCmdCopyQueryPoolResults(cb2, qpool, 0, 2, qres.buf, 128, 8,
                                   VK_QUERY_RESULT_WITH_AVAILABILITY_BIT | VK_QUERY_RESULT_WAIT_BIT);
      }
      CHECK(vkEndCommandBuffer(cb2));
      memset(pdst, 0, 2 * N * 4);
      memset((char *)qres.map + 128, 0xcc, 32);
      VkSubmitInfo si2 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb2};
      CHECK(vkQueueSubmit(q, 1, &si2, fence));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
      CHECK(vkResetFences(dev, 1, &fence));
      for (uint32_t i = 0; i < N; i++) {
         uint32_t want = i >= 128 ? 2 * i + 9 : 0;
         if (pdst[i] != want || pdst[N + i] != 0) {
            fprintf(stderr, "FAIL DispatchBase result[%u]: %u %u (want %u 0)\n", i, pdst[i], pdst[N + i], want);
            return 1;
         }
      }
      if (qpool) {
         uint32_t *q32 = (uint32_t *)((char *)qres.map + 128); /* {value, availability} x 2, 32-bit */
         /* the earlier resubmit of cb rewrote the timestamps: read the host values again */
         CHECK(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(host), host, 16,
                                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
         EXPECT(q32[0] == (uint32_t)host[0] && q32[1] == 1 && q32[2] == (uint32_t)host[2] && q32[3] == 1);
      }
      /* base != 0 on a pipeline without DISPATCH_BASE is invalid usage: the command buffer must fail, not run */
      VkCommandBuffer cb3;
      CHECK(vkAllocateCommandBuffers(dev, &cai, &cb3));
      CHECK(vkBeginCommandBuffer(cb3, &cbbi));
      vkCmdBindPipeline(cb3, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(cb3, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 1, &zero_dyn);
      vkCmdDispatchBase(cb3, 1, 0, 0, 1, 1, 1);
      EXPECT(vkEndCommandBuffer(cb3) != VK_SUCCESS);
      vkFreeCommandBuffers(dev, pool, 1, &cb3);
      vkFreeCommandBuffers(dev, pool, 1, &cb2);
      vkDestroyPipeline(dev, pipe_base, NULL);
   }

 /* ---- : secondary command buffer, inlined by vkCmdExecuteCommands. The primary dispatches before and after
    * it with its own state; nothing may leak in either direction (after vkCmdExecuteCommands the primary rebinds, as
    * the spec requires, and the driver must re-emit the CP state the inlined packets overwrote). ---- */
   {
      VkCommandBufferAllocateInfo sai = cai;
      sai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
      VkCommandBuffer sec, prim;
      CHECK(vkAllocateCommandBuffers(dev, &sai, &sec));
      CHECK(vkAllocateCommandBuffers(dev, &cai, &prim));
      VkCommandBufferInheritanceInfo inh = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
      VkCommandBufferBeginInfo sbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .pInheritanceInfo = &inh};
      uint32_t zero_dyn = 0, pc3[2] = {4, 3};
      CHECK(vkBeginCommandBuffer(sec, &sbi));
      vkCmdBindPipeline(sec, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(sec, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 1, &dyn1);
      vkCmdPushConstants(sec, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc1);
      vkCmdDispatch(sec, N / 64, 1, 1); /* dst[N + i] = 5 i + 1 */
      CHECK(vkEndCommandBuffer(sec));
      CHECK(vkBeginCommandBuffer(prim, &cbbi));
      vkCmdBindPipeline(prim, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(prim, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 1, &zero_dyn);
      vkCmdPushConstants(prim, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc0);
      vkCmdDispatch(prim, N / 64, 1, 1); /* dst[i] = 3 i + 7, overwritten below */
      vkCmdExecuteCommands(prim, 1, &sec);
      barrier(prim);
      vkCmdBindPipeline(prim, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(prim, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 1, &zero_dyn);
      vkCmdPushConstants(prim, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc3);
      vkCmdDispatch(prim, N / 64, 1, 1); /* dst[i] = 4 i + 3 */
      CHECK(vkEndCommandBuffer(prim));
      memset(pdst, 0, 2 * N * 4);
      VkSubmitInfo si4 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &prim};
      CHECK(vkQueueSubmit(q, 1, &si4, fence));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
      CHECK(vkResetFences(dev, 1, &fence));
      for (uint32_t i = 0; i < N; i++) {
         if (pdst[i] != 4 * i + 3 || pdst[N + i] != 5 * i + 1) {
            fprintf(stderr, "FAIL secondary result[%u]: %u %u (want %u %u)\n", i, pdst[i], pdst[N + i], 4 * i + 3,
                    5 * i + 1);
            return 1;
         }
      }
      vkFreeCommandBuffers(dev, pool, 1, &prim);
      vkFreeCommandBuffers(dev, pool, 1, &sec);
   }

   /* compile path without a CP (any backend): the real SPIR-V must go through NIR + ngcomp, and a layout that lacks
    * a binding the shader uses must be refused.  Skipped when ngcomp is not configured. */
   if (getenv("NGV_NGCOMP_DIR")) {
      VkShaderModuleCreateInfo rsmci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                        .codeSize = sizeof(ngv_scale_bias_real_spv), .pCode = ngv_scale_bias_real_spv};
      VkShaderModule rsm;
      CHECK(vkCreateShaderModule(dev, &rsmci, NULL, &rsm));
      VkComputePipelineCreateInfo rcp = cpci;
      rcp.stage.module = rsm;
      rcp.stage.pName = "ngv_scale_bias";
      VkPipeline rp;
      CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &rcp, NULL, &rp));
      vkDestroyPipeline(dev, rp, NULL);
      VkDescriptorSetLayoutBinding one = {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
      VkDescriptorSetLayoutCreateInfo odsl = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                              .bindingCount = 1, .pBindings = &one};
      VkDescriptorSetLayout dsl1;
      CHECK(vkCreateDescriptorSetLayout(dev, &odsl, NULL, &dsl1));
      VkPipelineLayoutCreateInfo opl = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                        .pSetLayouts = &dsl1};
      VkPipelineLayout pl1;
      CHECK(vkCreatePipelineLayout(dev, &opl, NULL, &pl1));
      rcp.layout = pl1;
      VkPipeline rp2 = VK_NULL_HANDLE;
      EXPECT(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &rcp, NULL, &rp2) != VK_SUCCESS && rp2 == VK_NULL_HANDLE);
      vkDestroyPipelineLayout(dev, pl1, NULL);
      vkDestroyDescriptorSetLayout(dev, dsl1, NULL);
      vkDestroyShaderModule(dev, rsm, NULL);
   }

   /* teardown */
   vkDestroyFence(dev, fence, NULL);
   vkDestroySemaphore(dev, tl, NULL);
   vkFreeCommandBuffers(dev, pool, 1, &cb);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyEvent(dev, ev, NULL);
   if (qpool)
      vkDestroyQueryPool(dev, qpool, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   struct buf *all[] = {&src, &dst, &args, &cpy, &qres};
   for (unsigned i = 0; i < 5; i++) {
      vkDestroyBuffer(dev, all[i]->buf, NULL);
      vkFreeMemory(dev, all[i]->mem, NULL);
   }
   vkDestroyDevice(dev, NULL);

   if (fault_test(pd))
      return 1;
   vkDestroyInstance(inst, NULL);
   printf("ngv_compute%s: OK\n", degraded ? " (degraded features)" : "");
   return 0;
}
