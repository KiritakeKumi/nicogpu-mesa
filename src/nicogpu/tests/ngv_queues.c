/* SPDX-License-Identifier: MIT */
/* ngv_queues.c -- queue families / multiple queues of the ngv driver.
 *
 * The driver exposes two queue families with the same capabilities (COMPUTE|TRANSFER): family 0 ("universal", 1
 * queue) and family 1 (compute-only, 2 queues); every VkQueue is its own winsys queue on the one CP ring.  Checks:
 *   1. the family properties (count, flags, queue counts, timestamp bits);
 *   2. bad VkDeviceCreateInfo queue requests are refused (unknown family, too many queues, duplicate family,
 *      PROTECTED), the CTS *_compute_only_queue device (family 0 x1 + family 1 x1) is accepted;
 *   3. a device with all three queues (family 1 listed first): distinct VkQueue handles, vkGetDeviceQueue2;
 *   4. a cross-queue chain q0 -> binary semaphore -> q1 -> timeline semaphore -> q2 on CONCURRENT buffers;
 *   5. wait-before-signal across queues (q2 waits for a timeline value that q1 only submits later), unless
 *      NGV_THREADED_SUBMIT=0;
 *   6. vkQueueWaitIdle per queue, vkDeviceWaitIdle;
 *   7. three application threads, one per queue, submitting / waiting concurrently;
 *   8. on a fresh two-family device: a fault on the family-1 queue is VK_ERROR_DEVICE_LOST on the family-0 queue
 *      too, and the device can still be destroyed.
 * Runs on the stub winsys (NGV_WINSYS=stub, software CP) and on the functional model (NGV_WINSYS=hal); on the board
 * (drm winsys) sections 1-7 run, the fault (8) only with NGV_QUEUES_FAULT=1.  Exit status 0 = all checks passed. */
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "shaders/scale_bias.spv.h"
#include "shaders/scale_bias_real.spv.h"
#include "ngv_test_ext.h"


#define N 1024           /* u32 per buffer */
#define MT_ITERS 200     /* per thread in section 7 */
#define TIMEOUT 5000000000ull

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties mp;
static const uint32_t both_families[2] = {0, 1};

struct buf {
   VkBuffer buf;
   VkDeviceMemory mem;
   uint32_t *map;
};

/* host-visible, coherent; CONCURRENT between the two families (no ownership transfers needed) */
static int
make_buf(struct buf *b)
{
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = N * 4,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             .sharingMode = VK_SHARING_MODE_CONCURRENT, .queueFamilyIndexCount = 2,
                             .pQueueFamilyIndices = both_families};
   CHECK(vkCreateBuffer(dev, &bci, NULL, &b->buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, b->buf, &req);
   uint32_t type = UINT32_MAX;
   for (uint32_t t = 0; t < mp.memoryTypeCount; t++)
      if ((req.memoryTypeBits & (1u << t)) &&
          (mp.memoryTypes[t].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
         type = t;
         break;
      }
   EXPECT(type != UINT32_MAX);
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                               .memoryTypeIndex = type};
   CHECK(vkAllocateMemory(dev, &mai, NULL, &b->mem));
   CHECK(vkBindBufferMemory(dev, b->buf, b->mem, 0));
   void *p;
   CHECK(vkMapMemory(dev, b->mem, 0, VK_WHOLE_SIZE, 0, &p));
   b->map = p;
   memset(b->map, 0, N * 4);
   return 0;
}

static void
free_buf(struct buf *b)
{
   vkDestroyBuffer(dev, b->buf, NULL);
   vkFreeMemory(dev, b->mem, NULL);
}

static int
check_words(const struct buf *b, uint32_t first, uint32_t count, uint32_t want, const char *what)
{
   for (uint32_t i = first; i < first + count; i++)
      if (b->map[i] != want) {
         fprintf(stderr, "FAIL %s: word %u = 0x%08x, want 0x%08x\n", what, i, b->map[i], want);
         return 1;
      }
   return 0;
}

static VkSemaphore
make_sem(VkSemaphoreType type, uint64_t initial)
{
   VkSemaphoreTypeCreateInfoKHR tci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR, .semaphoreType = type,
                                    .initialValue = initial};
   VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tci};
   VkSemaphore s = VK_NULL_HANDLE;
   if (vkCreateSemaphore(dev, &sci, NULL, &s) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return s;
}

static VkCommandPool
make_pool(VkDevice d, uint32_t family)
{
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                  .queueFamilyIndex = family};
   VkCommandPool p = VK_NULL_HANDLE;
   if (vkCreateCommandPool(d, &cpi, NULL, &p) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return p;
}

static VkCommandBuffer
make_cb(VkDevice d, VkCommandPool pool)
{
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb = VK_NULL_HANDLE;
   if (vkAllocateCommandBuffers(d, &cai, &cb) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return cb;
}

static void
transfer_barrier(VkCommandBuffer cb)
{
   VkMemoryBarrier2KHR mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
                          .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT_KHR,
                          .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR,
                          .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT_KHR,
                          .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT_KHR | VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR};
   VkDependencyInfoKHR di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
   xt.CmdPipelineBarrier2(cb, &di);
}

/* one command buffer per submit, synchronization2 */
static VkResult
submit(VkQueue q, VkCommandBuffer cb, VkSemaphore wait, uint64_t wait_value, VkSemaphore signal, uint64_t signal_value,
       VkFence fence)
{
   VkSemaphoreSubmitInfoKHR w = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR, .semaphore = wait,
                              .value = wait_value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR};
   VkSemaphoreSubmitInfoKHR s = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR, .semaphore = signal,
                              .value = signal_value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR};
   VkCommandBufferSubmitInfoKHR c = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR, .commandBuffer = cb};
   VkSubmitInfo2KHR si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
                       .waitSemaphoreInfoCount = wait ? 1 : 0, .pWaitSemaphoreInfos = &w,
                       .commandBufferInfoCount = cb ? 1 : 0, .pCommandBufferInfos = &c,
                       .signalSemaphoreInfoCount = signal ? 1 : 0, .pSignalSemaphoreInfos = &s};
   return xt.QueueSubmit2(q, 1, &si, fence);
}

static VkFence
make_fence(VkDevice d)
{
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence f = VK_NULL_HANDLE;
   if (vkCreateFence(d, &fci, NULL, &f) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return f;
}

static VkResult
create_device(VkPhysicalDevice pd, const VkDeviceQueueCreateInfo *qci, uint32_t nqci, VkDevice *out)
{
   const unsigned want = XT_TIMELINE | XT_SYNC2;
   *out = VK_NULL_HANDLE;
   if (xt_require(pd, want))
      return VK_ERROR_EXTENSION_NOT_PRESENT;
   struct xt_dev xd;
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = nqci,
                             .pQueueCreateInfos = qci};
   xt_dci(&dci, &xd, want);
   VkResult r = vkCreateDevice(pd, &dci, NULL, out);
   if (r == VK_SUCCESS)
      xt_load(*out);
   return r;
}

static const float prios[2] = {1.0f, 0.0f};

/* ---- 7: one application thread per queue ---------------------------------------------------------------------- */

struct mt_arg {
   VkQueue q;
   uint32_t family;
   uint32_t id;
   struct buf *b;
};
static atomic_int mt_failures;

static void *
mt_thread(void *p)
{
   struct mt_arg *a = p;
   VkCommandPool pool = make_pool(dev, a->family);
   VkCommandBuffer cb = pool ? make_cb(dev, pool) : VK_NULL_HANDLE;
   VkFence f = make_fence(dev);
   if (!cb || !f) {
      fprintf(stderr, "FAIL mt %u: setup\n", a->id);
      atomic_fetch_add(&mt_failures, 1);
      return NULL;
   }
   for (uint32_t it = 0; it < MT_ITERS; it++) {
      const uint32_t v = (a->id << 24) | it;
      VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      if (vkResetCommandBuffer(cb, 0) || vkBeginCommandBuffer(cb, &bi)) {
         atomic_fetch_add(&mt_failures, 1);
         break;
      }
      vkCmdFillBuffer(cb, a->b->buf, 0, VK_WHOLE_SIZE, v);
      if (vkEndCommandBuffer(cb) || vkResetFences(dev, 1, &f) || submit(a->q, cb, 0, 0, 0, 0, f) ||
          vkWaitForFences(dev, 1, &f, VK_TRUE, TIMEOUT)) {
         fprintf(stderr, "FAIL mt %u: iteration %u submit / wait\n", a->id, it);
         atomic_fetch_add(&mt_failures, 1);
         break;
      }
      if (a->b->map[0] != v || a->b->map[N - 1] != v) {
         fprintf(stderr, "FAIL mt %u: iteration %u: 0x%08x / 0x%08x, want 0x%08x\n", a->id, it, a->b->map[0],
                 a->b->map[N - 1], v);
         atomic_fetch_add(&mt_failures, 1);
         break;
      }
   }
   vkDestroyFence(dev, f, NULL);
   vkFreeCommandBuffers(dev, pool, 1, &cb);
   vkDestroyCommandPool(dev, pool, NULL);
   return NULL;
}

/* ---- 8: fault on the family-1 queue of a two-family device ---------------------------------------------------- */

static int
fault_test(VkPhysicalDevice pd, const uint32_t *code, size_t size, const char *entry)
{
   VkDeviceQueueCreateInfo qci[2] = {
      {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1,
       .pQueuePriorities = prios},
      {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 1, .queueCount = 1,
       .pQueuePriorities = prios}};
   VkDevice d;
   CHECK(create_device(pd, qci, 2, &d));
   VkQueue q0, q1;
   vkGetDeviceQueue(d, 0, 0, &q0);
   vkGetDeviceQueue(d, 1, 0, &q1);
   /* a dispatch with no descriptor set bound faults the (software) CP, as in ngv_compute's fault_test */
   VkDescriptorSetLayoutBinding b[2] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
   VkDescriptorSetLayoutCreateInfo dslci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 2, .pBindings = b};
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(d, &dslci, NULL, &dsl));
   VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
   VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                      .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(d, &plci, NULL, &pl));
   VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size,
                                    .pCode = code};
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(d, &smci, NULL, &sm));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm, .pName = entry},
      .layout = pl};
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   VkCommandPool pool1 = make_pool(d, 1), pool0 = make_pool(d, 0);
   EXPECT(pool1 && pool0);
   VkCommandBuffer bad = make_cb(d, pool1), empty = make_cb(d, pool0);
   EXPECT(bad && empty);
   VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(bad, &bi));
   vkCmdBindPipeline(bad, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdDispatch(bad, 1, 1, 1);
   CHECK(vkEndCommandBuffer(bad));
   CHECK(vkBeginCommandBuffer(empty, &bi));
   CHECK(vkEndCommandBuffer(empty));

   /* family 0 works before the fault */
   VkFence f = make_fence(d);
   EXPECT(f);
   VkSubmitInfo si0 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &empty};
   CHECK(vkQueueSubmit(q0, 1, &si0, f));
   CHECK(vkWaitForFences(d, 1, &f, VK_TRUE, TIMEOUT));
   CHECK(vkResetFences(d, 1, &f));

   VkSubmitInfo si1 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &bad};
   VkResult r = vkQueueSubmit(q1, 1, &si1, f);
   if (r == VK_SUCCESS)
      r = vkWaitForFences(d, 1, &f, VK_TRUE, TIMEOUT);
   if (r != VK_ERROR_DEVICE_LOST) {
      fprintf(stderr, "FAIL fault on family 1: result %d, want VK_ERROR_DEVICE_LOST\n", r);
      return 1;
   }
   EXPECT(vkQueueWaitIdle(q1) == VK_ERROR_DEVICE_LOST);
   /* the loss is device-wide: the innocent family-0 queue reports it too, and a submit there must not hang */
   r = vkQueueSubmit(q0, 1, &si0, VK_NULL_HANDLE);
   EXPECT(r == VK_SUCCESS || r == VK_ERROR_DEVICE_LOST);
   EXPECT(vkQueueWaitIdle(q0) == VK_ERROR_DEVICE_LOST);
   EXPECT(vkDeviceWaitIdle(d) == VK_ERROR_DEVICE_LOST);

   vkDestroyFence(d, f, NULL);
   vkFreeCommandBuffers(d, pool1, 1, &bad);
   vkFreeCommandBuffers(d, pool0, 1, &empty);
   vkDestroyCommandPool(d, pool1, NULL);
   vkDestroyCommandPool(d, pool0, NULL);
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
   const char *ws = getenv("NGV_WINSYS");
   const int stub = ws && !strcmp(ws, "stub");
   const int hal = ws && !strcmp(ws, "hal");
   const char *ts = getenv("NGV_THREADED_SUBMIT");
   const int threaded = !ts || strcmp(ts, "0");
   const char *fe = getenv("NGV_QUEUES_FAULT");
   const int do_fault = stub || hal || (fe && atoi(fe));

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice pd;
   VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
   EXPECT((er == VK_SUCCESS || er == VK_INCOMPLETE) && n == 1);
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   /* 1. families */
   uint32_t qn = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
   EXPECT(qn == 2);
   VkQueueFamilyProperties qp[3];
   memset(qp, 0, sizeof(qp));
   qn = 3; /* more room than needed: the driver writes back 2 */
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp);
   EXPECT(qn == 2);
   for (uint32_t f = 0; f < 2; f++) {
      EXPECT(qp[f].queueFlags == (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT));
      EXPECT(qp[f].timestampValidBits == qp[0].timestampValidBits);
      EXPECT(qp[f].minImageTransferGranularity.width == 1);
   }
   EXPECT(qp[0].queueCount == 1 && qp[1].queueCount == 2);
   {
      /* the 2-variant, and a short array (only family 0 is written) */
      VkQueueFamilyProperties2 q2[2] = {{.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2},
                                        {.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2}};
      uint32_t c = 2;
      vkGetPhysicalDeviceQueueFamilyProperties2(pd, &c, q2);
      EXPECT(c == 2 && q2[1].queueFamilyProperties.queueCount == 2);
      VkQueueFamilyProperties one = {0};
      c = 1;
      vkGetPhysicalDeviceQueueFamilyProperties(pd, &c, &one);
      EXPECT(c == 1 && one.queueCount == 1);
   }

   /* 2. queue requests */
   {
      VkDevice d;
      VkDeviceQueueCreateInfo bad_family = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                            .queueFamilyIndex = 2, .queueCount = 1, .pQueuePriorities = prios};
      EXPECT(create_device(pd, &bad_family, 1, &d) != VK_SUCCESS && d == VK_NULL_HANDLE);
      const float three[3] = {1.0f, 1.0f, 1.0f};
      VkDeviceQueueCreateInfo too_many = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                          .queueFamilyIndex = 1, .queueCount = 3, .pQueuePriorities = three};
      EXPECT(create_device(pd, &too_many, 1, &d) != VK_SUCCESS && d == VK_NULL_HANDLE);
      VkDeviceQueueCreateInfo dup[2] = {
         {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 1, .queueCount = 1,
          .pQueuePriorities = prios},
         {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 1, .queueCount = 1,
          .pQueuePriorities = prios}};
      EXPECT(create_device(pd, dup, 2, &d) != VK_SUCCESS && d == VK_NULL_HANDLE);
      VkDeviceQueueCreateInfo prot = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                      .flags = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT, .queueFamilyIndex = 0,
                                      .queueCount = 1, .pQueuePriorities = prios};
      EXPECT(create_device(pd, &prot, 1, &d) != VK_SUCCESS && d == VK_NULL_HANDLE);
      /* the CTS compute-only device (vktComputeIndirectComputeDispatchTests.cpp createCustomDevice) and a device
       * with only the compute-only family */
      VkDeviceQueueCreateInfo cts[2] = {
         {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1,
          .pQueuePriorities = prios},
         {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 1, .queueCount = 1,
          .pQueuePriorities = prios}};
      CHECK(create_device(pd, cts, 2, &d));
      VkQueue a = VK_NULL_HANDLE, b = VK_NULL_HANDLE;
      vkGetDeviceQueue(d, 0, 0, &a);
      vkGetDeviceQueue(d, 1, 0, &b);
      EXPECT(a && b && a != b);
      CHECK(vkQueueWaitIdle(b));
      vkDestroyDevice(d, NULL);
      CHECK(create_device(pd, &cts[1], 1, &d));
      vkGetDeviceQueue(d, 1, 0, &b);
      EXPECT(b);
      CHECK(vkQueueWaitIdle(b));
      vkDestroyDevice(d, NULL);
   }

   /* 3. all three queues; family 1 first in the create info (order must not matter) */
   VkDeviceQueueCreateInfo all[2] = {
      {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 1, .queueCount = 2,
       .pQueuePriorities = prios},
      {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1,
       .pQueuePriorities = prios}};
   CHECK(create_device(pd, all, 2, &dev));
   VkQueue q[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
   vkGetDeviceQueue(dev, 0, 0, &q[0]);
   vkGetDeviceQueue(dev, 1, 0, &q[1]);
   vkGetDeviceQueue(dev, 1, 1, &q[2]);
   EXPECT(q[0] && q[1] && q[2] && q[0] != q[1] && q[1] != q[2] && q[0] != q[2]);
   {
      VkDeviceQueueInfo2 qi2 = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2, .queueFamilyIndex = 1,
                                .queueIndex = 1};
      VkQueue q2 = VK_NULL_HANDLE;
      vkGetDeviceQueue2(dev, &qi2, &q2);
      EXPECT(q2 == q[2]);
   }

   struct buf x, y, z;
   if (make_buf(&x) || make_buf(&y) || make_buf(&z))
      return 1;
   VkCommandPool pool0 = make_pool(dev, 0), pool1 = make_pool(dev, 1);
   EXPECT(pool0 && pool1);
   VkCommandBuffer cba = make_cb(dev, pool0), cbb = make_cb(dev, pool1), cbc = make_cb(dev, pool1);
   EXPECT(cba && cbb && cbc);
   VkSemaphore bin = make_sem(VK_SEMAPHORE_TYPE_BINARY_KHR, 0), tl = make_sem(VK_SEMAPHORE_TYPE_TIMELINE_KHR, 0);
   VkFence fence = make_fence(dev);
   EXPECT(bin && tl && fence);
   VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   const VkBufferCopy whole = {0, 0, N * 4};

   /* 4. q0: fill X -> (binary) -> q1: copy X->Y, fill the second half of Y -> (timeline 1) -> q2: copy Y->Z */
   CHECK(vkBeginCommandBuffer(cba, &bi));
   vkCmdFillBuffer(cba, x.buf, 0, VK_WHOLE_SIZE, 0x11111111u);
   CHECK(vkEndCommandBuffer(cba));
   CHECK(vkBeginCommandBuffer(cbb, &bi));
   vkCmdCopyBuffer(cbb, x.buf, y.buf, 1, &whole);
   transfer_barrier(cbb);
   vkCmdFillBuffer(cbb, y.buf, N * 2, N * 2, 0x22222222u);
   CHECK(vkEndCommandBuffer(cbb));
   CHECK(vkBeginCommandBuffer(cbc, &bi));
   vkCmdCopyBuffer(cbc, y.buf, z.buf, 1, &whole);
   CHECK(vkEndCommandBuffer(cbc));
   CHECK(submit(q[0], cba, 0, 0, bin, 0, VK_NULL_HANDLE));
   CHECK(submit(q[1], cbb, bin, 0, tl, 1, VK_NULL_HANDLE));
   CHECK(submit(q[2], cbc, tl, 1, 0, 0, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, TIMEOUT));
   if (check_words(&z, 0, N / 2, 0x11111111u, "chain, first half") ||
       check_words(&z, N / 2, N / 2, 0x22222222u, "chain, second half"))
      return 1;
   uint64_t v = 0;
   CHECK(xt.GetSemaphoreCounterValue(dev, tl, &v));
   EXPECT(v == 1);

   /* 5. wait-before-signal across queues: q2 waits for timeline 3, which q1 only submits afterwards */
   if (threaded) {
      CHECK(vkResetFences(dev, 1, &fence));
      memset(z.map, 0, N * 4);
      CHECK(vkResetCommandBuffer(cbc, 0));
      CHECK(vkBeginCommandBuffer(cbc, &bi));
      vkCmdCopyBuffer(cbc, x.buf, z.buf, 1, &whole);
      CHECK(vkEndCommandBuffer(cbc));
      CHECK(vkResetCommandBuffer(cbb, 0));
      CHECK(vkBeginCommandBuffer(cbb, &bi));
      vkCmdFillBuffer(cbb, x.buf, 0, VK_WHOLE_SIZE, 0x33333333u);
      CHECK(vkEndCommandBuffer(cbb));
      CHECK(submit(q[2], cbc, tl, 3, 0, 0, fence));
      usleep(20000);
      EXPECT(vkGetFenceStatus(dev, fence) == VK_NOT_READY);
      EXPECT(z.map[0] == 0);
      CHECK(submit(q[1], cbb, 0, 0, tl, 3, VK_NULL_HANDLE));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, TIMEOUT));
      if (check_words(&z, 0, N, 0x33333333u, "wait-before-signal"))
         return 1;
      /* the host waits on a value of the other family's queue as well */
      VkSemaphoreWaitInfoKHR wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR, .semaphoreCount = 1,
                                .pSemaphores = &tl, .pValues = (uint64_t[]){3}};
      CHECK(xt.WaitSemaphores(dev, &wi, TIMEOUT));
   } else {
      printf("ngv_queues: NGV_THREADED_SUBMIT=0, wait-before-signal skipped\n");
   }

   /* 6. idle */
   for (int i = 0; i < 3; i++)
      CHECK(vkQueueWaitIdle(q[i]));
   CHECK(vkDeviceWaitIdle(dev));

   /* 7. one application thread per queue */
   {
      struct buf mb[3];
      for (int i = 0; i < 3; i++)
         if (make_buf(&mb[i]))
            return 1;
      struct mt_arg args[3] = {{q[0], 0, 1, &mb[0]}, {q[1], 1, 2, &mb[1]}, {q[2], 1, 3, &mb[2]}};
      pthread_t t[3];
      for (int i = 0; i < 3; i++)
         EXPECT(pthread_create(&t[i], NULL, mt_thread, &args[i]) == 0);
      for (int i = 0; i < 3; i++)
         pthread_join(t[i], NULL);
      EXPECT(atomic_load(&mt_failures) == 0);
      for (int i = 0; i < 3; i++) {
         if (check_words(&mb[i], 0, N, ((uint32_t)(i + 1) << 24) | (MT_ITERS - 1), "threads"))
            return 1;
         free_buf(&mb[i]);
      }
   }

   vkDestroyFence(dev, fence, NULL);
   vkDestroySemaphore(dev, bin, NULL);
   vkDestroySemaphore(dev, tl, NULL);
   vkFreeCommandBuffers(dev, pool0, 1, &cba);
   VkCommandBuffer cb1[2] = {cbb, cbc};
   vkFreeCommandBuffers(dev, pool1, 2, cb1);
   vkDestroyCommandPool(dev, pool0, NULL);
   vkDestroyCommandPool(dev, pool1, NULL);
   free_buf(&x);
   free_buf(&y);
   free_buf(&z);
   vkDestroyDevice(dev, NULL);

   /* 8. fault */
   if (do_fault) {
      const int real = !stub; /* the placeholder kernel only runs on the stub CP */
      if (fault_test(pd, real ? ngv_scale_bias_real_spv : ngv_scale_bias_spv,
                     real ? sizeof(ngv_scale_bias_real_spv) : sizeof(ngv_scale_bias_spv),
                     real ? "ngv_scale_bias" : "ngv_ph_scale_bias"))
         return 1;
   } else {
      printf("ngv_queues: fault section skipped (drm winsys; NGV_QUEUES_FAULT=1 runs it)\n");
   }

   vkDestroyInstance(inst, NULL);
   printf("ngv_queues: OK\n");
   return 0;
}
