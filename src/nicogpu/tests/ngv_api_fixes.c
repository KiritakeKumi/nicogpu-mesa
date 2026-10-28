/* SPDX-License-Identifier: MIT */
/* ngv_api_fixes.c -- regression test of the fixes (dEQP api.* / synchronization.* on the functional model).
 *
 *   gcc -O1 -g -o ngv_api_fixes ngv_api_fixes.c -lvulkan
 *   VK_DRIVER_FILES=<ngv icd json> NGV_WINSYS=hal|stub|drm NGV_NGCOMP_DIR=... ./ngv_api_fixes
 *
 * What it covers, each one a crash or a wrong result before :
 *   1  buffer sizes: UINT64_MAX and maxBufferSize + 1 either fail with OOM or report requirements >= size (was: success
 *      with a wrapped size)
 *   2  vkCreateBufferView / vkDestroyBufferView exist (was NULL: SIGSEGV)
 *   3  occlusion query pools can be created; PIPELINE_STATISTICS (feature off) fails cleanly
 *   4  vkCmdCopyBufferToImage / CopyImageToBuffer / CopyImage move the right bytes (was NULL: SIGSEGV): padded buffer
 *      rows, mip level, array layer, sub-rectangle, 3D sub-volume
 *   5  commands the device cannot execute (draw, blit, clear, begin query, ...) end the command buffer with
 *      VK_ERROR_FEATURE_NOT_PRESENT instead of calling a NULL pointer
 *   6  VK_ERROR_FORMAT_NOT_SUPPORTED comes with an all-zero VkImageFormatProperties from both query entry points
 * Exit status 0 = all checks passed. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define XT_NO_CHECK_MACROS /* CHECK exits, EXPECT counts failures here */
#include "ngv_test_ext.h"

static int failures;
#define CHECK(x)                                                              \
   do {                                                                       \
      VkResult r_ = (x);                                                      \
      if (r_ != VK_SUCCESS) {                                                 \
         fprintf(stderr, "FAIL %s:%d %s = %d\n", __FILE__, __LINE__, #x, r_); \
         exit(1);                                                             \
      }                                                                       \
   } while (0)
#define EXPECT(c, ...)                                                          \
   do {                                                                         \
      if (!(c)) {                                                               \
         fprintf(stderr, "FAIL %s:%d %s: ", __FILE__, __LINE__, #c);            \
         fprintf(stderr, "" __VA_ARGS__);                                       \
         fprintf(stderr, "\n");                                                 \
         failures++;                                                            \
      }                                                                         \
   } while (0)

static VkDevice dev;
static VkPhysicalDevice pd;
static VkPhysicalDeviceMemoryProperties mp;
static VkQueue queue;
static VkCommandPool pool;

static int
mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   return xt_mem_type(&mp, bits, want, 0);
}

struct buf {
   VkBuffer b;
   VkDeviceMemory m;
   uint8_t *map;
   VkDeviceSize size;
};

static struct buf
buf_new(VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct buf o = {0};
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
   CHECK(vkCreateBuffer(dev, &bci, NULL, &o.b));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, o.b, &req);
   int t = mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (t < 0) {
      fprintf(stderr, "no host visible coherent memory type\n");
      exit(1);
   }
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                               .memoryTypeIndex = (uint32_t)t};
   CHECK(vkAllocateMemory(dev, &mai, NULL, &o.m));
   CHECK(vkBindBufferMemory(dev, o.b, o.m, 0));
   void *p;
   CHECK(vkMapMemory(dev, o.m, 0, VK_WHOLE_SIZE, 0, &p));
   o.map = p;
   o.size = size;
   return o;
}

struct img {
   VkImage i;
   VkDeviceMemory m;
};

static struct img
img_new(VkImageType type, VkExtent3D ext, uint32_t levels, uint32_t layers)
{
   struct img o = {0};
   VkImageCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = type,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = ext, .mipLevels = levels,
                            .arrayLayers = layers, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
   CHECK(vkCreateImage(dev, &ici, NULL, &o.i));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, o.i, &req);
   int t = mem_type(req.memoryTypeBits, 0);
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                               .memoryTypeIndex = (uint32_t)t};
   CHECK(vkAllocateMemory(dev, &mai, NULL, &o.m));
   CHECK(vkBindImageMemory(dev, o.i, o.m, 0));
   return o;
}

static VkCommandBuffer
cb_begin(void)
{
   VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                     .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));
   VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cb, &bi));
   return cb;
}

static void
xfer_barrier(VkCommandBuffer cb)
{
   VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0,
                        NULL);
}

static void
cb_run(VkCommandBuffer cb)
{
   CHECK(vkEndCommandBuffer(cb));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence f;
   CHECK(vkCreateFence(dev, &fci, NULL, &f));
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
   CHECK(vkQueueSubmit(queue, 1, &si, f));
   CHECK(vkWaitForFences(dev, 1, &f, VK_TRUE, 60ull * 1000000000ull));
   vkDestroyFence(dev, f, NULL);
   vkFreeCommandBuffers(dev, pool, 1, &cb);
}

static VkImageSubresourceLayers
sub(uint32_t level, uint32_t layer, uint32_t layers)
{
   return (VkImageSubresourceLayers){VK_IMAGE_ASPECT_COLOR_BIT, level, layer, layers};
}

/* ---- 4: image transfers ---------------------------------------------------------------------------------- */

static void
test_image_transfers(void)
{
   const VkBufferUsageFlags bu = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
   struct buf src = buf_new(4096, bu), dst = buf_new(4096, bu);
   for (uint32_t i = 0; i < 4096; i++)
      src.map[i] = (uint8_t)(i * 7 + 3);
   memset(dst.map, 0xee, 4096);

   /* A: 8x6, 2 levels, 3 layers.  Level 0 layer 1 gets a padded buffer layout (row length 10, image height 8, offset
    * 16); level 1 layer 2 (4x3) a tight one; both are read back tight into dst at different offsets. */
   struct img a = img_new(VK_IMAGE_TYPE_2D, (VkExtent3D){8, 6, 1}, 2, 3);
   struct img b = img_new(VK_IMAGE_TYPE_2D, (VkExtent3D){8, 6, 1}, 1, 1);
   struct img v = img_new(VK_IMAGE_TYPE_3D, (VkExtent3D){4, 4, 4}, 1, 1);
   VkCommandBuffer cb = cb_begin();
   VkBufferImageCopy w0 = {.bufferOffset = 16, .bufferRowLength = 10, .bufferImageHeight = 8,
                           .imageSubresource = sub(0, 1, 1), .imageExtent = {8, 6, 1}};
   vkCmdCopyBufferToImage(cb, src.b, a.i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &w0);
   VkBufferImageCopy w1 = {.bufferOffset = 1024, .imageSubresource = sub(1, 2, 1), .imageExtent = {4, 3, 1}};
   vkCmdCopyBufferToImage(cb, src.b, a.i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &w1);
   VkBufferImageCopy w3 = {.bufferOffset = 2048, .imageSubresource = sub(0, 0, 1), .imageExtent = {4, 4, 4}};
   w3.imageSubresource.layerCount = 1;
   vkCmdCopyBufferToImage(cb, src.b, v.i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &w3);
   xfer_barrier(cb);
   /* read back: A L0/layer1 tight at dst+0, A L1/layer2 at dst+256, 3D sub-volume (1,1,1) 2x2x2 at dst+512 */
   VkBufferImageCopy r0 = {.bufferOffset = 0, .imageSubresource = sub(0, 1, 1), .imageExtent = {8, 6, 1}};
   vkCmdCopyImageToBuffer(cb, a.i, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.b, 1, &r0);
   VkBufferImageCopy r1 = {.bufferOffset = 256, .imageSubresource = sub(1, 2, 1), .imageExtent = {4, 3, 1}};
   vkCmdCopyImageToBuffer(cb, a.i, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.b, 1, &r1);
   VkBufferImageCopy r3 = {.bufferOffset = 512, .imageSubresource = sub(0, 0, 1), .imageOffset = {1, 1, 1},
                           .imageExtent = {2, 2, 2}};
   vkCmdCopyImageToBuffer(cb, v.i, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.b, 1, &r3);
   /* B: sub-rectangle (2,1) 4x4 of A L0/layer1 -> B at (1,2); read back B (1,2) 4x4 at dst+768 */
   VkImageCopy ic = {.srcSubresource = sub(0, 1, 1), .srcOffset = {2, 1, 0}, .dstSubresource = sub(0, 0, 1),
                     .dstOffset = {1, 2, 0}, .extent = {4, 4, 1}};
   vkCmdCopyImage(cb, a.i, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &ic);
   xfer_barrier(cb);
   VkBufferImageCopy r4 = {.bufferOffset = 768, .imageSubresource = sub(0, 0, 1), .imageOffset = {1, 2, 0},
                           .imageExtent = {4, 4, 1}};
   vkCmdCopyImageToBuffer(cb, b.i, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.b, 1, &r4);
   cb_run(cb);

   int bad = 0;
   for (uint32_t y = 0; y < 6; y++)
      for (uint32_t x = 0; x < 32; x++)
         if (dst.map[y * 32 + x] != src.map[16 + y * 40 + x])
            bad++;
   EXPECT(!bad, "2D level 0 layer 1 with padded rows: %d bytes differ", bad);
   bad = 0;
   for (uint32_t i = 0; i < 4 * 3 * 4; i++)
      if (dst.map[256 + i] != src.map[1024 + i])
         bad++;
   EXPECT(!bad, "2D level 1 layer 2: %d bytes differ", bad);
   bad = 0;
   for (uint32_t z = 0; z < 2; z++)
      for (uint32_t y = 0; y < 2; y++)
         for (uint32_t x = 0; x < 8; x++)
            if (dst.map[512 + z * 16 + y * 8 + x] != src.map[2048 + (z + 1) * 64 + (y + 1) * 16 + 4 + x])
               bad++;
   EXPECT(!bad, "3D sub-volume (1,1,1) 2x2x2: %d bytes differ", bad);
   bad = 0;
   for (uint32_t y = 0; y < 4; y++)
      for (uint32_t x = 0; x < 16; x++)
         if (dst.map[768 + y * 16 + x] != src.map[16 + (y + 1) * 40 + 2 * 4 + x])
            bad++;
   EXPECT(!bad, "vkCmdCopyImage sub-rectangle: %d bytes differ", bad);
   printf("image transfers: %s\n", failures ? "FAIL" : "PASS");
}

/* ---- 5: commands the device cannot execute ---------------------------------------------------------------- */

static void
expect_not_present(const char *what, VkCommandBuffer cb)
{
   VkResult r = vkEndCommandBuffer(cb);
   EXPECT(r == VK_ERROR_FEATURE_NOT_PRESENT, "%s: vkEndCommandBuffer = %d", what, r);
   vkFreeCommandBuffers(dev, pool, 1, &cb);
}

static void
test_unsupported_commands(VkQueryPool occlusion)
{
   struct buf bb = buf_new(4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
   struct img a = img_new(VK_IMAGE_TYPE_2D, (VkExtent3D){8, 6, 1}, 1, 1);
   struct img b = img_new(VK_IMAGE_TYPE_2D, (VkExtent3D){8, 6, 1}, 1, 1);
   VkCommandBuffer cb;
   cb = cb_begin();
   vkCmdDraw(cb, 3, 1, 0, 0);
   expect_not_present("vkCmdDraw", cb);
   cb = cb_begin();
   vkCmdDrawIndexed(cb, 3, 1, 0, 0, 0);
   expect_not_present("vkCmdDrawIndexed", cb);
   cb = cb_begin();
   vkCmdDrawIndirect(cb, bb.b, 0, 1, 16);
   expect_not_present("vkCmdDrawIndirect", cb);
   cb = cb_begin();
   vkCmdDrawIndexedIndirect(cb, bb.b, 0, 1, 20);
   expect_not_present("vkCmdDrawIndexedIndirect", cb);
   cb = cb_begin();
   VkDeviceSize off = 0;
   vkCmdBindVertexBuffers(cb, 0, 1, &bb.b, &off);
   expect_not_present("vkCmdBindVertexBuffers", cb);
   cb = cb_begin();
   vkCmdBindIndexBuffer(cb, bb.b, 0, VK_INDEX_TYPE_UINT16);
   expect_not_present("vkCmdBindIndexBuffer", cb);
   cb = cb_begin();
   VkImageBlit blit = {.srcSubresource = sub(0, 0, 1), .srcOffsets = {{0, 0, 0}, {8, 6, 1}},
                       .dstSubresource = sub(0, 0, 1), .dstOffsets = {{0, 0, 0}, {8, 6, 1}}};
   vkCmdBlitImage(cb, a.i, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                  VK_FILTER_NEAREST);
   expect_not_present("vkCmdBlitImage", cb);
   cb = cb_begin();
   VkImageResolve res = {.srcSubresource = sub(0, 0, 1), .dstSubresource = sub(0, 0, 1), .extent = {8, 6, 1}};
   vkCmdResolveImage(cb, a.i, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &res);
   expect_not_present("vkCmdResolveImage", cb);
   cb = cb_begin();
   VkClearColorValue cc = {{1, 0, 0, 1}};
   VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
   vkCmdClearColorImage(cb, a.i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cc, 1, &rng);
   expect_not_present("vkCmdClearColorImage", cb);
   cb = cb_begin();
   VkClearDepthStencilValue ds = {1.0f, 0};
   vkCmdClearDepthStencilImage(cb, a.i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &ds, 1, &rng);
   expect_not_present("vkCmdClearDepthStencilImage", cb);
   cb = cb_begin();
   VkClearAttachment ca = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT};
   VkClearRect cr = {.rect = {{0, 0}, {1, 1}}, .layerCount = 1};
   vkCmdClearAttachments(cb, 1, &ca, 1, &cr);
   expect_not_present("vkCmdClearAttachments", cb);
   cb = cb_begin();
   vkCmdBeginQuery(cb, occlusion, 0, 0);
   expect_not_present("vkCmdBeginQuery", cb);
   cb = cb_begin();
   vkCmdEndQuery(cb, occlusion, 0);
   expect_not_present("vkCmdEndQuery", cb);
   printf("unsupported commands: %s\n", failures ? "FAIL" : "PASS");
}

int
main(void)
{
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || n < 1) {
      fprintf(stderr, "no physical device\n");
      return 1;
   }
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
                                  .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = 0};
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));

   /* 1: buffer sizes */
   {
      VkPhysicalDeviceMaintenance3Properties m3 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
      VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m3};
      vkGetPhysicalDeviceProperties2(pd, &p2);
      const VkDeviceSize sizes[] = {UINT64_MAX, UINT64_MAX - 255, (VkDeviceSize)1 << 63, m3.maxMemoryAllocationSize + 1};
      for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
         VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizes[i],
                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
         VkBuffer b = VK_NULL_HANDLE;
         VkResult r = vkCreateBuffer(dev, &bci, NULL, &b);
         if (r == VK_SUCCESS) {
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(dev, b, &req);
            EXPECT(req.size >= sizes[i], "size %llu: requirements.size %llu", (unsigned long long)sizes[i],
                   (unsigned long long)req.size);
            vkDestroyBuffer(dev, b, NULL);
         } else {
            EXPECT(r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY, "size %llu: %d",
                   (unsigned long long)sizes[i], r);
         }
      }
      printf("buffer sizes: %s\n", failures ? "FAIL" : "PASS");
   }

   /* 2: buffer views */
   {
      struct buf tb = buf_new(4096, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT);
      VkBufferViewCreateInfo bvi = {.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = tb.b,
                                    .format = VK_FORMAT_R8G8B8A8_UNORM, .offset = 0, .range = VK_WHOLE_SIZE};
      VkBufferView bv = VK_NULL_HANDLE;
      CHECK(vkCreateBufferView(dev, &bvi, NULL, &bv));
      EXPECT(bv != VK_NULL_HANDLE, "buffer view handle");
      vkDestroyBufferView(dev, bv, NULL);
      vkDestroyBufferView(dev, VK_NULL_HANDLE, NULL);
      printf("buffer views: %s\n", failures ? "FAIL" : "PASS");
   }

   /* 3: query pools */
   VkQueryPool occlusion = VK_NULL_HANDLE;
   {
      VkQueryPoolCreateInfo qpi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                   .queryType = VK_QUERY_TYPE_OCCLUSION, .queryCount = 1};
      CHECK(vkCreateQueryPool(dev, &qpi, NULL, &occlusion));
      qpi.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
      qpi.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
      VkQueryPool ps = VK_NULL_HANDLE;
      VkResult r = vkCreateQueryPool(dev, &qpi, NULL, &ps);
      EXPECT(r != VK_SUCCESS, "pipeline statistics pool created although the feature is off");
      if (r == VK_SUCCESS)
         vkDestroyQueryPool(dev, ps, NULL);
      printf("query pools: %s\n", failures ? "FAIL" : "PASS");
   }

   test_image_transfers();
   test_unsupported_commands(occlusion);
   vkDestroyQueryPool(dev, occlusion, NULL);

   /* 6: format not supported -> zeroed properties */
   {
      VkImageFormatProperties p1;
      memset(&p1, 0xcd, sizeof(p1));
      VkResult r1 = vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_A2R10G10B10_SNORM_PACK32,
                                                              VK_IMAGE_TYPE_1D, VK_IMAGE_TILING_LINEAR,
                                                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, &p1);
      VkPhysicalDeviceImageFormatInfo2 fi = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                                             .format = VK_FORMAT_A2R10G10B10_SNORM_PACK32,
                                             .type = VK_IMAGE_TYPE_1D, .tiling = VK_IMAGE_TILING_LINEAR,
                                             .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
      VkImageFormatProperties2 p2;
      memset(&p2, 0xcd, sizeof(p2));
      p2.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
      p2.pNext = NULL;
      VkResult r2 = vkGetPhysicalDeviceImageFormatProperties2(pd, &fi, &p2);
      if (r1 == VK_ERROR_FORMAT_NOT_SUPPORTED) {
         VkImageFormatProperties z = {0};
         EXPECT(!memcmp(&p1, &z, sizeof(z)), "vkGetPhysicalDeviceImageFormatProperties not zero");
      }
      if (r2 == VK_ERROR_FORMAT_NOT_SUPPORTED) {
         VkImageFormatProperties z = {0};
         EXPECT(!memcmp(&p2.imageFormatProperties, &z, sizeof(z)), "vkGetPhysicalDeviceImageFormatProperties2 not zero");
      }
      EXPECT(r1 == r2, "the two queries disagree: %d vs %d", r1, r2);
      printf("unsupported format properties: %s\n", failures ? "FAIL" : "PASS");
   }

   vkDestroyCommandPool(dev, pool, NULL);
   printf("NGV_API_FIXES: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
   return failures ? 1 : 0;
}
