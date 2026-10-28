/* SPDX-License-Identifier: MIT */
/* ngv_shaders.c -- shader-coverage survey of the ngv driver on the functional model (ported from the earlier
 * parallel attempt).  Needs the real compile path (SPIR-V -> NIR -> ngcomp), so it runs with NGV_WINSYS=hal (functional
 * model) or, in hardware mode, with NGV_WINSYS=drm / auto / unset (real device via the DRM render node).  With any
 * other winsys (stub) it prints SKIP and exits 0.  Expected values are host-computed in both modes.
 * Hardware mode: NGV_HW_SKIP=a,b skips kernels (default "float,intops,atomic,scratch,...": SFU needed / not yet run on
 * the board, see main()), NGV_HW_TIMEOUT_MS sets the
 * per-kernel fence timeout (default 10000).  A fault / device loss / timeout is a FAIL, never an automatic SKIP.
 *
 * ngv_shaders <dir with ngv_{add,shared,loop,float,ubo2d,ids,intops,atomic,scratch}.spv plus the scratch / atomic ones:
 * ngv_{scratch_vec,scratch_sub,scratch_oob,scratch_big,atomic_ops,atomic_shared}.spv, ngv_half.spv,
 * ngv_numwg.spv, ngv_subgroup.spv / ngv_subgroup_shuffle.spv, ngv_half16.spv / ngv_int16.spv, ngv_sfu.spv, ngv_wide.spv>
 * NGV_EXTRA=1 also runs the opt-in cases (scratch_oob: GLSL-undefined out-of-bounds probe).
 *
 * Every case dispatches one GLSL shader (test/shaders/ngv_*.comp) on prepared buffers and compares every element:
 *   PASS    pipeline compiled, results match
 *   FAIL    wrong result, or the pipeline did not compile although the case is not marked as a known gap
 * SKIP known ngcomp/driver gap (section 14.4, 15): pipeline creation failed as expected. The
 *           shader and the check stay in place so the case turns on by itself when the gap is closed
 *   XPASS   a known-gap case compiled and is correct: update the table (the gap is closed)
 *   XFAIL!  a known-gap case compiled but computes the wrong result: that is a bug, counted as FAIL
 * Exit status 0 = no FAIL. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define XT_NO_CHECK_MACROS /* CHECK returns CASE_FAIL here */
#include "ngv_test_ext.h"

#define CHECK(x)                                                              \
   do {                                                                       \
      VkResult r_ = (x);                                                      \
      if (r_ != VK_SUCCESS) {                                                 \
         fprintf(stderr, "FAIL %s:%d %s = %d\n", __FILE__, __LINE__, #x, r_); \
         return CASE_FAIL;                                                    \
      }                                                                       \
   } while (0)

enum { CASE_PASS = 0, CASE_FAIL = 1, CASE_NOCOMPILE = 2, CASE_UNSUPPORTED = 3 /* device does not report it */ };

static VkDevice dev;
static VkQueue queue;
static VkPhysicalDeviceMemoryProperties mp;
static VkCommandPool pool;
static uint64_t fence_timeout_ns = 120ull * 1000000000ull; /* hal; hardware mode lowers it (NGV_HW_TIMEOUT_MS) */

struct buf {
   VkBuffer buffer;
   VkDeviceMemory mem;
   uint32_t *map;
};

static int
find_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   return xt_mem_type(&mp, bits, want, 0);
}

static VkResult
make_buffer(struct buf *b, VkDeviceSize size, VkBufferUsageFlags usage)
{
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
   VkResult r = vkCreateBuffer(dev, &bci, NULL, &b->buffer);
   if (r != VK_SUCCESS)
      return r;
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, b->buffer, &req);
   int t = find_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (t < 0)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                               .memoryTypeIndex = t};
   r = vkAllocateMemory(dev, &mai, NULL, &b->mem);
   if (r != VK_SUCCESS)
      return r;
   r = vkBindBufferMemory(dev, b->buffer, b->mem, 0);
   if (r != VK_SUCCESS)
      return r;
   return vkMapMemory(dev, b->mem, 0, VK_WHOLE_SIZE, 0, (void **)&b->map);
}

static void
free_buf(struct buf *b)
{
   vkDestroyBuffer(dev, b->buffer, NULL);
   vkFreeMemory(dev, b->mem, NULL);
}

static VkResult
make_pipeline(const char *spv_path, uint32_t push_size, VkDescriptorSetLayout dsl, VkPipelineLayout *pl, VkPipeline *pipe)
{
   size_t size;
   uint32_t *code = xt_load_spv(spv_path, &size);
   if (!code) {
      fprintf(stderr, "cannot read %s\n", spv_path);
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
   VkShaderModule sm;
   VkResult r = vkCreateShaderModule(dev, &smi, NULL, &sm);
   free(code);
   if (r != VK_SUCCESS)
      return r;
   VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_size};
   VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &dsl, .pushConstantRangeCount = push_size ? 1 : 0,
                                     .pPushConstantRanges = &pcr};
   r = vkCreatePipelineLayout(dev, &pli, NULL, pl);
   if (r != VK_SUCCESS) {
      vkDestroyShaderModule(dev, sm, NULL);
      return r;
   }
   VkComputePipelineCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm, .pName = "main"},
      .layout = *pl,
   };
   r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, pipe);
   vkDestroyShaderModule(dev, sm, NULL);
   if (r != VK_SUCCESS)
      vkDestroyPipelineLayout(dev, *pl, NULL);
   return r;
}

static VkDescriptorSetLayout
make_dsl(uint32_t nbind, const VkDescriptorType *types)
{
 VkDescriptorSetLayoutBinding b[8]; /* : up to 8 buffers (the pool has 8) */
   for (uint32_t i = 0; i < nbind; i++)
      b[i] = (VkDescriptorSetLayoutBinding){.binding = i,
                                            .descriptorType = types ? types[i] : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                         .bindingCount = nbind, .pBindings = b};
   VkDescriptorSetLayout l = VK_NULL_HANDLE;
   if (vkCreateDescriptorSetLayout(dev, &ci, NULL, &l) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return l;
}

static VkResult
make_set(VkDescriptorSetLayout dsl, VkDescriptorPool *dp, VkDescriptorSet *set, struct buf **bufs, uint32_t n,
         const VkDescriptorType *types)
{
   VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8}};
   VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2,
                                     .poolSizeCount = 2, .pPoolSizes = ps};
   VkResult r = vkCreateDescriptorPool(dev, &dpi, NULL, dp);
   if (r != VK_SUCCESS)
      return r;
   VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = *dp,
                                      .descriptorSetCount = 1, .pSetLayouts = &dsl};
   r = vkAllocateDescriptorSets(dev, &dai, set);
   if (r != VK_SUCCESS)
      return r;
   VkDescriptorBufferInfo bi[8];
   VkWriteDescriptorSet w[8];
   for (uint32_t i = 0; i < n; i++) {
      bi[i] = (VkDescriptorBufferInfo){.buffer = bufs[i]->buffer, .offset = 0, .range = VK_WHOLE_SIZE};
      w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = *set, .dstBinding = i,
                                    .descriptorCount = 1,
                                    .descriptorType = types ? types[i] : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    .pBufferInfo = &bi[i]};
   }
   vkUpdateDescriptorSets(dev, n, w, 0, NULL);
   return VK_SUCCESS;
}

/* When set, run_one dispatches with vkCmdDispatchIndirect(run_indirect->buffer, run_indirect_off) instead of the
 * grid arguments (numwg case). */
static struct buf *run_indirect;
static VkDeviceSize run_indirect_off;

/* Compile + run one shader on prepared buffers.  Returns CASE_PASS (ran), CASE_NOCOMPILE (pipeline creation failed:
 * the driver logged the reason on stderr) or CASE_FAIL (any other Vulkan error). */
static int
run_one(const char *spv, uint32_t nb, const VkDescriptorType *types, struct buf **bufs, const void *push,
        uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz)
{
   VkDescriptorSetLayout dsl = make_dsl(nb, types);
   if (dsl == VK_NULL_HANDLE)
      return CASE_FAIL;
   VkPipelineLayout pl;
   VkPipeline pipe;
   VkResult r = make_pipeline(spv, push_size, dsl, &pl, &pipe);
   if (r != VK_SUCCESS) {
      vkDestroyDescriptorSetLayout(dev, dsl, NULL);
      return CASE_NOCOMPILE;
   }
   VkDescriptorPool dp;
   VkDescriptorSet set;
   CHECK(make_set(dsl, &dp, &set, bufs, nb, types));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
   VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cb, &bi));
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
   if (push_size)
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
   if (run_indirect)
      vkCmdDispatchIndirect(cb, run_indirect->buffer, run_indirect_off);
   else
      vkCmdDispatch(cb, gx, gy, gz);
   /* make the shader writes visible to the host (compute -> host) */
   VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                         .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cb));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
   CHECK(vkQueueSubmit(queue, 1, &si, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, fence_timeout_ns));
   vkDestroyFence(dev, fence, NULL);
   vkFreeCommandBuffers(dev, pool, 1, &cb);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   return CASE_PASS;
}

#define SSBO VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

/* ---- cases: return CASE_PASS (ran, all elements match), CASE_FAIL (ran, mismatches), CASE_NOCOMPILE -------------- */

static int
case_add(const char *spv)
{
   enum { N = 1000, CAP = 1024 }; /* N not a multiple of 64: the last workgroup is partially masked */
   struct buf a, b, c;
   CHECK(make_buffer(&a, CAP * 4, SSBO));
   CHECK(make_buffer(&b, CAP * 4, SSBO));
   CHECK(make_buffer(&c, CAP * 4, SSBO));
   for (uint32_t i = 0; i < CAP; i++) {
      a.map[i] = i * 2654435761u + 7;
      b.map[i] = 0x10000u * i + 3 * i;
      c.map[i] = 0xdeadbeefu;
   }
   struct buf *bufs[3] = {&a, &b, &c};
   uint32_t n = N;
   int rc = run_one(spv, 3, NULL, bufs, &n, 4, (N + 63) / 64, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t i = 0; i < CAP; i++) {
      uint32_t want = i < N ? a.map[i] + b.map[i] : 0xdeadbeefu;
      if (c.map[i] != want && bad++ < 8)
         fprintf(stderr, "c[%u] = 0x%08x, want 0x%08x\n", i, c.map[i], want);
   }
   printf("  %u elements checked (n=%u), %u mismatches\n", (unsigned)CAP, (unsigned)N, bad);
   free_buf(&a);
   free_buf(&b);
   free_buf(&c);
   return bad ? CASE_FAIL : CASE_PASS;
}

static int
case_shared(const char *spv)
{
   enum { WG = 4, COUNT = WG * 64 };
   struct buf in, out;
   CHECK(make_buffer(&in, COUNT * 4, SSBO));
   CHECK(make_buffer(&out, COUNT * 4, SSBO));
   for (uint32_t i = 0; i < COUNT; i++) {
      in.map[i] = 1000 + i;
      out.map[i] = 0xaaaaaaaau;
   }
   struct buf *bufs[2] = {&in, &out};
   int rc = run_one(spv, 2, NULL, bufs, NULL, 0, WG, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t g = 0; g < COUNT; g++) {
      uint32_t wg = g / 64, l = g % 64;
      uint32_t want = in.map[wg * 64 + 63 - l] * 3u + wg;
      if (out.map[g] != want && bad++ < 8)
         fprintf(stderr, "out[%u] = 0x%08x, want 0x%08x\n", g, out.map[g], want);
   }
   printf("  %u elements checked, %u mismatches\n", (unsigned)COUNT, bad);
   free_buf(&in);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

static int
case_loop(const char *spv)
{
   enum { COUNT = 100, ITERS = 7 };
   struct buf in, out;
   CHECK(make_buffer(&in, COUNT * 16 * 4, SSBO));
   CHECK(make_buffer(&out, 128 * 4, SSBO));
   for (uint32_t i = 0; i < COUNT * 16; i++)
      in.map[i] = i * 2246822519u + 11;
   for (uint32_t i = 0; i < 128; i++)
      out.map[i] = 0xcccccccau;
   struct buf *bufs[2] = {&in, &out};
   uint32_t push[2] = {ITERS, COUNT};
   int rc = run_one(spv, 2, NULL, bufs, push, 8, 4, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t i = 0; i < 128; i++) {
      uint32_t want = 0xcccccccau;
      if (i < COUNT) {
         want = 0;
         for (uint32_t k = 0; k < ITERS; k++)
            want += in.map[i * 16 + k] * (k + 1);
      }
      if (out.map[i] != want && bad++ < 4)
         fprintf(stderr, "loop out[%u] = 0x%08x, want 0x%08x\n", i, out.map[i], want);
   }
   printf("  128 elements checked, %u mismatches\n", bad);
   free_buf(&in);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

static int
case_float(const char *spv)
{
   enum { COUNT = 128 };
   struct buf in, out;
   CHECK(make_buffer(&in, COUNT * 4, SSBO));
   CHECK(make_buffer(&out, COUNT * 4, SSBO));
   float *fin = (float *)in.map, *fout = (float *)out.map;
   for (uint32_t i = 0; i < COUNT; i++) {
      fin[i] = ((int)i - 64) * 0.37f + (i % 5) * 0.125f;
      fout[i] = 0;
   }
   struct buf *bufs[2] = {&in, &out};
   int rc = run_one(spv, 2, NULL, bufs, NULL, 0, 2, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t i = 0; i < COUNT; i++) {
      float x = fin[i];
      float t = fmaf(x, 0.5f, sqrtf(fabsf(x)));
      t = t + floorf(x);
      t = t + fmaxf(x, 1.5f);
      float want = t - fminf(x, -2.0f);
      float err = fabsf(fout[i] - want);
      if (!(err <= 1e-5f * fmaxf(1.0f, fabsf(want))) && bad++ < 4)
         fprintf(stderr, "float out[%u] = %.8g, want %.8g\n", i, fout[i], want);
   }
   printf("  %u elements checked (rel. error 1e-5), %u mismatches\n", (unsigned)COUNT, bad);
   free_buf(&in);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

static int
case_ubo2d(const char *spv)
{
   struct buf u, in, out;
   CHECK(make_buffer(&u, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
   CHECK(make_buffer(&in, 256 * 16, SSBO));
   CHECK(make_buffer(&out, 256 * 16, SSBO));
   float *fu = (float *)u.map, *fin = (float *)in.map, *fout = (float *)out.map;
   const float scale[4] = {2.0f, -0.5f, 3.25f, 1.0f};
   const uint32_t bias[4] = {5, 100, 0, 7};
   memcpy(fu, scale, 16);
   memcpy(fu + 4, bias, 16);
   for (uint32_t i = 0; i < 1024; i++) {
      fin[i] = (float)(i % 37) * 0.75f - 9.0f;
      fout[i] = -1.0f;
   }
   struct buf *bufs[3] = {&u, &in, &out};
   const VkDescriptorType types[3] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
   int rc = run_one(spv, 3, types, bufs, NULL, 0, 2, 2, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t i = 0; i < 1024; i++) {
      volatile float m = fin[i] * scale[i % 4];
      float want = m + (float)bias[i % 4];
      if (fout[i] != want && bad++ < 4)
         fprintf(stderr, "ubo2d out[%u] = %.8g, want %.8g\n", i, fout[i], want);
   }
   printf("  1024 floats checked (bit exact), %u mismatches\n", bad);
   free_buf(&u);
   free_buf(&in);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

static int
case_atomic(const char *spv)
{
   struct buf cnt;
   CHECK(make_buffer(&cnt, 64, SSBO));
   cnt.map[0] = 0;
   struct buf *bufs[1] = {&cnt};
   int rc = run_one(spv, 1, NULL, bufs, NULL, 0, 3, 1, 1);
   if (rc)
      return rc;
   printf("  counter = %u (want 192)\n", cnt.map[0]);
   unsigned ok = cnt.map[0] == 192;
   free_buf(&cnt);
   return ok ? CASE_PASS : CASE_FAIL;
}

static int
case_ids(const char *spv)
{
   enum { GX = 3, GY = 2, PER = 64, COUNT = GX * GY * PER };
   struct buf out;
   CHECK(make_buffer(&out, COUNT * 16, SSBO));
   for (uint32_t i = 0; i < COUNT * 4; i++)
      out.map[i] = 0xfefefefeu;
   struct buf *bufs[1] = {&out};
   int rc = run_one(spv, 1, NULL, bufs, NULL, 0, GX, GY, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t wy = 0; wy < GY; wy++)
      for (uint32_t wx = 0; wx < GX; wx++)
         for (uint32_t l = 0; l < PER; l++) {
            uint32_t g = (wy * GX + wx) * PER + l;
            uint32_t want[4] = {(l % 16) | ((l / 16) << 8), wx | (wy << 8),
                                (wx * 16 + (l % 16)) | ((wy * 4 + (l / 16)) << 16), 16 * 100 + 4};
            if (memcmp(&out.map[g * 4], want, 16) && bad++ < 4)
               fprintf(stderr, "ids[%u] = %x %x %x %x, want %x %x %x %x\n", g, out.map[g * 4], out.map[g * 4 + 1],
                       out.map[g * 4 + 2], out.map[g * 4 + 3], want[0], want[1], want[2], want[3]);
         }
   printf("  %u invocations checked, %u mismatches\n", (unsigned)COUNT, bad);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* : gl_NumWorkGroups. One direct dispatch (3 x 2 x 2 groups), then one indirect dispatch whose
 * VkDispatchIndirectCommand {2, 3, 1} sits at byte offset 12 of the parameter buffer (4 B aligned only, garbage
 * in front of it, as in dEQP-VK.compute.pipeline.indirect_dispatch.*).  dst[wg * 16 + lid] = {n.x, n.y, n.z, wg}. */
static unsigned
check_numwg(const struct buf *out, const uint32_t n[3], uint32_t total, const char *what)
{
   const uint32_t groups = n[0] * n[1] * n[2];
   unsigned bad = 0;
   for (uint32_t e = 0; e < total; e++) {
      const uint32_t wg = e / 16;
      uint32_t want[4] = {n[0], n[1], n[2], wg};
      if (wg >= groups)
         want[0] = want[1] = want[2] = want[3] = 0xfefefefeu; /* beyond the grid: untouched */
      if (memcmp(&out->map[e * 4], want, 16) && bad++ < 4)
         fprintf(stderr, "numwg %s [%u] = %u %u %u %u, want %u %u %u %u\n", what, e, out->map[e * 4],
                 out->map[e * 4 + 1], out->map[e * 4 + 2], out->map[e * 4 + 3], want[0], want[1], want[2], want[3]);
   }
   return bad;
}

static int
case_numwg(const char *spv)
{
   enum { MAXWG = 12, COUNT = MAXWG * 16 };
   static const uint32_t direct[3] = {3, 2, 2}, indirect[3] = {2, 3, 1};
   struct buf out, args;
   CHECK(make_buffer(&out, COUNT * 16, SSBO));
   CHECK(make_buffer(&args, 64, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT));
   for (uint32_t i = 0; i < COUNT * 4; i++)
      out.map[i] = 0xfefefefeu;
   struct buf *bufs[1] = {&out};
   int rc = run_one(spv, 1, NULL, bufs, NULL, 0, direct[0], direct[1], direct[2]);
   if (rc) {
      free_buf(&args);
      free_buf(&out);
      return rc;
   }
   unsigned bad = check_numwg(&out, direct, COUNT, "direct");
   for (uint32_t i = 0; i < COUNT * 4; i++)
      out.map[i] = 0xfefefefeu;
   for (uint32_t i = 0; i < 16; i++)
      args.map[i] = 0x7777u + i; /* would be a huge grid if the driver read the wrong offset */
   memcpy(&args.map[3], indirect, sizeof(indirect));
   run_indirect = &args;
   run_indirect_off = 12;
   rc = run_one(spv, 1, NULL, bufs, NULL, 0, 0, 0, 0);
   run_indirect = NULL;
   if (!rc)
      bad += check_numwg(&out, indirect, COUNT, "indirect");
   printf("  direct %ux%ux%u + indirect %ux%ux%u (offset 12) checked, %u mismatches\n", direct[0], direct[1],
          direct[2], indirect[0], indirect[1], indirect[2], bad);
   free_buf(&args);
   free_buf(&out);
   if (rc)
      return rc;
   return bad ? CASE_FAIL : CASE_PASS;
}

/* : reference IEEE binary16 conversions (round to nearest even; NaN -> 0x7e00 with the sign; f32 denormals -> 0) */
static uint16_t
ref_f2h(uint32_t x)
{
   uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
   if (e == 255)
      return (uint16_t)(sign | (m ? 0x7e00 : 0x7c00));
   if (e == 0)
      return (uint16_t)sign;
   int he = (int)e - 127 + 15;
   if (he >= 31)
      return (uint16_t)(sign | 0x7c00);
   if (he <= 0) { /* half denormal: value / 2^-24 = mant >> (126 - e) */
      uint32_t shift = 126 - e, mant = m | 0x800000;
      if (shift > 24)
         return (uint16_t)sign;
      uint32_t q = mant >> shift, rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
      if (rem > half || (rem == half && (q & 1)))
         q++;
      return (uint16_t)(sign | q);
   }
   uint32_t q = ((uint32_t)he << 10) | (m >> 13), rem = m & 0x1fff;
   if (rem > 0x1000 || (rem == 0x1000 && (q & 1)))
      q++; /* a carry into the exponent is the right result, up to inf */
   return (uint16_t)(sign | q);
}

static int
f32_is_nan(uint32_t b)
{
   return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu);
}

static uint32_t
ref_h2f(uint16_t h)
{
   uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
   if (e == 0) {
      float v = (float)m / 16777216.0f;
      uint32_t b;
      memcpy(&b, &v, 4);
      return b | s;
   }
   if (e == 31)
      return s | 0x7f800000 | (m << 13);
   return s | ((e + 112) << 23) | (m << 13);
}

static int
case_half(const char *spv)
{
   enum { COUNT = 128 };
   static const uint32_t special[] = {
      0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x477fe000, 0x477ff000, 0x477fefff, 0x47800000, /* 65504, ties */
      0x38800000, 0x387fc000, 0x33800000, 0x33000000, 0x33000001, 0x337fffff, 0x00000001, 0x80400000, /* denormals */
      0x3f801000, 0x3f803000, 0x3f800fff, 0x3f801001, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, /* RTE, inf, nan */
      0x3a800000, 0x387ff000, 0x38000000, 0x37800000, 0xc77ff000, 0x4b000000, 0x3eaaaaab, 0x40490fdb,
   };
   struct buf f, h, o;
   CHECK(make_buffer(&f, COUNT * 8, SSBO));
   CHECK(make_buffer(&h, COUNT * 4, SSBO));
   CHECK(make_buffer(&o, COUNT * 12, SSBO));
   uint32_t *uf = f.map, *uh = h.map, *uo = o.map;
   for (uint32_t i = 0; i < 2 * COUNT; i++) {
      uint32_t r = i * 2654435761u + 0x9e3779b9u;
      r ^= r >> 13;
      uf[i] = i < sizeof(special) / 4 ? special[i] : (r & 0x8fffffffu) + 0x30000000u; /* mostly in half range */
   }
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t r = i * 0x85ebca6bu + 0xc2b2ae35u;
      r ^= r >> 16;
      uh[i] = i < 32 ? (special[i] >> 16) | ((uint32_t)ref_f2h(special[i]) << 16) : r;
   }
   uh[0] = 0x00017c00; /* smallest denormal | +inf */
   uh[1] = 0x7e01fc00; /* nan with payload | -inf */
   uh[2] = 0x03ff8400; /* largest denormal | -smallest normal */
   for (uint32_t i = 0; i < COUNT * 3; i++)
      uo[i] = 0xdeadbeef;
   struct buf *bufs[3] = {&f, &h, &o};
   int rc = run_one(spv, 3, NULL, bufs, NULL, 0, COUNT / 64, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t want[3] = {(uint32_t)ref_f2h(uf[2 * i]) | ((uint32_t)ref_f2h(uf[2 * i + 1]) << 16),
                          ref_h2f((uint16_t)uh[i]), ref_h2f((uint16_t)(uh[i] >> 16))};
      uint32_t got[3] = {uo[2 * i], uo[2 * i + 1], uo[2 * COUNT + i]};
      static const char *what[3] = {"packHalf2x16", "unpackHalf2x16.x", "unpackHalf2x16.y"};
      for (int k = 0; k < 3; k++)
 /* an unpacked NaN may be any NaN (Vulkan keeps no payload; 's hardware gives the canonical 0x7fc00000) */
         if (got[k] != want[k] && !(k && f32_is_nan(want[k]) && f32_is_nan(got[k])) && bad++ < 8)
            fprintf(stderr, "  [%u] %s(%08x %08x): got %08x want %08x\n", i, what[k], k ? uh[i] : uf[2 * i],
                    k ? 0 : uf[2 * i + 1], got[k], want[k]);
   }
   printf("  %u packs + %u unpacks checked, %u mismatches\n", COUNT, 2 * COUNT, bad);
   free_buf(&f);
   free_buf(&h);
   free_buf(&o);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* ---- : 16-bit types. The device must report storageBuffer16BitAccess + shaderInt16 (+ shaderFloat16), main
 * enables them when it does; without them the cases report "not supported". */
static int have_16bit_storage, have_int16, have_float16;

static float
h2f(uint16_t h)
{
   uint32_t b = ref_h2f(h);
   float f;
   memcpy(&f, &b, 4);
   return f;
}

static uint16_t
f2h(float f)
{
   uint32_t b;
   memcpy(&b, &f, 4);
   return ref_f2h(b);
}

static int
h_is_nan(uint16_t h)
{
   return (h & 0x7c00) == 0x7c00 && (h & 0x3ff);
}

/* equal as the Vulkan spec lets the result differ: any NaN is one value, +0 == -0 */
static int
h_same(uint16_t got, uint16_t want)
{
   return got == want || (h_is_nan(got) && h_is_nan(want)) || (((got | want) & 0x7fff) == 0);
}

static int
case_half16(const char *spv)
{
   if (!have_16bit_storage || !have_float16 || !have_int16)
      return CASE_UNSUPPORTED;
   enum { COUNT = 128, K = 14 };
   static const uint16_t special[] = {0x0000, 0x8000, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x0001, 0x03ff, 0x0400, 0x8001,
                                      0x7c00, 0xfc00, 0x7e00, 0x3555, 0x4248, 0x5640};
   struct buf a, b, o, p;
   CHECK(make_buffer(&a, COUNT * 2, SSBO));
   CHECK(make_buffer(&b, COUNT * 2, SSBO));
   CHECK(make_buffer(&o, K * COUNT * 2, SSBO));
   CHECK(make_buffer(&p, COUNT * 8, SSBO));
   uint16_t *ha = (uint16_t *)a.map, *hb = (uint16_t *)b.map, *ho = (uint16_t *)o.map, *hp = (uint16_t *)p.map;
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t r = i * 2654435761u + 0x9e3779b9u, s = i * 0x85ebca6bu + 0xc2b2ae35u;
      r ^= r >> 15;
      s ^= s >> 13;
      /* mostly 2^-8 .. 2^8, a few of any magnitude */
      ha[i] = i < 16 ? special[i] : (uint16_t)(((r & 1) << 15) | ((7 + (r >> 4) % 16) << 10) | ((r >> 8) & 0x3ff));
      hb[i] = i < 16 ? special[(i * 5 + 3) & 15] : (uint16_t)(((s & 1) << 15) | ((7 + (s >> 4) % 16) << 10) | ((s >> 8) & 0x3ff));
      if ((i & 31) == 20)
         hb[i] = ha[i]; /* x == y */
   }
   memset(o.map, 0xaa, K * COUNT * 2);
   memset(p.map, 0xaa, COUNT * 8);
   struct buf *bufs[4] = {&a, &b, &o, &p};
   int rc = run_one(spv, 4, NULL, bufs, NULL, 0, COUNT / 64, 1, 1);
   if (rc)
      return rc;
   static const char *what[K] = {"x+y", "x-y", "x*y", "fma", "min", "max", "-x", "abs", "x/y", "sqrt|x|", "x*3",
                                 "i16->f16", "u16->f16", "f16->i16->f16"};
   unsigned bad = 0, checked = 0;
   for (uint32_t i = 0; i < COUNT; i++) {
      const uint16_t X = ha[i], Z = ha[(i + 1) % COUNT], Y = hb[i];
      const float x = h2f(X), y = h2f(Y), z = h2f(Z);
      const int nan_in = h_is_nan(X) || h_is_nan(Y) || h_is_nan(Z);
      const int inr = !(h_is_nan(X) || (X & 0x7fff) >= 0x7800); /* |x| < 32768, finite: f16 -> int16 is defined */
      uint16_t want[K];
      want[0] = f2h(x + y);
      want[1] = f2h(x - y);
      want[2] = f2h(x * y);
      want[3] = f2h(fmaf(x, y, z));
      want[4] = f2h(x < y ? x : y);
      want[5] = f2h(x > y ? x : y);
      want[6] = X ^ 0x8000;
      want[7] = X & 0x7fff;
      want[8] = f2h(x / y);
      want[9] = f2h(sqrtf(fabsf(x)));
      want[10] = f2h(x * 3.0f);
      want[11] = f2h((float)(int16_t)Y);
      want[12] = f2h((float)(uint16_t)Y);
      want[13] = inr ? f2h((float)(int16_t)truncf(x)) : 0;
      for (int k = 0; k < K; k++) {
         if ((k == 4 || k == 5) && nan_in)
            continue; /* min / max with a NaN operand: unspecified */
         if (k == 13 && !inr)
            continue;
         checked++;
         if (!h_same(ho[k * COUNT + i], want[k]) && bad++ < 12)
            fprintf(stderr, "  half16 [%u] %s(x=%04x y=%04x z=%04x): got %04x want %04x\n", i, what[k], X, Y, Z,
                    ho[k * COUNT + i], want[k]);
      }
      uint16_t wp[4] = {(uint16_t)(x < y), f2h(x * y),
                        (uint16_t)((h_is_nan(X) ? 1 : 0) | (((X & 0x7fff) == 0x7c00) ? 2 : 0) | (x == y ? 4 : 0)),
                        inr ? (uint16_t)(int16_t)truncf(x) : hp[i * 4 + 3]};
      for (int k = 0; k < 4; k++) {
         checked++;
         if (!(k == 1 ? h_same(hp[i * 4 + k], wp[k]) : hp[i * 4 + k] == wp[k]) && bad++ < 12)
            fprintf(stderr, "  half16 [%u] p.%d(x=%04x y=%04x): got %04x want %04x\n", i, k, X, Y, hp[i * 4 + k], wp[k]);
      }
   }
   printf("  %u values checked, %u mismatches\n", checked, bad);
   free_buf(&a);
   free_buf(&b);
   free_buf(&o);
   free_buf(&p);
   return bad ? CASE_FAIL : CASE_PASS;
}

static int
case_int16(const char *spv)
{
   if (!have_16bit_storage || !have_int16)
      return CASE_UNSUPPORTED;
   enum { COUNT = 128, K = 19, KU = 9 };
   struct buf x, y, o, u;
   CHECK(make_buffer(&x, COUNT * 2, SSBO));
   CHECK(make_buffer(&y, COUNT * 2, SSBO));
   CHECK(make_buffer(&o, K * COUNT * 2, SSBO));
   CHECK(make_buffer(&u, KU * COUNT * 4, SSBO));
   int16_t *ix = (int16_t *)x.map, *iy = (int16_t *)y.map;
   uint16_t *io = (uint16_t *)o.map;
   uint32_t *iu = u.map;
   static const int16_t special[] = {0, 1, -1, 2, -2, 32767, -32767, 255, 256, -256, 0x7f00, 0x00ff, 12345, -12345};
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t r = i * 2654435761u + 0x9e3779b9u, s = i * 0x85ebca6bu + 0xc2b2ae35u;
      r ^= r >> 15;
      s ^= s >> 13;
      int32_t a = i < 14 ? special[i] : (int32_t)(r % 65535u) - 32767;
      int32_t b = i < 14 ? special[(i * 3 + 1) % 14] : (int32_t)(s % 65535u) - 32767;
      if (i % 4 == 1)
         a = (int32_t)(r % 200u) - 100; /* small magnitudes */
      if (i % 4 == 2)
         b = (int32_t)(s % 40u) - 20;
      ix[i] = (int16_t)a;
      iy[i] = (int16_t)(b ? b : 7);
   }
   memset(o.map, 0xaa, K * COUNT * 2);
   memset(u.map, 0xaa, KU * COUNT * 4);
   struct buf *bufs[4] = {&x, &y, &o, &u};
   int rc = run_one(spv, 4, NULL, bufs, NULL, 0, COUNT / 64, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0, checked = 0;
   for (uint32_t i = 0; i < COUNT; i++) {
      const int32_t a = ix[i], b = iy[i], sh = b & 15;
      const uint16_t ua = (uint16_t)a, ub = (uint16_t)b;
      /* GLSL % on signed operands is undefined for negatives; glslang emits OpSMod (the result takes the sign of the divisor) */
      int32_t smod = a % b;
      if (smod && ((smod < 0) != (b < 0)))
         smod += b;
      uint16_t want[K] = {
         (uint16_t)(a + b), (uint16_t)(a - b), (uint16_t)(a * b), (uint16_t)(a / b), (uint16_t)smod,
         (uint16_t)((uint32_t)a << sh), (uint16_t)(a >> sh), (uint16_t)(ua >> sh), (uint16_t)(a < b ? a : b),
         (uint16_t)(a > b ? a : b), (uint16_t)(a < 0 ? -a : a), (uint16_t)-a, (uint16_t)(a & b), (uint16_t)(a | b),
         (uint16_t)(a ^ b), (uint16_t)~a, (uint16_t)(a < b ? 1 : -1), (uint16_t)(6 * a), (uint16_t)(a * 3000),
      };
      for (int k = 0; k < K; k++) {
         checked++;
         if (io[k * COUNT + i] != want[k] && bad++ < 12)
            fprintf(stderr, "  int16 [%u] op %d (a=%d b=%d): got %04x want %04x\n", i, k, a, b, io[k * COUNT + i],
                    want[k]);
      }
      float fa = (float)a, fb = (float)ub;
      uint32_t bfa, bfb;
      memcpy(&bfa, &fa, 4);
      memcpy(&bfb, &fb, 4);
      const uint32_t pk = ua | ((uint32_t)ub << 16), pk2 = pk ^ 0x00ff0f0fu;
      uint32_t wu[KU] = {ua, (uint32_t)a, (uint16_t)(ua * ub), (uint32_t)(a * b), bfa, bfb,
                         (uint32_t)(int32_t)(int16_t)(a * 3000), pk, (pk2 >> 16) * 3u + ((pk2 & 0xffffu) << 16)};
      for (int k = 0; k < KU; k++) {
         checked++;
         if (iu[k * COUNT + i] != wu[k] && bad++ < 12)
            fprintf(stderr, "  int16 [%u] wide %d (a=%d b=%d): got %08x want %08x\n", i, k, a, b, iu[k * COUNT + i],
                    wu[k]);
      }
   }
   printf("  %u values checked, %u mismatches\n", checked, bad);
   free_buf(&x);
   free_buf(&y);
   free_buf(&o);
   free_buf(&u);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* ---- : 8-bit and 64-bit integers (VK_KHR_8bit_storage + shaderInt8, shaderInt64). Without the features the case
 * reports "not supported". */
static int have_8bit_storage, have_int8, have_int64;

static int
case_wide(const char *spv)
{
   if (!have_8bit_storage || !have_int8 || !have_int64)
      return CASE_UNSUPPORTED;
   enum { COUNT = 128, K8 = 10, K64 = 6, KW = 3 };
   struct buf x, y, o, q, r, w;
   CHECK(make_buffer(&x, COUNT, SSBO));
   CHECK(make_buffer(&y, COUNT, SSBO));
   CHECK(make_buffer(&o, K8 * COUNT, SSBO));
   CHECK(make_buffer(&q, COUNT * 16, SSBO));
   CHECK(make_buffer(&r, K64 * COUNT * 16, SSBO));
   CHECK(make_buffer(&w, KW * COUNT * 4, SSBO));
   int8_t *ix = (int8_t *)x.map, *iy = (int8_t *)y.map, *io = (int8_t *)o.map;
   int64_t *iq = (int64_t *)q.map, *ir = (int64_t *)r.map;
   uint32_t *iw = w.map;
   static const int8_t special[] = {0, 1, -1, 2, -2, 127, -127, 64, -64, 100, -100, 7, -8, 33};
   static const int64_t special64[] = {0, 1, -1, 2, INT64_MAX, INT64_MIN + 1, 0x100000000ll, -0x100000000ll,
                                       0x7fffffffll, -0x80000000ll, 0xffffffffll, 12345678901234ll, -98765432109876ll};
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t t = i * 2654435761u + 0x9e3779b9u, u = i * 0x85ebca6bu + 0xc2b2ae35u;
      t ^= t >> 15;
      u ^= u >> 13;
      int32_t a = i < 14 ? special[i] : (int32_t)(t % 255u) - 127;
      int32_t b = i < 14 ? special[(i * 3 + 1) % 14] : (int32_t)(u % 255u) - 127;
      ix[i] = (int8_t)a;
      iy[i] = (int8_t)(b ? b : 5);
      uint64_t hi = ((uint64_t)t << 32) | u, lo = ((uint64_t)u << 32) | (t * 31u + i);
      int64_t p = i < 13 ? special64[i] : (int64_t)(i % 3 == 0 ? lo : hi);
      int64_t s = i < 13 ? special64[(i * 5 + 2) % 13] : (int64_t)(i % 4 == 0 ? (int64_t)(u % 1000u) - 500 : (int64_t)lo);
      if (s == 0 || (s == -1 && p == INT64_MIN))
         s = 3;
      iq[2 * i] = p;
      iq[2 * i + 1] = s;
   }
   memset(o.map, 0xaa, K8 * COUNT);
   memset(r.map, 0xaa, K64 * COUNT * 16);
   memset(w.map, 0xaa, KW * COUNT * 4);
   struct buf *bufs[6] = {&x, &y, &o, &q, &r, &w};
   int rc = run_one(spv, 6, NULL, bufs, NULL, 0, COUNT / 64, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0, checked = 0;
   for (uint32_t i = 0; i < COUNT; i++) {
      const int32_t a = ix[i], b = iy[i], sh = b & 7;
      const uint8_t ua = (uint8_t)a;
      int8_t want[K8] = {(int8_t)(a + b), (int8_t)(a - b), (int8_t)(a * b), (int8_t)(a / b), (int8_t)((uint32_t)a << sh),
                         (int8_t)(a >> sh), (int8_t)(ua >> sh), (int8_t)(a < b ? a : b), (int8_t)(a > b ? a : b),
                         (int8_t)(a < b ? 1 : -1)};
      for (int k = 0; k < K8; k++) {
         checked++;
         if (io[k * COUNT + i] != want[k] && bad++ < 12)
            fprintf(stderr, "  wide int8 [%u] op %d (a=%d b=%d): got %d want %d\n", i, k, a, b, io[k * COUNT + i], want[k]);
      }
      const int64_t p = iq[2 * i], s = iq[2 * i + 1];
      const int sh64 = (int)(s & 63);
      const int64_t want64[K64][2] = {{(int64_t)((uint64_t)p + (uint64_t)s), (int64_t)((uint64_t)p - (uint64_t)s)},
                                      {(int64_t)((uint64_t)p * (uint64_t)s), p / s},
                                      {(int64_t)((uint64_t)p << sh64), p >> sh64},
                                      {p < s ? p : s, p > s ? p : s},
                                      {p < 0 ? (int64_t)(0 - (uint64_t)p) : p, (int64_t)(0 - (uint64_t)p)},
                                      {p & s, p ^ s}};
      for (int k = 0; k < K64; k++)
         for (int c = 0; c < 2; c++) {
            checked++;
            if (ir[(k * COUNT + i) * 2 + c] != want64[k][c] && bad++ < 12)
               fprintf(stderr, "  wide int64 [%u] op %d.%d (p=%lld s=%lld): got %lld want %lld\n", i, k, c, (long long)p,
                       (long long)s, (long long)ir[(k * COUNT + i) * 2 + c], (long long)want64[k][c]);
         }
      const uint32_t ww[KW] = {(uint32_t)a, (uint32_t)ua, (uint32_t)((int64_t)a * 1000)};
      for (int k = 0; k < KW; k++) {
         checked++;
         if (iw[k * COUNT + i] != ww[k] && bad++ < 12)
            fprintf(stderr, "  wide widen %d [%u] (a=%d): got %08x want %08x\n", k, i, a, iw[k * COUNT + i], ww[k]);
      }
   }
   printf("  %u values checked, %u mismatches\n", checked, bad);
   free_buf(&x);
   free_buf(&y);
   free_buf(&o);
   free_buf(&q);
   free_buf(&r);
   free_buf(&w);
   return bad ? CASE_FAIL : CASE_PASS;
}

static int
case_intops(const char *spv)
{
   enum { COUNT = 128 };
   struct buf a, b, o;
   CHECK(make_buffer(&a, COUNT * 4, SSBO));
   CHECK(make_buffer(&b, COUNT * 4, SSBO));
   CHECK(make_buffer(&o, COUNT * 32, SSBO));
   int32_t *ia = (int32_t *)a.map, *ib = (int32_t *)b.map, *io = (int32_t *)o.map;
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t r = i * 2654435761u + 12345u;
      ia[i] = (int32_t)(r ^ (r >> 15));
      if (i == 5)
         ia[i] = 0;
      if (i == 6)
         ia[i] = -1;
      if (i == 7)
         ia[i] = INT32_MIN + 1;
      int32_t d = (int32_t)(i % 13) + 1;
      ib[i] = (i & 1) ? -d : d;
   }
   for (uint32_t i = 0; i < COUNT * 8; i++)
      io[i] = 0x13579bdf;
   struct buf *bufs[3] = {&a, &b, &o};
   int rc = run_one(spv, 3, NULL, bufs, NULL, 0, 2, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t i = 0; i < COUNT; i++) {
      int32_t x = ia[i], y = ib[i];
      int32_t want[8];
      want[0] = x / y;
      want[1] = x % y; /* glslang emits OpSMod for GLSL int %: the result takes the divisor's sign */
      if (want[1] != 0 && ((want[1] ^ y) < 0))
         want[1] += y;
      want[2] = __builtin_popcount((uint32_t)x);
      want[3] = x ^ y;
      want[4] = (int32_t)((uint32_t)x << (32 - 3 - 7)) >> (32 - 7);
      want[5] = (int32_t)(((uint32_t)x & ~(0xffu << 4)) | (((uint32_t)y << 4) & (0xffu << 4)));
      want[6] = x >> 3;
      want[7] = (int32_t)(((uint32_t)x >> 5) | ((uint32_t)y << 7));
      for (int k = 0; k < 8; k++)
         if (io[i * 8 + k] != want[k] && bad++ < 8)
            fprintf(stderr, "intops x=%d y=%d out[%d] = %d (0x%x), want %d (0x%x)\n", x, y, k, io[i * 8 + k],
                    io[i * 8 + k], want[k], want[k]);
   }
   printf("  %u results checked, %u mismatches\n", (unsigned)COUNT * 8, bad);
   free_buf(&a);
   free_buf(&b);
   free_buf(&o);
   return bad ? CASE_FAIL : CASE_PASS;
}

static int
case_scratch(const char *spv)
{
   enum { COUNT = 128 };
   struct buf in, out;
   CHECK(make_buffer(&in, COUNT * 4, SSBO));
   CHECK(make_buffer(&out, COUNT * 4, SSBO));
   for (uint32_t i = 0; i < COUNT; i++) {
      in.map[i] = i * 7 + 3;
      out.map[i] = 0;
   }
   struct buf *bufs[2] = {&in, &out};
   int rc = run_one(spv, 2, NULL, bufs, NULL, 0, 2, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t i = 0; i < COUNT; i++)
      if (out.map[i] != in.map[i] + (in.map[i] & 7))
         bad++;
   printf("  %u mismatches\n", bad);
   free_buf(&in);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* ---- board-checklist cases : scratch addressing / widths / growth, atomic variants ------- */

static uint32_t
hash32(uint32_t i)
{
   uint32_t r = i * 2654435761u + 12345u;
   r ^= r >> 13;
   r *= 0x5bd1e995u;
   return r ^ (r >> 15);
}

static int
cmp_u32(const void *a, const void *b)
{
   uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
   return x < y ? -1 : x > y;
}

/* multiset(a[0..n)) == multiset(b[0..n)); sorts both in place */
static int
same_multiset(uint32_t *a, uint32_t *b, unsigned n)
{
   qsort(a, n, 4, cmp_u32);
   qsort(b, n, 4, cmp_u32);
   return !memcmp(a, b, n * 4);
}

/* common driver for the "src[] -> dst[8*i or similar]" scratch kernels: 128 invocations, 2 workgroups */
static int
scratch_run(const char *spv, unsigned words_per_lane, struct buf *in, struct buf *out)
{
   enum { COUNT = 128 };
   CHECK(make_buffer(in, COUNT * 4, SSBO));
   CHECK(make_buffer(out, COUNT * words_per_lane * 4, SSBO));
   for (uint32_t i = 0; i < COUNT; i++)
      in->map[i] = hash32(i);
   for (uint32_t i = 0; i < COUNT * words_per_lane; i++)
      out->map[i] = 0xfefefefeu;
   struct buf *bufs[2] = {in, out};
   return run_one(spv, 2, NULL, bufs, NULL, 0, 2, 1, 1);
}

/* checklist 3: 32/64/96/128-bit scratch accesses at odd strides and offsets */
static int
case_scratch_vec(const char *spv)
{
   enum { COUNT = 128 };
   struct buf in, out;
   int rc = scratch_run(spv, 8, &in, &out);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t v = in.map[i];
      uint32_t j = (v >> 2) & 3u, m = (v >> 4) & 3u;
      if (m > 2)
         m = 2;
      uint32_t want[8] = {v + j,
                          v * 2u + j,
                          v ^ j,
                          v + m * 13u,
                          (v + m * 5u) + (v + m * 7u) + (v + m * 11u),
                          (v + m) + (v ^ (m * 77u)),
                          v * 3u + m,
                          (v ^ (m << 4)) + (v - m)};
      for (int k = 0; k < 8; k++)
         if (out.map[i * 8 + k] != want[k] && bad++ < 8)
            fprintf(stderr, "scratch_vec[%u][%d] = 0x%08x, want 0x%08x\n", i, k, out.map[i * 8 + k], want[k]);
   }
   printf("  %u results checked, %u mismatches\n", (unsigned)COUNT * 8, bad);
   free_buf(&in);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* checklist 3: 8/16-bit scratch accesses (needs the Int8/Int16 capabilities; a compile failure is a known gap) */
static int
case_scratch_sub(const char *spv)
{
   enum { COUNT = 128 };
   struct buf in, out;
   int rc = scratch_run(spv, 4, &in, &out);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t v = in.map[i];
      uint32_t j = v & 7u, k2 = (v >> 3) & 3u, m = (v >> 6) & 3u;
      if (m > 2)
         m = 2;
      uint32_t b8a = (uint8_t)(v + j * 37u), b8b = (uint8_t)(v + (j + 5u) * 37u);
      uint32_t h0 = (uint16_t)(v * 3u + k2), h1 = (uint16_t)(v * 3u + k2 + 3u);
      uint32_t t = (uint8_t)(v + m), h = (uint16_t)(v * 5u + m), u = (uint8_t)(v >> m);
      uint32_t want[4] = {b8a | (b8b << 8) | (h0 << 16), h1 | (t << 16) | (u << 24), h ^ (v + m * 3u),
                          (v ^ m) + v * m};
      for (int k = 0; k < 4; k++)
         if (out.map[i * 4 + k] != want[k] && bad++ < 8)
            fprintf(stderr, "scratch_sub[%u][%d] = 0x%08x, want 0x%08x\n", i, k, out.map[i * 4 + k], want[k]);
   }
   printf("  %u results checked, %u mismatches\n", (unsigned)COUNT * 4, bad);
   free_buf(&in);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* checklist 2: out-of-bounds scratch (read 0, write dropped).  Opt-in (NGV_EXTRA=1): the GLSL is UB by Vulkan rules */
static int
case_scratch_oob(const char *spv)
{
   enum { COUNT = 128 };
   struct buf in, out;
   int rc = scratch_run(spv, 4, &in, &out);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t v = in.map[i];
      uint32_t want[4] = {0u, v + (v & 7u), v + ((v >> 3) & 7u), v + 7u};
      for (int k = 0; k < 4; k++)
         if (out.map[i * 4 + k] != want[k] && bad++ < 8)
            fprintf(stderr, "scratch_oob[%u][%d] = 0x%08x, want 0x%08x\n", i, k, out.map[i * 4 + k], want[k]);
   }
   printf("  %u results checked, %u mismatches\n", (unsigned)COUNT * 4, bad);
   free_buf(&in);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* one prepared dispatch whose command buffer can be submitted repeatedly */
struct prog {
   VkDescriptorSetLayout dsl;
   VkPipelineLayout pl;
   VkPipeline pipe;
   VkDescriptorPool dp;
   VkDescriptorSet set;
   VkCommandBuffer cb;
};

static int
prog_make(struct prog *p, const char *spv, uint32_t nb, struct buf **bufs, uint32_t gx)
{
   p->dsl = make_dsl(nb, NULL);
   if (p->dsl == VK_NULL_HANDLE)
      return CASE_FAIL;
   if (make_pipeline(spv, 0, p->dsl, &p->pl, &p->pipe) != VK_SUCCESS) {
      vkDestroyDescriptorSetLayout(dev, p->dsl, NULL);
      return CASE_NOCOMPILE;
   }
   CHECK(make_set(p->dsl, &p->dp, &p->set, bufs, nb, NULL));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   CHECK(vkAllocateCommandBuffers(dev, &cai, &p->cb));
   VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(p->cb, &bi));
   vkCmdBindPipeline(p->cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
   vkCmdBindDescriptorSets(p->cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pl, 0, 1, &p->set, 0, NULL);
   vkCmdDispatch(p->cb, gx, 1, 1);
   VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                         .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(p->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0,
                        NULL);
   CHECK(vkEndCommandBuffer(p->cb));
   return CASE_PASS;
}

static int
prog_submit(struct prog *p)
{
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &p->cb};
   CHECK(vkQueueSubmit(queue, 1, &si, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, fence_timeout_ns));
   vkDestroyFence(dev, fence, NULL);
   return CASE_PASS;
}

static void
prog_free(struct prog *p)
{
   vkFreeCommandBuffers(dev, pool, 1, &p->cb);
   vkDestroyPipeline(dev, p->pipe, NULL);
   vkDestroyPipelineLayout(dev, p->pl, NULL);
   vkDestroyDescriptorPool(dev, p->dp, NULL);
   vkDestroyDescriptorSetLayout(dev, p->dsl, NULL);
}

/* checklist 6: scratch BO growth.  Record+run the small-scratch kernel (ngv_scratch, 32 B/lane), then record+run the
 * big one (ngv_scratch_big, 512 B/lane: the device scratch BO is replaced by a bigger one), then submit the SMALL
 * command buffer again (it still references the old BO's VA; the old BO must stay alive).  `spv` only gives the
 * directory; the case name has no .spv of its own. */
static int
case_scratch_grow(const char *spv)
{
   enum { COUNT = 128 };
   char dir[600], p_small[640], p_big[640];
   snprintf(dir, sizeof(dir), "%s", spv);
   char *sl = strrchr(dir, '/');
   if (sl)
      sl[1] = 0;
   else
      dir[0] = 0;
   snprintf(p_small, sizeof(p_small), "%sngv_scratch.spv", dir);
   snprintf(p_big, sizeof(p_big), "%sngv_scratch_big.spv", dir);

   struct buf in, o_small, o_big;
   CHECK(make_buffer(&in, COUNT * 4, SSBO));
   CHECK(make_buffer(&o_small, COUNT * 4, SSBO));
   CHECK(make_buffer(&o_big, COUNT * 4, SSBO));
   for (uint32_t i = 0; i < COUNT; i++) {
      in.map[i] = hash32(i) * 8u + (i & 7u);
      o_small.map[i] = 0;
      o_big.map[i] = 0;
   }
   struct buf *b_small[2] = {&in, &o_small}, *b_big[2] = {&in, &o_big};
   struct prog ps, pb;
   int rc = prog_make(&ps, p_small, 2, b_small, 2); /* recorded first: the scratch BO is sized for 32 B/lane */
   if (rc)
      return rc;
   int stage = 0;
   unsigned bad[3] = {0, 0, 0};
   rc = prog_submit(&ps);
   if (rc)
      return rc;
   for (uint32_t i = 0; i < COUNT; i++)
      if (o_small.map[i] != in.map[i] + (in.map[i] & 7u))
         bad[0]++;
   stage = 1;
   rc = prog_make(&pb, p_big, 2, b_big, 2); /* recorded second: needs 512 B/lane, the BO grows */
   if (rc)
      return rc;
   rc = prog_submit(&pb);
   if (rc)
      return rc;
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t v = in.map[i];
      uint32_t want = (v ^ (((v >> 3) & 127u) * 0x9e3779b1u)) + (v ^ ((v & 127u) * 0x9e3779b1u));
      if (o_big.map[i] != want && bad[1]++ < 4)
         fprintf(stderr, "scratch_grow big[%u] = 0x%08x, want 0x%08x\n", i, o_big.map[i], want);
   }
   stage = 2;
   for (uint32_t i = 0; i < COUNT; i++)
      o_small.map[i] = 0xdeadbeefu; /* replay must rewrite every element */
   rc = prog_submit(&ps);
   if (rc)
      return rc;
   for (uint32_t i = 0; i < COUNT; i++)
      if (o_small.map[i] != in.map[i] + (in.map[i] & 7u))
         bad[2]++;
   printf("  small=%u big=%u small-replay=%u mismatches (stage %d)\n", bad[0], bad[1], bad[2], stage);
   prog_free(&ps);
   prog_free(&pb);
   free_buf(&in);
   free_buf(&o_small);
   free_buf(&o_big);
   return bad[0] || bad[1] || bad[2] ? CASE_FAIL : CASE_PASS;
}

/* checklist 5/7: every buffer atomic op, uint and int.  Only order-independent facts are checked. */
static int
case_atomic_ops(const char *spv)
{
   enum { N = 192, NCELL = 14 };
   struct buf src, cells, rets;
   CHECK(make_buffer(&src, N * 4, SSBO));
   CHECK(make_buffer(&cells, 64, SSBO));
   CHECK(make_buffer(&rets, N * 5 * 4, SSBO));
   uint32_t v[N];
   for (uint32_t g = 0; g < N; g++)
      src.map[g] = v[g] = hash32(g + 1000);
   const uint32_t init[NCELL] = {0, 0xffffffffu, 0, 0xffffffffu, 0, 0, 0xcafe0000u, 0, 0,
                                 0x7fffffffu, 0x80000000u, 0, (uint32_t)-77, 0};
   memcpy(cells.map, init, sizeof(init));
   for (uint32_t i = 0; i < N * 5; i++)
      rets.map[i] = 0xfefefefeu;
   struct buf *bufs[3] = {&src, &cells, &rets};
   int rc = run_one(spv, 3, NULL, bufs, NULL, 0, 3, 1, 1);
   if (rc)
      return rc;

   uint32_t sum = 0, umin = 0xffffffffu, umax = 0, band = 0xffffffffu, bor = 0, bxor = 0;
   int32_t imin = INT32_MAX, imax = INT32_MIN;
   for (uint32_t g = 0; g < N; g++) {
      sum += v[g];
      umin = v[g] < umin ? v[g] : umin;
      umax = v[g] > umax ? v[g] : umax;
      band &= v[g] | 0x80000001u;
      bor |= v[g] & 0x0f0f0f0fu;
      bxor ^= v[g];
      imin = (int32_t)v[g] < imin ? (int32_t)v[g] : imin;
      imax = (int32_t)v[g] > imax ? (int32_t)v[g] : imax;
   }
   const uint32_t *c = cells.map;
   unsigned bad = 0;
#define EXPECT_CELL(idx, name, want)                                                         \
   do {                                                                                      \
      if (c[idx] != (uint32_t)(want)) {                                                      \
         bad++;                                                                              \
         fprintf(stderr, "atomic_ops %s = 0x%08x, want 0x%08x\n", name, c[idx], (uint32_t)(want)); \
      }                                                                                      \
   } while (0)
   EXPECT_CELL(0, "add", sum);
   EXPECT_CELL(1, "umin", umin);
   EXPECT_CELL(2, "umax", umax);
   EXPECT_CELL(3, "and", band);
   EXPECT_CELL(4, "or", bor);
   EXPECT_CELL(5, "xor", bxor);
   EXPECT_CELL(8, "ticket", N);
   EXPECT_CELL(9, "imin", imin);
   EXPECT_CELL(10, "imax", imax);
   EXPECT_CELL(13, "iadd", sum);
#undef EXPECT_CELL

   /* ticket: the atomicAdd(ticket, 1) return values are a permutation of 0..N-1 */
   static uint8_t seen[N];
   memset(seen, 0, sizeof(seen));
   for (uint32_t g = 0; g < N; g++) {
      uint32_t t = rets.map[g * 5];
      if (t >= N || seen[t]++) {
         bad++;
         fprintf(stderr, "atomic_ops ticket[%u] = %u (out of range or duplicated)\n", g, t);
         break;
      }
   }
   /* exchange (uint, int): multiset(returned olds + final cell) == multiset(written values + initial value) */
   for (int pass = 0; pass < 2; pass++) {
      uint32_t olds[N + 1], wrs[N + 1];
      for (uint32_t g = 0; g < N; g++) {
         olds[g] = rets.map[g * 5 + (pass ? 4 : 1)];
         wrs[g] = v[g];
      }
      olds[N] = c[pass ? 12 : 6];
      wrs[N] = init[pass ? 12 : 6];
      if (!same_multiset(olds, wrs, N + 1)) {
         bad++;
         fprintf(stderr, "atomic_ops %sxchg: returned/final values are not a permutation of the written ones\n",
                 pass ? "i" : "u");
      }
   }
   /* compSwap(cell, 0, tag): exactly one lane sees the old 0 and wins; every other lane sees the winner's tag */
   for (int pass = 0; pass < 2; pass++) {
      unsigned zeros = 0;
      uint32_t win = 0;
      uint32_t fin = c[pass ? 11 : 7];
      for (uint32_t g = 0; g < N; g++) {
         uint32_t o = rets.map[g * 5 + (pass ? 3 : 2)];
         if (o == 0) {
            zeros++;
            win = g;
         } else if (o != fin) {
            bad++;
            fprintf(stderr, "atomic_ops %scas[%u] old = 0x%08x, want final 0x%08x\n", pass ? "i" : "u", g, o, fin);
            break;
         }
      }
      uint32_t want_fin = pass ? (uint32_t)(-(int32_t)win - 1) : win + 1u;
      if (zeros != 1 || fin != want_fin) {
         bad++;
         fprintf(stderr, "atomic_ops %scas: %u winners, final 0x%08x (winner lane %u -> want 0x%08x)\n",
                 pass ? "i" : "u", zeros, fin, win, want_fin);
      }
   }
   printf("  14 cells + 192x5 return values checked, %u mismatches\n", bad);
   free_buf(&src);
   free_buf(&cells);
   free_buf(&rets);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* checklist 5/7: atomics on workgroup shared memory */
static int
case_atomic_shared(const char *spv)
{
   enum { WG = 3, N = WG * 64 };
   struct buf src, res, tick;
   CHECK(make_buffer(&src, N * 4, SSBO));
   CHECK(make_buffer(&res, WG * 8 * 4, SSBO));
   CHECK(make_buffer(&tick, N * 4, SSBO));
   for (uint32_t g = 0; g < N; g++) {
      src.map[g] = hash32(g + 5000);
      tick.map[g] = 0xfefefefeu;
   }
   for (uint32_t i = 0; i < WG * 8; i++)
      res.map[i] = 0xfefefefeu;
   struct buf *bufs[3] = {&src, &res, &tick};
   int rc = run_one(spv, 3, NULL, bufs, NULL, 0, WG, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t w = 0; w < WG; w++) {
      uint32_t mx = 0, mn = 0xffffffffu, orv = 0, sum = 0;
      uint64_t seen = 0;
      for (uint32_t l = 0; l < 64; l++) {
         uint32_t x = src.map[w * 64 + l];
         mx = x > mx ? x : mx;
         mn = x < mn ? x : mn;
         orv |= x & 0x00ff00ffu;
         sum += x;
         uint32_t t = tick.map[w * 64 + l];
         if (t >= 64 || (seen >> t & 1)) {
            if (bad++ < 8)
               fprintf(stderr, "atomic_shared wg%u lane %u ticket %u out of range / duplicated\n", w, l, t);
         } else {
            seen |= 1ull << t;
         }
      }
      const uint32_t *r = &res.map[w * 8];
      uint32_t want[6] = {64, mx, orv, sum, 0, mn};
      for (int k = 0; k < 6; k++) {
         if (k == 4) { /* compSwap winner: any lane's l+1 */
            if ((r[4] < 1 || r[4] > 64) && bad++ < 8)
               fprintf(stderr, "atomic_shared wg%u cas = %u, want 1..64\n", w, r[4]);
         } else if (r[k] != want[k] && bad++ < 8) {
            fprintf(stderr, "atomic_shared wg%u res[%d] = 0x%08x, want 0x%08x\n", w, k, r[k], want[k]);
         }
      }
   }
   printf("  %u workgroups checked, %u mismatches\n", (unsigned)WG, bad);
   free_buf(&src);
   free_buf(&res);
   free_buf(&tick);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* ---- : subgroups ---------------------------------------------------------------------- */

static uint32_t
sg_hash(uint32_t g)
{
   uint32_t h = g * 0x9E3779B1u ^ 0x5A5A0000u;
   h ^= h >> 15;
   h *= 0x85EBCA6Bu;
   return h ^ (h >> 13);
}

static unsigned
sg_popc(uint32_t m)
{
   unsigned n = 0;
   for (; m; m &= m - 1)
      n++;
   return n;
}

static unsigned
sg_ctz(uint32_t m)
{
   unsigned n = 0;
   if (!m)
      return 32;
   while (!(m & 1u)) {
      m >>= 1;
      n++;
   }
   return n;
}

static unsigned
sg_msb(uint32_t m)
{
   unsigned n = 31;
   if (!m)
      return ~0u;
   while (!(m & 0x80000000u)) {
      m <<= 1;
      n--;
   }
   return n;
}

/* BASIC / VOTE / BALLOT: 3 workgroups x 48 invocations = per workgroup one full subgroup and one with lanes 0..15
 * (ngv_subgroup.comp, 12 words per invocation). */
static int
case_subgroup(const char *spv)
{
   enum { WG = 3, PER = 48, COUNT = WG * PER, W = 12 };
   struct buf in, out;
   CHECK(make_buffer(&in, COUNT * 4, SSBO));
   CHECK(make_buffer(&out, COUNT * W * 4, SSBO));
   for (uint32_t g = 0; g < COUNT; g++)
      in.map[g] = sg_hash(g) | ((g % PER) % 32 == 0 ? 1u : 0u); /* lane 0 votes true: FindLSB/MSB are defined */
   memset(out.map, 0xfe, COUNT * W * 4);
   struct buf *bufs[2] = {&in, &out};
   int rc = run_one(spv, 2, NULL, bufs, NULL, 0, WG, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t w = 0; w < WG; w++)
      for (uint32_t s = 0; s < 2; s++) {
         const uint32_t base = w * PER + s * 32, n = s ? PER - 32 : 32;
         const uint32_t *x = &in.map[base];
         uint32_t b = 0, all = 1, any = 0, eq0 = 1, eqc = 1, eqf = 1, t = 0, t8 = 1;
         for (uint32_t j = 0; j < n; j++) {
            const uint32_t c = x[j] & 1u;
            b |= c << j;
            any |= c;
            all &= c;
            eq0 &= (x[j] & 0xF0u) == (x[0] & 0xF0u);
            eqc &= c == (x[0] & 1u);
            eqf &= (x[j] & 0x3F800000u) == (x[0] & 0x3F800000u);
            if (x[j] & 6u) {
               t |= 1u << j;
               t8 &= (x[j] & 8u) != 0;
            }
         }
         for (uint32_t l = 0; l < n; l++) {
            const uint32_t want[W] = {
               l | (s << 8) | (2u << 16) | (32u << 24),
               l == 0,
               b,
               0,
               any | (all << 1) | (eq0 << 2) | (eqc << 3) | (eqf << 4),
               x[0],
               x[3],
               sg_popc(b) | (sg_popc(b & (l == 31 ? ~0u : (2u << l) - 1)) << 8) | (sg_popc(b & ((1u << l) - 1)) << 16) |
                  (((b >> l) & 1u) << 24) | (((b >> 5) & 1u) << 25),
               sg_ctz(b) | (sg_msb(b) << 8),
               ~0u << l,
               (1u << l) - 1,
               (x[l] & 6u) ? sg_ctz(t) | ((l == sg_ctz(t)) << 8) | (t8 << 9) | (sg_popc(t) << 16) : 0xFFFFFFFFu,
            };
            const uint32_t *got = &out.map[(base + l) * W];
            for (int k = 0; k < W; k++)
               if (got[k] != want[k] && bad++ < 8)
                  fprintf(stderr, "subgroup wg%u sg%u lane%u word %d = 0x%08x, want 0x%08x\n", w, s, l, k, got[k],
                          want[k]);
         }
      }
   printf("  %u invocations checked, %u mismatches\n", (unsigned)COUNT, bad);
   free_buf(&in);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

static VkSubgroupFeatureFlags sg_ops; /* subgroupSupportedOperations of the device (main) */

/* SHUFFLE / SHUFFLE_RELATIVE / QUAD: only if the device reports them (XLANE unit; the model).  2 x 64 invocations,
 * full subgroups (ngv_subgroup_shuffle.comp, 10 words per invocation). */
static int
case_subgroup_shuffle(const char *spv)
{
   const VkSubgroupFeatureFlags need =
      VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT | VK_SUBGROUP_FEATURE_QUAD_BIT;
   if ((sg_ops & need) != need) {
      printf("  (subgroupSupportedOperations 0x%x: no SHUFFLE / SHUFFLE_RELATIVE / QUAD on this device)\n",
             (unsigned)sg_ops);
      return CASE_UNSUPPORTED;
   }
   enum { WG = 2, PER = 64, COUNT = WG * PER, W = 10 };
   struct buf in, out;
   CHECK(make_buffer(&in, COUNT * 4, SSBO));
   CHECK(make_buffer(&out, COUNT * W * 4, SSBO));
   for (uint32_t g = 0; g < COUNT; g++)
      in.map[g] = sg_hash(g + 1000);
   memset(out.map, 0xfe, COUNT * W * 4);
   struct buf *bufs[2] = {&in, &out};
   int rc = run_one(spv, 2, NULL, bufs, NULL, 0, WG, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0;
   for (uint32_t g = 0; g < COUNT; g++) {
      const uint32_t l = g % 32, *x = &in.map[g - l];
      const uint32_t r = x[l ^ 31];
      const uint32_t want[W] = {x[(l * 7 + 3) & 31], x[l ^ 5], l >= 3 ? x[l - 3] : 0, l < 28 ? x[l + 4] : 0,
                                x[(l & ~3u) + 2], x[l ^ 1], x[l ^ 2], x[l ^ 3], x[l ^ 1] & 1u, r ^ (~r << 1)};
      const uint32_t *got = &out.map[g * W];
      for (int k = 0; k < W; k++)
         if (got[k] != want[k] && bad++ < 8)
            fprintf(stderr, "subgroup_shuffle [%u] word %d = 0x%08x, want 0x%08x\n", g, k, got[k], want[k]);
   }
   printf("  %u invocations checked, %u mismatches\n", (unsigned)COUNT, bad);
   free_buf(&in);
   free_buf(&out);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* ---- : f32 transcendentals (sin / cos / tan, frexp, ldexp, exp*, log*, pow, sqrt, inverse trig, hyperbolics). The
 * tolerances follow the Vulkan precision rules where they exist (sin / cos: 2^-11 absolute on [-pi, pi]; exp / exp2 / pow:
 * 3 + 2|x| ulp; log: 2^-21 absolute in [0.5, 2], else 3 ulp; sqrt 2.5 ulp, inversesqrt 2 ulp) with some slack, and are
 * loose where the spec gives none (tan, inverse trig, hyperbolics).  frexp and ldexp are exact. */
static int
sfu_close(float got, double want, double abs_tol, double rel_tol)
{
   if (!(fabs(want) >= 1.2e-38 && fabs(want) <= 3.4e38) && want != 0.0)
      return 1; /* out of the normal range: flush / overflow behaviour is not what this case checks */
   double d = fabs((double)got - want);
   return d <= abs_tol || d <= rel_tol * fabs(want);
}

static int
case_sfu(const char *spv)
{
   enum { COUNT = 128, K = 19 };
   static const char *what[K] = {"sin",  "cos",  "tan",   "frexp.sig", "ldexp", "exp",  "exp2",  "log",   "log2", "pow",
                                 "sqrt", "rsqrt", "asin", "acos",      "atan",  "atan2", "sinh", "cosh", "tanh"};
   static const uint32_t special[] = {
      0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0x40000000, 0x40490fdb, 0xc0490fdb, /* 0 -0 1 -1 .5 2 pi -pi */
      0x3fc90fdb, 0x4096cbe4, 0x3a83126f, 0xba83126f, 0x40a00000, 0xc0e00000, 0x41200000, 0xc1a00000, /* pi/2 3pi/2 1e-3 -1e-3 5 -7 10 -20 */
      0x00012345, 0x80012345, 0x00800000, 0x007fffff, 0x7f7fffff, 0xff7fffff, 0x1e3ce508, 0x61a2e9d1, /* subnormals, min, max */
      0x3dcccccd, 0x3f666666, 0x4048f5c3, 0x42c80000, 0xc2c80000, 0x447a0000, 0x3727c5ac, 0x4b189680,
   };
   struct buf a, b, o, p;
   CHECK(make_buffer(&a, COUNT * 4, SSBO));
   CHECK(make_buffer(&b, COUNT * 4, SSBO));
   CHECK(make_buffer(&o, K * COUNT * 4, SSBO));
   CHECK(make_buffer(&p, COUNT * 4, SSBO));
   float *fa = (float *)a.map, *fb = (float *)b.map, *fo = (float *)o.map;
   int32_t *ip = (int32_t *)p.map;
   const double pi = 3.14159265358979323846;
   for (uint32_t i = 0; i < COUNT; i++) {
      uint32_t r = i * 2654435761u + 0x9e3779b9u, s = i * 0x85ebca6bu + 0xc2b2ae35u;
      r ^= r >> 15;
      s ^= s >> 13;
      const double ur = (r & 0xffffff) / 16777216.0, us = (s & 0xffffff) / 16777216.0; /* [0, 1) */
      if (i < 32) {
         memcpy(&fa[i], &special[i], 4);
      } else if (i < 64) {
         fa[i] = (float)((ur * 2 - 1) * pi); /* the precision-relevant range of sin / cos */
      } else if (i < 96) {
         fa[i] = (float)((ur * 2 - 1) * 20.0);
      } else {
         fa[i] = (float)ldexp(1.0 + ur, (int)(s % 61) - 30) * ((r & 0x1000000) ? -1.0f : 1.0f); /* 2^-30 .. 2^31 */
      }
      fb[i] = (float)((us * 2 - 1) * (i & 1 ? 20.0 : 6.0)); /* ldexp exponent (truncated) and pow exponent / atan2 y */
   }
   memset(o.map, 0xaa, K * COUNT * 4);
   memset(p.map, 0xaa, COUNT * 4);
   struct buf *bufs[4] = {&a, &b, &o, &p};
   int rc = run_one(spv, 4, NULL, bufs, NULL, 0, COUNT / 64, 1, 1);
   if (rc)
      return rc;
   unsigned bad = 0, checked = 0, badk[K] = {0};
   for (uint32_t i = 0; i < COUNT; i++) {
      const double x = fa[i], y = fb[i], ax = fabs(x);
      const double u = fmax(-1.0, fmin(1.0, x / 4.0));
      int e;
      const double m = frexp(x, &e);
      const double ulp = 1.1920929e-7; /* 2^-23 */
      double want[K], atol[K], rtol[K];
      int skip[K] = {0};
      want[0] = sin(x);
      atol[0] = ax <= pi ? 4.9e-4 : 1e-3; /* 2^-11 */
      rtol[0] = 0;
      want[1] = cos(x);
      atol[1] = atol[0];
      rtol[1] = 0;
      want[2] = tan(x);
      atol[2] = 1e-4;
      rtol[2] = 1e-3;
      skip[2] = fabs(cos(x)) < 0.1 || ax > 100;
      skip[0] = skip[1] = ax > 100;
      want[3] = m;
      atol[3] = rtol[3] = 0;
      want[4] = ldexp(x, (int)y);
      atol[4] = rtol[4] = 0;
      want[5] = exp(x);
      atol[5] = 0;
      rtol[5] = (4 + 2 * ax) * ulp;
      want[6] = exp2(x);
      atol[6] = 0;
      rtol[6] = (4 + 2 * ax) * ulp;
      want[7] = x != 0 ? log(ax) : 0;
      atol[7] = 1e-5;
      rtol[7] = 3.6e-7;
      skip[7] = x == 0;
      want[8] = x != 0 ? log2(ax) : 0;
      atol[8] = 1e-5;
      rtol[8] = 3.6e-7;
      skip[8] = x == 0;
      want[9] = pow(ax, y);
      atol[9] = 0;
      rtol[9] = (8 + 4 * fabs(y * log2(ax > 0 ? ax : 1))) * ulp;
      skip[9] = x == 0 || fabs(y * log2(ax)) > 120;
      want[10] = sqrt(ax);
      atol[10] = 0;
      rtol[10] = 3 * ulp;
      want[11] = x != 0 ? 1.0 / sqrt(ax) : 0;
      atol[11] = 0;
      rtol[11] = 3 * ulp;
      skip[11] = x == 0;
      want[12] = asin(u);
      want[13] = acos(u);
      want[14] = atan(x);
      want[15] = atan2(y, x);
      for (int k = 12; k <= 15; k++) {
         atol[k] = 1e-3;
         rtol[k] = 0;
      }
      skip[15] = x == 0 && y == 0;
      want[16] = ax <= 20 ? sinh(x) : 0;
      want[17] = ax <= 20 ? cosh(x) : 0;
      want[18] = tanh(x);
      for (int k = 16; k <= 18; k++) {
         atol[k] = 1e-5;
         rtol[k] = 1e-3;
         skip[k] = k < 18 && ax > 20;
      }
      for (int k = 0; k < K; k++) {
         if (skip[k])
            continue;
         const float g = fo[k * COUNT + i];
         checked++;
         if (!sfu_close(g, want[k], atol[k], rtol[k]) && !(want[k] != want[k]) && (badk[k]++, bad++ < 24))
            fprintf(stderr, "  sfu [%u] %s(x=%.9g y=%.9g): got %.9g want %.9g\n", i, what[k], x, y, (double)g, want[k]);
      }
      checked++;
      if (ip[i] != e && (badk[3]++, bad++ < 24))
         fprintf(stderr, "  sfu [%u] frexp.exp(x=%.9g): got %d want %d\n", i, x, ip[i], e);
   }
   printf("  %u values checked, %u mismatches", checked, bad);
   for (int k = 0; k < K; k++)
      if (badk[k])
         printf(" [%s %u]", what[k], badk[k]);
   printf("\n");
   free_buf(&a);
   free_buf(&b);
   free_buf(&o);
   free_buf(&p);
   return bad ? CASE_FAIL : CASE_PASS;
}

/* name, runner, known gap (NULL = must compile and pass), extra (only runs with NGV_EXTRA=1) */
static const struct {
   const char *name;
   int (*run)(const char *spv);
   const char *gap;
   int extra;
} cases[] = {
   {"add", case_add, NULL},
   {"shared", case_shared, NULL},
   {"loop", case_loop, NULL},
   {"float", case_float, NULL},
   {"ubo2d", case_ubo2d, NULL},
   {"ids", case_ids, NULL},
   {"intops", case_intops, NULL},
   {"atomic", case_atomic, NULL},
   {"scratch", case_scratch, NULL},
   /* checklist cases */
   {"scratch_vec", case_scratch_vec, NULL, 0},
   {"scratch_sub", case_scratch_sub, NULL, 0},
   {"scratch_grow", case_scratch_grow, NULL, 0},
   {"scratch_oob", case_scratch_oob, NULL, 1},
   {"atomic_ops", case_atomic_ops, NULL, 0},
   {"atomic_shared", case_atomic_shared, NULL, 0},
 /* */
   {"half", case_half, NULL, 0},
 /* */
   {"numwg", case_numwg, NULL, 0},
 /* */
   {"subgroup", case_subgroup, NULL, 0},
   {"subgroup_shuffle", case_subgroup_shuffle, NULL, 0},
 /* */
   {"half16", case_half16, NULL, 0},
   {"int16", case_int16, NULL, 0},
 /* */
   {"sfu", case_sfu, NULL, 0},
 /* */
   {"wide", case_wide, NULL, 0},
};

/* Is `name` in the comma separated list `list`?  (exact token match) */
static int
in_list(const char *list, const char *name)
{
   size_t n = strlen(name);
   while (*list) {
      const char *e = strchr(list, ',');
      size_t l = e ? (size_t)(e - list) : strlen(list);
      if (l == n && !strncmp(list, name, n))
         return 1;
      if (!e)
         break;
      list = e + 1;
   }
   return 0;
}

int
main(int argc, char **argv)
{
   const char *ws = getenv("NGV_WINSYS");
   /* hal = functional model; unset/auto/drm = real device through the DRM render node (hardware mode).  With auto the
    * driver falls back to hal when no device is present; the test cannot tell and simply runs in hardware mode (only
    * the skip list differs).  Anything else (stub, ...) cannot run real kernels. */
   int hw = !ws || !*ws || !strcmp(ws, "auto") || !strcmp(ws, "drm");
   if (!hw && strcmp(ws, "hal")) {
      printf("ngv_shaders: SKIP (needs NGV_WINSYS=hal (model) or drm/auto (hardware): the real NIR -> ngcomp path)\n");
      return 0;
   }
   /* Hardware mode: per-kernel timeout and skip list.  Default skip list: ngv_float needs f32 VALU + sqrt (SFU) and
    * ngv_intops needs integer div/mod (SFU rcp).  f32 VALU is in RM build 8003cb02, SFU only comes with M1b (the M1 RM
    * reports caps=0x00102011, no SFU), so both are skipped until then.  Faults are never auto-skipped: override the
    * list explicitly with NGV_HW_SKIP=a,b (empty value = skip nothing).  Names are without the "ngv_" prefix.
 * atomic / scratch : compiler (ngcomp) and driver (scratch BO) support are in, but they have not run
 * on the board yet (checklist in); they are skipped by default in hardware mode and run in the
    * functional-model mode (NGV_WINSYS=hal).  Run them on hardware with NGV_HW_SKIP=float,intops (or empty). */
   const char *hw_skip = "half16,int16,sfu,wide"; /* the rest passes on rm_sp b4 (2026-10-21); scratch_oob is opt-in via NGV_EXTRA=1;
 * the section 16-bit cases have only run on the model (NGV_HW_SKIP= runs them) */
   if (hw) {
      const char *e = getenv("NGV_HW_SKIP");
      if (e)
         hw_skip = e;
      const char *t = getenv("NGV_HW_TIMEOUT_MS");
      unsigned long ms = (t && *t) ? strtoul(t, NULL, 0) : 10000ul;
      fence_timeout_ns = (ms ? ms : 10000ul) * 1000000ull;
      printf("ngv_shaders: hardware mode (NGV_WINSYS=%s), skip list \"%s\", per-kernel timeout %lu ms\n",
             ws && *ws ? ws : "unset", hw_skip, (unsigned long)(fence_timeout_ns / 1000000ull));
   } else {
      hw_skip = "";
   }
   if (argc < 2) {
      fprintf(stderr, "usage: %s <dir with ngv_*.spv>\n", argv[0]);
      return 2;
   }
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS)
      return 1;
   uint32_t n = 1;
   VkPhysicalDevice pd;
   VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || n != 1)
      return 1;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   VkPhysicalDeviceSubgroupProperties sgp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
   VkPhysicalDeviceProperties2 pp2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgp};
   vkGetPhysicalDeviceProperties2(pd, &pp2);
   sg_ops = sgp.supportedOperations;
   printf("ngv_shaders: subgroupSize %u, subgroupSupportedOperations 0x%x\n", sgp.subgroupSize, (unsigned)sg_ops);
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
                                  .queueCount = 1, .pQueuePriorities = &prio};
 /* : turn on the 16-bit features the device reports (the case reports "not supported" without them) */
   VkPhysicalDevice16BitStorageFeatures s16 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
   VkPhysicalDevice8BitStorageFeatures s8 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
                                             .pNext = &s16};
   VkPhysicalDeviceShaderFloat16Int8FeaturesKHR fi8 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR, .pNext = &s8};
   VkPhysicalDeviceFeatures2 feat2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &fi8};
   uint32_t next = 0;
   int has_f16i8_ext = 0, has_8bit_ext = 0;
   vkEnumerateDeviceExtensionProperties(pd, NULL, &next, NULL);
   VkExtensionProperties *eps = calloc(next ? next : 1, sizeof(*eps));
   vkEnumerateDeviceExtensionProperties(pd, NULL, &next, eps);
   for (uint32_t e = 0; e < next; e++)
      if (!strcmp(eps[e].extensionName, "VK_KHR_shader_float16_int8"))
         has_f16i8_ext = 1;
      else if (!strcmp(eps[e].extensionName, "VK_KHR_8bit_storage"))
         has_8bit_ext = 1;
   free(eps);
   vkGetPhysicalDeviceFeatures2(pd, &feat2);
   have_16bit_storage = s16.storageBuffer16BitAccess && s16.uniformAndStorageBuffer16BitAccess;
   have_int16 = feat2.features.shaderInt16;
   have_float16 = has_f16i8_ext && fi8.shaderFloat16;
   have_8bit_storage = has_8bit_ext && s8.storageBuffer8BitAccess && s8.uniformAndStorageBuffer8BitAccess;
   have_int8 = has_f16i8_ext && fi8.shaderInt8;
   have_int64 = feat2.features.shaderInt64;
   printf("ngv_shaders: 16-bit features: storageBuffer16BitAccess %u, uniformAndStorageBuffer16BitAccess %u, "
          "shaderInt16 %u, shaderFloat16 %u (extension %s), shaderInt8 %u, storagePushConstant16 %u\n",
          s16.storageBuffer16BitAccess, s16.uniformAndStorageBuffer16BitAccess, feat2.features.shaderInt16,
          fi8.shaderFloat16, has_f16i8_ext ? "yes" : "no", fi8.shaderInt8, s16.storagePushConstant16);
   VkPhysicalDevice16BitStorageFeatures en16 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
                                                .storageBuffer16BitAccess = have_16bit_storage,
                                                .uniformAndStorageBuffer16BitAccess = have_16bit_storage};
   VkPhysicalDevice8BitStorageFeatures en8 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
                                              .pNext = &en16, .storageBuffer8BitAccess = have_8bit_storage,
                                              .uniformAndStorageBuffer8BitAccess = have_8bit_storage};
   VkPhysicalDeviceShaderFloat16Int8FeaturesKHR enf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR, .pNext = &en8,
      .shaderFloat16 = have_float16, .shaderInt8 = have_int8};
   VkPhysicalDeviceFeatures2 enf2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &enf,
                                     .features = {.shaderInt16 = have_int16, .shaderInt64 = have_int64}};
   const char *dev_exts[2] = {"VK_KHR_shader_float16_int8", "VK_KHR_8bit_storage"};
   uint32_t n_dev_exts = 0;
   if (has_f16i8_ext)
      dev_exts[n_dev_exts++] = "VK_KHR_shader_float16_int8";
   if (has_8bit_ext)
      dev_exts[n_dev_exts++] = "VK_KHR_8bit_storage";
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &enf2, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci, .enabledExtensionCount = n_dev_exts,
                             .ppEnabledExtensionNames = dev_exts};
   if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS)
      return 1;
   vkGetDeviceQueue(dev, 0, 0, &queue);
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = 0};
   if (vkCreateCommandPool(dev, &cpi, NULL, &pool) != VK_SUCCESS)
      return 1;

   unsigned npass = 0, nfail = 0, nskip = 0, nxpass = 0, nskipcfg = 0;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
      char path[600];
      snprintf(path, sizeof(path), "%s/ngv_%s.spv", argv[1], cases[i].name);
      printf("case %s:\n", cases[i].name);
      if (cases[i].extra && !getenv("NGV_EXTRA")) {
         printf("  SKIP (opt-in case: set NGV_EXTRA=1)\n");
         nskipcfg++;
         continue;
      }
      if (in_list(hw_skip, cases[i].name)) {
         printf("  SKIP (NGV_HW_SKIP: %s)\n",
                !strcmp(cases[i].name, "float") || !strcmp(cases[i].name, "intops")
                   ? "needs SFU (sqrt / rcp / div / mod), not in the current RM; f32 VALU is in RM 8003cb02, SFU is M1b"
                : !strncmp(cases[i].name, "atomic", 6) || !strncmp(cases[i].name, "scratch", 7)
 ? "not yet verified on the board; set NGV_HW_SKIP without it to run"
                : !strcmp(cases[i].name, "half16") || !strcmp(cases[i].name, "int16") || !strcmp(cases[i].name, "wide")
 ? "only run on the model so far (/); set NGV_HW_SKIP= to run on the board"
                   : "skipped by request");
         nskipcfg++;
         continue;
      }
      int rc = cases[i].run(path);
      if (rc == CASE_UNSUPPORTED) {
         printf("  SKIP (not supported by this device)\n");
         nskipcfg++;
      } else if (rc == CASE_NOCOMPILE && cases[i].gap) {
         printf("  SKIP (known gap: %s)\n", cases[i].gap);
         nskip++;
      } else if (rc == CASE_NOCOMPILE) {
         printf("  FAIL (pipeline creation failed, see the driver log above)\n");
         nfail++;
      } else if (rc == CASE_PASS && cases[i].gap) {
 printf(" XPASS (the known gap is closed: update the table in ngv_shaders.c and)\n");
         nxpass++;
      } else if (rc == CASE_PASS) {
         printf("  PASS\n");
         npass++;
      } else {
         printf("  %s\n", cases[i].gap ? "XFAIL! (compiled but wrong result)" : "FAIL");
         nfail++;
      }
   }
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("ngv_shaders: %u pass, %u skip (known gaps), %u skip (NGV_HW_SKIP), %u xpass, %u fail -> %s\n", npass, nskip,
          nskipcfg, nxpass, nfail, nfail ? "FAIL" : "OK");
   return nfail ? 1 : 0;
}
