/* SPDX-License-Identifier: MIT */
/* ngv_device.c -- instance, physical device, device and queue of the Nicogpu
 * Vulkan driver (skeleton). See */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ngv_private.h"
#include "ngv_units.h"

VKAPI_ATTR VkResult VKAPI_CALL
ngv_EnumerateInstanceVersion(uint32_t *pApiVersion)
{
   *pApiVersion = NGV_API_VERSION;
   return VK_SUCCESS;
}

const struct vk_instance_extension_table ngv_instance_extensions = {
   .KHR_device_group_creation = true,
   .KHR_external_fence_capabilities = true,
   .KHR_external_memory_capabilities = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_get_physical_device_properties2 = true,
   .EXT_debug_report = true,
   .EXT_debug_utils = true,
};

/* Device extensions: only what is implemented (section 2.2, section 2). The device is Vulkan 1.1
 * (NGV_API_VERSION), so the 1.2 / 1.3 functionality it has is exposed here as extensions:
 *   - timeline semaphores, host query reset, synchronization2: the Mesa runtime (timeline emulation on the binary
 * syncobj, host query reset) plus our packets;
 *   - private data: entirely the runtime (vk_object_base);
 *   - uniform buffer standard layout, scalar block layout: layout rules only -- spirv_to_nir applies the explicit
 *     offsets, buffer accesses are byte addressed and need no alignment (ISA v0.1.2 "no alignment requirement",
 * the descriptor spec section 3.1, the LSU splits line-crossing accesses);
 *   - subgroup size control: the subgroup size is always 32 (nicogpu_device_info.warp_size) and a workgroup is cut into
 *     warps in local-invocation-index order, so every flag / required size the extension allows is already met;
 * - SPIR-V 1.4 : spirv_to_nir takes the version from the module, nothing in the compiler depends on it;
 *     its required VK_KHR_shader_float_controls is reported with no controllable mode (ngv_get_float_controls): the
 *     SPIR-V FP capabilities (DenormPreserve / DenormFlushToZero / RoundingModeRTE / RoundingModeRTZ /
 *     SignedZeroInfNanPreserve) are not accepted, a shader using them fails to compile;
 * - zero-initialize workgroup memory: ngv_nir.c writes the zeros (pass, checks);
 *   - driver properties: driverName / driverInfo / driverID filled in ngv_uuid.c (driverID stays 0: no VkDriverId is registered);
 * - 16-bit types (): VK_KHR_16bit_storage (storageBuffer16BitAccess,
 * uniformAndStorageBuffer16BitAccess) and VK_KHR_shader_float16_int8 (shaderFloat16, shaderInt16 and, since,
 *     shaderInt8) -- 16-bit values are promoted to 32-bit by ngv_nir.c, memory accesses are ld/st u16, f16<->f32 is fcvt;
 * - : VK_KHR_8bit_storage (8-bit like 16-bit), shaderInt64 + VK_KHR_shader_atomic_int64
 *     (64-bit values are register pairs, 64-bit vectors are split into scalar accesses, atom.*.u64 / b64),
 *     VK_KHR_shader_subgroup_extended_types (8 / 16 / 64-bit subgroup operations), VK_KHR_shader_subgroup_rotate
 *     (rotate = shuffle, lowered by nir_lower_subgroups), VK_EXT_shader_subgroup_ballot / _vote (the ARB-style forms of
 *     ballot / vote), VK_KHR_workgroup_memory_explicit_layout, VK_KHR_buffer_device_address,
 *     VK_EXT_shader_replicated_composites, VK_EXT_shader_atomic_float (float32 load / store / exchange only);
 *   - vulkan memory model: on by default, NGV_VULKAN_MEMORY_MODEL=0 turns it off (ngv_get_device_extensions). */
static const struct vk_device_extension_table ngv_device_extensions = {
   .KHR_16bit_storage = true,
   .KHR_8bit_storage = true,
   .KHR_bind_memory2 = true,
   .KHR_buffer_device_address = true,
   .KHR_driver_properties = true,
   .KHR_get_memory_requirements2 = true,
   .KHR_sampler_mirror_clamp_to_edge = true,
   .KHR_shader_atomic_int64 = true,
   .KHR_shader_float16_int8 = true,
   .KHR_shader_float_controls = true,
   .KHR_shader_subgroup_extended_types = true,
   .KHR_shader_subgroup_rotate = true,
   .KHR_spirv_1_4 = true,
   .KHR_synchronization2 = true,
   .KHR_timeline_semaphore = true,
   .KHR_uniform_buffer_standard_layout = true,
   .KHR_workgroup_memory_explicit_layout = true,
   .KHR_zero_initialize_workgroup_memory = true,
   .EXT_host_query_reset = true,
   .EXT_image_view_min_lod = true,
   .EXT_non_seamless_cube_map = true,
   .EXT_private_data = true,
   .EXT_scalar_block_layout = true,
   .EXT_shader_atomic_float = true,
   .EXT_shader_replicated_composites = true,
   .EXT_shader_subgroup_ballot = true,
   .EXT_shader_subgroup_vote = true,
   .EXT_subgroup_size_control = true,
};

/* VK_KHR_vulkan_memory_model: default ON since 2026-10-24 (user ruling reversing the Q5 "default off"; the g4 hal gate with
 * the switch on had 0 regressions against g3: 833 more Pass, the 1504 new Fails are all storage-image cases waived to P2).
 * The compiler side is memmodel.py (conventions still PROVISIONAL). NGV_VULKAN_MEMORY_MODEL=0 turns the extension
 * and vulkanMemoryModel / vulkanMemoryModelDeviceScope off again; unset, empty or any other value keeps them on. */
bool
ngv_vulkan_memory_model_enabled(void)
{
   const char *e = getenv("NGV_VULKAN_MEMORY_MODEL");
   return !(e && strcmp(e, "0") == 0);
}

void
ngv_get_device_extensions(const struct ngv_physical_device *pdev, struct vk_device_extension_table *e)
{
   (void)pdev;
   *e = ngv_device_extensions;
   e->KHR_vulkan_memory_model = ngv_vulkan_memory_model_enabled();
}

/* ---- physical device -------------------------------------------------------- */

/* ---- subgroups ------------------------------------------------------------------- */

_Static_assert(NGV_SG_BASIC == VK_SUBGROUP_FEATURE_BASIC_BIT && NGV_SG_VOTE == VK_SUBGROUP_FEATURE_VOTE_BIT &&
                  NGV_SG_ARITHMETIC == VK_SUBGROUP_FEATURE_ARITHMETIC_BIT &&
                  NGV_SG_BALLOT == VK_SUBGROUP_FEATURE_BALLOT_BIT && NGV_SG_SHUFFLE == VK_SUBGROUP_FEATURE_SHUFFLE_BIT &&
                  NGV_SG_SHUFFLE_RELATIVE == VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT &&
                  NGV_SG_CLUSTERED == VK_SUBGROUP_FEATURE_CLUSTERED_BIT && NGV_SG_QUAD == VK_SUBGROUP_FEATURE_QUAD_BIT,
               "ngv_units.h: NGV_SG_* != VkSubgroupFeatureFlagBits");

/* The optional SP units the compiled code may use: the same resolution as the ngcomp `--units` argument
 * (ngv_ngcomp.c ngv_target_units: $NGV_NGCOMP_UNITS, else the device's report, else the legacy "sfu"), so what is
 * reported is what the compiler is allowed to emit. */
static uint32_t
ngv_sp_units(const struct ngv_physical_device *pdev)
{
   char buf[NGV_UNITS_BUFSZ];
   return ngv_units_parse(ngv_units_resolve(getenv("NGV_NGCOMP_UNITS"),
                                            (pdev->info.info_flags & NICOGPU_INFO_F_SP_UNITS) != 0, pdev->info.sp_units, buf,
                                            NULL));
}

static void
ngv_get_subgroup_features(const struct ngv_physical_device *pdev, struct vk_features *f)
{
   const bool arith = ngv_subgroup_arith_enabled(getenv("NGV_SUBGROUP_ARITH"));
   const uint32_t ops = ngv_subgroup_ops(ngv_sp_units(pdev), arith);
   /* OpGroupNonUniformBroadcast with a dynamically uniform, non-constant id: shfl.idx, or the bit-serial read */
   f->subgroupBroadcastDynamicId = true;
 /* VK_KHR_shader_subgroup_rotate : nir_lower_subgroups turns rotate / clustered rotate into shuffles
    * (ngv_nir.c lower_rotate_to_shuffle), so they exist exactly when SHUFFLE is reported. */
   f->shaderSubgroupRotate = !!(ops & NGV_SG_SHUFFLE);
   f->shaderSubgroupRotateClustered = !!(ops & NGV_SG_SHUFFLE);
 /* VK_KHR_shader_subgroup_extended_types : subgroup operations on the 16-bit types the device has
    * (shaderInt16 / shaderFloat16); 8-bit and 64-bit stay off with their base features. */
   f->shaderSubgroupExtendedTypes = true;
 /* VK_EXT_subgroup_size_control : the subgroup size is always warp_size and a workgroup is cut into warps
    * in local-invocation-index order, so every flag / required size the extension allows is already met. */
   f->subgroupSizeControl = true;
   f->computeFullSubgroups = true;
}

/* Subgroup = warp: fixed size, compute only; the operations depend on the unit set (ngv_units.h ngv_subgroup_ops). */
static void
ngv_get_subgroup_properties(const struct ngv_physical_device *pdev, struct vk_properties *p)
{
   const struct nicogpu_device_info *i = &pdev->info;
   const bool arith = ngv_subgroup_arith_enabled(getenv("NGV_SUBGROUP_ARITH")); /* =0: do not report ARITHMETIC / CLUSTERED */
   const uint32_t ws = i->warp_size ? i->warp_size : 32;
   p->subgroupSize = i->warp_size;
   p->subgroupSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
   p->subgroupSupportedOperations = ngv_subgroup_ops(ngv_sp_units(pdev), arith);
   p->subgroupQuadOperationsInAllStages = false;
 /* VK_EXT_subgroup_size_control : min = max = warp_size, and a workgroup is at most
    * ceil(max_workgroup_threads / warp_size) warps */
   p->minSubgroupSize = i->warp_size;
   p->maxSubgroupSize = i->warp_size;
   p->maxComputeWorkgroupSubgroups = (i->max_workgroup_threads + ws - 1) / ws;
   p->requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;
}

/* VK_KHR_shader_float_controls properties (needed by VK_KHR_spirv_1_4): the device can run the SPIR-V without any
 * float-control execution mode only -- the compiler does not honour DenormFlushToZero / RTE / RTZ /
 * SignedZeroInfNanPreserve (they are not in the accepted capability set), so those shaderXxxFloatNN are false and
 * the modes cannot be set independently. The one exception is shaderDenormPreserveFloat16 : 16-bit float
 * arithmetic is done in binary32 (MODE = 0 keeps subnormals) and rounded back with fcvt, so f16 subnormals are kept;
 * that report also makes the runtime accept the DenormPreserve capability (the execution mode itself changes nothing). */
static void
ngv_get_float_controls(struct vk_properties *p)
{
   p->denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE;
   p->roundingModeIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE;
   p->shaderSignedZeroInfNanPreserveFloat16 = p->shaderSignedZeroInfNanPreserveFloat32 =
      p->shaderSignedZeroInfNanPreserveFloat64 = false;
   p->shaderDenormPreserveFloat16 = true;
   p->shaderDenormPreserveFloat32 = p->shaderDenormPreserveFloat64 = false;
   p->shaderDenormFlushToZeroFloat16 = p->shaderDenormFlushToZeroFloat32 = p->shaderDenormFlushToZeroFloat64 = false;
   p->shaderRoundingModeRTEFloat16 = p->shaderRoundingModeRTEFloat32 = p->shaderRoundingModeRTEFloat64 = false;
   p->shaderRoundingModeRTZFloat16 = p->shaderRoundingModeRTZFloat32 = p->shaderRoundingModeRTZFloat64 = false;
}

/* Features: true only where the whole path (NIR lowering, the ngcomp printer and isel, the hardware) does it; the
 * table with the reasons for every 1.0 / 1.1 feature and the 1.2 / 1.3 ones behind extensions is section 1.
 * Subgroup-related reporting (incl. the size control) is all in ngv_get_subgroup_features /
 * ngv_get_subgroup_properties above; 16-bit types: -- storageBuffer16BitAccess,
 * uniformAndStorageBuffer16BitAccess, shaderInt16, shaderFloat16 on; storagePushConstant16 / 8 and storageInputOutput16
 * off (push constants and stage IO have no 8 / 16-bit path). 8-bit, 64-bit, BDA, explicit layout,...:. */
void
ngv_get_features(const struct ngv_physical_device *pdev, struct vk_features *f)
{
   (void)pdev;
   const bool mm = ngv_vulkan_memory_model_enabled();
   *f = (struct vk_features){0};

   /* Vulkan 1.0.  robustBufferAccess (required): every uniform / storage buffer access goes through ld/st/atom.buf
 * with the descriptor's size = the bound range, and the LSU bounds-checks each 4-byte segment (the descriptor spec section 3.2:
    * reads 0, writes and atomics dropped), in the model and in the RTL (ng_lsu.sv byte-valid mask).  Texel buffers
    * and vertex input have no path at all yet. */
   f->robustBufferAccess = true;

 /* : 16-bit types, see */
   f->shaderInt16 = true;
   f->storageBuffer16BitAccess = true;
   f->uniformAndStorageBuffer16BitAccess = true;
   f->shaderFloat16 = true;
 /* : 8-bit types take the 16-bit path (ngv_nir.c ngv_small_bits); storagePushConstant8 stays off like the
    * 16-bit one */
   f->shaderInt8 = true;
   f->storageBuffer8BitAccess = true;
   f->uniformAndStorageBuffer8BitAccess = true;
 /* : shaderInt64 -- 64-bit values live in register pairs (nir_lower_int64 for what isel lacks), 64-bit
    * vectors are read / written as scalar 64-bit accesses (ngv_mem_access_cb); atom.*.u64 / b64 exist in the ISA
    * and in ngcomp (isel_atomic.py) for buffers, global and shared.  The model runs all of it; the 64-bit atomics
 * on the board RM are not verified (PROVISIONAL). */
   f->shaderInt64 = true;
   f->shaderBufferInt64Atomics = true;
   f->shaderSharedInt64Atomics = true;
 /* : textures and images (tex.* / img.*). Features that the sampler / image descriptors and the
    * ngcomp lowering do all of, checked by dEQP-VK.texture.* (compute variants) and dEQP-VK.image.*: anisotropic filtering
    * (A1' of ERRATA v0.1.8 T8), cube array views, mirror-clamp addressing (address mode 4), non-seamless cube sampling
    * (sampler bit), the image view min_lod, storage images in every format of the table (extended formats) and loads / stores
    * without a format qualifier (the format is in the descriptor).  Not reported: textureCompressionBC (BLIT_SRC is not
 * implemented, which the feature requires for every BC format; ETC2 / ASTC are), sampledImage / storageImage
    * ArrayDynamicIndexing (a non-constant descriptor index needs a waterfall loop in ngcomp), shaderImageGatherExtended
    * (textureGatherOffsets has no dEQP compute coverage), shaderStorageImageMultisample (no multisample images), sparse
    * residency, VK_EXT_sampler_filter_minmax and VK_EXT_filter_cubic (no CTS evidence / no cubic filter in the unit). */
   f->samplerAnisotropy = true;
   f->imageCubeArray = true;
   f->samplerMirrorClampToEdge = true;
   f->nonSeamlessCubeMap = true;
   f->minLod = true;
   f->shaderStorageImageExtendedFormats = true;
   f->shaderStorageImageReadWithoutFormat = true;
   f->shaderStorageImageWriteWithoutFormat = true;
   /* Vulkan 1.1.  multiview (required): render passes are the runtime's (vk_render_pass.c), a view mask is accepted;
    * with no graphics queue a render pass instance can never be begun, so there is nothing else to implement. */
   f->multiview = true;

   /* Vulkan 1.2 functionality behind extensions */
   f->timelineSemaphore = true;
   f->hostQueryReset = true;
   f->uniformBufferStandardLayout = true;
   f->scalarBlockLayout = true;
   f->vulkanMemoryModel = mm;
   f->vulkanMemoryModelDeviceScope = mm;
 f->vulkanMemoryModelAvailabilityVisibilityChains = mm; /* : section 2 */

 /* : device addresses are the winsys VAs (ngv_BindBufferMemory2 stores them, the runtime returns them);
    * capture / replay and multi-device stay off */
   f->bufferDeviceAddress = true;
 /* Workgroup memory with explicit layout (VK_KHR_workgroup_memory_explicit_layout): shared memory is
    * byte addressed through a 32-bit offset (nir_address_format_32bit_offset), aliasing blocks are plain offsets;
    * 8 / 16-bit accesses are the scalar ones (ngv_mem_access_cb). */
   f->workgroupMemoryExplicitLayout = true;
   f->workgroupMemoryExplicitLayoutScalarBlockLayout = true;
   f->workgroupMemoryExplicitLayout8BitAccess = true;
   f->workgroupMemoryExplicitLayout16BitAccess = true;
 /* VK_EXT_shader_atomic_float : only the plain float32 load / store / exchange atomics (the 32-bit integer
    * ones on the same bits).  atom.*.fadd/fmin/fmax.f32 are flagged `optional` in the ISA and the driver has no
    * capability bit for them (the RTL ng_lsu_fatomic is a separate option), so the Add / MinMax features stay off;
    * float64 needs shaderFloat64 (no FP64 unit in the RM). */
   f->shaderBufferFloat32Atomics = true;
   f->shaderSharedFloat32Atomics = true;
   /* VK_EXT_shader_replicated_composites: OpConstantCompositeReplicateEXT etc. are expanded by spirv_to_nir */
   f->shaderReplicatedComposites = true;

   /* Vulkan 1.3 functionality behind extensions */
   f->synchronization2 = true;
   f->privateData = true;
   f->shaderZeroInitializeWorkgroupMemory = true;

   ngv_get_subgroup_features(pdev, f);
}

void
ngv_get_properties(const struct ngv_physical_device *pdev, struct vk_properties *p)
{
   const struct nicogpu_device_info *i = &pdev->info;
   const bool is_hw = !strcmp(i->backend, "drm");
   const VkSampleCountFlags none = VK_SAMPLE_COUNT_1_BIT;

   *p = (struct vk_properties){
      .apiVersion = NGV_API_VERSION,
      .driverVersion = vk_get_driver_version(),
      .vendorID = i->vendor_id,
      .deviceID = i->device_id,
      .deviceType = is_hw ? VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                    : !strcmp(i->backend, "sim") ? VK_PHYSICAL_DEVICE_TYPE_CPU
                                                 : VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,

 /* Buffers / descriptors (the descriptor spec section 3.3, 8; PROVISIONAL
       * counts: bindless in hardware, the driver only needs "large"). */
      .maxUniformBufferRange = UINT32_MAX,
      .maxStorageBufferRange = UINT32_MAX,
      .minUniformBufferOffsetAlignment = 16,
      .minStorageBufferOffsetAlignment = 4,
      .maxPushConstantsSize = 128,
      .maxMemoryAllocationCount = 4096,
      .maxSamplerAllocationCount = 4000,
      .bufferImageGranularity = 1,
      /* ngcomp reads the base VA of sets 0..3 from u2..u9 and has no root-table path for sets >= 4 (isel.py
 * sel_load_buffer_desc), so 4 -- the Vulkan minimum -- is what a pipeline can really use. */
      .maxBoundDescriptorSets = 4,
      .maxPerStageDescriptorSamplers = 1u << 20,
      .maxPerStageDescriptorUniformBuffers = 1u << 20,
      .maxPerStageDescriptorStorageBuffers = 1u << 20,
      .maxPerStageDescriptorSampledImages = 1u << 20,
      .maxPerStageDescriptorStorageImages = 1u << 20,
      .maxPerStageDescriptorInputAttachments = 1u << 20,
      .maxPerStageResources = 1u << 20,
      .maxDescriptorSetSamplers = 1u << 20,
      .maxDescriptorSetUniformBuffers = 1u << 20,
      .maxDescriptorSetUniformBuffersDynamic = 16,
      .maxDescriptorSetStorageBuffers = 1u << 20,
      .maxDescriptorSetStorageBuffersDynamic = 16,
      .maxDescriptorSetSampledImages = 1u << 20,
      .maxDescriptorSetStorageImages = 1u << 20,
      .maxDescriptorSetInputAttachments = 1u << 20,

 /* Images (descriptor format limits, the descriptor spec section 9). */
      .maxImageDimension1D = 16384,
      .maxImageDimension2D = 16384,
      .maxImageDimension3D = 2048,
      .maxImageDimensionCube = 16384,
      .maxImageArrayLayers = 2048,
      .maxTexelBufferElements = 1u << 28,
      .maxSamplerLodBias = 15,
      .maxSamplerAnisotropy = 16,
      .sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT,
      .sampledImageIntegerSampleCounts = none,
      .sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT,
      .sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT,
      .storageImageSampleCounts = none,
 /* Graphics-only limits : "Required Limits" gives them a minimum for every implementation, whatever
       * its queues.  This driver has no graphics queue and no graphics pipelines (P2), so no application can reach
       * them; they carry the Vulkan 1.0 minimums (the fine-grained features they depend on are all off: geometry,
       * tessellation, multiViewport, sampleRateShading, fullDrawIndexUint32, multiDrawIndirect, ...). */
      .framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .framebufferStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .framebufferNoAttachmentsSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .maxSampleMaskWords = 1,
      .maxVertexInputAttributes = 16,
      .maxVertexInputBindings = 16,
      .maxVertexInputAttributeOffset = 2047,
      .maxVertexInputBindingStride = 2048,
      .maxVertexOutputComponents = 64,
      .maxFragmentInputComponents = 64,
      .maxFragmentOutputAttachments = 4,
      .maxFragmentCombinedOutputResources = 4,
      .maxDrawIndexedIndexValue = (1u << 24) - 1,
      .maxDrawIndirectCount = 1,
      .subPixelPrecisionBits = 4,
      /* VkPhysicalDeviceMultiviewProperties: the minimums that go with multiview = true (see ngv_get_features) */
      .maxMultiviewViewCount = 6,
      .maxMultiviewInstanceIndex = (1u << 27) - 1,
      .minTexelOffset = -8,
      .maxTexelOffset = 7,
      .minTexelGatherOffset = -8,
      .maxTexelGatherOffset = 7,
      .mipmapPrecisionBits = 8,
      .subTexelPrecisionBits = 8,
      .maxViewports = 1,
      .maxViewportDimensions = {16384, 16384},
      .viewportBoundsRange = {-32768.0f, 32767.0f},
      .maxFramebufferWidth = 16384,
      .maxFramebufferHeight = 16384,
      .maxFramebufferLayers = 256,
      .maxColorAttachments = 8,
      .pointSizeRange = {1.0f, 1.0f},
      .lineWidthRange = {1.0f, 1.0f},

      /* Compute: the part this driver exists for.  Limits come from the
       * device (HAL info) and the CP/ABI specs. */
      .maxComputeSharedMemorySize = i->max_shared_bytes,
      .maxComputeWorkGroupCount = {65535, 65535, 65535}, /* PROVISIONAL: Vulkan minimum; HW allows 2^32-1 */
      .maxComputeWorkGroupInvocations = i->max_workgroup_threads,
      .maxComputeWorkGroupSize = {i->max_workgroup_threads, i->max_workgroup_threads, i->max_workgroup_threads},
      .maxTimelineSemaphoreValueDifference = INT64_MAX,
      .maxMemoryAllocationSize = 1ull << 31, /* PROVISIONAL */
      .maxPerSetDescriptors = 1u << 20,
      .maxBufferSize = 1ull << 31,

      /* Mapping / alignment / misc */
      .sparseAddressSpaceSize = 0,
      .minMemoryMapAlignment = NICOGPU_PAGE_SIZE,
      .minTexelBufferOffsetAlignment = 16,
      .optimalBufferCopyOffsetAlignment = 4,
      .optimalBufferCopyRowPitchAlignment = 4,
      .nonCoherentAtomSize = 64,
      .timestampComputeAndGraphics = !!(i->features & NICOGPU_FEATURE_TIMESTAMP),
      .timestampPeriod = i->timestamp_freq_hz ? 1.0e9f / (float)i->timestamp_freq_hz : 1.0f,
      .discreteQueuePriorities = 2,
      .standardSampleLocations = true,
      .strictLines = false,
   };
   ngv_get_float_controls(p);
 ngv_get_subgroup_properties(pdev, p); /* / : every subgroup property is set there, only there */
   /* One BO is one contiguous range: the kernel's per-BO cap (ABI v5 gtt_bo_max; 1 GiB since 2026-10-25, needs a
    * CMA pool that big: cma=2G on the board) bounds every single allocation/buffer.  Advertising more than the kernel
    * can allocate lets ncnn/vkpeak (copy-h2d) allocate a huge staging buffer, get OUT_OF_DEVICE_MEMORY and
    * dereference the NULL result.  VkPhysicalDeviceMaintenance3Properties::maxMemoryAllocationSize must be >= 2^30
 * (section 4). */
   if (i->gtt_bo_max && i->gtt_bo_max < p->maxMemoryAllocationSize) {
      p->maxMemoryAllocationSize = i->gtt_bo_max;
      p->maxBufferSize = i->gtt_bo_max;
   }
}
