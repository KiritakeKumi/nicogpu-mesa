/* SPDX-License-Identifier: MIT */
/* ngv_uuid.c -- the three physical device UUIDs and the driver identity (section 16).
 *
 *   driverUUID         sha1("nicogpu driver" + pointer size + build id of libvulkan_nicogpu.so): one value per driver build,
 *                      equal for every device and every process of the build.
 *   deviceUUID         sha1("nicogpu device" + backend + vendor / device / revision + location): one value per physical
 *                      device, stable across processes and reboots.  Location = the PCI address of the DRM render node's
 *                      parent (/sys/class/drm/<node>/device -> 0000:03:00.0) on the drm backend; the sim / stub backends
 *                      have exactly one device per process and no location (their value is a constant).
 *   pipelineCacheUUID  sha1("nicogpu pipeline cache" + driver build id + ISA + the compile-relevant device limits +
 * ngv_ngcomp_identity): changes when the driver binary, the compiler (ngcomp *.py, the ISA
 *                      table), the unit set / fp contraction mode / memory-model switch or the ISA version changes, so a
 *                      VkPipelineCache blob written by another build is rejected by the loader / runtime header check.
 * The build id is the GNU build-id note of the shared object (meson links with --build-id=sha1); without one
 * (disk_cache_get_function_identifier fails) the compile time of this file stands in -- stable per build, but it does not
 * follow edits to the other files.
 *
 * driverID: VkDriverId has no entry for this driver (the enum is a Khronos registry; nobody may borrow another vendor's
 * value), so it stays 0 (= "unregistered") until an id is registered.  driverName is "nicogpu". */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util/disk_cache.h"
#include "util/mesa-sha1.h"

#include "ngv_private.h"

static void
sha1_str(struct mesa_sha1 *ctx, const char *s)
{
   _mesa_sha1_update(ctx, s, strlen(s) + 1); /* with the NUL: ("ab","c") != ("a","bc") */
}

static void
sha1_u32(struct mesa_sha1 *ctx, uint32_t v)
{
   _mesa_sha1_update(ctx, &v, sizeof(v));
}

/* The identity of this binary.  Same function address => same object, so the note found is ours. */
static void
sha1_driver_build(struct mesa_sha1 *ctx)
{
   if (disk_cache_get_function_identifier((void *)ngv_physical_device_init_uuids, ctx))
      return;
   sha1_str(ctx, "no-build-id " __DATE__ " " __TIME__);
}

/* PCI address (or whatever the parent device is called in sysfs) of the DRM node in `uri` ("drm:/dev/dri/renderD128");
 * "" when the uri is not a DRM node or sysfs has no link. */
static void
device_location(const char *uri, char out[64])
{
   out[0] = 0;
   if (!uri || strncmp(uri, "drm:", 4))
      return;
   const char *node = strrchr(uri + 4, '/');
   node = node ? node + 1 : uri + 4;
   if (!*node || strchr(node, '.'))
      return;
   char link[PATH_MAX], target[PATH_MAX];
   snprintf(link, sizeof(link), "/sys/class/drm/%s/device", node);
   ssize_t n = readlink(link, target, sizeof(target) - 1);
   if (n <= 0)
      return;
   target[n] = 0;
   const char *base = strrchr(target, '/');
   snprintf(out, 64, "%s", base ? base + 1 : target);
}

void
ngv_physical_device_init_uuids(struct ngv_physical_device *pdev)
{
   struct vk_properties *p = &pdev->vk.properties;
   const struct nicogpu_device_info *i = &pdev->info;
   struct mesa_sha1 ctx;
   uint8_t sha[20];

   _mesa_sha1_init(&ctx);
   sha1_str(&ctx, "nicogpu driver");
   sha1_u32(&ctx, (uint32_t)sizeof(void *));
   sha1_driver_build(&ctx);
   _mesa_sha1_final(&ctx, sha);
   memcpy(p->driverUUID, sha, VK_UUID_SIZE);

   char loc[64];
   device_location(pdev->desc.uri, loc);
   _mesa_sha1_init(&ctx);
   sha1_str(&ctx, "nicogpu device");
   sha1_str(&ctx, i->backend);
   sha1_u32(&ctx, i->vendor_id);
   sha1_u32(&ctx, i->device_id);
   sha1_u32(&ctx, i->revision);
   sha1_str(&ctx, loc);
   _mesa_sha1_final(&ctx, sha);
   memcpy(p->deviceUUID, sha, VK_UUID_SIZE);

   _mesa_sha1_init(&ctx);
   sha1_str(&ctx, "nicogpu pipeline cache");
   sha1_driver_build(&ctx);
   sha1_str(&ctx, i->isa_name);
   sha1_str(&ctx, i->isa_version);
   sha1_u32(&ctx, i->warp_size);
   sha1_u32(&ctx, i->max_vregs);
   sha1_u32(&ctx, i->num_uregs);
   sha1_u32(&ctx, i->max_shared_bytes);
   ngv_ngcomp_identity(pdev, &ctx);
   _mesa_sha1_final(&ctx, sha);
   memcpy(p->pipelineCacheUUID, sha, VK_UUID_SIZE);

   /* VkPhysicalDeviceDriverProperties (VK_KHR_driver_properties): the id is unregistered, see the header comment. */
   p->driverID = 0;
   snprintf(p->driverName, sizeof(p->driverName), "nicogpu");
   snprintf(p->driverInfo, sizeof(p->driverInfo), "Mesa nicogpu (winsys %s, build %02x%02x%02x%02x)", pdev->ops->name,
            p->driverUUID[0], p->driverUUID[1], p->driverUUID[2], p->driverUUID[3]);
}
