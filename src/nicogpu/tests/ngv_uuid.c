/* SPDX-License-Identifier: MIT */
/* ngv_uuid.c -- driver identity and the three physical device UUIDs (section 16). Instance + physical device only
 * (no VkDevice).  Checks, within one process:
 *   - driverName == "nicogpu", driverInfo starts with "Mesa nicogpu", driverID == 0 (no registered VkDriverId);
 *   - the three UUIDs are non-zero, pairwise different, and the same when the instance is created a second time;
 *   - vkGetPhysicalDeviceProperties2 agrees with vkGetPhysicalDeviceProperties on pipelineCacheUUID.
 * Prints "uuid: device=<hex> driver=<hex> cache=<hex>" for ngv_uuid.sh (cross-process / cross-configuration checks).
 * Exit status 0 = all checks passed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define EXPECT(c)                                                    \
   do {                                                              \
      if (!(c)) {                                                    \
         fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
         return 1;                                                   \
      }                                                              \
   } while (0)

struct ids {
   uint8_t device[VK_UUID_SIZE], driver[VK_UUID_SIZE], cache[VK_UUID_SIZE];
   char name[VK_MAX_DRIVER_NAME_SIZE], info[VK_MAX_DRIVER_INFO_SIZE];
   VkDriverId id;
   int core_ids_match;
};

static int
query(struct ids *o)
{
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS)
      return 1;
   uint32_t n = 0;
   vkEnumeratePhysicalDevices(inst, &n, NULL);
   EXPECT(n == 1);
   VkPhysicalDevice pd;
   vkEnumeratePhysicalDevices(inst, &n, &pd);

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   memcpy(o->cache, props.pipelineCacheUUID, VK_UUID_SIZE);

   VkPhysicalDeviceIDProperties idp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
   VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, .pNext = &idp};
   VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv};
   vkGetPhysicalDeviceProperties2(pd, &p2);
   memcpy(o->device, idp.deviceUUID, VK_UUID_SIZE);
   memcpy(o->driver, idp.driverUUID, VK_UUID_SIZE);
   snprintf(o->name, sizeof(o->name), "%s", drv.driverName);
   snprintf(o->info, sizeof(o->info), "%s", drv.driverInfo);
   o->id = drv.driverID;
   o->core_ids_match = !memcmp(p2.properties.pipelineCacheUUID, props.pipelineCacheUUID, VK_UUID_SIZE);
   vkDestroyInstance(inst, NULL);
   return 0;
}

static int
nonzero(const uint8_t *u)
{
   for (int i = 0; i < VK_UUID_SIZE; i++)
      if (u[i])
         return 1;
   return 0;
}

static void
hex(const uint8_t *u, char *out)
{
   for (int i = 0; i < VK_UUID_SIZE; i++)
      sprintf(out + 2 * i, "%02x", u[i]);
}

int
main(void)
{
   struct ids a, b;
   if (query(&a) || query(&b)) {
      fprintf(stderr, "FAIL vkCreateInstance\n");
      return 1;
   }
   EXPECT(!strcmp(a.name, "nicogpu"));
   EXPECT(!strncmp(a.info, "Mesa nicogpu", 12));
   EXPECT(a.id == 0); /* unregistered: no VkDriverId may be borrowed */
   EXPECT(a.core_ids_match);
   EXPECT(nonzero(a.device) && nonzero(a.driver) && nonzero(a.cache));
   EXPECT(memcmp(a.device, a.driver, VK_UUID_SIZE) && memcmp(a.device, a.cache, VK_UUID_SIZE) &&
          memcmp(a.driver, a.cache, VK_UUID_SIZE));
   EXPECT(!memcmp(a.device, b.device, VK_UUID_SIZE) && !memcmp(a.driver, b.driver, VK_UUID_SIZE) &&
          !memcmp(a.cache, b.cache, VK_UUID_SIZE));
   char d[33], r[33], c[33];
   hex(a.device, d);
   hex(a.driver, r);
   hex(a.cache, c);
   printf("uuid: device=%s driver=%s cache=%s\n", d, r, c);
   printf("driver: name=\"%s\" info=\"%s\" id=%d\n", a.name, a.info, (int)a.id);
   printf("ngv_uuid: OK\n");
   return 0;
}
