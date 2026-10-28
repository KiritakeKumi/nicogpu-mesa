/* SPDX-License-Identifier: MIT */
/* ngv_core_procs.c -- every core Vulkan 1.0 / 1.1 command must be reachable through vkGetInstanceProcAddr /
 * vkGetDeviceProcAddr, also the ones whose feature is off.
 *
 * A conformance suite calls a core command through the pointer it fetched, without testing it for NULL: a NULL entry
 * point is a SIGSEGV in the application (dEQP api.null_handle.destroy_buffer_view, api.buffer_view.*, ...), whereas
 * a command that exists and fails cleanly (VK_ERROR_FEATURE_NOT_PRESENT, or a no-op for vkDestroy*) is a normal test
 * result.
 *
 *   ngv_core_procs [ngv_core_cmds.txt]
 *
 * ngv_core_cmds.txt lines: "<VK_VERSION_1_x> <command> <type of the first parameter>" (generated from vk.xml, see
 * the header of that file).  The test creates an apiVersion 1.1 instance and a device (first physical device, queue
 * family 0), then looks up every command: VkDevice / VkQueue / VkCommandBuffer commands with vkGetDeviceProcAddr,
 * the others with vkGetInstanceProcAddr.  Prints the NULL ones; exit 1 if there is any.
 *
 *   gcc -O1 -g -o ngv_core_procs ngv_core_procs.c -lvulkan
 */
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

int
main(int argc, char **argv)
{
   const char *list = argc > 1 ? argv[1] : "ngv_core_cmds.txt";
   VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai};
   VkInstance inst;
   if (vkCreateInstance(&ici, NULL, &inst)) {
      puts("ngv_core_procs: vkCreateInstance failed");
      return 1;
   }
   uint32_t n = 1;
   VkPhysicalDevice pd;
   vkEnumeratePhysicalDevices(inst, &n, &pd);
   float prio = 1.0f;
   VkDeviceQueueCreateInfo q = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                .queueCount = 1,
                                .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &q};
   VkDevice dev;
   if (vkCreateDevice(pd, &dci, NULL, &dev)) {
      puts("ngv_core_procs: vkCreateDevice failed");
      return 1;
   }
   FILE *f = fopen(list, "r");
   if (!f) {
      printf("ngv_core_procs: cannot open %s\n", list);
      return 1;
   }
   char line[512], ver[64], name[128], type[128];
   int total = 0, missing = 0;
   while (fgets(line, sizeof line, f)) {
      if (line[0] == '#' || sscanf(line, "%63s %127s %127s", ver, name, type) != 3)
         continue;
      int devlevel = !strcmp(type, "VkDevice") || !strcmp(type, "VkQueue") || !strcmp(type, "VkCommandBuffer");
      void *p = devlevel ? (void *)vkGetDeviceProcAddr(dev, name) : (void *)vkGetInstanceProcAddr(inst, name);
      total++;
      if (!p) {
         printf("NULL %s %s %s\n", ver, devlevel ? "device" : "instance", name);
         missing++;
      }
   }
   fclose(f);
   printf("ngv_core_procs: %d commands, %d NULL: %s\n", total, missing, missing ? "FAIL" : "PASS");
   return missing ? 1 : 0;
}
