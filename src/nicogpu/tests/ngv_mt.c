/* SPDX-License-Identifier: MIT */
/* ngv_mt.c -- multi-thread stress test of the ngv driver's winsys thread contract (batch 3).
 *
 * Several application threads create/signal/wait/destroy timeline semaphores and fences on one VkDevice (every one
 * of those is a winsys syncobj call from an arbitrary thread) while the main thread keeps the queue busy with
 * wait-before-signal submits: the runtime's submit thread blocks in the winsys waiting for a submit that the host
 * only signals later, from yet another thread.  Binds the contract "any winsys op from any thread, no lock held
 * across a sleep": a deadlock/livelock shows up as the timeout in main(), a race as a crash, a wrong result or a
 * ThreadSanitizer report (build this file and the ICD with -fsanitize=thread, see).
 *
 * Needs the stub winsys (NGV_WINSYS=stub) or the hal winsys (software CP: empty command buffers only).
 * Exit status 0 = all checks passed.  NGV_MT_ITERS=n scales the work (default 300). */
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "ngv_test_ext.h"

#define APP_THREADS 4

static VkDevice dev;
static VkQueue queue;
static int iters = 300;
static atomic_int failures;

#define TCHECK(x)                                                                  \
   do {                                                                            \
      VkResult r_ = (x);                                                           \
      if (r_ != VK_SUCCESS) {                                                      \
         fprintf(stderr, "FAIL %s:%d %s = %d\n", __FILE__, __LINE__, #x, r_);      \
         atomic_fetch_add(&failures, 1);                                           \
         return NULL;                                                              \
      }                                                                            \
   } while (0)

static VkSemaphore
make_timeline(uint64_t initial)
{
   VkSemaphoreTypeCreateInfoKHR tci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR, .initialValue = initial};
   VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tci};
   VkSemaphore s = VK_NULL_HANDLE;
   if (vkCreateSemaphore(dev, &sci, NULL, &s) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return s;
}

/* app thread: create / host-signal / wait / destroy, fences included, as fast as possible */
static void *
app_thread(void *arg)
{
   (void)arg;
   for (int i = 0; i < iters; i++) {
      VkSemaphore s = make_timeline(0);
      if (!s) {
         fprintf(stderr, "FAIL vkCreateSemaphore\n");
         atomic_fetch_add(&failures, 1);
         return NULL;
      }
      VkSemaphoreSignalInfoKHR si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR, .semaphore = s, .value = 1};
      TCHECK(xt.SignalSemaphore(dev, &si));
      uint64_t v = 0;
      TCHECK(xt.GetSemaphoreCounterValue(dev, s, &v));
      if (v != 1) {
         fprintf(stderr, "FAIL counter %llu want 1\n", (unsigned long long)v);
         atomic_fetch_add(&failures, 1);
         return NULL;
      }
      uint64_t val = 1;
      VkSemaphoreWaitInfoKHR wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR, .semaphoreCount = 1,
                                .pSemaphores = &s, .pValues = &val};
      TCHECK(xt.WaitSemaphores(dev, &wi, 5000000000ull));
      /* a wait that must time out, then a late signal from a second signal value */
      val = 2;
      VkResult r = xt.WaitSemaphores(dev, &wi, 0);
      if (r != VK_TIMEOUT) {
         fprintf(stderr, "FAIL poll of unsignaled value: %d want VK_TIMEOUT\n", r);
         atomic_fetch_add(&failures, 1);
         return NULL;
      }
      vkDestroySemaphore(dev, s, NULL);

      VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                               .flags = (i & 1) ? VK_FENCE_CREATE_SIGNALED_BIT : 0};
      VkFence f;
      TCHECK(vkCreateFence(dev, &fci, NULL, &f));
      if (i & 1) {
         TCHECK(vkWaitForFences(dev, 1, &f, VK_TRUE, 5000000000ull));
         TCHECK(vkResetFences(dev, 1, &f));
      }
      r = vkGetFenceStatus(dev, f);
      if (r != VK_NOT_READY) {
         fprintf(stderr, "FAIL fence status %d want VK_NOT_READY\n", r);
         atomic_fetch_add(&failures, 1);
         return NULL;
      }
      vkDestroyFence(dev, f, NULL);
   }
   return NULL;
}

/* signaler thread: signals the wait value of the main thread's submit i after a short delay (so the submit thread
 * is already blocked in the winsys waiting for it) */
struct sig_ctx {
   VkSemaphore sem;
   atomic_int next; /* highest submit index whose wait value may be signaled */
};

static void *
signaler_thread(void *arg)
{
   struct sig_ctx *c = arg;
   for (int i = 0; i < iters; i++) {
      while (atomic_load(&c->next) <= i)
         usleep(50);
      if (i % 4 == 0)
         usleep(300); /* every 4th: let the submit thread really block first */
      /* timeline values must increase: the previous submit's GPU-side signal (2i) comes first */
      uint64_t prev = 2 * (uint64_t)i;
      VkSemaphoreWaitInfoKHR wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR, .semaphoreCount = 1,
                                .pSemaphores = &c->sem, .pValues = &prev};
      TCHECK(xt.WaitSemaphores(dev, &wi, 10000000000ull));
      VkSemaphoreSignalInfoKHR si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR, .semaphore = c->sem,
                                  .value = 2 * (uint64_t)i + 1};
      TCHECK(xt.SignalSemaphore(dev, &si));
   }
   return NULL;
}

static void *
watchdog(void *arg)
{
   (void)arg;
   sleep(120);
   fprintf(stderr, "FAIL watchdog: stuck for 120 s (deadlock / livelock)\n");
   _exit(2);
}

int
main(void)
{
   if (getenv("NGV_MT_ITERS"))
      iters = atoi(getenv("NGV_MT_ITERS"));
   pthread_t wd;
   pthread_create(&wd, NULL, watchdog, NULL);
   pthread_detach(wd);

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS)
      return 1;
   uint32_t n = 1;
   VkPhysicalDevice pd;
   if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || n != 1)
      return 1;
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
                                  .queueCount = 1, .pQueuePriorities = &prio};
   if (xt_require(pd, XT_TIMELINE))
      return 1;
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   struct xt_dev xd;
   xt_dci(&dci, &xd, XT_TIMELINE);
   if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) {
      fprintf(stderr, "FAIL vkCreateDevice\n");
      return 1;
   }
   xt_load(dev);
   vkGetDeviceQueue(dev, 0, 0, &queue);

   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0};
   VkCommandPool pool;
   if (vkCreateCommandPool(dev, &cpi, NULL, &pool) != VK_SUCCESS)
      return 1;
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   VkCommandBufferBeginInfo cbbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   if (vkAllocateCommandBuffers(dev, &cai, &cb) != VK_SUCCESS || vkBeginCommandBuffer(cb, &cbbi) != VK_SUCCESS ||
       vkEndCommandBuffer(cb) != VK_SUCCESS)
      return 1;

   VkSemaphore tl = make_timeline(0);
   if (!tl)
      return 1;
   struct sig_ctx sc = {.sem = tl};
   atomic_init(&sc.next, 0);

   pthread_t app_t[APP_THREADS], sig_t;
   for (int i = 0; i < APP_THREADS; i++)
      pthread_create(&app_t[i], NULL, app_thread, NULL);
   pthread_create(&sig_t, NULL, signaler_thread, &sc);

   /* main thread = the one thread that uses the queue.  Submit i waits tl >= 2i+1 (signaled later, by the
    * signaler thread: wait-before-signal) and signals tl = 2i+2. */
   for (int i = 0; i < iters; i++) {
      uint64_t wait_v = 2 * (uint64_t)i + 1, sig_v = 2 * (uint64_t)i + 2;
      VkTimelineSemaphoreSubmitInfoKHR tsi = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
                                           .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &wait_v,
                                           .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &sig_v};
      VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &tsi, .waitSemaphoreCount = 1,
                         .pWaitSemaphores = &tl, .pWaitDstStageMask = &stage, .commandBufferCount = 1,
                         .pCommandBuffers = &cb, .signalSemaphoreCount = 1, .pSignalSemaphores = &tl};
      VkResult r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
      if (r != VK_SUCCESS) {
         fprintf(stderr, "FAIL vkQueueSubmit %d at %d\n", r, i);
         atomic_fetch_add(&failures, 1);
         break;
      }
      atomic_store(&sc.next, i + 1);
      if (i % 8 == 7) { /* and wait for the GPU-side signal from the app side now and then */
         uint64_t v = sig_v;
         VkSemaphoreWaitInfoKHR wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR, .semaphoreCount = 1,
                                   .pSemaphores = &tl, .pValues = &v};
         r = xt.WaitSemaphores(dev, &wi, 10000000000ull);
         if (r != VK_SUCCESS) {
            fprintf(stderr, "FAIL xt.WaitSemaphores(tl >= %llu) = %d\n", (unsigned long long)v, r);
            atomic_fetch_add(&failures, 1);
            break;
         }
      }
   }
   pthread_join(sig_t, NULL);
   for (int i = 0; i < APP_THREADS; i++)
      pthread_join(app_t[i], NULL);
   if (vkQueueWaitIdle(queue) != VK_SUCCESS) {
      fprintf(stderr, "FAIL vkQueueWaitIdle\n");
      atomic_fetch_add(&failures, 1);
   }
   uint64_t fin = 0;
   xt.GetSemaphoreCounterValue(dev, tl, &fin);
   if (fin != 2 * (uint64_t)iters) {
      fprintf(stderr, "FAIL final timeline value %llu want %llu\n", (unsigned long long)fin,
              (unsigned long long)(2 * (uint64_t)iters));
      atomic_fetch_add(&failures, 1);
   }
   vkDestroySemaphore(dev, tl, NULL);
   vkFreeCommandBuffers(dev, pool, 1, &cb);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   if (atomic_load(&failures)) {
      fprintf(stderr, "ngv_mt: %d failure(s)\n", atomic_load(&failures));
      return 1;
   }
   printf("ngv_mt: OK (%d iterations, %d app threads)\n", iters, APP_THREADS);
   return 0;
}
