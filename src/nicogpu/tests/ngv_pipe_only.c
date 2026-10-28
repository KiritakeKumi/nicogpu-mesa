/* SPDX-License-Identifier: MIT */
/* ngv_pipe_only <nbind> <spv>...: create a compute pipeline for each module (nbind storage-buffer bindings in set 0, spec constant 0 = 4), no dispatch.
 * With NGV_KEEP_TMP=1 NGV_TMPDIR=dir NGV_SHADER_CACHE=0 the driver leaves the NIR text ngcomp compiled in dir (: compare the generated ISA). */
#include "ngv_test_ext.h" /* xt_load_spv */
int main(int argc,char**argv){
 unsigned nb=atoi(argv[1]);
 VkApplicationInfo ai={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_1};
 VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&ai};
 VkInstance inst; if(vkCreateInstance(&ici,NULL,&inst)){fprintf(stderr,"inst\n");return 1;}
 uint32_t n=1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst,&n,&pd);
 VkPhysicalDeviceShaderFloat16Int8Features f16={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,.shaderFloat16=1};
 VkPhysicalDevice16BitStorageFeatures s16={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,.pNext=&f16,.storageBuffer16BitAccess=1,.uniformAndStorageBuffer16BitAccess=1};
 VkPhysicalDeviceFeatures2 f2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&s16};
 f2.features.shaderInt16=1;
 float pr=1; VkDeviceQueueCreateInfo q={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=0,.queueCount=1,.pQueuePriorities=&pr};
 const char*ex[]={"VK_KHR_16bit_storage","VK_KHR_shader_float16_int8"};
 VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&f2,.queueCreateInfoCount=1,.pQueueCreateInfos=&q,.enabledExtensionCount=2,.ppEnabledExtensionNames=ex};
 VkDevice dev; if(vkCreateDevice(pd,&dci,NULL,&dev)){fprintf(stderr,"dev\n");return 1;}
 VkDescriptorSetLayoutBinding b[8]; for(unsigned i=0;i<nb;i++) b[i]=(VkDescriptorSetLayoutBinding){i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL};
 VkDescriptorSetLayoutCreateInfo dl={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=nb,.pBindings=b};
 VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev,&dl,NULL,&dsl);
 VkPipelineLayoutCreateInfo pli={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl};
 VkPipelineLayout pl; vkCreatePipelineLayout(dev,&pli,NULL,&pl);
 for(int i=2;i<argc;i++){
  size_t sz; uint32_t*c=xt_load_spv(argv[i],&sz); if(!c){fprintf(stderr,"read %s\n",argv[i]);continue;}
  VkShaderModuleCreateInfo smi={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=c};
  VkShaderModule sm; VkResult r=vkCreateShaderModule(dev,&smi,NULL,&sm); if(r){fprintf(stderr,"%s module %d\n",argv[i],r);continue;}
  int32_t loop=4; VkSpecializationMapEntry me={0,0,4}; VkSpecializationInfo si={1,&me,4,&loop};
  VkComputePipelineCreateInfo cpi={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main",.pSpecializationInfo=getenv("PIPE_NOSPEC")?NULL:&si},.layout=pl};
  VkPipeline p; r=vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpi,NULL,&p);
  fprintf(stderr,"%s: pipeline %s (%d)\n",argv[i],r?"FAILED":"ok",r);
 }
 return 0;}
