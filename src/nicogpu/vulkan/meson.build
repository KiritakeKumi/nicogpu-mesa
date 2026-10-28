# SPDX-License-Identifier: MIT
# ngv: Nicogpu Vulkan driver.

ngv_entrypoints = custom_target(
  'ngv_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['ngv_entrypoints.h', 'ngv_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'ngv',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

ngv_files = files(
  'ngv_cmd_buffer.c',
  'ngv_descriptor.c',
  'ngv_drm_ioctl.c',
  'ngv_image.c',
  'ngv_instance.c',
  'ngv_device.c',
  'ngv_memory.c',
  'ngv_meta.c',
  'ngv_nir.c',
  'ngv_nir_print.c',
  'ngv_pipeline.c',
  'ngv_query.c',
  'ngv_stub_cp.c',
  'ngv_sync.c',
  'ngv_uuid.c',
  'ngv_winsys.c',
  'ngv_winsys_drm.c',
  'ngv_winsys_stub.c',
  'ngv_ngcomp.c',
)

ngv_inc = include_directories('.', '../include')

# The HAL backend forwards to the prebuilt libnicogpu_winsys functional-model libraries (winsys + hal + isa +
# model + softfloat), which are not part of this repository: place the five static libraries in
# src/nicogpu/halprebuilt.  With the libraries present the backend is linked into the ICD
# (NGV_WITH_HAL, NGV_WINSYS=hal); without them it is only compile-checked as before.
fs = import('fs')
hal_dir = meson.current_source_dir() / '..' / 'halprebuilt'
cc = meson.get_compiler('c')
ngv_hal_deps = []
ngv_hal_args = []
if fs.exists(hal_dir / 'libnicogpu_winsys.a')
  foreach l : ['nicogpu_winsys', 'nicogpu_hal', 'nicogpu_isa', 'nicogpu_model', 'nicogpu_softfloat']
    ngv_hal_deps += cc.find_library(l, dirs : [hal_dir], static : true)
  endforeach
  ngv_hal_deps += cc.find_library('stdc++')
  ngv_files += files('ngv_winsys_hal.c')
  ngv_hal_args = ['-DNGV_WITH_HAL']
else
  static_library(
    'ngv_winsys_hal_check',
    'ngv_winsys_hal.c',
    c_args : ['-DNGV_WITH_HAL'],
    include_directories : [ngv_inc],
    build_by_default : true,
  )
endif

libvulkan_nicogpu = shared_library(
  'vulkan_nicogpu',
  [ngv_files, ngv_entrypoints],
  include_directories : [inc_include, inc_src, inc_util, ngv_inc],
  c_args : ngv_hal_args,
  dependencies : [idep_nir, idep_mesautil, idep_vulkan_util, idep_vulkan_runtime, ngv_hal_deps],
  link_args : [vulkan_icd_link_args, ld_args_bsymbolic, ld_args_gc_sections],
  link_depends : vulkan_icd_link_depends,
  gnu_symbol_visibility : 'hidden',
  install : true,
)

icd_lib_path = join_paths(get_option('prefix'), get_option('libdir'))
ngv_icd = custom_target(
  'ngv_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'ngv_icd.@0@.json'.format(host_machine.cpu()),
  command : [
    prog_python, '@INPUT0@', '--api-version', '1.1', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--lib-path', join_paths(icd_lib_path, 'libvulkan_nicogpu.so'),
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
  install_dir : with_vulkan_icd_dir,
  install : true,
)

# ICD that points at the build tree, for smoke tests (VK_DRIVER_FILES).
ngv_dev_icd = custom_target(
  'ngv_devenv_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'ngv_devenv_icd.@0@.json'.format(host_machine.cpu()),
  command : [
    prog_python, '@INPUT0@', '--api-version', '1.1', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--lib-path', meson.current_build_dir() / 'libvulkan_nicogpu.so',
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)
