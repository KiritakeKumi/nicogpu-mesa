# nicogpu Mesa Vulkan driver ("ngv")

This directory contains the experimental Vulkan driver for nicoGPU, an
open-source RISC-V based GPU.  Compute-first; Vulkan 1.1 core plus a set of
extensions.  This driver is **experimental** and under active development.

## Building

Only a Vulkan driver is provided (no OpenGL/Vulkan-Gallium integration):

```
meson setup build -Dvulkan-drivers=nicogpu -Dgallium-drivers=[] -Dplatforms=[]
```

The driver builds as an ICD (`libvulkan_nicogpu.so`) plus its JSON manifest.

## Dependencies

- Kernel module: https://github.com/KiritakeKumi/nicogpu-module
- RTL: https://github.com/KiritakeKumi/nicogpu
- Firmware: https://github.com/KiritakeKumi/nicogpu-firmware

## Winsys backends

`NGV_WINSYS` selects the backend at run time:

- `stub` (default available): `vulkan/ngv_winsys_stub.c`, a software
  implementation with a fake device and a software command processor.  Useful
  for bring-up and testing without hardware.
- `hal`: `vulkan/ngv_winsys_hal.c`, the real hardware backend.  It forwards to
  prebuilt functional-model libraries that are **not** part of this repository.
  Place the five static libraries `libnicogpu_winsys.a`, `libnicogpu_hal.a`,
  `libnicogpu_isa.a`, `libnicogpu_model.a` and `libnicogpu_softfloat.a` in
  `src/nicogpu/halprebuilt/` before configuring; the directory is git-ignored.
  Without those libraries the backend is still compile-checked.

## Tests

`tests/` contains stand-alone host test programs and GLSL shaders.  They are
not built by the Mesa build system; each file documents how to build it.

## License

MIT; see `LICENSE`.  Exception: `src/include/nicogpu_uapi.h` keeps its
GPL-2.0-with-Linux-syscall-note SPDX tag (kernel UAPI header).
