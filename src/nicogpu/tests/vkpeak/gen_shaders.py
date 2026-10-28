#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the vkpeak fp32/int32 mad-chain kernels (after nihui/vkpeak, MIT) as GLSL compute shaders.

usage: gen_shaders.py <outdir> [repeat=16] [local_size_x=32]
Writes xgp_{fp32,int32}_{p1,p4}.comp; `loop` is a specialization constant. : also xgp_fp16_{p1,p4}.comp (vkpeak
fp16-scalar / fp16-vec4 as ncnn compiles them with fp16 arithmetic: float16_t chain, float result); ngv_peak does not run them,
they exist to compare the generated ISA (ngv_pipe_only + NGV_KEEP_TMP=1, tools/analysis/f16_stats.py).
"""
import os
import sys

out = sys.argv[1]
rep = int(sys.argv[2]) if len(sys.argv) > 2 else 16
lsx = int(sys.argv[3]) if len(sys.argv) > 3 else 32
body = "\n".join("        c = a * c + b;" for _ in range(rep))


def shader(ty, vec4):
    s = "#version 450\n\nlayout (local_size_x = %d) in;\nlayout (constant_id = 0) const int loop = 1;\n\n" % lsx
    s += "layout (binding = 0) writeonly buffer c_blob { %s c_blob_data[]; };\n\nvoid main()\n{\n" % ty
    s += "    const uint gx = gl_GlobalInvocationID.x;\n    const uint lx = gl_LocalInvocationID.x;\n\n"
    if not vec4:
        s += "    %s c = %s(gx);\n    %s a = c;\n    %s b = %s(lx);\n" % (ty, ty, ty, ty, ty)
    else:
        v = "vec4" if ty == "float" else "ivec4"
        s += "    %s c = %s(gx);\n" % (v, v)
        s += "    %s a = c + %s(0,1,2,-3);\n" % (v, v)
        s += "    %s b = %s(%s(lx)) + %s(2,3,5,-7);\n" % (v, v, ty, v)
    s += "\n    for (int i = 0; i < loop; i++)\n    {\n%s\n    }\n\n" % body
    if not vec4:
        s += "    c_blob_data[gx] = c;\n"
    else:
        s += "    c_blob_data[gx] = (c[0] + c[1]) + (c[2] + c[3]);\n"
    s += "}\n"
    return s


for ty in ("float", "int"):
    for vec4 in (False, True):
        n = "xgp_%s_%s.comp" % ("fp32" if ty == "float" else "int32", "p4" if vec4 else "p1")
        with open(os.path.join(out, n), "w", newline="\n") as f:
            f.write(shader(ty, vec4))


def shader16(vec4):
 """: vkpeak fp16-scalar / fp16-vec4 as ncnn compiles them with fp16 arithmetic (float16_t chain, float result)."""
    L = ["#version 450", "#extension GL_EXT_shader_16bit_storage : require",
         "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require", "",
         "layout (local_size_x = %d) in;" % lsx, "layout (constant_id = 0) const int loop = 1;", "",
         "layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };", "", "void main()", "{",
         "    const uint gx = gl_GlobalInvocationID.x;", "    const uint lx = gl_LocalInvocationID.x;", ""]
    if not vec4:
        L += ["    float16_t c = float16_t(gx);", "    float16_t a = c;", "    float16_t b = float16_t(lx);"]
    else:
        L += ["    f16vec4 c = f16vec4(float16_t(gx));", "    f16vec4 a = c + f16vec4(0,1,2,-3);",
              "    f16vec4 b = f16vec4(float16_t(lx)) + f16vec4(2,3,5,-7);"]
    L += ["", "    for (int i = 0; i < loop; i++)", "    {", body, "    }", ""]
    L += ["    c_blob_data[gx] = float(c);" if not vec4 else "    c_blob_data[gx] = float((c.x + c.y) + (c.z + c.w));",
          "}", ""]
    return "\n".join(L)


for vec4 in (False, True):
    with open(os.path.join(out, "xgp_fp16_%s.comp" % ("p4" if vec4 else "p1")), "w", newline="\n") as f:
        f.write(shader16(vec4))
