#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# (section 8): adversarial cross-process residue check on the board (NGV_WINSYS=drm).
#
#   ngv_residue.sh DIR      # DIR holds ngv_residue, ngv_residue.spv, xo/fill/main.xo, xo/read/main.xo
#
# Build (board host): cc -O2 ngv_residue.c -o ngv_residue -lvulkan; glslangValidator -V shaders/ngv_residue.comp
# -o ngv_residue.spv; python3 ngcomp/driver.py compile shaders/ngv_residue_regs_<k>.xir xo/<k>/main.xo (k =
# fill, read).  Needs the usual Vulkan env (VK_DRIVER_FILES, NGV_NGCOMP_DIR, NGV_WINSYS=drm) and nothing else
# submitting to the GPU meanwhile.
# Per resource (shared memory, vector registers):
#   control: fill + read in ONE process (same VM, no owner change, no scrub) -> pattern must be visible, else the
#            test cannot see residue and proves nothing;
#   cross:   fill in process A, read in process B -> only zeros (the owner-change scrub cleared it).
# Exit 0 = PASS, 1 = FAIL, 2 = usage / run error.
set -u
D=${1:?usage: ngv_residue.sh DIR}
B="$D/ngv_residue"; S="$D/ngv_residue.spv"
fail=0
count() { sed -n 's/.*pattern \([0-9]*\) zero \([0-9]*\) other \([0-9]*\).*/\1 \2 \3/p'; }
for res in shared regs; do
  x=(); [ $res = regs ] && x=("$D/xo")
  c=$("$B" "$S" both "${x[@]}" | count) || exit 2
  "$B" "$S" fill "${x[@]}" > /dev/null || exit 2
  r=$("$B" "$S" read "${x[@]}" | count) || exit 2
  [ -n "$c" ] && [ -n "$r" ] || { echo "$res: no result line"; exit 2; }
  set -- $c; cp=$1
  set -- $r; rp=$1 rz=$2 ro=$3
  v=PASS
  [ "$cp" -gt 0 ] || v="FAIL (control saw no residue: the test is blind)"
  [ "$rp" -eq 0 ] && [ "$ro" -eq 0 ] || v="FAIL (residue crossed processes)"
  [ "${v%% *}" = PASS ] || fail=1
  echo "$res: control pattern $cp | cross-process pattern $rp zero $rz other $ro -> $v"
done
[ $fail = 0 ] && echo "SEC6_RESIDUE: PASS" || echo "SEC6_RESIDUE: FAIL"
exit $fail
