#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
for path in "$root/Examples/Monocular/mono_tum_vi_native_fisheye_plslam" "$root/Examples/physical_three_view_core_smoke" "$root/Vocabulary/ORBvoc.txt" "$root/config/cam_b_safe89_kb8_adaptive_min30_feat2400.yaml"; do
  [[ -s "$path" ]] || { echo "Missing $path" >&2; exit 2; }
done
export LD_LIBRARY_PATH="${PANGOLIN_LIB_DIR:-}:$root/lib:$root/Thirdparty/DBoW2/lib:$root/Thirdparty/g2o/lib:${LD_LIBRARY_PATH:-}"
if ldd "$root/Examples/Monocular/mono_tum_vi_native_fisheye_plslam" | grep -q "not found"; then
  ldd "$root/Examples/Monocular/mono_tum_vi_native_fisheye_plslam" | grep "not found"
  exit 2
fi
python3 - "$root/config/cam_b_safe89_kb8_adaptive_min30_feat2400.yaml" <<"PY"
import sys
settings = open(sys.argv[1]).read()
for key in ("Camera.type: KannalaBrandt8", "NativeFisheye.VirtualViews: 3", "NativeFisheye.PointLineTracking: 1"):
    assert key in settings, key
print("CAMERA_AND_CONFIG_PASS")
PY
"$root/Examples/physical_three_view_core_smoke"
echo THREE_VIEW_FRONTEND_DEPLOYMENT_PASS
