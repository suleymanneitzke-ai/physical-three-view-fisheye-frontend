#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="${PANGOLIN_LIB_DIR:-}:$root/lib:$root/Thirdparty/DBoW2/lib:$root/Thirdparty/g2o/lib:${LD_LIBRARY_PATH:-}"
"$root/Examples/physical_three_view_core_smoke"
if [[ -n "${SMOKE_IMAGES_DIR:-}" ]]; then
  : "${SMOKE_TIMES_FILE:?Set SMOKE_TIMES_FILE}"
  : "${SMOKE_CAMERA_YAML:?Set SMOKE_CAMERA_YAML}"
  out="${SMOKE_OUTPUT_DIR:-$root/smoke_output}"
  mkdir -p "$out"
  CAM_B_INIT_MIN_GOOD_POINTS=20 CAM_B_INIT_MIN_PARALLAX_DEG=0.5 CAM_B_INIT_MIN_REPROJECTION_POINTS=25 CAM_B_INIT_MIN_TRACKED_MAPPOINTS=25 CAM_B_TRACK_LOCAL_MAP_EARLY_MIN_INLIERS=25 CAM_B_TRACK_LOCAL_MAP_MIN_INLIERS=20 "$root/Examples/Monocular/mono_tum_vi_native_fisheye_plslam" "$root/Vocabulary/ORBvoc.txt" "$SMOKE_CAMERA_YAML" "$SMOKE_IMAGES_DIR" "$SMOKE_TIMES_FILE" "$out" "${SMOKE_MAX_FRAMES:-904}"
  python3 "$root/scripts/check_runtime.py" "$out/native_loop_per_frame.csv"
else
  echo "SYNTHETIC_CORE_PASS: set SMOKE_IMAGES_DIR, SMOKE_TIMES_FILE, and SMOKE_CAMERA_YAML for real online validation"
fi
