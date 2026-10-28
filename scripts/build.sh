#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cmake -S Thirdparty/DBoW2 -B Thirdparty/DBoW2/build -DCMAKE_BUILD_TYPE=Release
cmake --build Thirdparty/DBoW2/build --parallel 4
cmake -S Thirdparty/g2o -B Thirdparty/g2o/build -DCMAKE_BUILD_TYPE=Release
cmake --build Thirdparty/g2o/build --parallel 4
cmake -S . -B build_handoff -DCMAKE_BUILD_TYPE=Release -DBUILD_MONO_NATIVE_ONLY=ON
cmake --build build_handoff --target mono_tum_vi_native_fisheye_plslam physical_three_view_core_smoke --parallel 4
if [[ ! -f Vocabulary/ORBvoc.txt ]]; then gzip -dc Vocabulary/ORBvoc.txt.gz > Vocabulary/ORBvoc.txt; fi
echo BUILD_PASS
