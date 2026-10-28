# Physical three-view fisheye line frontend

This handoff packages the current CAM_B Physical-PL frontend source, its bundled DBoW2/g2o dependencies, build scripts, calibration, and a synthetic core smoke test. It accepts fisheye PNG frames, timestamps, a KannalaBrandt8 YAML file, and an ORB vocabulary; it emits pose/map results and per-frame physical-line residual diagnostics.

Source branch: `experiment/oak_odin_room_corridor_cam_b`; base commit: `7b20b22ecb0822cb04fb64bcbf7c5d8973d406ed`, plus previously uncommitted source. Delivery classification: `CURRENT_SOURCE_REBUILT_AND_VALIDATED`; this does not claim historical binary identity.

Ubuntu 20.04, C++14, CMake, OpenCV 4, Eigen3, PCL and Pangolin 0.9.5 are required. Install Pangolin from its upstream source and set `Pangolin_DIR` if CMake cannot locate it.

```bash
git clone --recursive https://github.com/suleymanneitzke-ai/physical-three-view-fisheye-frontend.git
cd physical-three-view-fisheye-frontend
bash scripts/setup_ubuntu.sh
bash scripts/build.sh
PANGOLIN_LIB_DIR=/path/to/pangolin/lib bash scripts/verify_installation.sh
PANGOLIN_LIB_DIR=/path/to/pangolin/lib bash scripts/run_smoke_test.sh
```

Anonymous clone works only after the repository is approved for Public visibility and independently verified. For a fixed release, use `git clone --recursive --branch v1.0-three-view-handoff https://github.com/suleymanneitzke-ai/physical-three-view-fisheye-frontend.git` after that tag has been published.

For online optimization validation, provide licensed images and timestamps:

```bash
SMOKE_IMAGES_DIR=/path/to/images SMOKE_TIMES_FILE=/path/to/times.txt SMOKE_CAMERA_YAML=config/cam_b_safe89_kb8_adaptive_min30_feat2400.yaml SMOKE_MAX_FRAMES=904 bash scripts/run_smoke_test.sh
```

The script checks three-member physical observations, residual construction and acceptance, joint pose acceptance, and tracked poses. Without images, it runs a synthetic three-view builder test. The OAK/ODIN raw dataset is not distributed. See README_CN.md for details. The project is GPLv3; bundled DBoW2 and g2o retain BSD licenses.
