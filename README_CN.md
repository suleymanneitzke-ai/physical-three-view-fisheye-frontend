# 三视图鱼眼物理线前端

本项目从 Structure-SLAM-PointLine 的 CAM_B 工作树隔离交付。输入为按时间排序的鱼眼 PNG、时间戳、KannalaBrandt8 相机 YAML 和 ORB 词表。输出包含位姿、MapPoint/MapLine、逐帧三视图物理线及优化残差诊断。

## 来源与状态

- 来源分支：`experiment/oak_odin_room_corridor_cam_b`，基础提交 `7b20b22ecb0822cb04fb64bcbf7c5d8973d406ed`；另含工作树未提交源码。
- 交付类型：`CURRENT_SOURCE_REBUILT_AND_VALIDATED`。本交付不声明与历史二进制字节相同。
- 最新有效历史参考：2026-09-26 完成的 OAK/CAM_B 904 帧运行；三成员组 1987，接受残差 279。
- 新隔离构建复跑：904 帧，三成员组 1987，创建残差 893，接受残差 843，联合位姿接受 212；完整日志保留在本机移交审计目录，原始 OAK 数据不包含于仓库。

## 构建与验证

系统：Ubuntu 20.04 验证；C++14、CMake、OpenCV 4.5.5、Eigen3、PCL、Pangolin 0.9.5。DBoW2 与 g2o 源码随仓库提供。Pangolin 需按其上游安装并使 `Pangolin_DIR` 可由 CMake 找到；若库不在标准路径，设置 `PANGOLIN_LIB_DIR`。

```bash
git clone --recursive https://github.com/suleymanneitzke-ai/physical-three-view-fisheye-frontend.git
cd physical-three-view-fisheye-frontend
bash scripts/setup_ubuntu.sh
bash scripts/build.sh
PANGOLIN_LIB_DIR=/path/to/pangolin/lib bash scripts/verify_installation.sh
PANGOLIN_LIB_DIR=/path/to/pangolin/lib bash scripts/run_smoke_test.sh
```

以上 clone 地址只有在仓库经批准公开并匿名测试通过后才能作为无账号命令使用。当前若仓库仍为 Private，该命令需要授权。`verify_installation.sh` 的通过标识只表示二进制、配置、链接和合成三视图测试通过；实际在线优化还需运行真实序列。

固定版本发布后可使用：

```bash
git clone --recursive --branch v1.0-three-view-handoff https://github.com/suleymanneitzke-ai/physical-three-view-fisheye-frontend.git
```

仅在该 tag 实际发布后使用。

真实数据 smoke：

```bash
SMOKE_IMAGES_DIR=/path/to/images SMOKE_TIMES_FILE=/path/to/times.txt SMOKE_CAMERA_YAML=config/cam_b_safe89_kb8_adaptive_min30_feat2400.yaml SMOKE_MAX_FRAMES=904 bash scripts/run_smoke_test.sh
```

接收方须提供有权使用的匹配数据。脚本核查三成员组、残差创建/接受、联合优化接受和有效位姿。仓库不包含 OAK/ODIN 原始数据。

## 调用链

`mono_tum_vi_native_fisheye_plslam` → runner → `System`/`Tracking` → `FisheyeLineFrontend` → `PhysicalLineObservationBuilder` → `Frame` → `NativeFisheyeMapLineProjector::AssociatePhysical` → `Optimizer::PoseOptimizationWithKb8PointLine` → `PhysicalLinePlanePoseEdge` → 位姿/地图输出。三个虚拟视图用于同一物理线观测分组；世界线的预测相机平面与 KB8 观测平面的叉积形成残差。

开发分支切换：`git branch -a`，`git switch <branch>`；更新：`git pull --ff-only`。已知限制：私有采集数据未发布、Pangolin 需独立安装、公开前无法匿名克隆。项目主体按 GPLv3，捆绑 DBoW2/g2o 各保留 BSD 许可。
