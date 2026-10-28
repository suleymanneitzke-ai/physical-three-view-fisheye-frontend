#ifndef STRUCTURE_SLAM_FISHEYE_POINT_POSE_EDGE_H
#define STRUCTURE_SLAM_FISHEYE_POINT_POSE_EDGE_H

#include "camera/CentralCameraModel.h"
#include "optimization/OfflineFrameOptimizationTypes.h"

#include <Eigen/Core>

namespace StructureSLAM {

struct FisheyePointResidual {
    Eigen::Vector2d error = Eigen::Vector2d::Zero();
    Eigen::Matrix<double, 2, 6> jacobian =
        Eigen::Matrix<double, 2, 6>::Zero();
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    Eigen::Vector3d Pc = Eigen::Vector3d::Zero();
    double chi2 = 0.0;
    double error_norm = 0.0;
    bool valid = false;
    std::string invalid_reason;
};

class FisheyePointPoseEdge {
public:
    // Error convention matches Structure-SLAM/g2o point edges:
    // e = observed_pixel - project(Tcw * Pw).
    static FisheyePointResidual evaluate(const CentralCameraModel& camera,
                                         const OfflinePoseTcw& Tcw,
                                         const PointObservation& obs);
};

Eigen::Matrix3d skewMatrix(const Eigen::Vector3d& v);

}  // namespace StructureSLAM

#endif  // STRUCTURE_SLAM_FISHEYE_POINT_POSE_EDGE_H
