#ifndef STRUCTURE_SLAM_FISHEYE_BEARING_PLANE_LINE_EDGE_H
#define STRUCTURE_SLAM_FISHEYE_BEARING_PLANE_LINE_EDGE_H

#include "line_geometry/EndpointPluckerConverter.h"
#include "optimization/OfflineFrameOptimizationTypes.h"

#include <Eigen/Core>

#include <string>

namespace StructureSLAM {

struct BearingPlaneLineSample {
    int frame_id = -1;
    int line3d_id = -1;
    int view_id = -1;
    int line_id = -1;
    int sample_index = -1;
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    Eigen::Vector3d bearing = Eigen::Vector3d::Zero();
    PluckerLineNV line_w;
    double sigma_theta_rad = 1.0;
    double sample_weight = 1.0;
};

struct BearingPlaneLineResidual {
    bool valid = false;
    std::string invalid_reason;
    double raw_residual = 0.0;
    double normalized_residual = 0.0;
    double chi2 = 0.0;
    double norm_nc = 0.0;
    Eigen::Matrix<double, 1, 6> jacobian =
        Eigen::Matrix<double, 1, 6>::Zero();
};

class FisheyeBearingPlaneLineEdge {
public:
    static BearingPlaneLineResidual evaluate(
        const OfflinePoseTcw& pose,
        const BearingPlaneLineSample& sample,
        double min_plane_norm = 1e-8);
};

}  // namespace StructureSLAM

#endif  // STRUCTURE_SLAM_FISHEYE_BEARING_PLANE_LINE_EDGE_H
