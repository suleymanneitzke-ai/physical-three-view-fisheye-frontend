#ifndef STRUCTURE_SLAM_OFFLINE_FRAME_OPTIMIZATION_TYPES_H
#define STRUCTURE_SLAM_OFFLINE_FRAME_OPTIMIZATION_TYPES_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <string>
#include <vector>

namespace StructureSLAM {

struct OfflinePoseTcw {
    Eigen::Matrix3d Rcw = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tcw = Eigen::Vector3d::Zero();
};

struct PointObservation {
    int frame_id = -1;
    double timestamp = 0.0;
    long map_point_id = -1;
    Eigen::Vector3d Pw = Eigen::Vector3d::Zero();
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    int octave = -1;
    double inverse_sigma2 = 1.0;
    bool original_outlier = false;
    bool map_point_bad = false;
};

struct LineObservation {
    int frame_id = -1;
    int line3d_id = -1;
};

struct OptimizationDiagnostics {
    int total_observations = 0;
    int valid_observations = 0;
    int initial_inliers = 0;
    int final_inliers = 0;
    double initial_weighted_chi2 = 0.0;
    double final_weighted_chi2 = 0.0;
    double initial_mean_error = 0.0;
    double final_mean_error = 0.0;
    int numerical_rank = 0;
    double condition_number = 0.0;
};

struct FrameOptimizationInput {
    int frame_id = -1;
    double timestamp = 0.0;
    OfflinePoseTcw initial_Tcw;
    std::vector<PointObservation> point_observations;
    std::vector<LineObservation> line_observations;
};

struct FrameOptimizationResult {
    OfflinePoseTcw initial_Tcw;
    OfflinePoseTcw candidate_Tcw;
    bool accepted = false;
    std::string status;
    OptimizationDiagnostics diagnostics;
};

}  // namespace StructureSLAM

#endif  // STRUCTURE_SLAM_OFFLINE_FRAME_OPTIMIZATION_TYPES_H
