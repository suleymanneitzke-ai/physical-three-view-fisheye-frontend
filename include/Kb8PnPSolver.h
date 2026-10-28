#ifndef STRUCTURE_SLAM_KB8_PNP_SOLVER_H
#define STRUCTURE_SLAM_KB8_PNP_SOLVER_H

#include "camera/KannalaBrandt8Camera.h"

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include <vector>

namespace StructureSLAM
{

class Frame;
class MapPoint;

class Kb8PnPSolver
{
public:
    struct Result
    {
        bool success = false;
        cv::Mat Tcw;
        std::vector<bool> inliers;
        int inlier_count = 0;
        double mean_bearing_error = 0.0;
        int iterations = 0;
    };

    struct IterationDiagnostic
    {
        int iteration = 0;
        int candidate_count = 0;
        int inliers = 0;
        double mean_error = 0.0;
        bool success = false;
    };

    Kb8PnPSolver(const Frame& frame,
                 const std::vector<MapPoint*>& map_point_matches,
                 const KannalaBrandt8Camera::Parameters& camera_params);

    Kb8PnPSolver(const std::vector<Eigen::Vector3d>& world_points,
                 const std::vector<cv::KeyPoint>& keypoints,
                 const KannalaBrandt8Camera::Parameters& camera_params);

    void SetRansacParameters(double probability = 0.99,
                             int min_inliers = 8,
                             int max_iterations = 300,
                             int min_set = 6,
                             double epsilon = 0.4,
                             double th2 = 5.991);

    Result find();
    Result iterate(int n_iterations);

    const std::vector<IterationDiagnostic>& diagnostics() const
    {
        return diagnostics_;
    }

    int correspondenceCount() const
    {
        return static_cast<int>(world_points_.size());
    }

private:
    static cv::Mat ToCvTcw(const Eigen::Matrix3d& Rcw,
                           const Eigen::Vector3d& tcw);

    bool BuildCorrespondences(const std::vector<Eigen::Vector3d>& world_points,
                              const std::vector<cv::KeyPoint>& keypoints);

    bool EstimateFromSubset(const std::vector<int>& subset,
                            Eigen::Matrix3d& Rcw,
                            Eigen::Vector3d& tcw) const;

    bool OptimizeBearingPose(const std::vector<int>& indices,
                             Eigen::VectorXd& pose) const;

    Eigen::VectorXd ResidualForIndices(const std::vector<int>& indices,
                                       const Eigen::VectorXd& pose) const;

    void PoseVectorToRt(const Eigen::VectorXd& pose,
                        Eigen::Matrix3d& Rcw,
                        Eigen::Vector3d& tcw) const;

    void ScoreCandidate(const Eigen::Matrix3d& Rcw,
                        const Eigen::Vector3d& tcw,
                        std::vector<bool>& inliers,
                        int& inlier_count,
                        double& mean_error) const;

    KannalaBrandt8Camera::Parameters camera_params_;
    KannalaBrandt8Camera camera_;
    std::vector<Eigen::Vector3d> world_points_;
    std::vector<Eigen::Vector3d> bearings_;
    std::vector<int> output_indices_;
    std::vector<IterationDiagnostic> diagnostics_;
    int output_size_ = 0;

    double probability_ = 0.99;
    int min_inliers_ = 8;
    int max_iterations_ = 300;
    int min_set_ = 6;
    double epsilon_ = 0.4;
    double th2_ = 5.991;
    double max_bearing_error_rad_ = 0.02;
};

} // namespace StructureSLAM

#endif // STRUCTURE_SLAM_KB8_PNP_SOLVER_H
