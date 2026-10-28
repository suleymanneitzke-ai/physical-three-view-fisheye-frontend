#ifndef STRUCTURE_SLAM_VIRTUAL_VIEW_LINE_PIXEL_RESIDUAL_H
#define STRUCTURE_SLAM_VIRTUAL_VIEW_LINE_PIXEL_RESIDUAL_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <string>

namespace StructureSLAM
{

struct VirtualViewLineObservation
{
    int frame_id = -1;
    int view_id = -1;
    int line_id = -1;

    Eigen::Vector3d image_line_normalized = Eigen::Vector3d::Zero();
    Eigen::Vector2d observed_start_px = Eigen::Vector2d::Zero();
    Eigen::Vector2d observed_end_px = Eigen::Vector2d::Zero();
};

struct LinePixelResidualResult
{
    bool valid = false;

    Eigen::Vector2d residual_px = Eigen::Vector2d::Zero();
    Eigen::Matrix<double, 2, 6> jacobian_tcw =
        Eigen::Matrix<double, 2, 6>::Zero();

    Eigen::Vector2d projected_start_px = Eigen::Vector2d::Zero();
    Eigen::Vector2d projected_end_px = Eigen::Vector2d::Zero();
    Eigen::Vector2d endpoint_depth_virtual =
        Eigen::Vector2d::Constant(-1.0);

    double chi2 = 0.0;
    std::string invalid_reason;
};

Eigen::Vector3d NormalizeImageLineForPixelDistance(
    const Eigen::Vector2d& start_px,
    const Eigen::Vector2d& end_px,
    bool* ok = nullptr);

LinePixelResidualResult EvaluateVirtualViewLinePixelResidual(
    const Eigen::Isometry3d& Tcw,
    const Eigen::Vector3d& endpoint0_w,
    const Eigen::Vector3d& endpoint1_w,
    const VirtualViewLineObservation& observation,
    bool compute_analytic_jacobian);

Eigen::Isometry3d ApplyLeftPerturbationRxRyRzTxTyTz(
    const Eigen::Matrix<double, 6, 1>& dx,
    const Eigen::Isometry3d& Tcw);

} // namespace StructureSLAM

#endif
