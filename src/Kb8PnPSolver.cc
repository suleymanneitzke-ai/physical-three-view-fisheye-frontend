#include "Kb8PnPSolver.h"

#include "Frame.h"
#include "MapPoint.h"

#include <Eigen/Dense>
#include <opencv2/calib3d/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace StructureSLAM
{
namespace
{

bool FiniteVector(const Eigen::Vector3d& v)
{
    return std::isfinite(v.x()) && std::isfinite(v.y()) &&
           std::isfinite(v.z());
}

Eigen::Matrix3d RodriguesToMatrix(const Eigen::Vector3d& w)
{
    const double theta = w.norm();
    if(theta < 1e-12)
        return Eigen::Matrix3d::Identity();
    return Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
}

double ClampDot(const double value)
{
    return std::max(-1.0, std::min(1.0, value));
}

} // namespace

Kb8PnPSolver::Kb8PnPSolver(
    const Frame& frame,
    const std::vector<MapPoint*>& map_point_matches,
    const KannalaBrandt8Camera::Parameters& camera_params)
    : camera_params_(camera_params),
      camera_(camera_params)
{
    std::vector<Eigen::Vector3d> world_points;
    std::vector<cv::KeyPoint> keypoints;
    const size_t n = std::min(map_point_matches.size(), frame.mvKeysUn.size());
    world_points.reserve(n);
    keypoints.reserve(n);
    output_size_ = static_cast<int>(map_point_matches.size());
    std::vector<int> output_indices;
    output_indices.reserve(n);

    for(size_t i = 0; i < n; ++i)
    {
        MapPoint* pMP = map_point_matches[i];
        if(!pMP || pMP->isBad())
            continue;
        const cv::Mat pos = pMP->GetWorldPos();
        if(pos.empty() || pos.rows < 3)
            continue;
        world_points.push_back(Eigen::Vector3d(
            pos.at<float>(0), pos.at<float>(1), pos.at<float>(2)));
        keypoints.push_back(frame.mvKeysUn[i]);
        output_indices.push_back(static_cast<int>(i));
    }

    output_indices_ = output_indices;
    BuildCorrespondences(world_points, keypoints);
}

Kb8PnPSolver::Kb8PnPSolver(
    const std::vector<Eigen::Vector3d>& world_points,
    const std::vector<cv::KeyPoint>& keypoints,
    const KannalaBrandt8Camera::Parameters& camera_params)
    : camera_params_(camera_params),
      camera_(camera_params)
{
    output_size_ = static_cast<int>(std::min(world_points.size(), keypoints.size()));
    output_indices_.resize(output_size_);
    std::iota(output_indices_.begin(), output_indices_.end(), 0);
    BuildCorrespondences(world_points, keypoints);
}

void Kb8PnPSolver::SetRansacParameters(
    double probability,
    int min_inliers,
    int max_iterations,
    int min_set,
    double epsilon,
    double th2)
{
    probability_ = probability;
    min_inliers_ = min_inliers;
    max_iterations_ = std::max(1, max_iterations);
    min_set_ = std::max(3, min_set);
    epsilon_ = epsilon;
    th2_ = th2;
    const double focal =
        std::max(std::abs(camera_params_.fx), std::abs(camera_params_.fy));
    max_bearing_error_rad_ =
        focal > 0.0 ? std::sqrt(std::max(0.0, th2_)) / focal : 0.02;
}

Kb8PnPSolver::Result Kb8PnPSolver::find()
{
    return iterate(max_iterations_);
}

Kb8PnPSolver::Result Kb8PnPSolver::iterate(int n_iterations)
{
    diagnostics_.clear();

    Result best;
    if(!camera_.valid() ||
       static_cast<int>(world_points_.size()) < std::max(min_set_, min_inliers_))
    {
        return best;
    }

    std::vector<int> all_indices(world_points_.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);

    std::mt19937 rng(7);
    const int iterations = std::min(std::max(1, n_iterations), max_iterations_);
    for(int it = 0; it < iterations; ++it)
    {
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        std::vector<int> subset(
            all_indices.begin(), all_indices.begin() + min_set_);

        Eigen::Matrix3d Rcw;
        Eigen::Vector3d tcw;
        if(!EstimateFromSubset(subset, Rcw, tcw))
        {
            diagnostics_.push_back({it, correspondenceCount(), 0,
                                    std::numeric_limits<double>::infinity(),
                                    false});
            continue;
        }

        std::vector<bool> inliers;
        int inlier_count = 0;
        double mean_error = std::numeric_limits<double>::infinity();
        ScoreCandidate(Rcw, tcw, inliers, inlier_count, mean_error);

        diagnostics_.push_back({it, correspondenceCount(), inlier_count,
                                mean_error, inlier_count >= min_inliers_});

        if(inlier_count > best.inlier_count ||
           (inlier_count == best.inlier_count &&
            mean_error < best.mean_bearing_error))
        {
            std::vector<int> inlier_indices;
            for(size_t i = 0; i < inliers.size(); ++i)
            {
                if(inliers[i])
                    inlier_indices.push_back(static_cast<int>(i));
            }
            Eigen::VectorXd pose(6);
            Eigen::AngleAxisd aa(Rcw);
            pose.head<3>() = aa.angle() * aa.axis();
            pose.tail<3>() = tcw;
            if(static_cast<int>(inlier_indices.size()) >= min_set_)
                OptimizeBearingPose(inlier_indices, pose);
            PoseVectorToRt(pose, Rcw, tcw);
            ScoreCandidate(Rcw, tcw, inliers, inlier_count, mean_error);

            best.success = inlier_count >= min_inliers_;
            best.Tcw = ToCvTcw(Rcw, tcw);
            best.inliers = inliers;
            best.inlier_count = inlier_count;
            best.mean_bearing_error = mean_error;
            best.iterations = it + 1;
        }
    }

    if(!best.success)
        best.Tcw.release();
    return best;
}

cv::Mat Kb8PnPSolver::ToCvTcw(const Eigen::Matrix3d& Rcw,
                              const Eigen::Vector3d& tcw)
{
    cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
    for(int r = 0; r < 3; ++r)
    {
        for(int c = 0; c < 3; ++c)
            Tcw.at<float>(r, c) = static_cast<float>(Rcw(r, c));
        Tcw.at<float>(r, 3) = static_cast<float>(tcw(r));
    }
    return Tcw;
}

bool Kb8PnPSolver::BuildCorrespondences(
    const std::vector<Eigen::Vector3d>& world_points,
    const std::vector<cv::KeyPoint>& keypoints)
{
    world_points_.clear();
    bearings_.clear();
    const size_t n = std::min(world_points.size(), keypoints.size());
    std::vector<int> valid_output_indices;
    valid_output_indices.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        Eigen::Vector3d bearing;
        const Eigen::Vector2d uv(keypoints[i].pt.x, keypoints[i].pt.y);
        if(!FiniteVector(world_points[i]) || !camera_.unproject(uv, bearing))
            continue;
        world_points_.push_back(world_points[i]);
        bearings_.push_back(bearing.normalized());
        if(i < output_indices_.size())
            valid_output_indices.push_back(output_indices_[i]);
        else
            valid_output_indices.push_back(static_cast<int>(i));
    }
    output_indices_ = valid_output_indices;
    return !world_points_.empty();
}

bool Kb8PnPSolver::EstimateFromSubset(const std::vector<int>& subset,
                                      Eigen::Matrix3d& Rcw,
                                      Eigen::Vector3d& tcw) const
{
    Eigen::VectorXd pose = Eigen::VectorXd::Zero(6);
    if(!OptimizeBearingPose(subset, pose))
        return false;
    PoseVectorToRt(pose, Rcw, tcw);
    return true;
}

bool Kb8PnPSolver::OptimizeBearingPose(const std::vector<int>& indices,
                                       Eigen::VectorXd& pose) const
{
    if(indices.size() < 3)
        return false;

    const double lambda = 1e-6;
    for(int iter = 0; iter < 30; ++iter)
    {
        const Eigen::VectorXd r = ResidualForIndices(indices, pose);
        if(!r.allFinite())
            return false;

        Eigen::MatrixXd J(r.size(), 6);
        const double eps = 1e-6;
        for(int j = 0; j < 6; ++j)
        {
            Eigen::VectorXd plus = pose;
            plus(j) += eps;
            Eigen::VectorXd minus = pose;
            minus(j) -= eps;
            J.col(j) =
                (ResidualForIndices(indices, plus) -
                 ResidualForIndices(indices, minus)) / (2.0 * eps);
        }

        Eigen::Matrix<double, 6, 6> H =
            J.transpose() * J + lambda * Eigen::Matrix<double, 6, 6>::Identity();
        Eigen::Matrix<double, 6, 1> b = -J.transpose() * r;
        const Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
        if(!dx.allFinite())
            return false;
        pose += dx;
        if(dx.norm() < 1e-8)
            break;
    }
    return true;
}

Eigen::VectorXd Kb8PnPSolver::ResidualForIndices(
    const std::vector<int>& indices,
    const Eigen::VectorXd& pose) const
{
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    PoseVectorToRt(pose, Rcw, tcw);

    Eigen::VectorXd residual(indices.size() * 3);
    for(size_t i = 0; i < indices.size(); ++i)
    {
        const int idx = indices[i];
        const Eigen::Vector3d Pc = Rcw * world_points_[idx] + tcw;
        const double n = Pc.norm();
        if(!std::isfinite(n) || n <= 1e-12)
        {
            residual.segment<3>(3 * i).setConstant(1e6);
            continue;
        }
        residual.segment<3>(3 * i) = Pc / n - bearings_[idx];
    }
    return residual;
}

void Kb8PnPSolver::PoseVectorToRt(const Eigen::VectorXd& pose,
                                  Eigen::Matrix3d& Rcw,
                                  Eigen::Vector3d& tcw) const
{
    Rcw = RodriguesToMatrix(pose.head<3>());
    tcw = pose.tail<3>();
}

void Kb8PnPSolver::ScoreCandidate(const Eigen::Matrix3d& Rcw,
                                  const Eigen::Vector3d& tcw,
                                  std::vector<bool>& inliers,
                                  int& inlier_count,
                                  double& mean_error) const
{
    inliers.assign(output_size_ > 0 ? output_size_ : static_cast<int>(world_points_.size()),
                   false);
    inlier_count = 0;
    double error_sum = 0.0;

    for(size_t i = 0; i < world_points_.size(); ++i)
    {
        const Eigen::Vector3d Pc = Rcw * world_points_[i] + tcw;
        const double n = Pc.norm();
        if(!std::isfinite(n) || n <= 1e-12)
            continue;
        const Eigen::Vector3d predicted = Pc / n;
        const double angle = std::acos(
            ClampDot(predicted.dot(bearings_[i])));
        if(angle <= max_bearing_error_rad_)
        {
            const int out_idx =
                i < output_indices_.size() ? output_indices_[i] : static_cast<int>(i);
            if(out_idx >= 0 && out_idx < static_cast<int>(inliers.size()))
                inliers[out_idx] = true;
            ++inlier_count;
            error_sum += angle;
        }
    }

    mean_error = inlier_count > 0 ?
        error_sum / static_cast<double>(inlier_count) :
        std::numeric_limits<double>::infinity();
}

} // namespace StructureSLAM
