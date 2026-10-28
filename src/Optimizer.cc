/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/
/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/Dense>
#include<Eigen/StdVector>

#include "Converter.h"
#include "FisheyeVirtualCameraGeometry.h"
#include "line_geometry/EndpointPluckerConverter.h"
#include "optimization/FisheyeJointPoseOptimizer.h"
#include "optimization/Kb8PointPoseEdge.h"
#include "optimization/NativeFisheyeLinePoseEdge.h"
#include "optimization/PhysicalLinePlanePoseEdge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <memory>
#include <set>
#include <sstream>
#include <thread>

namespace StructureSLAM
{

namespace
{

struct EdgeCostSummary
{
    double cost = 0.0;
    int inliers = 0;
};

EdgeCostSummary EvaluatePointPoseEdges(
    const vector<g2o::EdgeSE3ProjectXYZOnlyPose*>& mono_edges,
    const vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>& stereo_edges)
{
    EdgeCostSummary summary;
    for(size_t i = 0; i < mono_edges.size(); ++i)
    {
        g2o::EdgeSE3ProjectXYZOnlyPose* e = mono_edges[i];
        e->computeError();
        const double chi2 = e->chi2();
        summary.cost += chi2;
        if(std::isfinite(chi2) && chi2 <= 5.991)
            ++summary.inliers;
    }
    for(size_t i = 0; i < stereo_edges.size(); ++i)
    {
        g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = stereo_edges[i];
        e->computeError();
        const double chi2 = e->chi2();
        summary.cost += chi2;
        if(std::isfinite(chi2) && chi2 <= 7.815)
            ++summary.inliers;
    }
    return summary;
}

EdgeCostSummary EvaluateKb8PointBundleEdges(
    const vector<Kb8PointBundleEdge*>& edges)
{
    EdgeCostSummary summary;
    for(size_t i = 0; i < edges.size(); ++i)
    {
        Kb8PointBundleEdge* e = edges[i];
        e->computeError();
        const double chi2 = e->chi2();
        summary.cost += chi2;
        if(std::isfinite(chi2) && chi2 <= 5.991 &&
           e->isDepthPositive())
            ++summary.inliers;
    }
    return summary;
}

EdgeCostSummary EvaluateKb8PointPoseEdges(
    const vector<Kb8PointPoseEdge*>& edges)
{
    EdgeCostSummary summary;
    for(size_t i = 0; i < edges.size(); ++i)
    {
        Kb8PointPoseEdge* e = edges[i];
        e->computeError();
        const double chi2 = e->chi2();
        summary.cost += chi2;
        if(std::isfinite(chi2) && chi2 <= 5.991 &&
           e->isDepthPositive())
            ++summary.inliers;
    }
    return summary;
}

class PhysicalPixelReplayDump
{
public:
    static PhysicalPixelReplayDump& Instance()
    {
        static PhysicalPixelReplayDump dump;
        return dump;
    }

    bool enabled() const { return enabled_; }

    void Write(const Frame& frame,
               const size_t observation_index,
               const PhysicalLineObservation& physical,
               const PhysicalLineObservationMember& member,
               const MapLine& mapline,
               const Eigen::Vector3d& endpoint0_w,
               const Eigen::Vector3d& endpoint1_w,
               const Eigen::Vector3d& image_line,
               const LinePixelResidualResult& residual)
    {
        if(!enabled_ || !out_.is_open())
            return;

        const cv::Mat& Tcw = frame.mTcw;
        if(Tcw.empty() || Tcw.rows < 3 || Tcw.cols < 4)
            return;

        auto mat = [&](const int r, const int c) {
            return Tcw.type() == CV_32F ?
                static_cast<double>(Tcw.at<float>(r, c)) :
                Tcw.at<double>(r, c);
        };

        out_ << frame.mnId << ','
             << std::setprecision(17) << frame.mTimeStamp << ','
             << observation_index << ','
             << physical.physical_line_observation_id << ','
             << mapline.mnId << ','
             << physical.representative_member_index << ','
             << member.view_id << ','
             << member.source_line_id << ','
             << member.virtual_observation_index << ','
             << member.p_start.x << ','
             << member.p_start.y << ','
             << member.p_end.x << ','
             << member.p_end.y << ','
             << endpoint0_w.x() << ','
             << endpoint0_w.y() << ','
             << endpoint0_w.z() << ','
             << endpoint1_w.x() << ','
             << endpoint1_w.y() << ','
             << endpoint1_w.z() << ','
             << image_line.x() << ','
             << image_line.y() << ','
             << image_line.z() << ','
             << residual.residual_px.x() << ','
             << residual.residual_px.y() << ','
             << residual.chi2;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 4; ++c)
                out_ << ',' << mat(r, c);
        out_ << '\n';
    }

private:
    PhysicalPixelReplayDump()
    {
        const char* path = std::getenv("T265_PIXEL_K_REPLAY_DUMP");
        enabled_ = path && *path;
        if(!enabled_)
            return;
        out_.open(path);
        if(!out_.is_open())
        {
            enabled_ = false;
            return;
        }
        out_ << "frame_id,timestamp,physical_index,physical_observation_id,"
                "mapline_id,representative_member_index,virtual_index,"
                "source_line_id,virtual_observation_index,u_s,v_s,u_e,v_e,"
                "p0_w_x,p0_w_y,p0_w_z,p1_w_x,p1_w_y,p1_w_z,"
                "obs_line_a,obs_line_b,obs_line_c,"
                "runtime_pre_residual_s,runtime_pre_residual_e,"
                "runtime_pre_chi2,"
                "Tcw_00,Tcw_01,Tcw_02,Tcw_03,"
                "Tcw_10,Tcw_11,Tcw_12,Tcw_13,"
                "Tcw_20,Tcw_21,Tcw_22,Tcw_23\n";
    }

    bool enabled_ = false;
    std::ofstream out_;
};

struct ProjectionLineInformation
{
    bool valid = false;
    double sigma2_start = 0.0;
    double sigma2_end = 0.0;
    double omega_raw_start = 0.0;
    double omega_raw_end = 0.0;
    double omega_start = 0.0;
    double omega_end = 0.0;
    double normalization = 1.0;
    double det_jfv_start = 0.0;
    double det_jfv_end = 0.0;
    double condition_start = 0.0;
    double condition_end = 0.0;
};

class ProjectionLineInformationDump
{
public:
    static ProjectionLineInformationDump& Instance()
    {
        static ProjectionLineInformationDump dump;
        return dump;
    }

    bool enabled() const { return enabled_; }

    void Write(const Frame& frame,
               const size_t observation_index,
               const PhysicalLineObservation& physical,
               const PhysicalLineObservationMember& member,
               const MapLine& mapline,
               const Eigen::Vector3d& image_line,
               const ProjectionLineInformation& info)
    {
        if(!enabled_ || !out_.is_open())
            return;
        out_ << frame.mnId << ','
             << std::setprecision(17) << frame.mTimeStamp << ','
             << observation_index << ','
             << physical.physical_line_observation_id << ','
             << mapline.mnId << ','
             << physical.representative_member_index << ','
             << member.view_id << ','
             << member.p_start.x << ','
             << member.p_start.y << ','
             << member.p_end.x << ','
             << member.p_end.y << ','
             << image_line.x() << ','
             << image_line.y() << ','
             << info.sigma2_start << ','
             << info.sigma2_end << ','
             << info.omega_raw_start << ','
             << info.omega_raw_end << ','
             << info.omega_start << ','
             << info.omega_end << ','
             << info.normalization << ','
             << info.det_jfv_start << ','
             << info.det_jfv_end << ','
             << info.condition_start << ','
             << info.condition_end << ','
             << (info.valid ? 1 : 0) << '\n';
    }

private:
    ProjectionLineInformationDump()
    {
        const char* path = std::getenv("T265_LINE_INFORMATION_DUMP");
        enabled_ = path && *path;
        if(!enabled_)
            return;
        out_.open(path);
        if(!out_.is_open())
        {
            enabled_ = false;
            return;
        }
        out_ << "frame_id,timestamp,physical_index,physical_observation_id,"
                "mapline_id,representative_member_index,virtual_index,"
                "u_s,v_s,u_e,v_e,line_a,line_b,"
                "sigma2_s,sigma2_e,omega_raw_s,omega_raw_e,"
                "omega_s,omega_e,normalization_factor,"
                "det_jfv_s,det_jfv_e,condition_s,condition_e,valid\n";
    }

    bool enabled_ = false;
    std::ofstream out_;
};

double ProjectionInformationNormalization()
{
    const char* env = std::getenv("T265_PROJECTION_INFO_MEDIAN");
    if(!env || !*env)
        return 1.0;
    char* end = NULL;
    const double value = std::strtod(env, &end);
    if(end == env || !std::isfinite(value) || value <= 0.0)
        return 1.0;
    return value;
}

bool VirtualPixelToRawFisheye(const KannalaBrandt8Camera& camera,
                              const int view_id,
                              const Eigen::Vector2d& pv,
                              Eigen::Vector2d& pf)
{
    pf.setConstant(std::numeric_limits<double>::quiet_NaN());
    const double yaw = NativeFisheyeVirtualYawRad(view_id);
    if(!std::isfinite(yaw) || !pv.allFinite())
        return false;
    const Eigen::Vector3d bearing_v(
        (pv.x() - NativeFisheyeVirtualCx()) / NativeFisheyeVirtualFx(),
        (pv.y() - NativeFisheyeVirtualCy()) / NativeFisheyeVirtualFy(),
        1.0);
    const Eigen::Vector3d bearing_c = RotationCvFromYaw(yaw) * bearing_v;
    return camera.project(bearing_c, pf);
}

bool EndpointProjectionCovariance(const KannalaBrandt8Camera& camera,
                                  const int view_id,
                                  const Eigen::Vector2d& pv,
                                  const Eigen::Vector2d& line_normal,
                                  double& sigma2,
                                  double& omega_raw,
                                  double& det_jfv,
                                  double& condition)
{
    sigma2 = std::numeric_limits<double>::quiet_NaN();
    omega_raw = std::numeric_limits<double>::quiet_NaN();
    det_jfv = std::numeric_limits<double>::quiet_NaN();
    condition = std::numeric_limits<double>::quiet_NaN();
    if(!pv.allFinite() || !line_normal.allFinite())
        return false;

    const double h = 1e-3;
    Eigen::Vector2d pu_plus, pu_minus, pv_plus, pv_minus;
    if(!VirtualPixelToRawFisheye(camera, view_id,
                                 pv + Eigen::Vector2d(h, 0.0), pu_plus) ||
       !VirtualPixelToRawFisheye(camera, view_id,
                                 pv - Eigen::Vector2d(h, 0.0), pu_minus) ||
       !VirtualPixelToRawFisheye(camera, view_id,
                                 pv + Eigen::Vector2d(0.0, h), pv_plus) ||
       !VirtualPixelToRawFisheye(camera, view_id,
                                 pv - Eigen::Vector2d(0.0, h), pv_minus))
        return false;

    Eigen::Matrix2d j_fv;
    j_fv.col(0) = (pu_plus - pu_minus) / (2.0 * h);
    j_fv.col(1) = (pv_plus - pv_minus) / (2.0 * h);
    det_jfv = j_fv.determinant();
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(j_fv);
    const Eigen::Vector2d s = svd.singularValues();
    if(!j_fv.allFinite() || !std::isfinite(det_jfv) ||
       std::abs(det_jfv) <= 1e-12 ||
       s.minCoeff() <= 1e-12 || !s.allFinite())
        return false;
    condition = s.maxCoeff() / s.minCoeff();
    const Eigen::Matrix2d j_vf = j_fv.inverse();
    const Eigen::Matrix2d sigma_v = j_vf * j_vf.transpose();
    sigma2 = line_normal.transpose() * sigma_v * line_normal;
    constexpr double kVarianceFloor = 1e-12;
    if(!std::isfinite(sigma2) || sigma2 <= 0.0)
        return false;
    omega_raw = 1.0 / std::max(sigma2, kVarianceFloor);
    return std::isfinite(omega_raw) && omega_raw > 0.0 &&
           std::isfinite(condition);
}

ProjectionLineInformation ComputeProjectionLineInformation(
    const KannalaBrandt8Camera& camera,
    const PhysicalLineObservationMember& member,
    const Eigen::Vector3d& image_line,
    const double normalization)
{
    ProjectionLineInformation info;
    info.normalization = normalization > 0.0 ? normalization : 1.0;
    const Eigen::Vector2d n2d(image_line.x(), image_line.y());
    const Eigen::Vector2d ps(member.p_start.x, member.p_start.y);
    const Eigen::Vector2d pe(member.p_end.x, member.p_end.y);
    if(!EndpointProjectionCovariance(camera, member.view_id, ps, n2d,
                                     info.sigma2_start,
                                     info.omega_raw_start,
                                     info.det_jfv_start,
                                     info.condition_start) ||
       !EndpointProjectionCovariance(camera, member.view_id, pe, n2d,
                                     info.sigma2_end,
                                     info.omega_raw_end,
                                     info.det_jfv_end,
                                     info.condition_end))
        return info;
    info.omega_start = info.omega_raw_start / info.normalization;
    info.omega_end = info.omega_raw_end / info.normalization;
    info.valid =
        std::isfinite(info.omega_start) && std::isfinite(info.omega_end) &&
        info.omega_start > 0.0 && info.omega_end > 0.0;
    return info;
}

EdgeCostSummary EvaluateNativeLinePoseEdges(
    const vector<NativeFisheyeLinePoseEdge*>& line_edges)
{
    EdgeCostSummary summary;
    for(size_t i = 0; i < line_edges.size(); ++i)
    {
        NativeFisheyeLinePoseEdge* e = line_edges[i];
        e->computeError();
        const double chi2 = e->chi2();
        summary.cost += chi2;
        if(std::isfinite(chi2) && chi2 <= 5.991 &&
           e->isProjectionValid())
            ++summary.inliers;
    }
    return summary;
}

EdgeCostSummary EvaluatePhysicalLinePlanePoseEdges(
    const vector<PhysicalLinePlanePoseEdge*>& line_edges)
{
    EdgeCostSummary summary;
    for(size_t i = 0; i < line_edges.size(); ++i)
    {
        PhysicalLinePlanePoseEdge* e = line_edges[i];
        e->computeError();
        const double chi2 = e->chi2();
        summary.cost += chi2;
        if(std::isfinite(chi2) && chi2 <= 5.991 && e->isValid())
            ++summary.inliers;
    }
    return summary;
}

struct LocalBaRuntimeStats
{
    unsigned long current_keyframe_id = 0;
    int local_keyframes = 0;
    int fixed_keyframes = 0;
    int local_mappoints = 0;
    int pose_vertices = 0;
    int point_vertices = 0;
    int kb8_point_edges = 0;
    int legacy_pinhole_point_edges = 0;
    int stereo_point_edges = 0;
    int line_edges = 0;
    int point_edges_with_robust_kernel = 0;
    int point_edges_without_robust_kernel = 0;
    int optimization_round1_ran = 0;
    int optimization_round2_ran = 0;
    int point_inliers_before = 0;
    int point_outliers_before = 0;
    int point_inliers_after = 0;
    int point_outliers_after = 0;
    double point_cost_before = 0.0;
    double point_cost_after = 0.0;
    double line_cost_before = 0.0;
    double line_cost_after = 0.0;
    int keyframe_poses_updated = 0;
    int mappoint_positions_updated = 0;
    int observations_erased = 0;
    int stop_flag_triggered = 0;
    int optimizer_success = 0;
};

std::mutex& LocalBaDiagnosticsMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::string EnvPath(const char* name)
{
    const char* path = std::getenv(name);
    return path ? std::string(path) : std::string();
}

long long OptimizerNowMicros()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string OptimizerThreadIdString()
{
    std::ostringstream os;
    os << std::this_thread::get_id();
    return os.str();
}

void AppendOptimizerDiagnosticLine(const char* env_name,
                                   const std::string& line)
{
    const std::string path = EnvPath(env_name);
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(LocalBaDiagnosticsMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << line << '\n';
}

void LogEssentialGraphRuntimeEvent(const std::string& event,
                                   const int vertex_count,
                                   const int edge_count,
                                   const std::string& result)
{
    std::ostringstream os;
    os << OptimizerNowMicros() << ','
       << OptimizerThreadIdString() << ','
       << event << ','
       << vertex_count << ','
       << edge_count << ','
       << result;
    AppendOptimizerDiagnosticLine("KB8_ESSENTIAL_GRAPH_RUNTIME_CSV",
                                  os.str());
}

void LogEssentialGraphStructureVertex(const unsigned long vertex_id)
{
    std::ostringstream os;
    os << vertex_id << ",VertexSim3Expmap,-1,NONE,NONE";
    AppendOptimizerDiagnosticLine("KB8_ESSENTIAL_GRAPH_STRUCTURE_CSV",
                                  os.str());
}

void LogEssentialGraphStructureEdge(const int edge_id,
                                    const unsigned long vertex0,
                                    const unsigned long vertex1)
{
    std::ostringstream os;
    os << "-1,NONE," << edge_id
       << ",g2o::EdgeSim3,"
       << vertex0 << '-' << vertex1;
    AppendOptimizerDiagnosticLine("KB8_ESSENTIAL_GRAPH_STRUCTURE_CSV",
                                  os.str());
}

void LogEssentialGraphOptimization(const double before_error,
                                   const double after_error,
                                   const int iterations,
                                   const std::string& result)
{
    std::ostringstream os;
    os << before_error << ','
       << after_error << ','
       << iterations << ','
       << result;
    AppendOptimizerDiagnosticLine("KB8_ESSENTIAL_GRAPH_OPTIMIZATION_CSV",
                                  os.str());
}

void AppendLocalBaRuntimeStats(const LocalBaRuntimeStats& stats)
{
    const std::string path =
        EnvPath("STRUCTURE_SLAM_KB8_LOCAL_BA_STATS");
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(LocalBaDiagnosticsMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << stats.current_keyframe_id << ','
        << stats.local_keyframes << ','
        << stats.fixed_keyframes << ','
        << stats.local_mappoints << ','
        << stats.pose_vertices << ','
        << stats.point_vertices << ','
        << stats.kb8_point_edges << ','
        << stats.legacy_pinhole_point_edges << ','
        << stats.stereo_point_edges << ','
        << stats.line_edges << ','
        << stats.point_edges_with_robust_kernel << ','
        << stats.point_edges_without_robust_kernel << ','
        << stats.optimization_round1_ran << ','
        << stats.optimization_round2_ran << ','
        << stats.point_inliers_before << ','
        << stats.point_outliers_before << ','
        << stats.point_inliers_after << ','
        << stats.point_outliers_after << ','
        << stats.point_cost_before << ','
        << stats.point_cost_after << ','
        << stats.line_cost_before << ','
        << stats.line_cost_after << ','
        << stats.keyframe_poses_updated << ','
        << stats.mappoint_positions_updated << ','
        << stats.observations_erased << ','
        << stats.stop_flag_triggered << ','
        << stats.optimizer_success << '\n';
}

void AppendLocalBaObservationErasure(const unsigned long keyframe_id,
                                     const unsigned long mappoint_id,
                                     const double chi2,
                                     const bool depth_positive,
                                     const char* edge_type)
{
    const std::string path =
        EnvPath("STRUCTURE_SLAM_KB8_LOCAL_BA_ERASURES");
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(LocalBaDiagnosticsMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << keyframe_id << ','
        << mappoint_id << ','
        << edge_type << ','
        << chi2 << ','
        << (depth_positive ? 1 : 0) << '\n';
}

struct GlobalBaRuntimeStats
{
    int keyframes = 0;
    int mappoints = 0;
    int maplines = 0;
    int pose_vertices = 0;
    int point_vertices = 0;
    int line_vertices = 0;
    int kb8_point_edges = 0;
    int legacy_pinhole_point_edges = 0;
    int stereo_point_edges = 0;
    int line_edges = 0;
    int robust_point_edges = 0;
    int robust_line_edges = 0;
    int iterations_requested = 0;
    int optimization_ran = 0;
    int point_inliers_before = 0;
    int point_outliers_before = 0;
    int point_inliers_after = 0;
    int point_outliers_after = 0;
    double point_cost_before = 0.0;
    double point_cost_after = 0.0;
    double line_cost_before = 0.0;
    double line_cost_after = 0.0;
    int keyframe_pose_writebacks = 0;
    int mappoint_position_writebacks = 0;
    int mapline_position_writebacks = 0;
    int stop_flag_triggered = 0;
    unsigned long loop_keyframe_id = 0;
    int robust_enabled = 0;
    int with_line_feature = 0;
    int optimizer_success = 0;
};

struct OptimizeSim3RuntimeStats
{
    unsigned long keyframe1_id = 0;
    unsigned long keyframe2_id = 0;
    int correspondences = 0;
    int kb8_forward_edges = 0;
    int kb8_inverse_edges = 0;
    int legacy_forward_edges = 0;
    int legacy_inverse_edges = 0;
    int optimization_round1_ran = 0;
    int optimization_round2_ran = 0;
    int bad_after_round1 = 0;
    int final_inliers = 0;
    double bidirectional_cost_before = 0.0;
    double bidirectional_cost_after = 0.0;
    int optimizer_success = 0;
};

std::mutex& GlobalBaDiagnosticsMutex()
{
    static std::mutex mutex;
    return mutex;
}

void AppendGlobalBaRuntimeStats(const GlobalBaRuntimeStats& stats)
{
    const std::string path =
        EnvPath("STRUCTURE_SLAM_KB8_GLOBAL_BA_STATS");
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(GlobalBaDiagnosticsMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << stats.keyframes << ','
        << stats.mappoints << ','
        << stats.maplines << ','
        << stats.pose_vertices << ','
        << stats.point_vertices << ','
        << stats.line_vertices << ','
        << stats.kb8_point_edges << ','
        << stats.legacy_pinhole_point_edges << ','
        << stats.stereo_point_edges << ','
        << stats.line_edges << ','
        << stats.robust_point_edges << ','
        << stats.robust_line_edges << ','
        << stats.iterations_requested << ','
        << stats.optimization_ran << ','
        << stats.point_inliers_before << ','
        << stats.point_outliers_before << ','
        << stats.point_inliers_after << ','
        << stats.point_outliers_after << ','
        << stats.point_cost_before << ','
        << stats.point_cost_after << ','
        << stats.line_cost_before << ','
        << stats.line_cost_after << ','
        << stats.keyframe_pose_writebacks << ','
        << stats.mappoint_position_writebacks << ','
        << stats.mapline_position_writebacks << ','
        << stats.stop_flag_triggered << ','
        << stats.loop_keyframe_id << ','
        << stats.robust_enabled << ','
        << stats.with_line_feature << ','
        << stats.optimizer_success << '\n';
}

void AppendOptimizeSim3RuntimeStats(const OptimizeSim3RuntimeStats& stats)
{
    const std::string path =
        EnvPath("STRUCTURE_SLAM_KB8_OPTIMIZE_SIM3_STATS");
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(GlobalBaDiagnosticsMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << stats.keyframe1_id << ','
        << stats.keyframe2_id << ','
        << stats.correspondences << ','
        << stats.kb8_forward_edges << ','
        << stats.kb8_inverse_edges << ','
        << stats.legacy_forward_edges << ','
        << stats.legacy_inverse_edges << ','
        << stats.optimization_round1_ran << ','
        << stats.optimization_round2_ran << ','
        << stats.bad_after_round1 << ','
        << stats.final_inliers << ','
        << stats.bidirectional_cost_before << ','
        << stats.bidirectional_cost_after << ','
        << stats.optimizer_success << '\n';
}

double EdgeCost(const vector<g2o::OptimizableGraph::Edge*>& edges)
{
    double cost = 0.0;
    for(size_t i = 0; i < edges.size(); ++i)
    {
        if(!edges[i])
            continue;
        edges[i]->computeError();
        const double chi2 = edges[i]->chi2();
        if(std::isfinite(chi2))
            cost += chi2;
        else
            cost += 1e12;
    }
    return cost;
}

bool AllEdgesUsePoseVertex(
    g2o::OptimizableGraph::Vertex* pose_vertex,
    const vector<g2o::EdgeSE3ProjectXYZOnlyPose*>& mono_edges,
    const vector<Kb8PointPoseEdge*>& kb8_mono_edges,
    const vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>& stereo_edges,
    const vector<NativeFisheyeLinePoseEdge*>& line_edges)
{
    if(!pose_vertex)
        return false;
    for(size_t i = 0; i < mono_edges.size(); ++i)
    {
        if(mono_edges[i]->vertex(0) != pose_vertex)
            return false;
    }
    for(size_t i = 0; i < kb8_mono_edges.size(); ++i)
    {
        if(kb8_mono_edges[i]->vertex(0) != pose_vertex)
            return false;
    }
    for(size_t i = 0; i < stereo_edges.size(); ++i)
    {
        if(stereo_edges[i]->vertex(0) != pose_vertex)
            return false;
    }
    for(size_t i = 0; i < line_edges.size(); ++i)
    {
        if(line_edges[i]->vertex(0) != pose_vertex)
            return false;
    }
    return true;
}

bool AllPhysicalPlaneEdgesUsePoseVertex(
    g2o::OptimizableGraph::Vertex* pose_vertex,
    const vector<Kb8PointPoseEdge*>& kb8_mono_edges,
    const vector<PhysicalLinePlanePoseEdge*>& line_edges)
{
    if(!pose_vertex)
        return false;
    for(size_t i = 0; i < kb8_mono_edges.size(); ++i)
    {
        if(kb8_mono_edges[i]->vertex(0) != pose_vertex)
            return false;
    }
    for(size_t i = 0; i < line_edges.size(); ++i)
    {
        if(line_edges[i]->vertex(0) != pose_vertex)
            return false;
    }
    return true;
}

bool AllEdgesUsePoseVertex(
    g2o::OptimizableGraph::Vertex* pose_vertex,
    const vector<g2o::EdgeSE3ProjectXYZOnlyPose*>& mono_edges,
    const vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>& stereo_edges,
    const vector<NativeFisheyeLinePoseEdge*>& line_edges)
{
    const vector<Kb8PointPoseEdge*> no_kb8_edges;
    return AllEdgesUsePoseVertex(pose_vertex, mono_edges, no_kb8_edges,
                                 stereo_edges, line_edges);
}

int SelectRepresentativePhysicalMemberIndex(
    const PhysicalLineObservation& physical)
{
    if(physical.representative_member_index >= 0 &&
       physical.representative_member_index <
           static_cast<int>(physical.members.size()))
        return physical.representative_member_index;
    if(physical.members.empty())
        return -1;

    const auto supportScore =
        [](const PhysicalLineObservationMember& member) -> int
        {
            if(member.visible_support_valid)
                return member.visible_support_count;
            return (member.start_mask_valid ? 1 : 0) +
                   (member.end_mask_valid ? 1 : 0);
        };
    const auto borderScore =
        [](const PhysicalLineObservationMember& member) -> double
        {
            if(member.border_margin_px > 0.0)
                return member.border_margin_px;
            const int width = member.virtual_width > 0 ?
                                  member.virtual_width : 512;
            const int height = member.virtual_height > 0 ?
                                   member.virtual_height : 512;
            const auto endpoint_margin =
                [width, height](const cv::Point2f& p) -> double
                {
                    if(!std::isfinite(p.x) || !std::isfinite(p.y))
                        return 0.0;
                    return std::min(
                        std::min(static_cast<double>(p.x),
                                 static_cast<double>(width - 1) - p.x),
                        std::min(static_cast<double>(p.y),
                                 static_cast<double>(height - 1) - p.y));
                };
            return std::max(0.0,
                            std::min(endpoint_margin(member.p_start),
                                     endpoint_margin(member.p_end)));
        };

    int best = 0;
    for(size_t i = 1; i < physical.members.size(); ++i)
    {
        const PhysicalLineObservationMember& candidate =
            physical.members[i];
        const PhysicalLineObservationMember& current =
            physical.members[best];
        const int candidate_support = supportScore(candidate);
        const int current_support = supportScore(current);
        if(candidate_support > current_support)
        {
            best = static_cast<int>(i);
            continue;
        }
        if(candidate_support < current_support)
            continue;
        const double candidate_border = borderScore(candidate);
        const double current_border = borderScore(current);
        if(candidate_border > current_border)
        {
            best = static_cast<int>(i);
            continue;
        }
        if(candidate_border < current_border)
            continue;
        if(candidate.virtual_observation_index <
           current.virtual_observation_index)
            best = static_cast<int>(i);
    }
    return best;
}

void FillPoseDiagnostics(const cv::Mat& initial_pose,
                         const cv::Mat& final_pose,
                         Optimizer::NativeFisheyePoseOptimizationResult*
                         result)
{
    if(!result)
        return;
    result->pose_finite =
        !final_pose.empty() && cv::checkRange(final_pose);
    if(!result->pose_finite)
        return;

    const cv::Mat R0 = initial_pose.rowRange(0, 3).colRange(0, 3);
    const cv::Mat t0 = initial_pose.rowRange(0, 3).col(3);
    const cv::Mat R1 = final_pose.rowRange(0, 3).colRange(0, 3);
    const cv::Mat t1 = final_pose.rowRange(0, 3).col(3);
    result->pose_delta_translation = cv::norm(t1 - t0);

    const cv::Mat dR = R1 * R0.t();
    const double trace =
        static_cast<double>(dR.at<float>(0, 0)) +
        static_cast<double>(dR.at<float>(1, 1)) +
        static_cast<double>(dR.at<float>(2, 2));
    const double cos_angle =
        std::max(-1.0, std::min(1.0, (trace - 1.0) * 0.5));
    result->pose_delta_rotation_deg =
        std::acos(cos_angle) * 180.0 / 3.14159265358979323846;

    const cv::Mat RtR = R1.t() * R1;
    const double det = cv::determinant(R1);
    const double ortho_error =
        cv::norm(RtR - cv::Mat::eye(3, 3, RtR.type()));
    result->rotation_valid =
        std::isfinite(det) && std::isfinite(ortho_error) &&
        std::abs(det - 1.0) <= 1e-3 && ortho_error <= 1e-3;
}

} // namespace


void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, const bool bWithLineFeature, bool* pbStopFlag,  const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    if(bWithLineFeature)
    {
        vector<MapLine*> vpML = pMap->GetAllMapLines();
        cout << "***** GlobalBA with points & lines, only for corner optimization *****" << endl;
        BundleAdjustment(vpKFs,vpMP,vpML, nIterations,pbStopFlag, nLoopKF, bRobust);
    } else
    {
        cout << "***** GlobalBA with only points ***** " << endl;
        BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
    }

}


//只有点特征的BA
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    GlobalBaRuntimeStats runtime_stats;
    runtime_stats.keyframes = static_cast<int>(vpKFs.size());
    runtime_stats.mappoints = static_cast<int>(vpMP.size());
    runtime_stats.iterations_requested = nIterations;
    runtime_stats.loop_keyframe_id = nLoopKF;
    runtime_stats.robust_enabled = bRobust ? 1 : 0;
    runtime_stats.with_line_feature = 0;

    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0);
        optimizer.addVertex(vSE3);
        ++runtime_stats.pose_vertices;
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);
    map<KeyFrame*, std::unique_ptr<KannalaBrandt8Camera> > kb8Cameras;
    vector<Kb8PointBundleEdge*> vpEdgesKb8Mono;

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        ++runtime_stats.point_vertices;

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if(pKF->mvuRight[mit->second]<0)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                if(pKF->mbUseKb8PointObservations)
                {
                    if(kb8Cameras.find(pKF) == kb8Cameras.end())
                    {
                        kb8Cameras[pKF].reset(
                            new KannalaBrandt8Camera(
                                pKF->mKb8PointCameraParameters));
                    }
                    Kb8PointBundleEdge* e = new Kb8PointBundleEdge();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                    if(bRobust)
                    {
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                        ++runtime_stats.robust_point_edges;
                    }
                    e->pCamera = kb8Cameras[pKF].get();
                    optimizer.addEdge(e);
                    vpEdgesKb8Mono.push_back(e);
                    ++runtime_stats.kb8_point_edges;
                }
                else
                {
                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    if(bRobust)
                    {
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                        ++runtime_stats.robust_point_edges;
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;

                    optimizer.addEdge(e);
                    ++runtime_stats.legacy_pinhole_point_edges;
                }
            }
            else
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                    ++runtime_stats.robust_point_edges;
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
                ++runtime_stats.stereo_point_edges;
            }
        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    const EdgeCostSummary initial_kb8 =
        EvaluateKb8PointBundleEdges(vpEdgesKb8Mono);
    runtime_stats.point_cost_before = initial_kb8.cost;
    runtime_stats.point_inliers_before = initial_kb8.inliers;
    runtime_stats.point_outliers_before =
        runtime_stats.kb8_point_edges - initial_kb8.inliers;
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    runtime_stats.optimization_ran = 1;
    const EdgeCostSummary final_kb8 =
        EvaluateKb8PointBundleEdges(vpEdgesKb8Mono);
    runtime_stats.point_cost_after = final_kb8.cost;
    runtime_stats.point_inliers_after = final_kb8.inliers;
    runtime_stats.point_outliers_after =
        runtime_stats.kb8_point_edges - final_kb8.inliers;
    runtime_stats.optimizer_success =
        runtime_stats.kb8_point_edges == 0 ||
        runtime_stats.point_cost_after <= runtime_stats.point_cost_before ||
        runtime_stats.point_inliers_after > 0;

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
        ++runtime_stats.keyframe_pose_writebacks;
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
        ++runtime_stats.mappoint_position_writebacks;
    }

    AppendGlobalBaRuntimeStats(runtime_stats);
}

/**
* @brief 包含线特征的BA
* @param vpKFs 关键帧
* @param vpMP MapPoints
* @param vpML MapLines
* @param nIterations 迭代次数，20次
* @param pbStopFlag 是否强制暂停
* @param nLoopKF 关键帧的个数
* @param bRobust 是否使用核函数
*
* 3D-2D 最小化重投影误差 e = (u,v)-project(Tcw*Pw)
* 修改为：e = points的重投影误差 + lines的重投影误差
*          = (u, v)-project(Tcw*Pw) + line * project(Tcw*LineStartPointW) + line * project(Tcw*LineEndPointW)
* 1.Vertex:  g2o::VertexSE3Expmap(),即当前帧的Tcw
*            g2o::VertexSBAPointXYZ(),MapPoint的mWorldPos
*            g2o::VertexLinePointXYZ(), MapLine的端点世界坐标
* 2.Edge:
*      -g2o::EdgeSE3ProjectXYZ(): public  BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>
*          + Vertex：待优化当前帧的Tcw
*          + measurement: MapPoint在当前帧中的二维位置(u, v)
*          + InfoMatrix：invSigma2（与特征点所在的尺度有关）
*      -EdgeLineProjectXYZ : public BaseBinaryEdge<3, Vector3d, g2o::VertexSE3Expmap, g2o::VertexLinePointXYZ>
*          + Vertex1:待优化当前帧的Tcw
*          + Vertex2:待优化的MapLine的一个端点
*          + measuremtn: MapLine的一个端点在当前帧中的二维位置(u,v)
*/
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP, const vector<MapLine *> &vpML,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    GlobalBaRuntimeStats runtime_stats;
    runtime_stats.keyframes = static_cast<int>(vpKFs.size());
    runtime_stats.mappoints = static_cast<int>(vpMP.size());
    runtime_stats.maplines = static_cast<int>(vpML.size());
    runtime_stats.iterations_requested = nIterations;
    runtime_stats.loop_keyframe_id = nLoopKF;
    runtime_stats.robust_enabled = bRobust ? 1 : 0;
    runtime_stats.with_line_feature = 1;


    vector<bool> vbNotIncludedMP, vbNotIncludedML, vbNotIncludedMPL;
    vbNotIncludedMP.resize(vpMP.size());
    vbNotIncludedML.resize(vpML.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId == 0);
        optimizer.addVertex(vSE3);
        ++runtime_stats.pose_vertices;
        if (pKF->mnId > maxKFid)
            maxKFid = pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vector<Kb8PointBundleEdge*> vpEdgesKb8Mono;
    map<KeyFrame*, std::unique_ptr<KannalaBrandt8Camera> > kb8Cameras;

    long unsigned int maxMapPointId = maxKFid;

    int totalEdges = 0;
    // Set MapPoint vertices
    for (size_t i = 0; i < vpMP.size(); i++) {
        MapPoint *pMP = vpMP[i];

        if (pMP->isBad())
            continue;

        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        ++runtime_stats.point_vertices;

        if (id > maxMapPointId) {
            maxMapPointId = id;
        }

        const map<KeyFrame *, size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {

            KeyFrame *pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if (pKF->mvuRight[mit->second] < 0) {
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                if(pKF->mbUseKb8PointObservations)
                {
                    if(kb8Cameras.find(pKF) == kb8Cameras.end())
                    {
                        kb8Cameras[pKF].reset(
                            new KannalaBrandt8Camera(
                                pKF->mKb8PointCameraParameters));
                    }
                    Kb8PointBundleEdge* e = new Kb8PointBundleEdge();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    if (bRobust) {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                        ++runtime_stats.robust_point_edges;
                    }
                    e->pCamera = kb8Cameras[pKF].get();
                    optimizer.addEdge(e);
                    vpEdgesKb8Mono.push_back(e);
                    ++runtime_stats.kb8_point_edges;
                }
                else
                {
                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    if (bRobust) {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                        ++runtime_stats.robust_point_edges;
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    ++runtime_stats.legacy_pinhole_point_edges;
                }
            } else {
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust) {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                    ++runtime_stats.robust_point_edges;
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
                vpEdgesStereo.push_back(e);
                ++runtime_stats.stereo_point_edges;
            }
        }

        totalEdges += nEdges;

        if (nEdges == 0) {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        } else {
            vbNotIncludedMP[i] = false;
        }
    }

    cout << "GBA: Total point edges: " << totalEdges << endl;
    cout << "GBA: Max Point id: " << maxMapPointId << endl;

    int maxMapLineId = maxMapPointId;

    for (size_t i = 0; i < vpML.size(); i++) {
        MapLine *pML = vpML[i];

        if (pML->isBad())
            continue;

        g2o::VertexSBAPointXYZ *vStartPoint = new g2o::VertexSBAPointXYZ();
        vStartPoint->setEstimate(pML->GetWorldPos().head(3));
        const int id1 = (2 * pML->mnId) + 1 + maxMapPointId;
        vStartPoint->setId(id1);
        vStartPoint->setMarginalized(true);
        optimizer.addVertex(vStartPoint);
        runtime_stats.line_vertices += 2;

        g2o::VertexSBAPointXYZ *vEndPoint = new g2o::VertexSBAPointXYZ();
        vEndPoint->setEstimate(pML->GetWorldPos().tail(3));
        const int id2 = (2 * (pML->mnId + 1)) + maxMapPointId;
        vEndPoint->setId(id2);
        vEndPoint->setMarginalized(true);
        optimizer.addVertex(vEndPoint);

        if (id2 > maxMapLineId) {
            maxMapLineId = id2;
        }

        cout << "GBA: Line id1: " << id1 << ", id2: " << id2 << ", Max: " << maxMapLineId << endl;

        const map<KeyFrame *, size_t> observations = pML->GetObservations();

        int nEdges = 0;

        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {
            KeyFrame *pKF = mit->first;

            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nEdges++;

            Eigen::Vector3d lineObs = pKF->mvKeyLineFunctions[mit->second];

            EdgeLineProjectXYZ *es = new EdgeLineProjectXYZ();
            es->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
            es->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
            es->setMeasurement(lineObs);
            es->setInformation(Eigen::Matrix3d::Identity());

            if (bRobust) {
                g2o::RobustKernelHuber *rks = new g2o::RobustKernelHuber;
                es->setRobustKernel(rks);
                rks->setDelta(thHuber3D);
                ++runtime_stats.robust_line_edges;
            }

            es->fx = pKF->fx;
            es->fy = pKF->fy;
            es->cx = pKF->cx;
            es->cy = pKF->cy;

            optimizer.addEdge(es);
            ++runtime_stats.line_edges;

            EdgeLineProjectXYZ *ee = new EdgeLineProjectXYZ();
            ee->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
            ee->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
            ee->setMeasurement(lineObs);
            ee->setInformation(Eigen::Matrix3d::Identity());

            if (bRobust) {
                g2o::RobustKernelHuber *rke = new g2o::RobustKernelHuber;
                ee->setRobustKernel(rke);
                rke->setDelta(thHuber3D);
                ++runtime_stats.robust_line_edges;
            }

            ee->fx = pKF->fx;
            ee->fy = pKF->fy;
            ee->cx = pKF->cx;
            ee->cy = pKF->cy;

            optimizer.addEdge(ee);
            ++runtime_stats.line_edges;
        }

        if (nEdges == 0) {
            optimizer.removeVertex(vStartPoint);
            optimizer.removeVertex(vEndPoint);
            vbNotIncludedML[i] = true;
        } else {
            vbNotIncludedML[i] = false;
        }
    }


    // Optimize!
    const EdgeCostSummary initial_kb8 =
        EvaluateKb8PointBundleEdges(vpEdgesKb8Mono);
    runtime_stats.point_cost_before = initial_kb8.cost;
    runtime_stats.point_inliers_before = initial_kb8.inliers;
    runtime_stats.point_outliers_before =
        runtime_stats.kb8_point_edges - initial_kb8.inliers;
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    runtime_stats.optimization_ran = 1;
    const EdgeCostSummary final_kb8 =
        EvaluateKb8PointBundleEdges(vpEdgesKb8Mono);
    runtime_stats.point_cost_after = final_kb8.cost;
    runtime_stats.point_inliers_after = final_kb8.inliers;
    runtime_stats.point_outliers_after =
        runtime_stats.kb8_point_edges - final_kb8.inliers;
    runtime_stats.optimizer_success =
        runtime_stats.kb8_point_edges == 0 ||
        runtime_stats.point_cost_after <= runtime_stats.point_cost_before ||
        runtime_stats.point_inliers_after > 0;

    int bad = 0;
    int PNMono = 0;
    double PEMono = 0;
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
        g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];


        const float chi2 = e->chi2();
        //cout<<"optimize chi2"<<chi2<<endl;
        PNMono++;
        PEMono += chi2;

        if (chi2 > thHuber2D * thHuber2D) {
            bad++;
            cout << " GBA: Bad point: " << chi2 << endl;
        }
    }

    if (PNMono == 0)
        cout << "GBA: No mono points " << " ";
    else
        cout << "GBA: Mono points: " << PEMono / PNMono << " ";

    int PNStereo = 0;
    double PEStereo = 0;
    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];

        const float chi2 = e->chi2();
        //cout<<"optimize chi2"<<chi2<<endl;
        PNStereo++;
        PEStereo += chi2;

        if (chi2 > thHuber3D * thHuber3D) {
            bad++;
            cout << "GBA: Bad stereo point: " << chi2 << endl;
        }
    }
    if (PNStereo == 0)
        cout << "GBA: No stereo points " << " ";
    else
        cout << "GBA: Stereo points: " << PEStereo / PNStereo << endl;

    cout << "GBA: Total bad point edges: " << bad << endl;

    // Recover optimized data

    //Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if (nLoopKF == 0) {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        } else {
            pKF->mTcwGBA.create(4, 4, CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
        ++runtime_stats.keyframe_pose_writebacks;
    }

    //Points
    for (size_t i = 0; i < vpMP.size(); i++) {
        if (vbNotIncludedMP[i])
            continue;

        MapPoint *pMP = vpMP[i];

        if (pMP->isBad())
            continue;

        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));

        double dis = cv::norm(Converter::toCvMat(vPoint->estimate()) - pMP->GetWorldPos());
        if (dis > 0.5) {
            std::cout << "Point id: " << pMP->mnId << ", bad: " << pMP->isBad()  << ", pose - before: " << pMP->GetWorldPos().t()
                      << ", after: " << Converter::toCvMat(vPoint->estimate()).t() << std::endl;
        }

        if (nLoopKF == 0) {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        } else {
            pMP->mPosGBA.create(3, 1, CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
        ++runtime_stats.mappoint_position_writebacks;
    }

    //Lines
    for (size_t i = 0; i < vpML.size(); i++) {

        if (vbNotIncludedML[i])
            continue;

        MapLine *pML = vpML[i];

        if (pML->isBad())
            continue;

        g2o::VertexSBAPointXYZ *vStartPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                (2 * pML->mnId) + 1 + maxMapPointId));
        g2o::VertexSBAPointXYZ *vEndPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                2 * (pML->mnId + 1) + maxMapPointId));

        if (nLoopKF == 0) {
            Vector6d linePos;
            linePos << vStartPoint->estimate(), vEndPoint->estimate();
            pML->SetWorldPos(linePos);
            pML->UpdateAverageDir();
        } else {
            pML->mPosGBA.create(6, 1, CV_32F);
            Converter::toCvMat(vStartPoint->estimate()).copyTo(pML->mPosGBA.rowRange(0, 3));
            Converter::toCvMat(vEndPoint->estimate()).copyTo(pML->mPosGBA.rowRange(3, 6));
            pML->mnBAGlobalForKF = nLoopKF;
        }
        ++runtime_stats.mapline_position_writebacks;
    }

    AppendGlobalBaRuntimeStats(runtime_stats);
}

/**
 * 该优化函数主要用于Tracking线程中
 *
 * 3D-2D 最小化重投影误差 e = (u, v) - project(Tcw*Pw)
 * 修改为： e = points的重投影误差 + lines的重投影误差
 *          = (u, v) - project(Tcw*Pw) + line * project(Tcw * LineStartPointw) + line * project(Tcw * LineEndPointW)
 * 只优化pFrame的Tcw，不优化MapPoints和MapLines的坐标
 *
 * 1.Vertex: g2o::VertexSE3Expmap，即当前帧的Tcw
 * 2.Edge:
 *      - g2o::EdgeSE3ProjectXYZOnlyPose: public  BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>
 *          + Vertex：待优化当前帧的Tcw
 *          + measurement：MapPoint在当前帧中的二维位置(u,v)
 *          + InfoMatrix：invSigma2（与特征点所在的尺度有关）
 *
 *      添加相机位姿与线特征之间的误差边：
 *      - g2o::
 *
 * @param pFrame 图像帧
 * @return
 */
int Optimizer::PoseOptimization(Frame *pFrame,bool isCorner)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    // Set Frame vertex
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<Kb8PointPoseEdge *> vpEdgesKb8Mono;
    vector<size_t> vnIndexEdgeKb8Mono;
    vpEdgesKb8Mono.reserve(N);
    vnIndexEdgeKb8Mono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    vector<double> vMonoPointInfo(N, 1);
    vector<double> vSteroPointInfo(N, 1);
    std::unique_ptr<KannalaBrandt8Camera> kb8CameraHolder;
    const CentralCameraModel* kb8Camera = nullptr;
    if(pFrame->HasKb8PointObservations())
    {
        kb8CameraHolder.reset(
            new KannalaBrandt8Camera(pFrame->mKb8PointCameraParameters));
        if(!kb8CameraHolder->valid())
            return 0;
        kb8Camera = kb8CameraHolder.get();
    }

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);


        for (int i = 0; i < N; i++) {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (pMP) {
                // Monocular observation
                if (pFrame->mvuRight[i] < 0) {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;
                    cv::Mat Xw = pMP->GetWorldPos();
                    if(kb8Camera)
                    {
                        Kb8PointPoseEdge *e = new Kb8PointPoseEdge();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);
                        e->pCamera = kb8Camera;
                        e->Xw = Eigen::Vector3d(Xw.at<float>(0), Xw.at<float>(1), Xw.at<float>(2));
                        optimizer.addEdge(e);
                        vpEdgesKb8Mono.push_back(e);
                        vnIndexEdgeKb8Mono.push_back(i);
                    }
                    else
                    {
                        g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                } else  // Stereo observation
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    //SET EDGE
                    Eigen::Matrix<double, 3, 1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }

        }
    }

    const int NL = pFrame->NL;

    vector<EdgeLineProjectXYZOnlyPose *> vpEdgesLineSp;
    vector<size_t> vnIndexLineEdgeSp;
    vpEdgesLineSp.reserve(NL);
    vnIndexLineEdgeSp.reserve(NL);

    vector<EdgeLineProjectXYZOnlyPose *> vpEdgesLineEp;
    vector<size_t> vnIndexLineEdgeEp;
    vpEdgesLineEp.reserve(NL);
    vnIndexLineEdgeEp.reserve(NL);

    vector<double> vMonoStartPointInfo(NL, 1);
    vector<double> vMonoEndPointInfo(NL, 1);
    vector<double> vSteroStartPointInfo(NL, 1);
    vector<double> vSteroEndPointInfo(NL, 1);

    // Set MapLine vertices
    {
        unique_lock<mutex> lock(MapLine::mGlobalMutex);

        for (int i = 0; i < NL; i++) {
            MapLine *pML = pFrame->mvpMapLines[i];
            if (pML) {
                nInitialCorrespondences++;
                pFrame->mvbLineOutlier[i] = false;

                Eigen::Vector3d line_obs;
                line_obs = pFrame->mvKeyLineFunctions[i];

                EdgeLineProjectXYZOnlyPose *els = new EdgeLineProjectXYZOnlyPose();

                els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                els->setMeasurement(line_obs);
                els->setInformation(Eigen::Matrix3d::Identity());

                g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                els->setRobustKernel(rk_line_s);
                rk_line_s->setDelta(deltaStereo);

                els->fx = pFrame->fx;
                els->fy = pFrame->fy;
                els->cx = pFrame->cx;
                els->cy = pFrame->cy;

                els->Xw = pML->mWorldPos.head(3);
                optimizer.addEdge(els);

                vpEdgesLineSp.push_back(els);
                vnIndexLineEdgeSp.push_back(i);

                EdgeLineProjectXYZOnlyPose *ele = new EdgeLineProjectXYZOnlyPose();

                ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                ele->setMeasurement(line_obs);
                ele->setInformation(Eigen::Matrix3d::Identity());

                g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                ele->setRobustKernel(rk_line_e);
                rk_line_e->setDelta(deltaStereo);

                ele->fx = pFrame->fx;
                ele->fy = pFrame->fy;
                ele->cx = pFrame->cx;
                ele->cy = pFrame->cy;

                ele->Xw = pML->mWorldPos.tail(3);

                optimizer.addEdge(ele);

                vpEdgesLineEp.push_back(ele);
                vnIndexLineEdgeEp.push_back(i);
            }
        }
    }

    if (nInitialCorrespondences < 3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
    const int its[4] = {10, 10, 10, 10};

    int nBad = 0;

    for (size_t it = 0; it < 4; it++) {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;

        int PNMono = 0;
        double PEMono = 0, PMaxMono = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (pFrame->mvbOutlier[idx]) {
                e->computeError();
            }

            const float chi2 = e->chi2();
            //cout<<"optimize chi2"<<chi2<<endl;
            PNMono++;
            PEMono += chi2;
            PMaxMono = PMaxMono > chi2 ? PMaxMono : chi2;

            if (chi2 > chi2Mono[it]) {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            } else {
                pFrame->mvbOutlier[idx] = false;
                vMonoPointInfo[i] = 1.0 / sqrt(chi2);
                e->setLevel(0);
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        for(size_t i = 0, iend = vpEdgesKb8Mono.size(); i < iend; ++i)
        {
            Kb8PointPoseEdge* e = vpEdgesKb8Mono[i];
            const size_t idx = vnIndexEdgeKb8Mono[i];
            if(pFrame->mvbOutlier[idx])
                e->computeError();
            const float chi2 = e->chi2();
            if(chi2 > chi2Mono[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
                vMonoPointInfo[i] = 1.0 / sqrt(chi2);
            }
            if(it==2)
                e->setRobustKernel(0);
        }

        if (PNMono == 0)
            cout << "No mono points " << " ";
        else
            cout << " Mono points: " << PEMono / PNMono << " ";

        int PNStereo = 0;
        double PEStereo = 0, PMaxStereo = 0;
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->mvbOutlier[idx]) {
                e->computeError();
            }

            const float chi2 = e->chi2();
            //cout<<"optimize chi2"<<chi2<<endl;
            PNStereo++;
            PEStereo += chi2;
            PMaxStereo = PMaxStereo > chi2 ? PMaxStereo : chi2;

            if (chi2 > chi2Stereo[it]) {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            } else {
                e->setLevel(0);
                pFrame->mvbOutlier[idx] = false;
                vSteroPointInfo[i] = 1.0 / sqrt(chi2);
            }

            if (it == 2)
                e->setRobustKernel(0);
        }
        if (PNStereo == 0)
            cout << "No stereo points " << " ";
        else
            cout << " Stereo points: " << PEStereo / PNStereo << endl;

        int PNLine = 0;
        double PELine = 0, PMaxLine = 0;
        for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
            EdgeLineProjectXYZOnlyPose *e1 = vpEdgesLineSp[i];  //线段起始点
            EdgeLineProjectXYZOnlyPose *e2 = vpEdgesLineEp[i];  //线段终止点

            const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

            if (pFrame->mvbLineOutlier[idx]) {
                e1->computeError();
                e2->computeError();
            }
            e1->computeError();
            e2->computeError();

            const float chi2_s = e1->chiline();//e1->chi2();
            const float chi2_e = e2->chiline();//e2->chi2();
//                cout<<"Optimization: chi2_s "<<chi2_s<<", chi2_e "<<chi2_e<<endl;

            PNLine++;
            PELine += chi2_s + chi2_e;
            PMaxLine = PMaxLine > chi2_s + chi2_e ? PMaxLine : chi2_s + chi2_e;


            if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
                pFrame->mvbLineOutlier[idx] = true;
                e1->setLevel(1);
                e2->setLevel(1);
                nBad++;
            } else {
                pFrame->mvbLineOutlier[idx] = false;
                e1->setLevel(0);
                e2->setLevel(0);
                vSteroEndPointInfo[i] = 1.0 / sqrt(chi2_e);
                vSteroStartPointInfo[i] = 1.0 / sqrt(chi2_s);
            }

            if (it == 2) {
                e1->setRobustKernel(0);
                e2->setRobustKernel(0);
            }
        }

        if (PNLine == 0)
            cout << "No lines " << " ";
        else
            cout << " Lines: " << PELine / PNLine << endl;

        int PN = 0;
        double PE = 0, PMax = 0;


        if (PN == 0)
            cout << "No plane " << " ";
        else
            cout << " Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

        if (optimizer.edges().size() < 10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    pFrame->SetPose(Converter::toCvMat(SE3quat_recov));

    return nInitialCorrespondences - nBad;
    }

int Optimizer::PoseOptimization(Frame *pFrame)
{
        pFrame->mPoseOptimizationPointEdges = 0;
        pFrame->mPoseOptimizationInitialInliers = 0;
        pFrame->mPoseOptimizationFinalInliers = 0;
        pFrame->mPoseOptimizationOutliers = 0;
        pFrame->mPoseOptimizationReprojectionBefore = 0.0;
        pFrame->mPoseOptimizationReprojectionAfter = 0.0;
        pFrame->mbPoseOptimizationSuccess = false;
        pFrame->mbPoseOptimizationUsedKb8PointEdge = false;

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences=0;

        // Set Frame vertex
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<Kb8PointPoseEdge*> vpEdgesKb8Mono;
        vector<size_t> vnIndexEdgeKb8Mono;
        vpEdgesKb8Mono.reserve(N);
        vnIndexEdgeKb8Mono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);
        std::unique_ptr<KannalaBrandt8Camera> kb8CameraHolder;
        const CentralCameraModel* kb8Camera = nullptr;
        if(pFrame->HasKb8PointObservations())
        {
            kb8CameraHolder.reset(
                new KannalaBrandt8Camera(pFrame->mKb8PointCameraParameters));
            if(!kb8CameraHolder->valid())
                return 0;
            kb8Camera = kb8CameraHolder.get();
        }


        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<N; i++)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(pMP)
                {
                    // Monocular observation
                    if(pFrame->mvuRight[i]<0)
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double,2,1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;
                        cv::Mat Xw = pMP->GetWorldPos();
                        if(kb8Camera)
                        {
                            Kb8PointPoseEdge* e = new Kb8PointPoseEdge();
                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                            e->setMeasurement(obs);
                            const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(deltaMono);
                            e->pCamera = kb8Camera;
                            e->Xw = Eigen::Vector3d(Xw.at<float>(0), Xw.at<float>(1), Xw.at<float>(2));
                            optimizer.addEdge(e);
                            vpEdgesKb8Mono.push_back(e);
                            vnIndexEdgeKb8Mono.push_back(i);
                        }
                        else
                        {
                            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                            e->setMeasurement(obs);
                            const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(deltaMono);

                            e->fx = pFrame->fx;
                            e->fy = pFrame->fy;
                            e->cx = pFrame->cx;
                            e->cy = pFrame->cy;
                            e->Xw[0] = Xw.at<float>(0);
                            e->Xw[1] = Xw.at<float>(1);
                            e->Xw[2] = Xw.at<float>(2);

                            optimizer.addEdge(e);
                            vpEdgesMono.push_back(e);
                            vnIndexEdgeMono.push_back(i);
                        }
                    }
                    else  // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        //SET EDGE
                        Eigen::Matrix<double,3,1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }

            }
        }


        pFrame->mPoseOptimizationPointEdges = nInitialCorrespondences;
        pFrame->mPoseOptimizationInitialInliers = nInitialCorrespondences;
        pFrame->mbPoseOptimizationUsedKb8PointEdge =
            !vpEdgesKb8Mono.empty();

        const auto meanResidual =
            [&vpEdgesMono, &vpEdgesKb8Mono, &vpEdgesStereo]() -> double
            {
                double sum = 0.0;
                int count = 0;
                for(size_t i = 0; i < vpEdgesMono.size(); ++i)
                {
                    vpEdgesMono[i]->computeError();
                    sum += vpEdgesMono[i]->error().norm();
                    ++count;
                }
                for(size_t i = 0; i < vpEdgesKb8Mono.size(); ++i)
                {
                    vpEdgesKb8Mono[i]->computeError();
                    sum += vpEdgesKb8Mono[i]->error().norm();
                    ++count;
                }
                for(size_t i = 0; i < vpEdgesStereo.size(); ++i)
                {
                    vpEdgesStereo[i]->computeError();
                    sum += vpEdgesStereo[i]->error().norm();
                    ++count;
                }
                return count > 0 ? sum / static_cast<double>(count) : 0.0;
            };
        pFrame->mPoseOptimizationReprojectionBefore = meanResidual();

        if(nInitialCorrespondences<3)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4]={5.991,5.991,5.991,5.991};
        const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
        const int its[4]={10,10,10,10};

        int nBad=0;
        for(size_t it=0; it<4; it++)
        {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad=0;
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                //cout<<"optimize point chi2, "<<chi2<<endl;
                if(chi2>chi2Mono[it])
                {
                    pFrame->mvbOutlier[idx]=true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    pFrame->mvbOutlier[idx]=false;
                    e->setLevel(0);
                }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesKb8Mono.size(); i<iend; i++)
        {
            Kb8PointPoseEdge* e = vpEdgesKb8Mono[i];
            const size_t idx = vnIndexEdgeKb8Mono[i];

            if(pFrame->mvbOutlier[idx])
                e->computeError();

            const float chi2 = e->chi2();
            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Stereo[it])
                {
                    pFrame->mvbOutlier[idx]=true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx]=false;
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

            if(optimizer.edges().size()<10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        const int finalInliers = nInitialCorrespondences - nBad;
        pFrame->mPoseOptimizationFinalInliers = finalInliers;
        pFrame->mPoseOptimizationOutliers = nBad;
        pFrame->mPoseOptimizationReprojectionAfter = meanResidual();
        pFrame->mbPoseOptimizationSuccess = finalInliers > 0;

        return finalInliers;
    }

Optimizer::NativeFisheyePoseOptimizationResult
Optimizer::PoseOptimizationWithNativeFisheyeLines(
    Frame *pFrame, const int minValidLineCount)
{
        NativeFisheyePoseOptimizationResult result;
        result.status = "NOT_ATTEMPTED";
        if(!pFrame)
        {
            result.status = "NULL_FRAME";
            return result;
        }

        result.attempted = true;
        const cv::Mat point_only_pose = pFrame->mTcw.clone();
        const std::vector<bool> point_outlier_before = pFrame->mvbOutlier;
        std::vector<bool> line_outliers = pFrame->mvbFisheyeLineOutlier;
        if(line_outliers.size() < pFrame->mvpFisheyeMapLines.size())
            line_outliers.resize(pFrame->mvpFisheyeMapLines.size(), false);

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
        linearSolver = new g2o::LinearSolverDense<
            g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 * solver_ptr =
            new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver =
            new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(point_only_pose));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        result.pose_vertex_count = 1;

        const int N = pFrame->N;
        int nInitialCorrespondences = 0;
        std::vector<bool> local_point_outliers = pFrame->mvbOutlier;
        if(local_point_outliers.size() < static_cast<size_t>(N))
            local_point_outliers.resize(static_cast<size_t>(N), false);

        vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);
        const float deltaLine = sqrt(5.991);
        vector<NativeFisheyeLinePoseEdge*> vpEdgesLine;
        vector<size_t> vnIndexEdgeLine;
        vpEdgesLine.reserve(pFrame->mvpFisheyeMapLines.size());
        vnIndexEdgeLine.reserve(pFrame->mvpFisheyeMapLines.size());

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);
            for(int i=0; i<N; i++)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(!pMP)
                    continue;

                if(pFrame->mvuRight[i]<0)
                {
                    nInitialCorrespondences++;
                    local_point_outliers[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZOnlyPose* e =
                        new g2o::EdgeSE3ProjectXYZOnlyPose();
                    e->setVertex(0, dynamic_cast<
                        g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 =
                        pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*
                                      invSigma2);
                    g2o::RobustKernelHuber* rk =
                        new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);
                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else
                {
                    nInitialCorrespondences++;
                    local_point_outliers[i] = false;

                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e =
                        new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                    e->setVertex(0, dynamic_cast<
                        g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 =
                        pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info =
                        Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);
                    g2o::RobustKernelHuber* rk =
                        new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);
                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);
                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
        }

        if(nInitialCorrespondences < 3)
        {
            result.point_edge_count =
                static_cast<int>(vpEdgesMono.size() +
                                 vpEdgesStereo.size());
            result.total_edge_count = result.point_edge_count;
            result.uses_single_shared_pose_vertex =
                AllEdgesUsePoseVertex(optimizer.vertex(0), vpEdgesMono,
                                      vpEdgesStereo, vpEdgesLine);
            const EdgeCostSummary point_summary =
                EvaluatePointPoseEdges(vpEdgesMono, vpEdgesStereo);
            result.initial_point_cost = point_summary.cost;
            result.final_point_cost = point_summary.cost;
            result.initial_point_inliers = point_summary.inliers;
            result.final_point_inliers = point_summary.inliers;
            FillPoseDiagnostics(point_only_pose, point_only_pose, &result);
            result.status = "NOT_ENOUGH_POINT_CORRESPONDENCES";
            pFrame->mvbOutlier = point_outlier_before;
            return result;
        }

        const size_t nNative =
            std::min(pFrame->mvpFisheyeMapLines.size(),
                     pFrame->mvFisheyeLineObservations.size());
        for(size_t i = 0; i < nNative; ++i)
        {
            MapLine* pML = pFrame->mvpFisheyeMapLines[i];
            if(!pML)
                continue;
            if(pML->isBad())
            {
                line_outliers[i] = true;
                pFrame->mvpFisheyeMapLines[i] =
                    static_cast<MapLine*>(NULL);
                continue;
            }

            const FisheyeLineObservation& native_obs =
                pFrame->mvFisheyeLineObservations[i];
            bool line_ok = false;
            const Eigen::Vector3d line =
                NormalizeImageLineForPixelDistance(
                    Eigen::Vector2d(native_obs.virtual_start.x,
                                    native_obs.virtual_start.y),
                    Eigen::Vector2d(native_obs.virtual_end.x,
                                    native_obs.virtual_end.y),
                    &line_ok);
            if(!line_ok)
            {
                line_outliers[i] = true;
                continue;
            }

            const Vector6d world = pML->GetWorldPos();
            Eigen::Vector3d endpoint0(world(0), world(1), world(2));
            Eigen::Vector3d endpoint1(world(3), world(4), world(5));
            if(!endpoint0.allFinite() || !endpoint1.allFinite() ||
               (endpoint0 - endpoint1).norm() <= 1e-9)
            {
                line_outliers[i] = true;
                continue;
            }

            VirtualViewLineObservation observation;
            observation.frame_id = static_cast<int>(pFrame->mnId);
            observation.view_id = native_obs.view_id;
            observation.line_id = native_obs.line_id;
            observation.image_line_normalized = line;
            observation.observed_start_px =
                Eigen::Vector2d(native_obs.virtual_start.x,
                                native_obs.virtual_start.y);
            observation.observed_end_px =
                Eigen::Vector2d(native_obs.virtual_end.x,
                                native_obs.virtual_end.y);

            NativeFisheyeLinePoseEdge* e =
                new NativeFisheyeLinePoseEdge();
            e->setVertex(0, dynamic_cast<
                g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->endpoint0_w = endpoint0;
            e->endpoint1_w = endpoint1;
            e->observation = observation;
            const double info = std::isfinite(native_obs.information) &&
                                native_obs.information > 0.0 ?
                                native_obs.information : 1.0;
            e->setInformation(Eigen::Matrix2d::Identity() * info);
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaLine);

            if(!e->isProjectionValid())
            {
                line_outliers[i] = true;
                delete e;
                continue;
            }

            line_outliers[i] = false;
            optimizer.addEdge(e);
            vpEdgesLine.push_back(e);
            vnIndexEdgeLine.push_back(i);
        }

        result.line_initial = static_cast<int>(vpEdgesLine.size());
        result.point_edge_count =
            static_cast<int>(vpEdgesMono.size() + vpEdgesStereo.size());
        result.line_edge_count = static_cast<int>(vpEdgesLine.size());
        result.total_edge_count =
            result.point_edge_count + result.line_edge_count;
        result.uses_single_shared_pose_vertex =
            AllEdgesUsePoseVertex(optimizer.vertex(0), vpEdgesMono,
                                  vpEdgesStereo, vpEdgesLine);
        const EdgeCostSummary initial_point_summary =
            EvaluatePointPoseEdges(vpEdgesMono, vpEdgesStereo);
        const EdgeCostSummary initial_line_summary =
            EvaluateNativeLinePoseEdges(vpEdgesLine);
        result.initial_point_cost = initial_point_summary.cost;
        result.initial_line_cost = initial_line_summary.cost;
        result.initial_point_inliers = initial_point_summary.inliers;
        result.initial_line_inliers = initial_line_summary.inliers;
        if(result.line_initial < std::max(1, minValidLineCount))
        {
            result.final_point_cost = result.initial_point_cost;
            result.final_line_cost = result.initial_line_cost;
            result.final_point_inliers = result.initial_point_inliers;
            result.final_line_inliers = result.initial_line_inliers;
            FillPoseDiagnostics(point_only_pose, point_only_pose, &result);
            result.status = "NOT_ENOUGH_VALID_NATIVE_LINES";
            pFrame->mvbOutlier = point_outlier_before;
            pFrame->mvbFisheyeLineOutlier = line_outliers;
            return result;
        }

        const float chi2Mono[4]={5.991,5.991,5.991,5.991};
        const float chi2Stereo[4]={7.815,7.815,7.815,7.815};
        const float chi2Line[4]={5.991,5.991,5.991,5.991};
        const int its[4]={10,10,10,10};

        int nBadPoints = 0;
        int nBadLines = 0;
        for(size_t it=0; it<4; it++)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBadPoints = 0;
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];
                const size_t idx = vnIndexEdgeMono[i];
                if(local_point_outliers[idx])
                    e->computeError();
                const float chi2 = e->chi2();
                if(chi2>chi2Mono[it])
                {
                    local_point_outliers[idx]=true;
                    e->setLevel(1);
                    nBadPoints++;
                }
                else
                {
                    local_point_outliers[idx]=false;
                    e->setLevel(0);
                }
                if(it==2)
                    e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
            {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e =
                    vpEdgesStereo[i];
                const size_t idx = vnIndexEdgeStereo[i];
                if(local_point_outliers[idx])
                    e->computeError();
                const float chi2 = e->chi2();
                if(chi2>chi2Stereo[it])
                {
                    local_point_outliers[idx]=true;
                    e->setLevel(1);
                    nBadPoints++;
                }
                else
                {
                    local_point_outliers[idx]=false;
                    e->setLevel(0);
                }
                if(it==2)
                    e->setRobustKernel(0);
            }

            nBadLines = 0;
            for(size_t i=0, iend=vpEdgesLine.size(); i<iend; i++)
            {
                NativeFisheyeLinePoseEdge* e = vpEdgesLine[i];
                const size_t idx = vnIndexEdgeLine[i];
                if(line_outliers[idx])
                    e->computeError();
                const float chi2 = e->chi2();
                if(chi2>chi2Line[it] || !e->isProjectionValid())
                {
                    line_outliers[idx]=true;
                    e->setLevel(1);
                    nBadLines++;
                }
                else
                {
                    line_outliers[idx]=false;
                    e->setLevel(0);
                }
                if(it==2)
                    e->setRobustKernel(0);
            }
        }

        result.point_inliers =
            nInitialCorrespondences - nBadPoints;
        result.line_bad = nBadLines;
        result.line_inliers =
            result.line_initial - nBadLines;
        const EdgeCostSummary final_point_summary =
            EvaluatePointPoseEdges(vpEdgesMono, vpEdgesStereo);
        const EdgeCostSummary final_line_summary =
            EvaluateNativeLinePoseEdges(vpEdgesLine);
        result.final_point_cost = final_point_summary.cost;
        result.final_line_cost = final_line_summary.cost;
        result.final_point_inliers = final_point_summary.inliers;
        result.final_line_inliers = final_line_summary.inliers;
        pFrame->mvbFisheyeLineOutlier = line_outliers;

        if(result.line_inliers < std::max(1, minValidLineCount))
        {
            FillPoseDiagnostics(point_only_pose, point_only_pose, &result);
            result.status = "ALL_OR_TOO_MANY_NATIVE_LINE_OUTLIERS";
            pFrame->mvbOutlier = point_outlier_before;
            return result;
        }

        g2o::VertexSE3Expmap* vSE3_recov =
            static_cast<g2o::VertexSE3Expmap*>(
                optimizer.vertex(0));
        const cv::Mat pose =
            Converter::toCvMat(vSE3_recov->estimate());
        FillPoseDiagnostics(point_only_pose, pose, &result);
        if(pose.empty() || !cv::checkRange(pose))
        {
            result.status = "NON_FINITE_POSE";
            pFrame->mvbOutlier = point_outlier_before;
            return result;
        }

        const cv::Mat R = pose.rowRange(0, 3).colRange(0, 3);
        cv::Mat RtR = R.t() * R;
        const double det = cv::determinant(R);
        const double ortho_error =
            cv::norm(RtR - cv::Mat::eye(3, 3, RtR.type()));
        if(!std::isfinite(det) || !std::isfinite(ortho_error) ||
           std::abs(det - 1.0) > 1e-3 || ortho_error > 1e-3)
        {
            result.status = "INVALID_ROTATION";
            result.rotation_valid = false;
            pFrame->mvbOutlier = point_outlier_before;
            return result;
        }
        result.pose_finite = true;
        result.rotation_valid = true;

        pFrame->mvbOutlier = local_point_outliers;
        pFrame->SetPose(pose);
        result.success = true;
        result.wrote_pose = true;
        result.status = "OK";
        return result;
    }

Optimizer::NativeFisheyePoseOptimizationResult
Optimizer::PoseOptimizationWithPhysicalPixelLines(
    Frame *pFrame, const int minValidLineCount,
    const bool useProjectionCovariance)
{
        NativeFisheyePoseOptimizationResult result;
        result.status = "NOT_ATTEMPTED";
        if(!pFrame)
        {
            result.status = "NULL_FRAME";
            return result;
        }
        if(!pFrame->HasKb8PointObservations())
        {
            result.status = "NOT_KB8_FRAME";
            return result;
        }

        result.attempted = true;
        const cv::Mat point_only_pose = pFrame->mTcw.clone();
        const std::vector<bool> point_outlier_before = pFrame->mvbOutlier;
        std::vector<bool> local_point_outliers = pFrame->mvbOutlier;
        if(local_point_outliers.size() < static_cast<size_t>(pFrame->N))
            local_point_outliers.resize(static_cast<size_t>(pFrame->N), false);

        KannalaBrandt8Camera camera(pFrame->mKb8PointCameraParameters);
        if(!camera.valid())
        {
            result.status = "INVALID_KB8_CAMERA";
            return result;
        }

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
        linearSolver = new g2o::LinearSolverDense<
            g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3* solver_ptr =
            new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver =
            new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(point_only_pose));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        result.pose_vertex_count = 1;

        const float deltaMono = std::sqrt(5.991f);
        const float deltaLine = std::sqrt(5.991f);
        vector<Kb8PointPoseEdge*> vpEdgesKb8Mono;
        vector<size_t> vnIndexEdgeKb8Mono;
        vpEdgesKb8Mono.reserve(static_cast<size_t>(pFrame->N));
        vnIndexEdgeKb8Mono.reserve(static_cast<size_t>(pFrame->N));

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);
            for(int i = 0; i < pFrame->N; ++i)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(!pMP || pMP->isBad() ||
                   i >= static_cast<int>(pFrame->mvKeysUn.size()))
                    continue;
                if(i < static_cast<int>(pFrame->mvuRight.size()) &&
                   pFrame->mvuRight[i] >= 0)
                    continue;

                local_point_outliers[i] = false;
                const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;
                Kb8PointPoseEdge* e = new Kb8PointPoseEdge();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                    optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 =
                    kpUn.octave >= 0 &&
                    kpUn.octave <
                        static_cast<int>(pFrame->mvInvLevelSigma2.size())
                        ? pFrame->mvInvLevelSigma2[kpUn.octave]
                        : 1.0f;
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                g2o::RobustKernelHuber* rk =
                    new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);
                e->pCamera = &camera;
                const cv::Mat Xw = pMP->GetWorldPos();
                e->Xw = Eigen::Vector3d(Xw.at<float>(0),
                                        Xw.at<float>(1),
                                        Xw.at<float>(2));
                optimizer.addEdge(e);
                vpEdgesKb8Mono.push_back(e);
                vnIndexEdgeKb8Mono.push_back(static_cast<size_t>(i));
            }
        }

        result.point_edge_count = static_cast<int>(vpEdgesKb8Mono.size());
        if(result.point_edge_count < 3)
        {
            result.total_edge_count = result.point_edge_count;
            result.initial_point_cost =
                EvaluateKb8PointPoseEdges(vpEdgesKb8Mono).cost;
            result.final_point_cost = result.initial_point_cost;
            FillPoseDiagnostics(point_only_pose, point_only_pose, &result);
            result.status = "NOT_ENOUGH_POINT_CORRESPONDENCES";
            pFrame->mvbOutlier = point_outlier_before;
            return result;
        }

        result.physical_observation_count =
            static_cast<int>(pFrame->mvPhysicalLineObservations.size());
        result.physical_match_count =
            pFrame->CountMatchedPhysicalLineObservations();
        pFrame->ResetPhysicalLineResidualState();
        if(pFrame->mvbPhysicalLineOutlier.size() <
           pFrame->mvPhysicalLineObservations.size())
        {
            pFrame->mvbPhysicalLineOutlier.assign(
                pFrame->mvPhysicalLineObservations.size(), false);
        }

        vector<PhysicalLinePlanePoseEdge*> vpEdgesLine;
        vector<size_t> vnIndexEdgeLine;
        vpEdgesLine.reserve(pFrame->mvPhysicalLineObservations.size());
        vnIndexEdgeLine.reserve(pFrame->mvPhysicalLineObservations.size());
        std::set<MapLine*> residual_maplines;
        bool provenance_residuals_logged = false;
        auto log_physical_line_residuals =
            [&](const std::string& forced_reject_reason) {
                if(provenance_residuals_logged)
                    return;
                provenance_residuals_logged = true;
                for(size_t edge_i = 0; edge_i < vnIndexEdgeLine.size(); ++edge_i)
                {
                    const size_t obs_idx = vnIndexEdgeLine[edge_i];
                    if(obs_idx >= pFrame->mvpPhysicalMapLines.size())
                        continue;
                    MapLine* pML = pFrame->mvpPhysicalMapLines[obs_idx];
                    if(!pML)
                        continue;
                    if(!forced_reject_reason.empty())
                    {
                        pML->MarkOptimizerResidualRejected(
                            static_cast<long>(pFrame->mnId),
                            static_cast<long>(obs_idx),
                            forced_reject_reason);
                        continue;
                    }
                    const bool rejected =
                        obs_idx < pFrame->mvbPhysicalLineOutlier.size() &&
                        pFrame->mvbPhysicalLineOutlier[obs_idx];
                    if(rejected)
                    {
                        pML->MarkOptimizerResidualRejected(
                            static_cast<long>(pFrame->mnId),
                            static_cast<long>(obs_idx),
                            "CHI2_OR_INVALID");
                    }
                    else
                    {
                        pML->MarkOptimizerResidualAccepted(
                            static_cast<long>(pFrame->mnId),
                            static_cast<long>(obs_idx));
                    }
                }
            };

        const bool use_physical_lines =
            !pFrame->mvPhysicalLineObservations.empty() &&
            pFrame->mvpPhysicalMapLines.size() ==
                pFrame->mvPhysicalLineObservations.size();
        if(!use_physical_lines)
        {
            result.total_edge_count = result.point_edge_count;
            FillPoseDiagnostics(point_only_pose, point_only_pose, &result);
            result.status = "PHYSICAL_LINE_INPUT_MISMATCH";
            pFrame->mvbOutlier = point_outlier_before;
            return result;
        }

        for(size_t i = 0; i < pFrame->mvPhysicalLineObservations.size(); ++i)
        {
            MapLine* pML = pFrame->mvpPhysicalMapLines[i];
            if(!pML)
            {
                pFrame->mvbPhysicalLineOutlier[i] = true;
                continue;
            }
            if(pML->isBad())
            {
                pFrame->mvbPhysicalLineOutlier[i] = true;
                pFrame->mvpPhysicalMapLines[i] = static_cast<MapLine*>(NULL);
                continue;
            }
            if(residual_maplines.count(pML))
            {
                ++result.duplicate_physical_residual_violations;
                pFrame->mvbPhysicalLineOutlier[i] = true;
                continue;
            }

            PhysicalLineObservation& physical =
                pFrame->mvPhysicalLineObservations[i];
            if(!physical.plane_valid || physical.members.empty())
            {
                pFrame->mvbPhysicalLineOutlier[i] = true;
                continue;
            }
            const Vector6d world = pML->GetWorldPos();
            const Eigen::Vector3d endpoint0(world(0), world(1), world(2));
            const Eigen::Vector3d endpoint1(world(3), world(4), world(5));
            if(!endpoint0.allFinite() || !endpoint1.allFinite() ||
               (endpoint0 - endpoint1).norm() <= 1e-9)
            {
                pFrame->mvbPhysicalLineOutlier[i] = true;
                continue;
            }

            const Eigen::Vector3d direction = endpoint1 - endpoint0;
            const double direction_norm = direction.norm();
            if(direction_norm <= 1e-12 || !std::isfinite(direction_norm))
            {
                pFrame->mvbPhysicalLineOutlier[i] = true;
                continue;
            }

            PhysicalLinePlanePoseEdge* e =
                new PhysicalLinePlanePoseEdge();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                optimizer.vertex(0)));
            e->setMeasurement(Eigen::Vector3d::Zero());
            e->line_v_w = direction / direction_norm;
            e->line_n_w = endpoint0.cross(e->line_v_w);
            e->observed_plane_c = physical.physical_plane;
            e->setInformation(Eigen::Matrix3d::Identity());
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaLine);

            if(!e->isValid())
            {
                pFrame->mvbPhysicalLineOutlier[i] = true;
                delete e;
                continue;
            }

            optimizer.addEdge(e);
            vpEdgesLine.push_back(e);
            vnIndexEdgeLine.push_back(i);
            pFrame->mvPhysicalLineResidualCount[i] = 1;
            pML->MarkOptimizerResidualCreated(
                static_cast<long>(pFrame->mnId),
                static_cast<long>(i));
            result.max_residuals_per_physical_match =
                std::max(result.max_residuals_per_physical_match, 1);
            residual_maplines.insert(pML);
            pFrame->mvbPhysicalLineOutlier[i] = false;
        }

        result.line_initial = static_cast<int>(vpEdgesLine.size());
        result.line_edge_count = static_cast<int>(vpEdgesLine.size());
        result.total_edge_count =
            result.point_edge_count + result.line_edge_count;
        result.uses_single_shared_pose_vertex =
            AllPhysicalPlaneEdgesUsePoseVertex(optimizer.vertex(0),
                                               vpEdgesKb8Mono,
                                               vpEdgesLine);
        const EdgeCostSummary initial_point_summary =
            EvaluateKb8PointPoseEdges(vpEdgesKb8Mono);
        const EdgeCostSummary initial_line_summary =
            EvaluatePhysicalLinePlanePoseEdges(vpEdgesLine);
        result.initial_point_cost = initial_point_summary.cost;
        result.initial_line_cost = initial_line_summary.cost;
        result.initial_point_inliers = initial_point_summary.inliers;
        result.initial_line_inliers = initial_line_summary.inliers;

        if(result.line_initial < std::max(1, minValidLineCount))
        {
            result.final_point_cost = result.initial_point_cost;
            result.final_line_cost = result.initial_line_cost;
            result.final_point_inliers = result.initial_point_inliers;
            result.final_line_inliers = result.initial_line_inliers;
            FillPoseDiagnostics(point_only_pose, point_only_pose, &result);
            result.status = "NOT_ENOUGH_VALID_PHYSICAL_PLANE_LINES";
            log_physical_line_residuals(result.status);
            pFrame->mvbOutlier = point_outlier_before;
            return result;
        }

        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Line[4] = {5.991, 5.991, 5.991, 5.991};
        const int its[4] = {10, 10, 10, 10};
        int nBadPoints = 0;
        int nBadLines = 0;

        for(size_t it = 0; it < 4; ++it)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBadPoints = 0;
            for(size_t i = 0; i < vpEdgesKb8Mono.size(); ++i)
            {
                Kb8PointPoseEdge* e = vpEdgesKb8Mono[i];
                const size_t idx = vnIndexEdgeKb8Mono[i];
                if(local_point_outliers[idx])
                    e->computeError();
                const float chi2 = e->chi2();
                if(chi2 > chi2Mono[it] || !e->isDepthPositive())
                {
                    local_point_outliers[idx] = true;
                    e->setLevel(1);
                    ++nBadPoints;
                }
                else
                {
                    local_point_outliers[idx] = false;
                    e->setLevel(0);
                }
                if(it == 2)
                    e->setRobustKernel(0);
            }

            nBadLines = 0;
            for(size_t i = 0; i < vpEdgesLine.size(); ++i)
            {
                PhysicalLinePlanePoseEdge* e = vpEdgesLine[i];
                const size_t idx = vnIndexEdgeLine[i];
                if(pFrame->mvbPhysicalLineOutlier[idx])
                    e->computeError();
                const float chi2 = e->chi2();
                if(chi2 > chi2Line[it] || !e->isValid())
                {
                    pFrame->mvbPhysicalLineOutlier[idx] = true;
                    e->setLevel(1);
                    ++nBadLines;
                }
                else
                {
                    pFrame->mvbPhysicalLineOutlier[idx] = false;
                    e->setLevel(0);
                }
                if(it == 2)
                    e->setRobustKernel(0);
            }
        }

        result.point_inliers = result.point_edge_count - nBadPoints;
        result.line_bad = nBadLines;
        result.line_inliers = result.line_initial - nBadLines;
        const EdgeCostSummary final_point_summary =
            EvaluateKb8PointPoseEdges(vpEdgesKb8Mono);
        const EdgeCostSummary final_line_summary =
            EvaluatePhysicalLinePlanePoseEdges(vpEdgesLine);
        result.final_point_cost = final_point_summary.cost;
        result.final_line_cost = final_line_summary.cost;
        result.final_point_inliers = final_point_summary.inliers;
        result.final_line_inliers = final_line_summary.inliers;

        if(result.line_inliers < std::max(1, minValidLineCount))
        {
            FillPoseDiagnostics(point_only_pose, point_only_pose, &result);
            result.status = "ALL_OR_TOO_MANY_PHYSICAL_PLANE_LINE_OUTLIERS";
            log_physical_line_residuals("");
            pFrame->mvbOutlier = point_outlier_before;
            pFrame->SetPose(point_only_pose);
            return result;
        }

        g2o::VertexSE3Expmap* vSE3_recov =
            static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        const cv::Mat pose = Converter::toCvMat(vSE3_recov->estimate());
        FillPoseDiagnostics(point_only_pose, pose, &result);
        if(pose.empty() || !cv::checkRange(pose))
        {
            result.status = "NON_FINITE_POSE";
            log_physical_line_residuals(result.status);
            pFrame->mvbOutlier = point_outlier_before;
            pFrame->SetPose(point_only_pose);
            return result;
        }
        if(!result.rotation_valid)
        {
            result.status = "INVALID_ROTATION";
            log_physical_line_residuals(result.status);
            pFrame->mvbOutlier = point_outlier_before;
            pFrame->SetPose(point_only_pose);
            return result;
        }

        log_physical_line_residuals("");
        pFrame->mvbOutlier = local_point_outliers;
        pFrame->SetPose(pose);
        result.success = true;
        result.wrote_pose = true;
        result.status = "OK";
        return result;
}

Optimizer::NativeFisheyePoseOptimizationResult
Optimizer::PoseOptimizationWithKb8PointLine(
    Frame *pFrame, const int minValidLineCount)
{
        NativeFisheyePoseOptimizationResult result;
        result.status = "NOT_ATTEMPTED";
        if(!pFrame)
        {
            result.status = "NULL_FRAME";
            return result;
        }
        if(!pFrame->HasKb8PointObservations())
        {
            result.status = "NOT_KB8_FRAME";
            return result;
        }

        result.attempted = true;
        const cv::Mat point_only_pose = pFrame->mTcw.clone();
        const std::vector<bool> point_outlier_before = pFrame->mvbOutlier;
        std::vector<bool> line_outliers = pFrame->mvbFisheyeLineOutlier;
        if(line_outliers.size() < pFrame->mvpFisheyeMapLines.size())
            line_outliers.resize(pFrame->mvpFisheyeMapLines.size(), false);

        KannalaBrandt8Camera camera(pFrame->mKb8PointCameraParameters);
        if(!camera.valid())
        {
            result.status = "INVALID_KB8_CAMERA";
            return result;
        }

        FisheyeFrameOptimizationSnapshot snapshot;
        snapshot.frame_id = static_cast<long>(pFrame->mnId);
        snapshot.timestamp = pFrame->mTimeStamp;
        snapshot.initial_pose_Tcw.Rcw =
            Converter::toMatrix3d(point_only_pose.rowRange(0, 3).colRange(0, 3));
        snapshot.initial_pose_Tcw.tcw =
            Eigen::Vector3d(point_only_pose.at<float>(0, 3),
                            point_only_pose.at<float>(1, 3),
                            point_only_pose.at<float>(2, 3));
        snapshot.point_pose_Tcw = snapshot.initial_pose_Tcw;

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);
            const int N = pFrame->N;
            for(int i = 0; i < N; ++i)
            {
                MapPoint* pMP = pFrame->mvpMapPoints[i];
                if(!pMP || pMP->isBad() || i >= static_cast<int>(pFrame->mvKeysUn.size()))
                    continue;
                if(i < static_cast<int>(pFrame->mvuRight.size()) &&
                   pFrame->mvuRight[i] >= 0)
                    continue;

                const cv::KeyPoint& kp = pFrame->mvKeysUn[i];
                PointObservationSnapshot obs;
                obs.point_id = static_cast<long>(pMP->mnId);
                obs.feature_index = i;
                const cv::Mat Xw = pMP->GetWorldPos();
                obs.world_point = Eigen::Vector3d(Xw.at<float>(0),
                                                  Xw.at<float>(1),
                                                  Xw.at<float>(2));
                obs.observed_pixel = Eigen::Vector2d(kp.pt.x, kp.pt.y);
                obs.octave = kp.octave;
                if(kp.octave >= 0 &&
                   kp.octave < static_cast<int>(pFrame->mvInvLevelSigma2.size()))
                    obs.inverse_sigma2 = pFrame->mvInvLevelSigma2[kp.octave];
                snapshot.points.push_back(obs);
            }
        }

        result.point_edge_count = static_cast<int>(snapshot.points.size());
        result.pose_vertex_count = 1;
        result.uses_single_shared_pose_vertex = true;
        if(snapshot.points.size() < 3)
        {
            result.status = "NOT_ENOUGH_POINT_CORRESPONDENCES";
            pFrame->mvbOutlier = point_outlier_before;
            return result;
        }

        EndpointPluckerConverter converter;
        int line_sample_edges = 0;
        result.physical_observation_count =
            static_cast<int>(pFrame->mvPhysicalLineObservations.size());
        result.physical_match_count =
            pFrame->CountMatchedPhysicalLineObservations();
        pFrame->ResetPhysicalLineResidualState();

        const bool use_physical_lines =
            !pFrame->mvPhysicalLineObservations.empty() &&
            pFrame->mvpPhysicalMapLines.size() ==
                pFrame->mvPhysicalLineObservations.size();

        if(use_physical_lines)
        {
            std::set<MapLine*> residual_maplines;
            for(size_t i = 0; i < pFrame->mvPhysicalLineObservations.size();
                ++i)
            {
                MapLine* pML = pFrame->mvpPhysicalMapLines[i];
                if(!pML)
                {
                    pFrame->mvbPhysicalLineOutlier[i] = true;
                    continue;
                }
                if(pML->isBad())
                {
                    pFrame->mvbPhysicalLineOutlier[i] = true;
                    pFrame->mvpPhysicalMapLines[i] =
                        static_cast<MapLine*>(NULL);
                    continue;
                }
                if(residual_maplines.count(pML))
                {
                    ++result.duplicate_physical_residual_violations;
                    pFrame->mvbPhysicalLineOutlier[i] = true;
                    continue;
                }

                const PhysicalLineObservation& physical =
                    pFrame->mvPhysicalLineObservations[i];
                if(!physical.plane_valid || physical.members.empty())
                {
                    pFrame->mvbPhysicalLineOutlier[i] = true;
                    continue;
                }

                const Vector6d world = pML->GetWorldPos();
                PluckerLineNV line_w;
                std::string reject_reason;
                if(!converter.convert(
                       Eigen::Vector3d(world(0), world(1), world(2)),
                       Eigen::Vector3d(world(3), world(4), world(5)),
                       line_w, reject_reason))
                {
                    pFrame->mvbPhysicalLineOutlier[i] = true;
                    continue;
                }

                const PhysicalLineObservationMember& member =
                    physical.members.front();
                const cv::Point2f midpoint(
                    0.5f * (member.p_start.x + member.p_end.x),
                    0.5f * (member.p_start.y + member.p_end.y));
                Eigen::Vector3d bearing;
                if(!camera.unproject(Eigen::Vector2d(midpoint.x,
                                                     midpoint.y),
                                     bearing))
                {
                    pFrame->mvbPhysicalLineOutlier[i] = true;
                    continue;
                }

                LineObservationSnapshot line;
                line.line_id = static_cast<long>(pML->mnId);
                line.view_id = member.view_id;
                line.image_line_id = physical.physical_line_observation_id;
                line.association_occurrence = static_cast<int>(i);
                line.line_w = line_w;
                line.line_weight = 1.0;
                line.bearing_samples.push_back(bearing);
                line.source_sample_indices.push_back(0);
                line.optimizer_sample_ordinals.push_back(0);
                snapshot.lines.push_back(line);
                pFrame->mvPhysicalLineResidualCount[i] = 1;
                result.max_residuals_per_physical_match =
                    std::max(result.max_residuals_per_physical_match, 1);
                residual_maplines.insert(pML);
                pFrame->mvbPhysicalLineOutlier[i] = false;
                ++line_sample_edges;
            }
        }
        else
        {
            const size_t nNative =
                std::min(pFrame->mvpFisheyeMapLines.size(),
                         pFrame->mvFisheyeLineObservations.size());
            for(size_t i = 0; i < nNative; ++i)
            {
                MapLine* pML = pFrame->mvpFisheyeMapLines[i];
                if(!pML)
                {
                    line_outliers[i] = true;
                    continue;
                }
                if(pML->isBad())
                {
                    line_outliers[i] = true;
                    pFrame->mvpFisheyeMapLines[i] = static_cast<MapLine*>(NULL);
                    continue;
                }

                const FisheyeLineObservation& native_obs =
                    pFrame->mvFisheyeLineObservations[i];
                const Vector6d world = pML->GetWorldPos();
                PluckerLineNV line_w;
                std::string reject_reason;
                if(!converter.convert(Eigen::Vector3d(world(0), world(1), world(2)),
                                      Eigen::Vector3d(world(3), world(4), world(5)),
                                      line_w, reject_reason))
                {
                    line_outliers[i] = true;
                    continue;
                }

                LineObservationSnapshot line;
                line.line_id = static_cast<long>(pML->mnId);
                line.view_id = native_obs.view_id;
                line.image_line_id = native_obs.native_line_id >= 0
                                         ? native_obs.native_line_id
                                         : static_cast<int>(i);
                line.association_occurrence = static_cast<int>(i);
                line.line_w = line_w;
                line.line_weight =
                    std::isfinite(native_obs.information) &&
                    native_obs.information > 0.0 ? native_obs.information : 1.0;

                for(size_t s = 0; s < native_obs.fisheye_samples.size(); ++s)
                {
                    if(s < native_obs.sample_valid.size() &&
                       !native_obs.sample_valid[s])
                        continue;
                    const cv::Point2f& px = native_obs.fisheye_samples[s];
                    Eigen::Vector3d bearing;
                    if(camera.unproject(Eigen::Vector2d(px.x, px.y), bearing))
                    {
                        line.bearing_samples.push_back(bearing);
                        line.source_sample_indices.push_back(static_cast<int>(s));
                        line.optimizer_sample_ordinals.push_back(
                            static_cast<int>(line.bearing_samples.size()) - 1);
                    }
                }
                if(line.bearing_samples.empty())
                {
                    const cv::Point2f endpoints[2] = {
                        native_obs.fisheye_start, native_obs.fisheye_end};
                    for(int s = 0; s < 2; ++s)
                    {
                        Eigen::Vector3d bearing;
                        if(camera.unproject(Eigen::Vector2d(endpoints[s].x,
                                                            endpoints[s].y),
                                            bearing))
                        {
                            line.bearing_samples.push_back(bearing);
                            line.source_sample_indices.push_back(s);
                            line.optimizer_sample_ordinals.push_back(s);
                        }
                    }
                }
                if(line.bearing_samples.size() < 2)
                {
                    line_outliers[i] = true;
                    continue;
                }
                line_sample_edges += static_cast<int>(line.bearing_samples.size());
                snapshot.lines.push_back(line);
                line_outliers[i] = false;
            }
        }

        result.line_initial = static_cast<int>(snapshot.lines.size());
        result.line_edge_count = line_sample_edges;
        result.total_edge_count = result.point_edge_count + result.line_edge_count;
        if(static_cast<int>(snapshot.lines.size()) < std::max(1, minValidLineCount))
        {
            result.status = "NOT_ENOUGH_VALID_NATIVE_LINES";
            pFrame->mvbOutlier = point_outlier_before;
            pFrame->mvbFisheyeLineOutlier = line_outliers;
            return result;
        }

        FisheyeJointSafetyConfig config;
        config.min_valid_lines = std::max(1, minValidLineCount);
        FisheyeJointPoseOptimizer optimizer(config);
        const FisheyeJointOptimizationResult strict_result =
            optimizer.run(snapshot, camera);

        result.initial_point_cost = strict_result.point_at_point.chi2;
        result.final_point_cost = strict_result.point_at_joint.chi2;
        result.initial_line_cost = strict_result.line_at_point.chi2;
        result.final_line_cost = strict_result.line_at_joint.chi2;
        result.initial_point_inliers = strict_result.point_at_point.inliers;
        result.final_point_inliers = strict_result.point_at_joint.inliers;
        result.initial_line_inliers = strict_result.line_at_point.inliers;
        result.final_line_inliers = strict_result.line_at_joint.inliers;
        result.point_inliers = strict_result.point_at_joint.inliers;
        result.line_inliers = strict_result.line_at_joint.inliers;
        result.line_bad = std::max(0, result.line_edge_count - result.line_inliers);
        result.pose_delta_translation =
            strict_result.safety_metrics.translation_delta_m;
        result.pose_delta_rotation_deg =
            strict_result.safety_metrics.rotation_delta_deg;
        result.pose_finite = strict_result.safety_metrics.finite;
        result.rotation_valid = strict_result.safety_metrics.finite;

        Eigen::Matrix<double, 4, 4> Tcw = Eigen::Matrix<double, 4, 4>::Identity();
        Tcw.block<3, 3>(0, 0) = strict_result.safe_output_pose.Rcw;
        Tcw.block<3, 1>(0, 3) = strict_result.safe_output_pose.tcw;
        const cv::Mat safe_pose = Converter::toCvMat(Tcw);
        FillPoseDiagnostics(point_only_pose, safe_pose, &result);

        result.status = strict_result.joint_accepted
                            ? "OK"
                            : toString(strict_result.reject_reason);
        result.success = strict_result.joint_accepted;
        pFrame->mvbFisheyeLineOutlier = line_outliers;

        if(strict_result.joint_accepted)
        {
            pFrame->SetPose(safe_pose);
            result.wrote_pose = true;
        }
        else
        {
            pFrame->SetPose(point_only_pose);
            pFrame->mvbOutlier = point_outlier_before;
            result.wrote_pose = false;
        }
        return result;
}


/**
 * @brief Local BA
 *
 * 1.Vertex:
 *      - g2o::VertexSE3Expmap，LocalKeyFrames，即当前关键帧的位姿，与当前关键帧相连的关键帧的位姿
 *      - g2o::VertexSE3Expmap，FixedCameras，即能观测到LocalMapPoints的关键帧（并且不属于LocalMappingKeyFrames）的位姿，在优化中这些关键帧的位姿不变
 *      - g2o：：VertexSBAPointXYZ()，LocalMapPoints, 即LocalKeyFrames能观测到的所有MapPoints的位置
 * 2.Edge:
 *      -g2o::EdgeSE3ProjectXYZ(), BaseBinaryEdge
 *          + Vertex：关键帧的Tcw，MapPoint的Pw
 *          + measurement:MapPoint在关键帧中的二维位置(u,v)
 *          + InforMatrix：invSigma2,与特征点所在的尺度有关
 *      -g2o::EdgeStereoSE3ProjectXYZ(), BaseBinaryEdge
 *          + Vertex：关键帧的Tcw, MapPoint的Pw
 *          + measurement: MapPoint在关键帧中的二维位置（ul,v,ur）
 *          + InfoMatrix： invSigma2，与特征点所在的尺度有关
 *
 * @param pKF   KeyFrame
 * @param pbStopFlag    是否停止优化的标识
 * @param pMap          在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{
    LocalBaRuntimeStats runtime_stats;
    runtime_stats.current_keyframe_id = pKF ? pKF->mnId : 0;

    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    // step1：将当前关键帧加入到lLocalKeyFrames
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    // step2：找到关键帧连接的关键帧（一级相连），加入lLocalKeyFrames中
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;   //Q:为什么是pKF的ID
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    // step3：遍历lLocalKeyFrames中关键帧，将它们观测的MapPoints加入到lLocalMapPoints
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // step4：得到能被局部MapPoints观测到，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时不优化
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // step5：构造g2o优化器
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    // step6：添加顶点，Pose of Local KeyFrame
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // step7：添加顶点，Pose of Fixed KeyFrame，注意这里调用了vSE3->setFixed(true)
    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }
    runtime_stats.local_keyframes =
        static_cast<int>(lLocalKeyFrames.size());
    runtime_stats.fixed_keyframes =
        static_cast<int>(lFixedCameras.size());
    runtime_stats.local_mappoints =
        static_cast<int>(lLocalMapPoints.size());
    runtime_stats.pose_vertices =
        runtime_stats.local_keyframes + runtime_stats.fixed_keyframes;

    // Set MapPoint vertices
    // step7：添加3D顶点
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<Kb8PointBundleEdge*> vpEdgesKb8Mono;
    vpEdgesKb8Mono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFKb8Mono;
    vpEdgeKFKb8Mono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeKb8Mono;
    vpMapPointEdgeKb8Mono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);
    map<KeyFrame*, std::unique_ptr<KannalaBrandt8Camera> > kb8Cameras;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        ++runtime_stats.point_vertices;

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    if(pKFi->mbUseKb8PointObservations)
                    {
                        if(kb8Cameras.find(pKFi) == kb8Cameras.end())
                        {
                            kb8Cameras[pKFi].reset(
                                new KannalaBrandt8Camera(
                                    pKFi->mKb8PointCameraParameters));
                        }
                        Kb8PointBundleEdge* e = new Kb8PointBundleEdge();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->pCamera = kb8Cameras[pKFi].get();

                        optimizer.addEdge(e);
                        vpEdgesKb8Mono.push_back(e);
                        vpEdgeKFKb8Mono.push_back(pKFi);
                        vpMapPointEdgeKb8Mono.push_back(pMP);
                        ++runtime_stats.kb8_point_edges;
                    }
                    else
                    {
                        g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                        ++runtime_stats.legacy_pinhole_point_edges;
                    }
                    ++runtime_stats.point_edges_with_robust_kernel;
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    ++runtime_stats.stereo_point_edges;
                    ++runtime_stats.point_edges_with_robust_kernel;
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
        {
            runtime_stats.stop_flag_triggered = 1;
            AppendLocalBaRuntimeStats(runtime_stats);
            return;
        }

    const EdgeCostSummary initial_kb8 =
        EvaluateKb8PointBundleEdges(vpEdgesKb8Mono);
    runtime_stats.point_cost_before = initial_kb8.cost;
    runtime_stats.point_inliers_before = initial_kb8.inliers;
    runtime_stats.point_outliers_before =
        runtime_stats.kb8_point_edges - initial_kb8.inliers;
    optimizer.initializeOptimization();
    optimizer.optimize(5);
    runtime_stats.optimization_round1_ran = 1;

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
        {
            bDoMore = false;
            runtime_stats.stop_flag_triggered = 1;
        }

    if(bDoMore)
    {

        // Check inlier observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesKb8Mono.size(); i<iend;i++)
        {
            Kb8PointBundleEdge* e = vpEdgesKb8Mono[i];
            MapPoint* pMP = vpMapPointEdgeKb8Mono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
            ++runtime_stats.point_edges_without_robust_kernel;
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }
        runtime_stats.point_edges_without_robust_kernel =
            runtime_stats.kb8_point_edges +
            runtime_stats.legacy_pinhole_point_edges +
            runtime_stats.stereo_point_edges;

        // Optimize again without the outliers

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
        runtime_stats.optimization_round2_ran = 1;

    }

    const EdgeCostSummary final_kb8 =
        EvaluateKb8PointBundleEdges(vpEdgesKb8Mono);
    runtime_stats.point_cost_after = final_kb8.cost;
    runtime_stats.point_inliers_after = final_kb8.inliers;
    runtime_stats.point_outliers_after =
        runtime_stats.kb8_point_edges - final_kb8.inliers;
    runtime_stats.optimizer_success =
        runtime_stats.kb8_point_edges == 0 ||
        runtime_stats.point_cost_after <= runtime_stats.point_cost_before ||
        runtime_stats.point_inliers_after > 0;

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesKb8Mono.size()+vpEdgesStereo.size());

    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            AppendLocalBaObservationErasure(
                pKFi->mnId, pMP->mnId, e->chi2(),
                e->isDepthPositive(), "legacy_pinhole_mono");
        }
    }

    for(size_t i=0, iend=vpEdgesKb8Mono.size(); i<iend;i++)
    {
        Kb8PointBundleEdge* e = vpEdgesKb8Mono[i];
        MapPoint* pMP = vpMapPointEdgeKb8Mono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFKb8Mono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            AppendLocalBaObservationErasure(
                pKFi->mnId, pMP->mnId, e->chi2(),
                e->isDepthPositive(), "kb8_mono");
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            AppendLocalBaObservationErasure(
                pKFi->mnId, pMP->mnId, e->chi2(),
                e->isDepthPositive(), "stereo");
        }
    }
    runtime_stats.observations_erased = static_cast<int>(vToErase.size());

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
        ++runtime_stats.keyframe_poses_updated;
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
        ++runtime_stats.mappoint_positions_updated;
    }

    AppendLocalBaRuntimeStats(runtime_stats);
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();
    int essential_graph_vertex_count = 0;
    int essential_graph_edge_count = 0;
    int essential_graph_edge_id = 0;
    int essential_graph_pose_writebacks = 0;
    int essential_graph_mappoint_writebacks = 0;
    LogEssentialGraphRuntimeEvent(
        "OPTIMIZE_ESSENTIAL_GRAPH_BEGIN", 0, 0, "BEGIN");

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);
        ++essential_graph_vertex_count;
        LogEssentialGraphStructureVertex(nIDi);

        vpVertices[nIDi]=VSim3;
    }
    LogEssentialGraphRuntimeEvent(
        "CREATE_SIM3_VERTICES", essential_graph_vertex_count, 0, "PASS");


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);
            ++essential_graph_edge_count;
            LogEssentialGraphStructureEdge(
                ++essential_graph_edge_id, nIDi, nIDj);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
            ++essential_graph_edge_count;
            LogEssentialGraphStructureEdge(
                ++essential_graph_edge_id, nIDi, nIDj);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
                ++essential_graph_edge_count;
                LogEssentialGraphStructureEdge(
                    ++essential_graph_edge_id, nIDi, pLKF->mnId);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                    ++essential_graph_edge_count;
                    LogEssentialGraphStructureEdge(
                        ++essential_graph_edge_id, nIDi, pKFn->mnId);
                }
            }
        }
    }
    LogEssentialGraphRuntimeEvent(
        "CREATE_SIM3_EDGES", essential_graph_vertex_count,
        essential_graph_edge_count, "PASS");

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    const double before_error = optimizer.activeChi2();
    const int optimized_iterations = optimizer.optimize(20);
    optimizer.computeActiveErrors();
    const double after_error = optimizer.activeChi2();
    LogEssentialGraphOptimization(
        before_error, after_error, 20,
        optimized_iterations >= 0 ? "PASS" : "FAIL");
    LogEssentialGraphRuntimeEvent(
        "OPTIMIZATION_DONE", essential_graph_vertex_count,
        essential_graph_edge_count,
        optimized_iterations >= 0 ? "PASS" : "FAIL");

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
        ++essential_graph_pose_writebacks;
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
        ++essential_graph_mappoint_writebacks;
    }
    LogEssentialGraphRuntimeEvent(
        "WRITEBACK_DONE", essential_graph_pose_writebacks,
        essential_graph_mappoint_writebacks, "PASS");
}




int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    // Capability: KB8_OPTIMIZE_SIM3_POINT_EDGE_READY.
    OptimizeSim3RuntimeStats runtime_stats;
    runtime_stats.keyframe1_id = pKF1 ? pKF1->mnId : 0;
    runtime_stats.keyframe2_id = pKF2 ? pKF2->mnId : 0;
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);
    const bool useKb8ForwardEdge = pKF1->mbUseKb8PointObservations;
    const bool useKb8InverseEdge = pKF2->mbUseKb8PointObservations;
    std::unique_ptr<KannalaBrandt8Camera> kb8Camera1;
    std::unique_ptr<KannalaBrandt8Camera> kb8Camera2;
    if(useKb8ForwardEdge)
        kb8Camera1.reset(new KannalaBrandt8Camera(
            pKF1->mKb8PointCameraParameters));
    if(useKb8InverseEdge)
        kb8Camera2.reset(new KannalaBrandt8Camera(
            pKF2->mKb8PointCameraParameters));

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::OptimizableGraph::Edge*> vpEdges12;
    vector<g2o::OptimizableGraph::Edge*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        g2o::OptimizableGraph::Edge* e12 = NULL;
        if(useKb8ForwardEdge)
        {
            Kb8Sim3ProjectXYZEdge* e = new Kb8Sim3ProjectXYZEdge();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs1);
            e->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);
            g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk1);
            rk1->setDelta(deltaHuber);
            e->pCamera = kb8Camera1.get();
            optimizer.addEdge(e);
            e12 = e;
            ++runtime_stats.kb8_forward_edges;
        }
        else
        {
            g2o::EdgeSim3ProjectXYZ* e = new g2o::EdgeSim3ProjectXYZ();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs1);
            e->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

            g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk1);
            rk1->setDelta(deltaHuber);
            optimizer.addEdge(e);
            e12 = e;
            ++runtime_stats.legacy_forward_edges;
        }

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        g2o::OptimizableGraph::Edge* e21 = NULL;
        if(useKb8InverseEdge)
        {
            Kb8InverseSim3ProjectXYZEdge* e =
                new Kb8InverseSim3ProjectXYZEdge();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs2);
            e->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

            g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk2);
            rk2->setDelta(deltaHuber);
            e->pCamera = kb8Camera2.get();
            optimizer.addEdge(e);
            e21 = e;
            ++runtime_stats.kb8_inverse_edges;
        }
        else
        {
            g2o::EdgeInverseSim3ProjectXYZ* e =
                new g2o::EdgeInverseSim3ProjectXYZ();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs2);
            e->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

            g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk2);
            rk2->setDelta(deltaHuber);
            optimizer.addEdge(e);
            e21 = e;
            ++runtime_stats.legacy_inverse_edges;
        }

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }
    runtime_stats.correspondences = nCorrespondences;
    runtime_stats.bidirectional_cost_before =
        EdgeCost(vpEdges12) + EdgeCost(vpEdges21);

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);
    runtime_stats.optimization_round1_ran = 1;

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::OptimizableGraph::Edge* e12 = vpEdges12[i];
        g2o::OptimizableGraph::Edge* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=NULL;
            vpEdges21[i]=NULL;
            nBad++;
        }
    }
    runtime_stats.bad_after_round1 = nBad;

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
    {
        runtime_stats.bidirectional_cost_after =
            EdgeCost(vpEdges12) + EdgeCost(vpEdges21);
        runtime_stats.final_inliers = 0;
        AppendOptimizeSim3RuntimeStats(runtime_stats);
        return 0;
    }

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);
    runtime_stats.optimization_round2_ran = 1;

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::OptimizableGraph::Edge* e12 = vpEdges12[i];
        g2o::OptimizableGraph::Edge* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }
    runtime_stats.final_inliers = nIn;
    runtime_stats.bidirectional_cost_after =
        EdgeCost(vpEdges12) + EdgeCost(vpEdges21);
    runtime_stats.optimizer_success = nIn > 0 ? 1 : 0;

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();
    AppendOptimizeSim3RuntimeStats(runtime_stats);

    return nIn;
}


} //namespace StructureSLAM
