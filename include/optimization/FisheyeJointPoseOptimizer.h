#ifndef STRUCTURE_SLAM_FISHEYE_JOINT_POSE_OPTIMIZER_H
#define STRUCTURE_SLAM_FISHEYE_JOINT_POSE_OPTIMIZER_H

#include "camera/CentralCameraModel.h"
#include "optimization/FisheyeBearingPlaneLineEdge.h"
#include "optimization/FisheyeJointSafetyConfig.h"
#include "optimization/JointSafetyFilter.h"
#include "optimization/OfflineFrameOptimizationTypes.h"

#include <Eigen/Core>

#include <map>
#include <string>
#include <vector>

namespace StructureSLAM {

struct PointObservationSnapshot {
    long point_id = -1;
    int feature_index = -1;
    Eigen::Vector3d world_point = Eigen::Vector3d::Zero();
    Eigen::Vector2d observed_pixel = Eigen::Vector2d::Zero();
    int octave = -1;
    double inverse_sigma2 = 1.0;
};

struct LineObservationSnapshot {
    long line_id = -1;
    int view_id = -1;
    int image_line_id = -1;
    int association_occurrence = -1;
    PluckerLineNV line_w;
    std::vector<Eigen::Vector3d> bearing_samples;
    std::vector<int> source_sample_indices;
    std::vector<int> optimizer_sample_ordinals;
    double line_weight = 1.0;
};

struct FisheyeFrameOptimizationSnapshot {
    long frame_id = -1;
    double timestamp = 0.0;
    OfflinePoseTcw initial_pose_Tcw;
    OfflinePoseTcw point_pose_Tcw;
    std::vector<PointObservationSnapshot> points;
    std::vector<LineObservationSnapshot> lines;
};

struct CommonSetEvaluation {
    int count = 0;
    int inliers = 0;
    int invalid = 0;
    double raw = 0.0;
    double chi2 = 0.0;
    double robust = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double max = 0.0;
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    std::map<std::pair<long, int>, Eigen::Matrix<double, 6, 1> > per_line_g;
    bool finite = true;
};

struct HessianSummary {
    int rank = 0;
    double condition = 0.0;
    Eigen::Matrix<double, 6, 1> eigenvalues =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> weakest_eigenvector =
        Eigen::Matrix<double, 6, 1>::Zero();
};

struct FisheyeJointRuntimeMetrics {
    double point_common_eval_ms = 0.0;
    double line_precheck_ms = 0.0;
    double probe_solve_ms = 0.0;
    double probe_eval_ms = 0.0;
    double joint_graph_build_ms = 0.0;
    double joint_optimize_ms = 0.0;
    double post_eval_ms = 0.0;
    double core_total_ms = 0.0;
};

struct FisheyeJointRoundTrace {
    int round = -1;
    int iteration_count = 0;
    bool robust_kernel_enabled = true;
    int active_point_count = 0;
    int active_line_count = 0;
    int active_line_sample_count = 0;
    OfflinePoseTcw pose;
    double point_cost = 0.0;
    double line_cost = 0.0;
    double joint_cost = 0.0;
    std::string solver_status = "OK";
};

struct FisheyeJointOptimizationResult {
    OfflinePoseTcw point_pose;
    OfflinePoseTcw probe_pose;
    OfflinePoseTcw joint_candidate_pose;
    OfflinePoseTcw safe_output_pose;
    bool pre_pass = false;
    bool probe_pass = false;
    bool post_pass = false;
    bool joint_attempted = false;
    bool joint_accepted = false;
    JointRejectReason reject_reason = JointRejectReason::None;
    JointSafetyMetrics safety_metrics;
    CommonSetEvaluation point_at_point;
    CommonSetEvaluation line_at_point;
    CommonSetEvaluation point_at_probe;
    CommonSetEvaluation point_at_joint;
    CommonSetEvaluation line_at_joint;
    HessianSummary joint_hessian_at_point;
    int joint_iterations = 0;
    std::string status = "OK";
    FisheyeJointRuntimeMetrics timing;
    std::vector<FisheyeJointRoundTrace> round_trace;
};

class FisheyeJointPoseOptimizer {
public:
    explicit FisheyeJointPoseOptimizer(const FisheyeJointSafetyConfig& config);

    FisheyeJointOptimizationResult run(
        const FisheyeFrameOptimizationSnapshot& snapshot,
        const CentralCameraModel& camera) const;

    CommonSetEvaluation evaluatePoints(
        const FisheyeFrameOptimizationSnapshot& snapshot,
        const OfflinePoseTcw& pose,
        const CentralCameraModel& camera) const;

    CommonSetEvaluation evaluateLines(
        const FisheyeFrameOptimizationSnapshot& snapshot,
        const OfflinePoseTcw& pose,
        double lambda) const;

    static HessianSummary summarizeHessian(
        const Eigen::Matrix<double, 6, 6>& H);
    static OfflinePoseTcw expLeft(
        const Eigen::Matrix<double, 6, 1>& dx,
        const OfflinePoseTcw& pose);
    static double rotationDifferenceDeg(
        const OfflinePoseTcw& a,
        const OfflinePoseTcw& b);

private:
    OfflinePoseTcw optimizeJoint(
        const FisheyeFrameOptimizationSnapshot& snapshot,
        const CentralCameraModel& camera,
        int* iterations,
        std::string* status,
        std::vector<FisheyeJointRoundTrace>* trace) const;

    Eigen::Matrix<double, 6, 1> solveDx(
        const Eigen::Matrix<double, 6, 6>& H,
        const Eigen::Matrix<double, 6, 1>& g,
        double damping_scale) const;

    FisheyeJointSafetyConfig config_;
    JointSafetyFilter filter_;
};

}  // namespace StructureSLAM

#endif  // STRUCTURE_SLAM_FISHEYE_JOINT_POSE_OPTIMIZER_H
