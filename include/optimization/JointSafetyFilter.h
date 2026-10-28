#ifndef STRUCTURE_SLAM_JOINT_SAFETY_FILTER_H
#define STRUCTURE_SLAM_JOINT_SAFETY_FILTER_H

#include "optimization/FisheyeJointSafetyConfig.h"

#include <string>
#include <vector>

namespace StructureSLAM {

enum class JointRejectReason {
    None,
    InsufficientLines,
    InvalidLineHessian,
    JointRankDeficient,
    SingleLineGradientDominated,
    ProbePointChi2Increase,
    ProbePointInlierLoss,
    ProbePointP95Increase,
    ProbeInvalidProjection,
    JointObjectiveNotReduced,
    PostPointChi2Increase,
    PostPointInlierLoss,
    PostPointP95Increase,
    LineObjectiveNotImproved,
    ExcessivePoseUpdate,
    NumericalFailure
};

std::string toString(JointRejectReason reason);

struct JointSafetyMetrics {
    int valid_lines = 0;
    int joint_rank = 0;
    double max_single_line_gradient_fraction = 0.0;
    double probe_point_chi2_relative_increase = 0.0;
    int probe_point_inlier_loss = 0;
    double probe_point_p95_relative_increase = 0.0;
    int probe_invalid_projection = 0;
    double joint_objective_drop = 0.0;
    double post_point_chi2_relative_increase = 0.0;
    int post_point_inlier_loss = 0;
    double post_point_p95_relative_increase = 0.0;
    double line_chi2_relative_change = 0.0;
    double translation_delta_m = 0.0;
    double rotation_delta_deg = 0.0;
    bool finite = true;
};

struct JointSafetyDecision {
    bool pre_pass = false;
    bool probe_pass = false;
    bool post_pass = false;
    bool accepted = false;
    JointRejectReason primary_reason = JointRejectReason::None;
    std::vector<JointRejectReason> failed_reasons;
};

class JointSafetyFilter {
public:
    explicit JointSafetyFilter(const FisheyeJointSafetyConfig& config);
    JointSafetyDecision evaluate(const JointSafetyMetrics& m) const;

private:
    FisheyeJointSafetyConfig config_;
};

}  // namespace StructureSLAM

#endif
