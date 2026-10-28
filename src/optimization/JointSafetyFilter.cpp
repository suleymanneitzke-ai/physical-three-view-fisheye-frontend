#include "optimization/JointSafetyFilter.h"

namespace StructureSLAM {

std::string toString(JointRejectReason r) {
    switch (r) {
        case JointRejectReason::None: return "None";
        case JointRejectReason::InsufficientLines: return "InsufficientLines";
        case JointRejectReason::InvalidLineHessian: return "InvalidLineHessian";
        case JointRejectReason::JointRankDeficient: return "JointRankDeficient";
        case JointRejectReason::SingleLineGradientDominated: return "SingleLineGradientDominated";
        case JointRejectReason::ProbePointChi2Increase: return "ProbePointChi2Increase";
        case JointRejectReason::ProbePointInlierLoss: return "ProbePointInlierLoss";
        case JointRejectReason::ProbePointP95Increase: return "ProbePointP95Increase";
        case JointRejectReason::ProbeInvalidProjection: return "ProbeInvalidProjection";
        case JointRejectReason::JointObjectiveNotReduced: return "JointObjectiveNotReduced";
        case JointRejectReason::PostPointChi2Increase: return "PostPointChi2Increase";
        case JointRejectReason::PostPointInlierLoss: return "PostPointInlierLoss";
        case JointRejectReason::PostPointP95Increase: return "PostPointP95Increase";
        case JointRejectReason::LineObjectiveNotImproved: return "LineObjectiveNotImproved";
        case JointRejectReason::ExcessivePoseUpdate: return "ExcessivePoseUpdate";
        case JointRejectReason::NumericalFailure: return "NumericalFailure";
    }
    return "Unknown";
}

JointSafetyFilter::JointSafetyFilter(const FisheyeJointSafetyConfig& config)
    : config_(config) {}

JointSafetyDecision JointSafetyFilter::evaluate(const JointSafetyMetrics& m) const {
    JointSafetyDecision d;
    if (!m.finite) d.failed_reasons.push_back(JointRejectReason::NumericalFailure);
    if (m.valid_lines < config_.min_valid_lines)
        d.failed_reasons.push_back(JointRejectReason::InsufficientLines);
    if (m.joint_rank < config_.required_joint_rank)
        d.failed_reasons.push_back(JointRejectReason::JointRankDeficient);
    if (m.max_single_line_gradient_fraction >= config_.max_single_line_gradient_ratio)
        d.failed_reasons.push_back(JointRejectReason::SingleLineGradientDominated);
    d.pre_pass = d.failed_reasons.empty();

    const size_t after_pre = d.failed_reasons.size();
    if (m.probe_point_chi2_relative_increase >
        config_.probe_max_point_chi2_relative_increase)
        d.failed_reasons.push_back(JointRejectReason::ProbePointChi2Increase);
    if (m.probe_point_inlier_loss > config_.probe_max_point_inlier_loss)
        d.failed_reasons.push_back(JointRejectReason::ProbePointInlierLoss);
    if (m.probe_point_p95_relative_increase >
        config_.probe_max_point_p95_relative_increase)
        d.failed_reasons.push_back(JointRejectReason::ProbePointP95Increase);
    if (m.probe_invalid_projection > config_.probe_max_new_invalid_projections)
        d.failed_reasons.push_back(JointRejectReason::ProbeInvalidProjection);
    d.probe_pass = d.pre_pass && d.failed_reasons.size() == after_pre;

    const size_t after_probe = d.failed_reasons.size();
    if (config_.post_requires_joint_objective_drop && m.joint_objective_drop <= 0.0)
        d.failed_reasons.push_back(JointRejectReason::JointObjectiveNotReduced);
    if (m.post_point_chi2_relative_increase >
        config_.post_max_point_chi2_relative_increase)
        d.failed_reasons.push_back(JointRejectReason::PostPointChi2Increase);
    if (m.post_point_inlier_loss > config_.post_max_point_inlier_loss)
        d.failed_reasons.push_back(JointRejectReason::PostPointInlierLoss);
    if (config_.protocol_version == SafetyProtocolVersion::P9StrictCpp &&
        m.post_point_p95_relative_increase >
            config_.post_max_point_p95_relative_increase)
        d.failed_reasons.push_back(JointRejectReason::PostPointP95Increase);
    if (config_.post_requires_line_chi2_decrease && m.line_chi2_relative_change >= 0.0)
        d.failed_reasons.push_back(JointRejectReason::LineObjectiveNotImproved);
    if (m.translation_delta_m > config_.max_translation_delta_m ||
        m.rotation_delta_deg > config_.max_rotation_delta_deg)
        d.failed_reasons.push_back(JointRejectReason::ExcessivePoseUpdate);
    d.post_pass = d.probe_pass && d.failed_reasons.size() == after_probe;
    d.accepted = d.pre_pass && d.probe_pass && d.post_pass;
    d.primary_reason = d.failed_reasons.empty() ? JointRejectReason::None
                                                : d.failed_reasons.front();
    return d;
}

}  // namespace StructureSLAM
