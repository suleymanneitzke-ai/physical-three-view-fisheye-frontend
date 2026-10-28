#ifndef STRUCTURE_SLAM_FISHEYE_JOINT_SAFETY_CONFIG_H
#define STRUCTURE_SLAM_FISHEYE_JOINT_SAFETY_CONFIG_H

#include <map>
#include <string>

namespace StructureSLAM {

enum class SafetyProtocolVersion {
    P8Historical,
    P9StrictCpp
};

struct FisheyeJointSafetyConfig {
    SafetyProtocolVersion protocol_version = SafetyProtocolVersion::P9StrictCpp;
    double line_lambda = 0.25;
    double probe_beta = 0.25;
    int min_valid_lines = 10;
    int required_joint_rank = 6;
    double max_single_line_gradient_ratio = 0.35;
    double probe_max_point_chi2_relative_increase = 0.01;
    int probe_max_point_inlier_loss = 1;
    double probe_max_point_p95_relative_increase = 0.02;
    int probe_max_new_invalid_projections = 0;
    double post_max_point_chi2_relative_increase = 0.01;
    int post_max_point_inlier_loss = 1;
    double post_max_point_p95_relative_increase = 0.02;
    bool post_requires_joint_objective_drop = true;
    bool post_requires_line_chi2_decrease = true;
    double max_translation_delta_m = 0.20;
    double max_rotation_delta_deg = 10.0;
    double point_degraded_safe_output_chi2_relative_threshold = 0.01;
    int point_degraded_safe_output_inlier_loss_threshold = 1;
};

struct FisheyeJointSafetyConfigLoadResult {
    bool ok = false;
    std::string error;
    std::string sha256;
    FisheyeJointSafetyConfig config;
    std::map<std::string, std::string> raw_values;
};

FisheyeJointSafetyConfigLoadResult loadFisheyeJointSafetyConfig(
    const std::string& yaml_path,
    const std::string& expected_sha256);

std::string sha256File(const std::string& path);

}  // namespace StructureSLAM

#endif
