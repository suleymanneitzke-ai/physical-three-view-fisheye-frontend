#include "optimization/FisheyeJointSafetyConfig.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

namespace StructureSLAM {
namespace {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

bool parseBool(const std::string& s) {
    return s == "true" || s == "1" || s == "True" || s == "TRUE";
}

double getDouble(const std::map<std::string, std::string>& m,
                 const std::string& key, bool& ok) {
    auto it = m.find(key);
    if (it == m.end()) {
        ok = false;
        return 0.0;
    }
    char* end = nullptr;
    const double v = std::strtod(it->second.c_str(), &end);
    if (!end || *end != '\0' || !std::isfinite(v)) ok = false;
    return v;
}

int getInt(const std::map<std::string, std::string>& m,
           const std::string& key, bool& ok) {
    return static_cast<int>(getDouble(m, key, ok));
}

}  // namespace

std::string sha256File(const std::string& path) {
    std::array<char, 256> buf{};
    std::string cmd = "sha256sum '" + path + "'";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    std::string out;
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) out += buf.data();
    pclose(pipe);
    std::istringstream iss(out);
    std::string hash;
    iss >> hash;
    return hash;
}

FisheyeJointSafetyConfigLoadResult loadFisheyeJointSafetyConfig(
    const std::string& yaml_path,
    const std::string& expected_sha256) {
    FisheyeJointSafetyConfigLoadResult res;
    res.sha256 = sha256File(yaml_path);
    if (res.sha256.empty()) {
        res.error = "failed_to_compute_sha256";
        return res;
    }
    if (!expected_sha256.empty() && res.sha256 != expected_sha256) {
        res.error = "sha256_mismatch";
        return res;
    }
    std::ifstream in(yaml_path);
    if (!in) {
        res.error = "cannot_open_config";
        return res;
    }
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const size_t pos = line.find(':');
        if (pos == std::string::npos) continue;
        res.raw_values[trim(line.substr(0, pos))] = trim(line.substr(pos + 1));
    }
    const std::set<std::string> required = {
        "lambda_line", "probe_beta", "min_valid_lines",
        "max_single_line_gradient_fraction", "joint_rank_required",
        "probe_point_chi2_relative_increase_max",
        "probe_point_post_inlier_loss_max",
        "probe_point_p95_relative_increase_max",
        "probe_invalid_projection_max",
        "post_point_chi2_relative_increase_max",
        "post_point_post_inlier_loss_max",
        "post_requires_joint_objective_drop",
        "post_requires_line_chi2_decrease",
        "post_translation_delta_max_m",
        "post_rotation_delta_max_deg",
        "point_degraded_safe_output_chi2_relative_threshold",
        "point_degraded_safe_output_inlier_loss_threshold"};
    for (const auto& key : required) {
        if (!res.raw_values.count(key)) {
            res.error = "missing_config_field:" + key;
            return res;
        }
    }
    bool ok = true;
    auto& c = res.config;
    c.line_lambda = getDouble(res.raw_values, "lambda_line", ok);
    c.probe_beta = getDouble(res.raw_values, "probe_beta", ok);
    c.min_valid_lines = getInt(res.raw_values, "min_valid_lines", ok);
    c.required_joint_rank = getInt(res.raw_values, "joint_rank_required", ok);
    c.max_single_line_gradient_ratio =
        getDouble(res.raw_values, "max_single_line_gradient_fraction", ok);
    c.probe_max_point_chi2_relative_increase =
        getDouble(res.raw_values, "probe_point_chi2_relative_increase_max", ok);
    c.probe_max_point_inlier_loss =
        getInt(res.raw_values, "probe_point_post_inlier_loss_max", ok);
    c.probe_max_point_p95_relative_increase =
        getDouble(res.raw_values, "probe_point_p95_relative_increase_max", ok);
    c.probe_max_new_invalid_projections =
        getInt(res.raw_values, "probe_invalid_projection_max", ok);
    c.post_max_point_chi2_relative_increase =
        getDouble(res.raw_values, "post_point_chi2_relative_increase_max", ok);
    c.post_max_point_inlier_loss =
        getInt(res.raw_values, "post_point_post_inlier_loss_max", ok);
    c.post_requires_joint_objective_drop =
        parseBool(res.raw_values["post_requires_joint_objective_drop"]);
    c.post_requires_line_chi2_decrease =
        parseBool(res.raw_values["post_requires_line_chi2_decrease"]);
    c.max_translation_delta_m =
        getDouble(res.raw_values, "post_translation_delta_max_m", ok);
    c.max_rotation_delta_deg =
        getDouble(res.raw_values, "post_rotation_delta_max_deg", ok);
    c.point_degraded_safe_output_chi2_relative_threshold =
        getDouble(res.raw_values,
                  "point_degraded_safe_output_chi2_relative_threshold", ok);
    c.point_degraded_safe_output_inlier_loss_threshold =
        getInt(res.raw_values,
               "point_degraded_safe_output_inlier_loss_threshold", ok);
    if (!ok || c.line_lambda <= 0.0 || c.probe_beta <= 0.0 ||
        c.min_valid_lines < 0 || c.required_joint_rank < 0 ||
        c.max_translation_delta_m <= 0.0 || c.max_rotation_delta_deg <= 0.0) {
        res.error = "invalid_config_value";
        return res;
    }
    res.ok = true;
    return res;
}

}  // namespace StructureSLAM
