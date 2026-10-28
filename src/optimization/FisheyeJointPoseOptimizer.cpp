#include "optimization/FisheyeJointPoseOptimizer.h"

#include "optimization/FisheyePointPoseEdge.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace StructureSLAM {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kPointChi2 = 5.991;
constexpr double kLineChi2 = 3.841;
constexpr double kPointHuber = 2.4476519360399265;
constexpr double kLineHuber = 1.9598462270537148;
constexpr double kSigmaTheta = kPi / 180.0;

using Mat3 = Eigen::Matrix3d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Clock = std::chrono::steady_clock;

double msSince(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double huberCost(double chi2, double delta, double* rho1) {
    if (chi2 <= delta * delta) {
        *rho1 = 1.0;
        return chi2;
    }
    const double s = std::sqrt(std::max(0.0, chi2));
    *rho1 = delta / std::max(1e-15, s);
    return 2.0 * delta * s - delta * delta;
}

void setDistribution(const std::vector<double>& values, CommonSetEvaluation* out) {
    if (values.empty()) return;
    std::vector<double> v = values;
    std::sort(v.begin(), v.end());
    out->mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    out->median = v[v.size() / 2];
    const size_t p95 = std::min(v.size() - 1,
                                static_cast<size_t>(std::ceil(0.95 * v.size())) - 1);
    out->p95 = v[p95];
    out->max = v.back();
}

BearingPlaneLineSample makeLineSample(
    const FisheyeFrameOptimizationSnapshot& snapshot,
    const LineObservationSnapshot& line,
    size_t sample_ordinal,
    const Eigen::Vector3d& bearing) {
    BearingPlaneLineSample sample;
    sample.frame_id = static_cast<int>(snapshot.frame_id);
    sample.line3d_id = static_cast<int>(line.line_id);
    sample.view_id = line.view_id;
    sample.line_id = line.image_line_id;
    sample.sample_index =
        sample_ordinal < line.source_sample_indices.size()
            ? line.source_sample_indices[sample_ordinal]
            : static_cast<int>(sample_ordinal);
    sample.bearing = bearing;
    sample.line_w = line.line_w;
    sample.sigma_theta_rad = kSigmaTheta;
    sample.sample_weight = line.line_weight /
        std::max(1, static_cast<int>(line.bearing_samples.size()));
    return sample;
}

}  // namespace

FisheyeJointPoseOptimizer::FisheyeJointPoseOptimizer(
    const FisheyeJointSafetyConfig& config)
    : config_(config), filter_(config) {}

CommonSetEvaluation FisheyeJointPoseOptimizer::evaluatePoints(
    const FisheyeFrameOptimizationSnapshot& snapshot,
    const OfflinePoseTcw& pose,
    const CentralCameraModel& camera) const {
    CommonSetEvaluation out;
    std::vector<double> errors;
    for (const auto& p : snapshot.points) {
        PointObservation obs;
        obs.frame_id = static_cast<int>(snapshot.frame_id);
        obs.map_point_id = p.point_id;
        obs.Pw = p.world_point;
        obs.uv = p.observed_pixel;
        obs.octave = p.octave;
        obs.inverse_sigma2 = p.inverse_sigma2;
        const auto r = FisheyePointPoseEdge::evaluate(camera, pose, obs);
        if (!r.valid) {
            ++out.invalid;
            continue;
        }
        double rho1 = 1.0;
        const double robust = huberCost(r.chi2, kPointHuber, &rho1);
        out.H += rho1 * p.inverse_sigma2 * r.jacobian.transpose() * r.jacobian;
        out.g += rho1 * p.inverse_sigma2 * r.jacobian.transpose() * r.error;
        out.raw += r.error.squaredNorm();
        out.chi2 += r.chi2;
        out.robust += robust;
        ++out.count;
        out.inliers += r.chi2 <= kPointChi2 ? 1 : 0;
        errors.push_back(r.error_norm);
    }
    setDistribution(errors, &out);
    out.finite = out.finite && out.H.allFinite() && out.g.allFinite();
    return out;
}

CommonSetEvaluation FisheyeJointPoseOptimizer::evaluateLines(
    const FisheyeFrameOptimizationSnapshot& snapshot,
    const OfflinePoseTcw& pose,
    double lambda) const {
    CommonSetEvaluation out;
    std::vector<double> errors;
    for (const auto& line : snapshot.lines) {
        Vec6 line_g = Vec6::Zero();
        for (size_t sample_index = 0; sample_index < line.bearing_samples.size();
             ++sample_index) {
            const auto sample =
                makeLineSample(snapshot, line, sample_index,
                               line.bearing_samples[sample_index]);
            const auto r = FisheyeBearingPlaneLineEdge::evaluate(pose, sample);
            if (!r.valid) {
                ++out.invalid;
                continue;
            }
            double rho1 = 1.0;
            const double robust = huberCost(r.chi2, kLineHuber, &rho1);
            const double w = sample.sample_weight;
            const Vec6 gi = lambda * w * rho1 *
                r.jacobian.transpose() *
                Eigen::Matrix<double, 1, 1>(r.normalized_residual);
            out.H += lambda * w * rho1 * r.jacobian.transpose() * r.jacobian;
            out.g += gi;
            line_g += gi;
            out.raw += lambda * w * r.raw_residual * r.raw_residual;
            out.chi2 += lambda * w * r.chi2;
            out.robust += lambda * w * robust;
            ++out.count;
            out.inliers += r.chi2 <= kLineChi2 ? 1 : 0;
            errors.push_back(std::abs(r.raw_residual));
        }
        const auto key = std::make_pair(line.line_id, line.image_line_id);
        auto it = out.per_line_g.find(key);
        if (it == out.per_line_g.end()) {
            out.per_line_g.emplace(key, line_g);
        } else {
            it->second += line_g;
        }
    }
    setDistribution(errors, &out);
    out.finite = out.finite && out.H.allFinite() && out.g.allFinite();
    return out;
}

HessianSummary FisheyeJointPoseOptimizer::summarizeHessian(const Mat6& H) {
    HessianSummary out;
    Eigen::SelfAdjointEigenSolver<Mat6> es(H);
    if (es.info() != Eigen::Success || !es.eigenvalues().allFinite()) return out;
    out.eigenvalues = es.eigenvalues();
    out.weakest_eigenvector = es.eigenvectors().col(0);
    const double mx = std::max(0.0, out.eigenvalues.maxCoeff());
    const double threshold = mx * 1e-8;
    double min_pos = 1e300;
    for (int i = 0; i < 6; ++i) {
        if (out.eigenvalues[i] > threshold) {
            ++out.rank;
            min_pos = std::min(min_pos, out.eigenvalues[i]);
        }
    }
    out.condition = min_pos < 1e299 ? mx / min_pos : 0.0;
    return out;
}

OfflinePoseTcw FisheyeJointPoseOptimizer::expLeft(const Vec6& dx,
                                                  const OfflinePoseTcw& pose) {
    const Vec3 w = dx.head<3>();
    const Vec3 u = dx.tail<3>();
    const double theta = w.norm();
    const Mat3 W = skewMatrix(w);
    Mat3 R;
    Mat3 V;
    if (theta < 1e-12) {
        R = Mat3::Identity() + W + 0.5 * W * W;
        V = Mat3::Identity() + 0.5 * W + (1.0 / 6.0) * W * W;
    } else {
        const Mat3 W2 = W * W;
        R = Mat3::Identity() + std::sin(theta) / theta * W +
            (1.0 - std::cos(theta)) / (theta * theta) * W2;
        V = Mat3::Identity() +
            (1.0 - std::cos(theta)) / (theta * theta) * W +
            (theta - std::sin(theta)) / (theta * theta * theta) * W2;
    }
    OfflinePoseTcw out;
    out.Rcw = R * pose.Rcw;
    out.tcw = R * pose.tcw + V * u;
    return out;
}

double FisheyeJointPoseOptimizer::rotationDifferenceDeg(const OfflinePoseTcw& a,
                                                        const OfflinePoseTcw& b) {
    const Mat3 dR = a.Rcw * b.Rcw.transpose();
    const double c = std::max(-1.0, std::min(1.0, 0.5 * (dR.trace() - 1.0)));
    return std::acos(c) * 180.0 / kPi;
}

Vec6 FisheyeJointPoseOptimizer::solveDx(const Mat6& H, const Vec6& g,
                                        double damping_scale) const {
    Mat6 A = H;
    for (int i = 0; i < 6; ++i) {
        A(i, i) += damping_scale * std::max(1.0, std::abs(H(i, i)));
    }
    Eigen::JacobiSVD<Mat6> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vec6 inv = Vec6::Zero();
    for (int i = 0; i < 6; ++i) {
        if (svd.singularValues()[i] > 1e-12) inv[i] = 1.0 / svd.singularValues()[i];
    }
    return -svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose() * g;
}

OfflinePoseTcw FisheyeJointPoseOptimizer::optimizeJoint(
    const FisheyeFrameOptimizationSnapshot& snapshot,
    const CentralCameraModel& camera,
    int* iterations,
    std::string* status,
    std::vector<FisheyeJointRoundTrace>* trace) const {
    OfflinePoseTcw pose = snapshot.initial_pose_Tcw;
    std::vector<bool> point_active(snapshot.points.size(), true);
    std::vector<std::vector<bool> > line_active;
    line_active.reserve(snapshot.lines.size());
    for (const auto& line : snapshot.lines) {
        line_active.push_back(std::vector<bool>(line.bearing_samples.size(), true));
    }

    auto activeSnapshot = [&]() {
        FisheyeFrameOptimizationSnapshot active = snapshot;
        active.points.clear();
        for (size_t i = 0; i < snapshot.points.size(); ++i) {
            if (point_active[i]) active.points.push_back(snapshot.points[i]);
        }
        active.lines.clear();
        for (size_t i = 0; i < snapshot.lines.size(); ++i) {
            LineObservationSnapshot line = snapshot.lines[i];
            line.bearing_samples.clear();
            line.source_sample_indices.clear();
            line.optimizer_sample_ordinals.clear();
            for (size_t j = 0; j < snapshot.lines[i].bearing_samples.size(); ++j) {
                if (line_active[i][j]) {
                    line.bearing_samples.push_back(snapshot.lines[i].bearing_samples[j]);
                    if (j < snapshot.lines[i].source_sample_indices.size()) {
                        line.source_sample_indices.push_back(
                            snapshot.lines[i].source_sample_indices[j]);
                    }
                    if (j < snapshot.lines[i].optimizer_sample_ordinals.size()) {
                        line.optimizer_sample_ordinals.push_back(
                            snapshot.lines[i].optimizer_sample_ordinals[j]);
                    }
                }
            }
            if (!line.bearing_samples.empty()) active.lines.push_back(line);
        }
        return active;
    };

    auto active = activeSnapshot();
    double current = evaluatePoints(active, pose, camera).robust +
                     evaluateLines(active, pose, config_.line_lambda).robust;
    constexpr int kStages[3] = {5, 5, 8};
    for (int stage = 0; stage < 3; ++stage) {
        const bool robust_enabled = true;
        for (int it = 0; it < kStages[stage]; ++it) {
            active = activeSnapshot();
            const auto pe = evaluatePoints(active, pose, camera);
            const auto le = evaluateLines(active, pose, config_.line_lambda);
            const Vec6 dx = solveDx(pe.H + le.H, pe.g + le.g, 1e-6);
            if (!dx.allFinite()) {
                *status = "NAN_INF";
                return pose;
            }
            const auto candidate = expLeft(dx, pose);
            const double candidate_cost =
                evaluatePoints(active, candidate, camera).robust +
                evaluateLines(active, candidate, config_.line_lambda).robust;
            if (std::isfinite(candidate_cost) && candidate_cost <= current) {
                pose = candidate;
                current = candidate_cost;
                ++(*iterations);
                if (dx.norm() < 1e-10) break;
            }
        }
        if (trace) {
            active = activeSnapshot();
            const auto pe = evaluatePoints(active, pose, camera);
            const auto le = evaluateLines(active, pose, config_.line_lambda);
            FisheyeJointRoundTrace row;
            row.round = stage;
            row.iteration_count = *iterations;
            row.robust_kernel_enabled = robust_enabled;
            row.active_point_count = static_cast<int>(active.points.size());
            row.active_line_count = static_cast<int>(active.lines.size());
            for (const auto& line : active.lines) {
                row.active_line_sample_count +=
                    static_cast<int>(line.bearing_samples.size());
            }
            row.pose = pose;
            row.point_cost = pe.robust;
            row.line_cost = le.robust;
            row.joint_cost = pe.robust + le.robust;
            row.solver_status = *status;
            trace->push_back(row);
        }
        if (stage == 0) {
            for (size_t i = 0; i < snapshot.points.size(); ++i) {
                FisheyeFrameOptimizationSnapshot one = snapshot;
                one.points.assign(1, snapshot.points[i]);
                const auto pr = evaluatePoints(one, pose, camera);
                point_active[i] = pr.count == 1 && pr.chi2 < kPointChi2;
            }
            for (size_t i = 0; i < snapshot.lines.size(); ++i) {
                int active_count = 0;
                for (size_t j = 0; j < snapshot.lines[i].bearing_samples.size(); ++j) {
                    FisheyeFrameOptimizationSnapshot one = snapshot;
                    LineObservationSnapshot line = snapshot.lines[i];
                    line.bearing_samples.assign(1, snapshot.lines[i].bearing_samples[j]);
                    line.source_sample_indices.clear();
                    line.optimizer_sample_ordinals.clear();
                    if (j < snapshot.lines[i].source_sample_indices.size()) {
                        line.source_sample_indices.push_back(
                            snapshot.lines[i].source_sample_indices[j]);
                    }
                    if (j < snapshot.lines[i].optimizer_sample_ordinals.size()) {
                        line.optimizer_sample_ordinals.push_back(
                            snapshot.lines[i].optimizer_sample_ordinals[j]);
                    }
                    one.lines.assign(1, line);
                    const auto lr = evaluateLines(one, pose, 1.0);
                    line_active[i][j] = lr.count == 1 && lr.chi2 < kLineChi2;
                    if (line_active[i][j]) ++active_count;
                }
                const int required = std::max(
                    2, static_cast<int>(std::ceil(0.4 * snapshot.lines[i].bearing_samples.size())));
                if (active_count < required) {
                    std::fill(line_active[i].begin(), line_active[i].end(), false);
                }
            }
        }
    }
    return pose;
}

FisheyeJointOptimizationResult FisheyeJointPoseOptimizer::run(
    const FisheyeFrameOptimizationSnapshot& snapshot,
    const CentralCameraModel& camera) const {
    const auto total_start = Clock::now();
    FisheyeJointOptimizationResult out;
    out.point_pose = snapshot.point_pose_Tcw;
    out.safe_output_pose = snapshot.point_pose_Tcw;

    auto t = Clock::now();
    out.point_at_point = evaluatePoints(snapshot, snapshot.point_pose_Tcw, camera);
    out.timing.point_common_eval_ms = msSince(t);

    t = Clock::now();
    out.line_at_point = evaluateLines(snapshot, snapshot.point_pose_Tcw, config_.line_lambda);
    out.joint_hessian_at_point =
        summarizeHessian(out.point_at_point.H + out.line_at_point.H);
    double total_g = 0.0;
    double max_g = 0.0;
    for (const auto& kv : out.line_at_point.per_line_g) {
        const double n = kv.second.norm();
        total_g += n;
        max_g = std::max(max_g, n);
    }
    out.safety_metrics.valid_lines =
        static_cast<int>(out.line_at_point.per_line_g.size());
    out.safety_metrics.joint_rank = out.joint_hessian_at_point.rank;
    out.safety_metrics.max_single_line_gradient_fraction =
        total_g > 1e-12 ? max_g / total_g : 0.0;
    out.safety_metrics.finite = out.point_at_point.finite && out.line_at_point.finite;
    out.timing.line_precheck_ms = msSince(t);

    t = Clock::now();
    const Vec6 dx = solveDx(out.point_at_point.H + out.line_at_point.H,
                           out.point_at_point.g + out.line_at_point.g,
                           1e-8);
    out.probe_pose = expLeft(config_.probe_beta * dx, snapshot.point_pose_Tcw);
    out.timing.probe_solve_ms = msSince(t);

    t = Clock::now();
    out.point_at_probe = evaluatePoints(snapshot, out.probe_pose, camera);
    out.safety_metrics.probe_point_chi2_relative_increase =
        (out.point_at_probe.chi2 - out.point_at_point.chi2) /
        std::max(1.0, out.point_at_point.chi2);
    out.safety_metrics.probe_point_inlier_loss =
        out.point_at_point.inliers - out.point_at_probe.inliers;
    out.safety_metrics.probe_point_p95_relative_increase =
        (out.point_at_probe.p95 - out.point_at_point.p95) /
        std::max(1e-12, out.point_at_point.p95);
    out.safety_metrics.probe_invalid_projection = out.point_at_probe.invalid;
    out.timing.probe_eval_ms = msSince(t);

    auto pre_probe_decision = filter_.evaluate(out.safety_metrics);
    out.pre_pass = pre_probe_decision.pre_pass;
    out.probe_pass = pre_probe_decision.probe_pass;

    t = Clock::now();
    if (out.pre_pass && out.probe_pass) {
        out.joint_attempted = true;
        out.joint_candidate_pose =
            optimizeJoint(snapshot, camera, &out.joint_iterations, &out.status,
                          &out.round_trace);
    } else {
        out.joint_candidate_pose = snapshot.point_pose_Tcw;
        out.status = "SKIPPED_BY_PRE_PROBE";
    }
    out.timing.joint_optimize_ms = msSince(t);

    t = Clock::now();
    out.point_at_joint = evaluatePoints(snapshot, out.joint_candidate_pose, camera);
    out.line_at_joint = evaluateLines(snapshot, out.joint_candidate_pose,
                                      config_.line_lambda);
    out.safety_metrics.post_point_chi2_relative_increase =
        (out.point_at_joint.chi2 - out.point_at_point.chi2) /
        std::max(1.0, out.point_at_point.chi2);
    out.safety_metrics.post_point_inlier_loss =
        out.point_at_point.inliers - out.point_at_joint.inliers;
    out.safety_metrics.post_point_p95_relative_increase =
        (out.point_at_joint.p95 - out.point_at_point.p95) /
        std::max(1e-12, out.point_at_point.p95);
    out.safety_metrics.line_chi2_relative_change =
        (out.line_at_joint.chi2 - out.line_at_point.chi2) /
        std::max(1.0, out.line_at_point.chi2);
    out.safety_metrics.joint_objective_drop =
        (out.point_at_point.robust + out.line_at_point.robust) -
        (out.point_at_joint.robust + out.line_at_joint.robust);
    out.safety_metrics.translation_delta_m =
        (out.joint_candidate_pose.tcw - snapshot.point_pose_Tcw.tcw).norm();
    out.safety_metrics.rotation_delta_deg =
        rotationDifferenceDeg(out.joint_candidate_pose, snapshot.point_pose_Tcw);
    out.safety_metrics.finite =
        out.safety_metrics.finite && out.point_at_probe.finite &&
        out.point_at_joint.finite && out.line_at_joint.finite &&
        out.joint_candidate_pose.Rcw.allFinite() &&
        out.joint_candidate_pose.tcw.allFinite();
    const auto final_decision = filter_.evaluate(out.safety_metrics);
    out.pre_pass = final_decision.pre_pass;
    out.probe_pass = final_decision.probe_pass;
    out.post_pass = final_decision.post_pass;
    out.joint_accepted = final_decision.accepted;
    out.reject_reason = final_decision.primary_reason;
    out.safe_output_pose =
        out.joint_accepted ? out.joint_candidate_pose : snapshot.point_pose_Tcw;
    out.timing.post_eval_ms = msSince(t);
    out.timing.core_total_ms = msSince(total_start);
    return out;
}

}  // namespace StructureSLAM
