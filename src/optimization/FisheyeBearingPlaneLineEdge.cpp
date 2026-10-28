#include "optimization/FisheyeBearingPlaneLineEdge.h"

#include <cmath>
#include <limits>

namespace StructureSLAM {
namespace {

using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;

bool finiteVec(const Vec3& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

Mat3 skew(const Vec3& v) {
    Mat3 S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return S;
}

}  // namespace

BearingPlaneLineResidual FisheyeBearingPlaneLineEdge::evaluate(
    const OfflinePoseTcw& pose,
    const BearingPlaneLineSample& sample,
    double min_plane_norm) {
    BearingPlaneLineResidual out;
    if (!pose.Rcw.allFinite() || !pose.tcw.allFinite() ||
        !sample.bearing.allFinite() || !sample.line_w.n.allFinite() ||
        !sample.line_w.v.allFinite() || sample.sigma_theta_rad <= 0.0 ||
        !std::isfinite(sample.sigma_theta_rad)) {
        out.invalid_reason = "NONFINITE_INPUT";
        return out;
    }
    const Vec3 vc = pose.Rcw * sample.line_w.v;
    const Vec3 nc = pose.Rcw * sample.line_w.n + pose.tcw.cross(vc);
    const double norm_nc = nc.norm();
    if (!finiteVec(nc) || !finiteVec(vc) || !std::isfinite(norm_nc) ||
        norm_nc < min_plane_norm) {
        out.invalid_reason = "DEGENERATE_EXPLANATION_PLANE";
        return out;
    }
    const Vec3 b = sample.bearing.normalized();
    const Vec3 nh = nc / norm_nc;
    out.raw_residual = nh.dot(b);
    out.normalized_residual = out.raw_residual / sample.sigma_theta_rad;
    out.chi2 = out.normalized_residual * out.normalized_residual;
    out.norm_nc = norm_nc;

    const Eigen::RowVector3d dr_dnc =
        b.transpose() * (Mat3::Identity() - nh * nh.transpose()) / norm_nc;
    Eigen::Matrix<double, 3, 6> dnc_dxi;
    dnc_dxi.block<3, 3>(0, 0) = -skew(nc);
    dnc_dxi.block<3, 3>(0, 3) = -skew(vc);
    out.jacobian = (dr_dnc * dnc_dxi) / sample.sigma_theta_rad;
    out.valid = std::isfinite(out.raw_residual) && out.jacobian.allFinite();
    if (!out.valid) {
        out.invalid_reason = "NONFINITE_RESIDUAL_OR_JACOBIAN";
    }
    return out;
}

}  // namespace StructureSLAM
