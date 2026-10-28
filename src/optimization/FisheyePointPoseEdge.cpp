#include "optimization/FisheyePointPoseEdge.h"

#include <cmath>

namespace StructureSLAM {

Eigen::Matrix3d skewMatrix(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
    return S;
}

FisheyePointResidual FisheyePointPoseEdge::evaluate(
    const CentralCameraModel& camera,
    const OfflinePoseTcw& Tcw,
    const PointObservation& obs) {
    FisheyePointResidual out;
    if (!obs.Pw.allFinite() || !obs.uv.allFinite() ||
        !std::isfinite(obs.inverse_sigma2) || obs.inverse_sigma2 <= 0.0) {
        out.invalid_reason = "NON_FINITE_OR_INVALID_OBSERVATION";
        return out;
    }
    out.Pc = Tcw.Rcw * obs.Pw + Tcw.tcw;
    if (!out.Pc.allFinite() || out.Pc.z() <= 0.0) {
        out.invalid_reason = "POINT_NOT_IN_FORWARD_CAMERA_DOMAIN";
        return out;
    }
    if (!camera.project(out.Pc, out.predicted)) {
        out.invalid_reason = "CAMERA_PROJECT_REJECT";
        return out;
    }
    if (!camera.isInImage(out.predicted)) {
        out.invalid_reason = "PREDICTED_PIXEL_OUTSIDE_IMAGE";
        return out;
    }
    Eigen::Vector3d bearing;
    if (!camera.unproject(obs.uv, bearing)) {
        out.invalid_reason = "OBSERVED_PIXEL_CAMERA_REJECT";
        return out;
    }
    Eigen::Matrix<double, 2, 3> Jpi;
    if (!camera.projectionJacobian(out.Pc, Jpi)) {
        out.invalid_reason = "PROJECTION_JACOBIAN_REJECT";
        return out;
    }
    Eigen::Matrix<double, 3, 6> dPc_dxi;
    dPc_dxi.block<3, 3>(0, 0) = -skewMatrix(out.Pc);
    dPc_dxi.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    out.error = obs.uv - out.predicted;
    out.jacobian = -Jpi * dPc_dxi;
    out.chi2 = obs.inverse_sigma2 * out.error.squaredNorm();
    out.error_norm = out.error.norm();
    out.valid = out.error.allFinite() && out.jacobian.allFinite() &&
                std::isfinite(out.chi2);
    if (!out.valid) {
        out.invalid_reason = "NON_FINITE_RESIDUAL_OR_JACOBIAN";
    }
    return out;
}

}  // namespace StructureSLAM
