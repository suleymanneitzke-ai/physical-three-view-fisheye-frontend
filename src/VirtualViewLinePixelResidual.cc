#include "VirtualViewLinePixelResidual.h"

#include "FisheyeVirtualCameraGeometry.h"

#include <Eigen/Dense>

#include <cmath>
#include <limits>

namespace StructureSLAM
{
namespace
{

constexpr double kMinDepth = 1e-9;
constexpr double kLineNormEps = 1e-12;

bool FiniteTransform(const Eigen::Isometry3d& T)
{
    return T.matrix().allFinite();
}

Eigen::Matrix3d Skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return S;
}

double ViewYawRad(int view_id)
{
    return NativeFisheyeVirtualYawRad(view_id);
}

bool ProjectVirtual(const Eigen::Vector3d& Pv, Eigen::Vector2d& px)
{
    px.setConstant(std::numeric_limits<double>::quiet_NaN());
    if(!Pv.allFinite() || Pv.z() <= kMinDepth)
        return false;
    px.x() = NativeFisheyeVirtualFx() * Pv.x() / Pv.z() +
             NativeFisheyeVirtualCx();
    px.y() = NativeFisheyeVirtualFy() * Pv.y() / Pv.z() +
             NativeFisheyeVirtualCy();
    return px.allFinite();
}

Eigen::Matrix<double, 2, 3> PinholeJacobian(const Eigen::Vector3d& Pv)
{
    Eigen::Matrix<double, 2, 3> J;
    const double z = Pv.z();
    J << NativeFisheyeVirtualFx() / z,
         0.0,
         -NativeFisheyeVirtualFx() * Pv.x() / (z * z),
         0.0,
         NativeFisheyeVirtualFy() / z,
         -NativeFisheyeVirtualFy() * Pv.y() / (z * z);
    return J;
}

bool FillEndpoint(const Eigen::Isometry3d& Tcw,
                  const Eigen::Matrix3d& Rvc,
                  const Eigen::Vector3d& Pw,
                  const Eigen::Vector3d& line,
                  bool compute_jacobian,
                  Eigen::Vector2d& residual_and_px,
                  double& depth,
                  Eigen::Matrix<double, 1, 6>& Jrow,
                  std::string& invalid_reason)
{
    if(!Pw.allFinite())
    {
        invalid_reason = "endpoint_not_finite";
        return false;
    }
    const Eigen::Vector3d Pc = Tcw.linear() * Pw + Tcw.translation();
    const Eigen::Vector3d Pv = Rvc * Pc;
    depth = Pv.z();
    Eigen::Vector2d px;
    if(!ProjectVirtual(Pv, px))
    {
        invalid_reason = "virtual_projection_invalid_or_behind_camera";
        return false;
    }
    residual_and_px = px;
    if(compute_jacobian)
    {
        Eigen::Matrix<double, 3, 6> dPc_dxi;
        dPc_dxi.block<3, 3>(0, 0) = -Skew(Pc);
        dPc_dxi.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        const Eigen::Matrix<double, 1, 2> de_dpx = line.head<2>().transpose();
        Jrow = de_dpx * PinholeJacobian(Pv) * Rvc * dPc_dxi;
    }
    return true;
}

} // namespace

Eigen::Vector3d NormalizeImageLineForPixelDistance(
    const Eigen::Vector2d& start_px,
    const Eigen::Vector2d& end_px,
    bool* ok)
{
    if(ok)
        *ok = false;
    if(!start_px.allFinite() || !end_px.allFinite())
        return Eigen::Vector3d::Zero();
    const Eigen::Vector3d s(start_px.x(), start_px.y(), 1.0);
    const Eigen::Vector3d e(end_px.x(), end_px.y(), 1.0);
    Eigen::Vector3d line = s.cross(e);
    const double n = std::hypot(line.x(), line.y());
    if(!std::isfinite(n) || n <= kLineNormEps)
        return Eigen::Vector3d::Zero();
    line /= n;
    if(ok)
        *ok = line.allFinite();
    return line;
}

LinePixelResidualResult EvaluateVirtualViewLinePixelResidual(
    const Eigen::Isometry3d& Tcw,
    const Eigen::Vector3d& endpoint0_w,
    const Eigen::Vector3d& endpoint1_w,
    const VirtualViewLineObservation& observation,
    bool compute_analytic_jacobian)
{
    LinePixelResidualResult out;
    if(observation.view_id < 0 || observation.view_id >= 3)
    {
        out.invalid_reason = "invalid_view_id";
        return out;
    }
    if(!FiniteTransform(Tcw))
    {
        out.invalid_reason = "Tcw_not_finite";
        return out;
    }
    if(!observation.image_line_normalized.allFinite())
    {
        out.invalid_reason = "image_line_not_finite";
        return out;
    }
    const double ab_norm = std::hypot(observation.image_line_normalized.x(),
                                     observation.image_line_normalized.y());
    if(!std::isfinite(ab_norm) || std::abs(ab_norm - 1.0) > 1e-8)
    {
        out.invalid_reason = "image_line_not_pixel_normalized";
        return out;
    }

    const Eigen::Matrix3d Rvc = RotationVcFromYaw(ViewYawRad(observation.view_id));
    Eigen::Matrix<double, 1, 6> J0 = Eigen::Matrix<double, 1, 6>::Zero();
    Eigen::Matrix<double, 1, 6> J1 = Eigen::Matrix<double, 1, 6>::Zero();
    std::string reason;
    if(!FillEndpoint(Tcw, Rvc, endpoint0_w, observation.image_line_normalized,
                     compute_analytic_jacobian, out.projected_start_px,
                     out.endpoint_depth_virtual.x(), J0, reason))
    {
        out.invalid_reason = "endpoint0_" + reason;
        return out;
    }
    if(!FillEndpoint(Tcw, Rvc, endpoint1_w, observation.image_line_normalized,
                     compute_analytic_jacobian, out.projected_end_px,
                     out.endpoint_depth_virtual.y(), J1, reason))
    {
        out.invalid_reason = "endpoint1_" + reason;
        return out;
    }
    out.residual_px.x() = observation.image_line_normalized.dot(
        Eigen::Vector3d(out.projected_start_px.x(), out.projected_start_px.y(), 1.0));
    out.residual_px.y() = observation.image_line_normalized.dot(
        Eigen::Vector3d(out.projected_end_px.x(), out.projected_end_px.y(), 1.0));
    if(compute_analytic_jacobian)
    {
        out.jacobian_tcw.row(0) = J0;
        out.jacobian_tcw.row(1) = J1;
    }
    out.chi2 = out.residual_px.squaredNorm();
    out.valid = out.residual_px.allFinite() && out.projected_start_px.allFinite() &&
                out.projected_end_px.allFinite() && out.jacobian_tcw.allFinite() &&
                std::isfinite(out.chi2);
    if(!out.valid)
        out.invalid_reason = "non_finite_residual_or_jacobian";
    return out;
}

Eigen::Isometry3d ApplyLeftPerturbationRxRyRzTxTyTz(
    const Eigen::Matrix<double, 6, 1>& dx,
    const Eigen::Isometry3d& Tcw)
{
    const Eigen::Vector3d w = dx.head<3>();
    const Eigen::Vector3d upsilon = dx.tail<3>();
    const double theta = w.norm();
    Eigen::Matrix3d Omega = Skew(w);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d V = Eigen::Matrix3d::Identity();
    if(theta < 1e-12)
    {
        R += Omega;
        V += 0.5 * Omega;
    }
    else
    {
        const double theta2 = theta * theta;
        R = Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
        V += (1.0 - std::cos(theta)) / theta2 * Omega +
             (theta - std::sin(theta)) / (theta2 * theta) * Omega * Omega;
    }
    Eigen::Isometry3d dT = Eigen::Isometry3d::Identity();
    dT.linear() = R;
    dT.translation() = V * upsilon;
    return dT * Tcw;
}

} // namespace StructureSLAM
