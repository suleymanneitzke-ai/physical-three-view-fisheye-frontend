#include "optimization/PhysicalLinePlanePoseEdge.h"

#include <Eigen/Geometry>
#include <cmath>

namespace StructureSLAM
{
namespace
{

Eigen::Matrix3d Skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d s;
    s << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
    return s;
}

bool Normalize(const Eigen::Vector3d& in, Eigen::Vector3d& out)
{
    if(!in.allFinite())
        return false;
    const double n = in.norm();
    if(n <= 1e-12)
        return false;
    out = in / n;
    return out.allFinite();
}

} // namespace

PhysicalLinePlanePoseEdge::PhysicalLinePlanePoseEdge()
{
    setMeasurement(Eigen::Vector3d::Zero());
}

bool PhysicalLinePlanePoseEdge::isValid() const
{
    Eigen::Vector3d obs_hat;
    Eigen::Vector3d v_hat;
    return Normalize(observed_plane_c, obs_hat) &&
           Normalize(line_v_w, v_hat) &&
           line_n_w.allFinite();
}

void PhysicalLinePlanePoseEdge::computeError()
{
    const g2o::VertexSE3Expmap* v =
        static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    if(!v || !isValid())
    {
        _error = Eigen::Vector3d::Constant(1e6);
        return;
    }

    Eigen::Vector3d obs_hat;
    Normalize(observed_plane_c, obs_hat);

    const Eigen::Matrix3d R = v->estimate().rotation().toRotationMatrix();
    const Eigen::Vector3d t = v->estimate().translation();
    const Eigen::Vector3d vc = R * line_v_w;
    const Eigen::Vector3d nc = R * line_n_w + t.cross(vc);
    Eigen::Vector3d pred_hat;
    if(!Normalize(nc, pred_hat))
    {
        _error = Eigen::Vector3d::Constant(1e6);
        return;
    }

    // A plane normal is unoriented.  Align the predicted normal with the
    // observed representative before forming the directed cross-product
    // residual; keep this sign fixed for the local linearization below.
    if(obs_hat.dot(pred_hat) < 0.0)
        pred_hat = -pred_hat;

    _error = obs_hat.cross(pred_hat) - _measurement;
}

void PhysicalLinePlanePoseEdge::linearizeOplus()
{
    const g2o::VertexSE3Expmap* v =
        static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    if(!v || !isValid())
    {
        _jacobianOplusXi.setZero();
        return;
    }

    Eigen::Vector3d obs_hat;
    Normalize(observed_plane_c, obs_hat);

    const Eigen::Matrix3d R = v->estimate().rotation().toRotationMatrix();
    const Eigen::Vector3d t = v->estimate().translation();
    const Eigen::Vector3d vc = R * line_v_w;
    const Eigen::Vector3d nc = R * line_n_w + t.cross(vc);
    const double nc_norm = nc.norm();
    if(nc_norm <= 1e-12 || !std::isfinite(nc_norm))
    {
        _jacobianOplusXi.setZero();
        return;
    }

    Eigen::Vector3d pred_hat = nc / nc_norm;
    double plane_sign = 1.0;
    if(obs_hat.dot(pred_hat) < 0.0)
    {
        plane_sign = -1.0;
        pred_hat = -pred_hat;
    }
    const Eigen::Matrix3d d_hat_d_nc =
        (Eigen::Matrix3d::Identity() - pred_hat * pred_hat.transpose()) /
        nc_norm * plane_sign;

    Eigen::Matrix<double, 3, 6> d_nc_d_xi;
    d_nc_d_xi.block<3, 3>(0, 0) = -Skew(nc);
    d_nc_d_xi.block<3, 3>(0, 3) = -Skew(vc);

    _jacobianOplusXi = Skew(obs_hat) * d_hat_d_nc * d_nc_d_xi;
}

bool PhysicalLinePlanePoseEdge::read(std::istream&)
{
    return false;
}

bool PhysicalLinePlanePoseEdge::write(std::ostream&) const
{
    return false;
}

} // namespace StructureSLAM
