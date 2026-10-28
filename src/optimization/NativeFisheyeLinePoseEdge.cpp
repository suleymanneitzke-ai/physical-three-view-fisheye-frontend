#include "optimization/NativeFisheyeLinePoseEdge.h"

#include <Eigen/Geometry>

namespace StructureSLAM
{
namespace
{

Eigen::Isometry3d ToIsometry(const g2o::SE3Quat& pose)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = pose.rotation().toRotationMatrix();
    T.translation() = pose.translation();
    return T;
}

} // namespace

NativeFisheyeLinePoseEdge::NativeFisheyeLinePoseEdge()
{
    setMeasurement(Eigen::Vector2d::Zero());
}

LinePixelResidualResult NativeFisheyeLinePoseEdge::Evaluate(
    const bool compute_jacobian) const
{
    const g2o::VertexSE3Expmap* v =
        static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    if(!v)
    {
        LinePixelResidualResult out;
        out.invalid_reason = "missing_pose_vertex";
        return out;
    }
    return EvaluateVirtualViewLinePixelResidual(
        ToIsometry(v->estimate()), endpoint0_w, endpoint1_w,
        observation, compute_jacobian);
}

void NativeFisheyeLinePoseEdge::computeError()
{
    const LinePixelResidualResult residual = Evaluate(false);
    if(!residual.valid)
        _error = Eigen::Vector2d::Constant(1e6);
    else
        _error = residual.residual_px - _measurement;
}

void NativeFisheyeLinePoseEdge::linearizeOplus()
{
    const LinePixelResidualResult residual = Evaluate(true);
    if(!residual.valid)
        _jacobianOplusXi.setZero();
    else
        _jacobianOplusXi = residual.jacobian_tcw;
}

bool NativeFisheyeLinePoseEdge::isProjectionValid() const
{
    return Evaluate(false).valid;
}

bool NativeFisheyeLinePoseEdge::read(std::istream&)
{
    return false;
}

bool NativeFisheyeLinePoseEdge::write(std::ostream&) const
{
    return false;
}

} // namespace StructureSLAM
