#include "optimization/Kb8PointPoseEdge.h"

#include <Eigen/Geometry>

#include <cmath>

namespace StructureSLAM
{
namespace
{

Eigen::Matrix<double, 3, 6> PoseJacobian(const Eigen::Vector3d& Pc)
{
    Eigen::Matrix<double, 3, 6> J;
    J << 0.0, Pc.z(), -Pc.y(), 1.0, 0.0, 0.0,
        -Pc.z(), 0.0, Pc.x(), 0.0, 1.0, 0.0,
        Pc.y(), -Pc.x(), 0.0, 0.0, 0.0, 1.0;
    return J;
}

} // namespace

Kb8PointPoseEdge::Kb8PointPoseEdge()
{
}

void Kb8PointPoseEdge::computeError()
{
    const g2o::VertexSE3Expmap* v =
        static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    if(!v || !pCamera)
    {
        _error = Eigen::Vector2d::Constant(1e6);
        return;
    }

    const Eigen::Vector3d Pc = v->estimate().map(Xw);
    Eigen::Vector2d uv;
    if(!Pc.allFinite() || !pCamera->project(Pc, uv))
    {
        _error = Eigen::Vector2d::Constant(1e6);
        return;
    }
    _error = _measurement - uv;
}

void Kb8PointPoseEdge::linearizeOplus()
{
    const g2o::VertexSE3Expmap* v =
        static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    if(!v || !pCamera)
    {
        _jacobianOplusXi.setZero();
        return;
    }

    const Eigen::Vector3d Pc = v->estimate().map(Xw);
    Eigen::Matrix<double, 2, 3> Jpi;
    if(!Pc.allFinite() || !pCamera->projectionJacobian(Pc, Jpi))
    {
        _jacobianOplusXi.setZero();
        return;
    }

    _jacobianOplusXi = -Jpi * PoseJacobian(Pc);
}

bool Kb8PointPoseEdge::isDepthPositive() const
{
    const g2o::VertexSE3Expmap* v =
        static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    return v && (v->estimate().map(Xw))(2) > 0.0;
}

Eigen::Matrix<double, 2, 6> Kb8PointPoseEdge::jacobian() const
{
    return _jacobianOplusXi;
}

bool Kb8PointPoseEdge::read(std::istream&)
{
    return false;
}

bool Kb8PointPoseEdge::write(std::ostream&) const
{
    return false;
}

Kb8PointBundleEdge::Kb8PointBundleEdge()
{
}

void Kb8PointBundleEdge::computeError()
{
    const g2o::VertexSE3Expmap* vPose =
        static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vPoint =
        static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    if(!vPose || !vPoint || !pCamera)
    {
        _error = Eigen::Vector2d::Constant(1e6);
        return;
    }

    const Eigen::Vector3d Pc = vPose->estimate().map(vPoint->estimate());
    Eigen::Vector2d uv;
    if(!Pc.allFinite() || !pCamera->project(Pc, uv))
    {
        _error = Eigen::Vector2d::Constant(1e6);
        return;
    }
    _error = _measurement - uv;
}

void Kb8PointBundleEdge::linearizeOplus()
{
    Eigen::Matrix<double, 2, 3> point_jacobian;
    Eigen::Matrix<double, 2, 6> pose_jacobian;
    if(!computeJacobians(point_jacobian, pose_jacobian))
    {
        _jacobianOplusXi.setZero();
        _jacobianOplusXj.setZero();
        return;
    }

    _jacobianOplusXi = point_jacobian;
    _jacobianOplusXj = pose_jacobian;
}

bool Kb8PointBundleEdge::computeJacobians(
    Eigen::Matrix<double, 2, 3>& point_jacobian,
    Eigen::Matrix<double, 2, 6>& pose_jacobian) const
{
    const g2o::VertexSE3Expmap* vPose =
        static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vPoint =
        static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    point_jacobian.setZero();
    pose_jacobian.setZero();
    if(!vPose || !vPoint || !pCamera)
        return false;

    const Eigen::Vector3d Pc = vPose->estimate().map(vPoint->estimate());
    Eigen::Matrix<double, 2, 3> Jpi;
    if(!Pc.allFinite() || !pCamera->projectionJacobian(Pc, Jpi))
        return false;

    point_jacobian =
        -Jpi * vPose->estimate().rotation().toRotationMatrix();
    pose_jacobian = -Jpi * PoseJacobian(Pc);
    return true;
}

bool Kb8PointBundleEdge::isDepthPositive() const
{
    const g2o::VertexSE3Expmap* vPose =
        static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vPoint =
        static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    return vPose && vPoint &&
           (vPose->estimate().map(vPoint->estimate()))(2) > 0.0;
}

Eigen::Matrix<double, 2, 3> Kb8PointBundleEdge::pointJacobian() const
{
    return _jacobianOplusXi;
}

Eigen::Matrix<double, 2, 6> Kb8PointBundleEdge::poseJacobian() const
{
    return _jacobianOplusXj;
}

bool Kb8PointBundleEdge::read(std::istream&)
{
    return false;
}

bool Kb8PointBundleEdge::write(std::ostream&) const
{
    return false;
}

Kb8Sim3ProjectXYZEdge::Kb8Sim3ProjectXYZEdge()
{
}

void Kb8Sim3ProjectXYZEdge::computeError()
{
    const g2o::VertexSim3Expmap* vSim3 =
        static_cast<const g2o::VertexSim3Expmap*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vPoint =
        static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    if(!vSim3 || !vPoint || !pCamera)
    {
        _error = Eigen::Vector2d::Constant(1e6);
        return;
    }

    const Eigen::Vector3d Pc = vSim3->estimate().map(vPoint->estimate());
    Eigen::Vector2d uv;
    if(!Pc.allFinite() || !pCamera->project(Pc, uv))
    {
        _error = Eigen::Vector2d::Constant(1e6);
        return;
    }
    _error = _measurement - uv;
}

bool Kb8Sim3ProjectXYZEdge::read(std::istream&)
{
    return false;
}

bool Kb8Sim3ProjectXYZEdge::write(std::ostream&) const
{
    return false;
}

Kb8InverseSim3ProjectXYZEdge::Kb8InverseSim3ProjectXYZEdge()
{
}

void Kb8InverseSim3ProjectXYZEdge::computeError()
{
    const g2o::VertexSim3Expmap* vSim3 =
        static_cast<const g2o::VertexSim3Expmap*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vPoint =
        static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    if(!vSim3 || !vPoint || !pCamera)
    {
        _error = Eigen::Vector2d::Constant(1e6);
        return;
    }

    const Eigen::Vector3d Pc =
        vSim3->estimate().inverse().map(vPoint->estimate());
    Eigen::Vector2d uv;
    if(!Pc.allFinite() || !pCamera->project(Pc, uv))
    {
        _error = Eigen::Vector2d::Constant(1e6);
        return;
    }
    _error = _measurement - uv;
}

bool Kb8InverseSim3ProjectXYZEdge::read(std::istream&)
{
    return false;
}

bool Kb8InverseSim3ProjectXYZEdge::write(std::ostream&) const
{
    return false;
}

} // namespace StructureSLAM
