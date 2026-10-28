#ifndef STRUCTURE_SLAM_KB8_POINT_POSE_EDGE_H
#define STRUCTURE_SLAM_KB8_POINT_POSE_EDGE_H

#include "camera/CentralCameraModel.h"

#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <Eigen/Core>

namespace StructureSLAM
{

class Kb8PointPoseEdge
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d,
                                g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Kb8PointPoseEdge();

    void computeError() override;
    void linearizeOplus() override;
    bool read(std::istream& is) override;
    bool write(std::ostream& os) const override;

    bool isDepthPositive() const;
    Eigen::Matrix<double, 2, 6> jacobian() const;

    Eigen::Vector3d Xw = Eigen::Vector3d::Zero();
    const CentralCameraModel* pCamera = nullptr;
};

class Kb8PointBundleEdge
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,
                                 g2o::VertexSBAPointXYZ,
                                 g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Kb8PointBundleEdge();

    void computeError() override;
    void linearizeOplus() override;
    bool read(std::istream& is) override;
    bool write(std::ostream& os) const override;

    bool computeJacobians(Eigen::Matrix<double, 2, 3>& point_jacobian,
                          Eigen::Matrix<double, 2, 6>& pose_jacobian) const;
    bool isDepthPositive() const;
    Eigen::Matrix<double, 2, 3> pointJacobian() const;
    Eigen::Matrix<double, 2, 6> poseJacobian() const;

    const CentralCameraModel* pCamera = nullptr;
};

class Kb8Sim3ProjectXYZEdge
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,
                                 g2o::VertexSBAPointXYZ,
                                 g2o::VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Kb8Sim3ProjectXYZEdge();

    void computeError() override;
    bool read(std::istream& is) override;
    bool write(std::ostream& os) const override;

    const CentralCameraModel* pCamera = nullptr;
};

class Kb8InverseSim3ProjectXYZEdge
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,
                                 g2o::VertexSBAPointXYZ,
                                 g2o::VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Kb8InverseSim3ProjectXYZEdge();

    void computeError() override;
    bool read(std::istream& is) override;
    bool write(std::ostream& os) const override;

    const CentralCameraModel* pCamera = nullptr;
};

} // namespace StructureSLAM

#endif // STRUCTURE_SLAM_KB8_POINT_POSE_EDGE_H
