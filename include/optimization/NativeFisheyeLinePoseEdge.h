#ifndef STRUCTURE_SLAM_NATIVE_FISHEYE_LINE_POSE_EDGE_H
#define STRUCTURE_SLAM_NATIVE_FISHEYE_LINE_POSE_EDGE_H

#include "VirtualViewLinePixelResidual.h"

#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

#include <Eigen/Core>

namespace StructureSLAM
{

class NativeFisheyeLinePoseEdge
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d,
                                g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    NativeFisheyeLinePoseEdge();

    void computeError() override;
    void linearizeOplus() override;
    bool read(std::istream& is) override;
    bool write(std::ostream& os) const override;

    bool isProjectionValid() const;
    LinePixelResidualResult Evaluate(bool compute_jacobian) const;

    Eigen::Vector3d endpoint0_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d endpoint1_w = Eigen::Vector3d::Zero();
    VirtualViewLineObservation observation;
};

} // namespace StructureSLAM

#endif
