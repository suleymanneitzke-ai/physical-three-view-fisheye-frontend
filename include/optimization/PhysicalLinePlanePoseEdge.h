#ifndef STRUCTURE_SLAM_PHYSICAL_LINE_PLANE_POSE_EDGE_H
#define STRUCTURE_SLAM_PHYSICAL_LINE_PLANE_POSE_EDGE_H

#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

#include <Eigen/Core>

namespace StructureSLAM
{

class PhysicalLinePlanePoseEdge
    : public g2o::BaseUnaryEdge<3, Eigen::Vector3d,
                                g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PhysicalLinePlanePoseEdge();

    void computeError() override;
    void linearizeOplus() override;
    bool read(std::istream& is) override;
    bool write(std::ostream& os) const override;

    bool isValid() const;

    Eigen::Vector3d line_v_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d line_n_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d observed_plane_c = Eigen::Vector3d::Zero();
};

} // namespace StructureSLAM

#endif
