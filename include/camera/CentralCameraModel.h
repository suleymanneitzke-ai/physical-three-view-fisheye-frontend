#ifndef STRUCTURE_SLAM_CENTRAL_CAMERA_MODEL_H
#define STRUCTURE_SLAM_CENTRAL_CAMERA_MODEL_H

#include <Eigen/Core>

namespace StructureSLAM {

class CentralCameraModel {
public:
    virtual ~CentralCameraModel() = default;

    virtual bool project(const Eigen::Vector3d& Pc,
                         Eigen::Vector2d& uv) const = 0;

    virtual bool unproject(const Eigen::Vector2d& uv,
                           Eigen::Vector3d& bearing) const = 0;

    virtual bool projectionJacobian(
        const Eigen::Vector3d& Pc,
        Eigen::Matrix<double, 2, 3>& J) const = 0;

    virtual bool isInImage(const Eigen::Vector2d& uv,
                           double border = 0.0) const = 0;
};

}  // namespace StructureSLAM

#endif  // STRUCTURE_SLAM_CENTRAL_CAMERA_MODEL_H
