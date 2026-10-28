#ifndef STRUCTURE_SLAM_KANNALA_BRANDT8_CAMERA_H
#define STRUCTURE_SLAM_KANNALA_BRANDT8_CAMERA_H

#include "CentralCameraModel.h"

#include <array>
#include <string>

namespace StructureSLAM {

class KannalaBrandt8Camera final : public CentralCameraModel {
public:
    struct Parameters {
        double fx = 0.0;
        double fy = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        double k1 = 0.0;
        double k2 = 0.0;
        double k3 = 0.0;
        double k4 = 0.0;
        int width = 0;
        int height = 0;
        double newton_tolerance = 1e-6;
        int newton_max_iterations = 10;
        double min_positive_z = 1e-12;
        double min_valid_radius = 1e-12;
        // CAM_B trusted monotonic physical branch.  Safe-FoV experiments
        // override this with a smaller bearing-domain limit.
        double physical_alpha_max_deg = 113.201946;
        double max_theta_d = 1.9416861378997647;
    };

    explicit KannalaBrandt8Camera(const Parameters& params);

    bool project(const Eigen::Vector3d& Pc,
                 Eigen::Vector2d& uv) const override;

    bool unproject(const Eigen::Vector2d& uv,
                   Eigen::Vector3d& bearing) const override;

    bool projectionJacobian(
        const Eigen::Vector3d& Pc,
        Eigen::Matrix<double, 2, 3>& J) const override;

    bool isInImage(const Eigen::Vector2d& uv,
                   double border = 0.0) const override;

    bool valid() const { return valid_; }
    const std::string& validationError() const { return validation_error_; }
    const Parameters& parameters() const { return params_; }

    double distortedTheta(double theta) const;
    double distortedThetaDerivative(double theta) const;

private:
    bool finiteVector(const Eigen::Vector2d& v) const;
    bool finiteVector(const Eigen::Vector3d& v) const;
    bool finiteScalar(double v) const;

    Parameters params_;
    bool valid_ = false;
    std::string validation_error_;
};

}  // namespace StructureSLAM

#endif  // STRUCTURE_SLAM_KANNALA_BRANDT8_CAMERA_H
