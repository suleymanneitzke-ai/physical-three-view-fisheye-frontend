#include "camera/KannalaBrandt8Camera.h"

#include <cmath>
#include <limits>
#include <sstream>

namespace StructureSLAM {

KannalaBrandt8Camera::KannalaBrandt8Camera(const Parameters& params)
    : params_(params) {
    if (!finiteScalar(params_.fx) || !finiteScalar(params_.fy) ||
        !finiteScalar(params_.cx) || !finiteScalar(params_.cy) ||
        !finiteScalar(params_.k1) || !finiteScalar(params_.k2) ||
        !finiteScalar(params_.k3) || !finiteScalar(params_.k4)) {
        validation_error_ = "non-finite parameter";
        return;
    }
    if (params_.fx <= 0.0 || params_.fy <= 0.0) {
        validation_error_ = "fx/fy must be positive";
        return;
    }
    if (params_.width <= 0 || params_.height <= 0) {
        validation_error_ = "image width/height must be positive";
        return;
    }
    if (params_.newton_max_iterations <= 0 ||
        params_.newton_tolerance <= 0.0 ||
        params_.min_positive_z <= 0.0 ||
        params_.min_valid_radius <= 0.0 ||
        params_.max_theta_d <= 0.0) {
        validation_error_ = "invalid numeric guard parameter";
        return;
    }
    valid_ = true;
}

bool KannalaBrandt8Camera::project(const Eigen::Vector3d& Pc,
                                   Eigen::Vector2d& uv) const {
    uv.setConstant(std::numeric_limits<double>::quiet_NaN());
    if (!valid_ || !finiteVector(Pc) || Pc.norm() <= params_.min_positive_z) {
        return false;
    }

    const double r2 = Pc.x() * Pc.x() + Pc.y() * Pc.y();
    const double r = std::sqrt(r2);
    if (r < params_.min_valid_radius) {
        if (Pc.z() <= 0.0) return false;
        uv.x() = params_.cx;
        uv.y() = params_.cy;
        return true;
    }

    const double theta = std::atan2(r, Pc.z());
    const double theta_d = distortedTheta(theta);
    const double scale = theta_d / r;
    uv.x() = params_.fx * scale * Pc.x() + params_.cx;
    uv.y() = params_.fy * scale * Pc.y() + params_.cy;
    return finiteVector(uv);
}

bool KannalaBrandt8Camera::unproject(const Eigen::Vector2d& uv,
                                     Eigen::Vector3d& bearing) const {
    bearing.setConstant(std::numeric_limits<double>::quiet_NaN());
    if (!valid_ || !finiteVector(uv)) {
        return false;
    }

    const double mx = (uv.x() - params_.cx) / params_.fx;
    const double my = (uv.y() - params_.cy) / params_.fy;
    double theta_d = std::sqrt(mx * mx + my * my);

    if (!finiteScalar(theta_d)) {
        return false;
    }
    if (theta_d > params_.max_theta_d) return false;

    if (theta_d > params_.min_valid_radius) {
        double theta = theta_d;
        bool converged = false;
        for (int i = 0; i < params_.newton_max_iterations; ++i) {
            const double f = distortedTheta(theta) - theta_d;
            const double fd = distortedThetaDerivative(theta);
            if (std::abs(fd) < 1e-15 || !finiteScalar(fd)) {
                return false;
            }
            const double step = f / fd;
            theta -= step;
            if (!finiteScalar(theta)) {
                return false;
            }
            if (std::abs(step) < params_.newton_tolerance) {
                converged = true;
                break;
            }
        }
        if (!converged &&
            std::abs(distortedTheta(theta) - theta_d) > 10.0 * params_.newton_tolerance) {
            return false;
        }
        const double radial = std::sin(theta);
        bearing = Eigen::Vector3d(mx * radial / theta_d,
                                  my * radial / theta_d,
                                  std::cos(theta));
        return finiteVector(bearing);
    }

    bearing = Eigen::Vector3d(0.0, 0.0, 1.0);
    const double n = bearing.norm();
    if (!finiteScalar(n) || n <= 0.0) {
        return false;
    }
    bearing /= n;
    return finiteVector(bearing);
}

bool KannalaBrandt8Camera::projectionJacobian(
    const Eigen::Vector3d& Pc,
    Eigen::Matrix<double, 2, 3>& J) const {
    J.setConstant(std::numeric_limits<double>::quiet_NaN());
    if (!valid_ || !finiteVector(Pc) || Pc.norm() <= params_.min_positive_z) {
        return false;
    }

    const double x2 = Pc.x() * Pc.x();
    const double y2 = Pc.y() * Pc.y();
    const double z2 = Pc.z() * Pc.z();
    const double r2 = x2 + y2;
    const double r = std::sqrt(r2);

    if (r < params_.min_valid_radius) {
        J.setZero();
        J(0, 0) = params_.fx / Pc.z();
        J(1, 1) = params_.fy / Pc.z();
        return true;
    }

    const double r3 = r2 * r;
    const double theta = std::atan2(r, Pc.z());
    const double f = distortedTheta(theta);
    const double fd = distortedThetaDerivative(theta);

    J(0, 0) = params_.fx *
              (fd * Pc.z() * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
    J(1, 0) = params_.fy *
              (fd * Pc.z() * Pc.y() * Pc.x() / (r2 * (r2 + z2)) -
               f * Pc.y() * Pc.x() / r3);
    J(0, 1) = params_.fx *
              (fd * Pc.z() * Pc.y() * Pc.x() / (r2 * (r2 + z2)) -
               f * Pc.y() * Pc.x() / r3);
    J(1, 1) = params_.fy *
              (fd * Pc.z() * y2 / (r2 * (r2 + z2)) + f * x2 / r3);
    J(0, 2) = -params_.fx * fd * Pc.x() / (r2 + z2);
    J(1, 2) = -params_.fy * fd * Pc.y() / (r2 + z2);
    return J.allFinite();
}

bool KannalaBrandt8Camera::isInImage(const Eigen::Vector2d& uv,
                                     double border) const {
    if (!valid_ || !finiteVector(uv) || !finiteScalar(border) || border < 0.0) {
        return false;
    }
    return uv.x() >= border && uv.y() >= border &&
           uv.x() < static_cast<double>(params_.width) - border &&
           uv.y() < static_cast<double>(params_.height) - border;
}

double KannalaBrandt8Camera::distortedTheta(double theta) const {
    const double theta2 = theta * theta;
    const double theta3 = theta * theta2;
    const double theta5 = theta3 * theta2;
    const double theta7 = theta5 * theta2;
    const double theta9 = theta7 * theta2;
    return theta + params_.k1 * theta3 + params_.k2 * theta5 +
           params_.k3 * theta7 + params_.k4 * theta9;
}

double KannalaBrandt8Camera::distortedThetaDerivative(double theta) const {
    const double theta2 = theta * theta;
    const double theta4 = theta2 * theta2;
    const double theta6 = theta4 * theta2;
    const double theta8 = theta4 * theta4;
    return 1.0 + 3.0 * params_.k1 * theta2 + 5.0 * params_.k2 * theta4 +
           7.0 * params_.k3 * theta6 + 9.0 * params_.k4 * theta8;
}

bool KannalaBrandt8Camera::finiteVector(const Eigen::Vector2d& v) const {
    return finiteScalar(v.x()) && finiteScalar(v.y());
}

bool KannalaBrandt8Camera::finiteVector(const Eigen::Vector3d& v) const {
    return finiteScalar(v.x()) && finiteScalar(v.y()) && finiteScalar(v.z());
}

bool KannalaBrandt8Camera::finiteScalar(double v) const {
    return std::isfinite(v);
}

}  // namespace StructureSLAM
