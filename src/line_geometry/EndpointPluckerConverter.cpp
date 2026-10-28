#include "line_geometry/EndpointPluckerConverter.h"

#include <cmath>
#include <Eigen/Geometry>

namespace StructureSLAM {

EndpointPluckerConverter::EndpointPluckerConverter(double min_length,
                                                   double max_orthogonality)
    : min_length_(min_length), max_orthogonality_(max_orthogonality) {}

bool EndpointPluckerConverter::convert(const Eigen::Vector3d& start,
                                       const Eigen::Vector3d& end,
                                       PluckerLineNV& line,
                                       std::string& reject_reason) const {
    line = PluckerLineNV();
    reject_reason.clear();
    if (!finiteVector(start) || !finiteVector(end)) {
        reject_reason = "NON_FINITE_ENDPOINT";
        return false;
    }
    const Eigen::Vector3d d = end - start;
    const double length = d.norm();
    if (!std::isfinite(length) || length < min_length_) {
        reject_reason = "SHORT_OR_INVALID_SEGMENT";
        return false;
    }

    line.v = d / length;
    line.n = start.cross(line.v);
    line.length = length;
    line.orthogonality = std::abs(line.n.dot(line.v));
    if (!finiteVector(line.n) || !finiteVector(line.v) ||
        !std::isfinite(line.orthogonality)) {
        reject_reason = "NON_FINITE_PLUCKER";
        return false;
    }
    if (line.orthogonality > max_orthogonality_) {
        reject_reason = "ORTHOGONALITY_FAILURE";
        return false;
    }
    return true;
}

bool EndpointPluckerConverter::finiteVector(const Eigen::Vector3d& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

}  // namespace StructureSLAM
