#ifndef STRUCTURE_SLAM_ENDPOINT_PLUCKER_CONVERTER_H
#define STRUCTURE_SLAM_ENDPOINT_PLUCKER_CONVERTER_H

#include <Eigen/Core>
#include <string>

namespace StructureSLAM {

struct PluckerLineNV {
    Eigen::Vector3d n = Eigen::Vector3d::Zero();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    double length = 0.0;
    double orthogonality = 0.0;
};

class EndpointPluckerConverter {
public:
    explicit EndpointPluckerConverter(double min_length = 1e-8,
                                      double max_orthogonality = 1e-10);

    bool convert(const Eigen::Vector3d& start,
                 const Eigen::Vector3d& end,
                 PluckerLineNV& line,
                 std::string& reject_reason) const;

    double minLength() const { return min_length_; }
    double maxOrthogonality() const { return max_orthogonality_; }

private:
    static bool finiteVector(const Eigen::Vector3d& v);

    double min_length_;
    double max_orthogonality_;
};

}  // namespace StructureSLAM

#endif  // STRUCTURE_SLAM_ENDPOINT_PLUCKER_CONVERTER_H
