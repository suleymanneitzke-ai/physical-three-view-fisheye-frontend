#ifndef STRUCTURE_SLAM_JOINT_POINT_LINE_DIAGNOSTICS_H
#define STRUCTURE_SLAM_JOINT_POINT_LINE_DIAGNOSTICS_H

#include <Eigen/Core>

#include <string>
#include <vector>

namespace StructureSLAM {

struct ResidualDistribution {
    int count = 0;
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double max = 0.0;
};

struct HessianDiagnostics {
    std::vector<double> eigenvalues;
    int rank = 0;
    double condition = 0.0;
    double trace = 0.0;
    double log_determinant = 0.0;
    Eigen::Matrix<double, 6, 1> weakest_eigenvector =
        Eigen::Matrix<double, 6, 1>::Zero();
};

struct GradientConflictDiagnostics {
    double full_cosine = 0.0;
    double rotation_cosine = 0.0;
    double translation_cosine = 0.0;
    double point_gradient_norm = 0.0;
    double line_gradient_norm = 0.0;
    double scaled_line_gradient_norm = 0.0;
    double gradient_norm_ratio = 0.0;
};

struct JointCandidateClassification {
    std::string class_name;
    std::string reason;
};

}  // namespace StructureSLAM

#endif  // STRUCTURE_SLAM_JOINT_POINT_LINE_DIAGNOSTICS_H
