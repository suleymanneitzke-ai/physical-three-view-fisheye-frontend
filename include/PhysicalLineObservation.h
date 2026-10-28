#ifndef STRUCTURE_SLAM_PHYSICAL_LINE_OBSERVATION_H
#define STRUCTURE_SLAM_PHYSICAL_LINE_OBSERVATION_H

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <cstddef>
#include <vector>

namespace StructureSLAM {

struct PhysicalLineObservationMember {
    int view_id = -1;
    int source_line_id = -1;
    std::size_t virtual_observation_index = static_cast<std::size_t>(-1);
    Eigen::Vector3d homogeneous_line = Eigen::Vector3d::Zero();
    cv::Point2f p_start = cv::Point2f(-1.f, -1.f);
    cv::Point2f p_end = cv::Point2f(-1.f, -1.f);
    int virtual_width = 0;
    int virtual_height = 0;
    std::vector<cv::Point2f> internal_points;
    int visible_support_count = 0;
    double border_margin_px = 0.0;
    bool start_mask_valid = false;
    bool end_mask_valid = false;
    bool visible_support_valid = false;
};

struct PhysicalLineObservation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum State { SINGLE_VIEW, TWO_VIEW, THREE_VIEW, AMBIGUOUS, CONFLICT };
    unsigned long physical_frame_id = static_cast<unsigned long>(-1);
    double timestamp = 0.0;
    int physical_line_observation_id = -1;
    State ambiguity_state = SINGLE_VIEW;
    Eigen::Vector3d physical_plane = Eigen::Vector3d::Zero();
    std::vector<PhysicalLineObservationMember> members;
    int representative_member_index = -1;
    bool plane_valid = false;
    bool production_mask_consumed = false;
    bool conditional_80deg_used = false;
};

} // namespace StructureSLAM
#endif
