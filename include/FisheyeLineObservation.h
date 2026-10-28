#ifndef STRUCTURE_SLAM_FISHEYE_LINE_OBSERVATION_H
#define STRUCTURE_SLAM_FISHEYE_LINE_OBSERVATION_H

#include <opencv2/core.hpp>

#include <limits>
#include <vector>

namespace StructureSLAM
{

// A lightweight 2D line observation in the original monocular
// fisheye image coordinate system.
//
// This structure is deliberately independent of KeyLine, MapLine,
// descriptors, matching and optimization.
struct FisheyeLineObservation
{
    static constexpr long unsigned int kInvalidTrackId =
        std::numeric_limits<long unsigned int>::max();

    // Native observation identity. native_line_id is stable inside the
    // original fisheye frame; mapline_id is filled only after association
    // with a formal MapLine. A value of -1 means "not assigned".
    int native_line_id = -1;
    long unsigned int temporal_track_id = kInvalidTrackId;
    long unsigned int mapline_id = static_cast<long unsigned int>(-1);

    long unsigned int frame_id = static_cast<long unsigned int>(-1);
    long unsigned int keyframe_id = static_cast<long unsigned int>(-1);
    double timestamp = 0.0;

    int view_id = -1;
    int line_id = -1;
    int duplicate_group_id = -1;

    // Raw coordinates are in the original fisheye image. Virtual coordinates
    // are in the selected perspective virtual view. Both are pixel units.
    cv::Point2f fisheye_start = cv::Point2f(-1.0f, -1.0f);
    cv::Point2f fisheye_end = cv::Point2f(-1.0f, -1.0f);
    cv::Point2f virtual_start = cv::Point2f(-1.0f, -1.0f);
    cv::Point2f virtual_end = cv::Point2f(-1.0f, -1.0f);

    double virtual_length_px = 0.0;
    double virtual_angle_deg = 0.0;
    double response = 0.0;
    double width = 0.0;
    double precision = 0.0;
    double sigma_px = 0.5;
    double information = 4.0;
    double valid_sample_ratio = 0.0;
    bool inlier = true;
    bool outlier = false;
    bool descriptor_valid = false;

    std::vector<cv::Point2f> fisheye_samples;
    std::vector<cv::Point2f> virtual_samples;
    std::vector<unsigned char> sample_valid;
    cv::Mat lbd_descriptor;
};

struct NativeFisheyeLineProjectionDiagnostic
{
    bool attempted = false;
    bool visible = false;
    bool matched = false;
    bool temporal_track_compatible = false;
    bool descriptor_compatible = false;

    long unsigned int mapline_id = static_cast<long unsigned int>(-1);
    int view_id = -1;

    cv::Point2f projected_fisheye_start = cv::Point2f(-1.0f, -1.0f);
    cv::Point2f projected_fisheye_end = cv::Point2f(-1.0f, -1.0f);
    cv::Point2f projected_virtual_start = cv::Point2f(-1.0f, -1.0f);
    cv::Point2f projected_virtual_end = cv::Point2f(-1.0f, -1.0f);

    double endpoint_depth_min = -1.0;
    double range_m = -1.0;
    double angle_difference_deg = -1.0;
    double overlap_ratio = 0.0;
    double descriptor_distance = -1.0;
    double score = 0.0;
};

} // namespace StructureSLAM

#endif
