#ifndef FISHEYE_LINE_MATCHER_H
#define FISHEYE_LINE_MATCHER_H

#include <opencv2/core.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include <string>
#include <vector>

#include "FisheyeLineFrontend.h"

namespace StructureSLAM
{

class FisheyeLineMatcher
{
public:
    enum class Mode
    {
        DescriptorOnly,
        DescriptorAngleLength,
        FullGeometry
    };

    struct Options
    {
        double max_descriptor_distance = 80.0;
        double ratio_test_threshold = 0.80;
        bool mutual_check = true;
        double min_line_length_px = 30.0;
        double max_angle_difference_deg = 15.0;
        double min_length_ratio = 0.5;
        double max_length_ratio = 2.0;
        double max_midpoint_distance_px = 120.0;
        double max_symmetric_line_distance_px = 30.0;
        double min_overlap_ratio = 0.2;
    };

    struct ViewFeatures
    {
        int view_id = -1;
        std::vector<FisheyeLineFrontend::ViewLine> lines;
        std::vector<cv::line_descriptor::KeyLine> keylines;
        cv::Mat descriptors;
    };

    struct FrameFeatures
    {
        int frame_id = -1;
        double timestamp = 0.0;
        int orb_keypoints = 0;
        std::vector<ViewFeatures> views;
    };

    struct Match
    {
        int frame_id_1 = -1;
        int frame_id_2 = -1;
        double timestamp_1 = 0.0;
        double timestamp_2 = 0.0;
        int view_id = -1;
        int line_id_1 = -1;
        int line_id_2 = -1;
        int index_1 = -1;
        int index_2 = -1;
        double descriptor_distance = 0.0;
        double ratio_value = 1.0;
        bool mutual_valid = false;
        double angle_difference_deg = 0.0;
        double length_ratio = 0.0;
        double midpoint_distance_px = 0.0;
        double symmetric_line_distance_px = 0.0;
        double overlap_ratio = 0.0;
        double descriptor_score = 0.0;
        double angle_score = 0.0;
        double length_score = 0.0;
        double midpoint_score = 0.0;
        double overlap_score = 0.0;
        double final_match_score = 0.0;
        bool accepted = false;
        std::string rejection_reason;
    };

    struct PairResult
    {
        int frame_id_1 = -1;
        int frame_id_2 = -1;
        double timestamp_1 = 0.0;
        double timestamp_2 = 0.0;
        int line_count_1 = 0;
        int line_count_2 = 0;
        int raw_descriptor_matches = 0;
        int ratio_matches = 0;
        int mutual_matches = 0;
        int geometry_matches = 0;
        int final_matches = 0;
        double match_ratio_1 = 0.0;
        double match_ratio_2 = 0.0;
        double mean_descriptor_distance = 0.0;
        double median_descriptor_distance = 0.0;
        double mean_angle_difference_deg = 0.0;
        double mean_midpoint_displacement_px = 0.0;
        double mean_overlap_ratio = 0.0;
        double descriptor_time_ms = 0.0;
        double matching_time_ms = 0.0;
        double geometry_filter_time_ms = 0.0;
        std::vector<Match> matches;
    };

    FisheyeLineMatcher();
    explicit FisheyeLineMatcher(const Options& options);

    FrameFeatures BuildFrameFeatures(int frame_id,
                                     double timestamp,
                                     const FisheyeLineFrontend::FrameResult& frontend_result,
                                     int orb_keypoints) const;

    PairResult MatchPair(const FrameFeatures& first,
                         const FrameFeatures& second,
                         Mode mode) const;

    const Options& GetOptions() const { return options_; }

    static const char* ModeName(Mode mode);

private:
    std::vector<cv::line_descriptor::KeyLine>
    BuildKeyLines(const std::vector<FisheyeLineFrontend::ViewLine>& lines) const;

    Match EvaluateMatch(const FrameFeatures& first,
                        const FrameFeatures& second,
                        int view_id,
                        int index_1,
                        int index_2,
                        double descriptor_distance,
                        double ratio_value,
                        bool mutual_valid,
                        Mode mode) const;

    double DescriptorDistance(const cv::Mat& desc1, int idx1,
                              const cv::Mat& desc2, int idx2) const;
    double AngleDifferenceDeg(double a, double b) const;
    double SymmetricLineDistance(const cv::Vec4f& a, const cv::Vec4f& b) const;
    double ProjectionOverlapRatio(const cv::Vec4f& a, const cv::Vec4f& b) const;
    double LengthScore(double ratio) const;

    Options options_;
};

} // namespace StructureSLAM

#endif
