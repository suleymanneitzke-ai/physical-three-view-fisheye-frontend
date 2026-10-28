#include "FisheyeLineMatcher.h"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace StructureSLAM
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

double NowMs()
{
    using Clock = std::chrono::steady_clock;
    static const Clock::time_point start = Clock::now();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

cv::Point2f MidPoint(const cv::Vec4f& line)
{
    return cv::Point2f(0.5f * (line[0] + line[2]), 0.5f * (line[1] + line[3]));
}

double LineLength(const cv::Vec4f& line)
{
    const double dx = line[2] - line[0];
    const double dy = line[3] - line[1];
    return std::sqrt(dx * dx + dy * dy);
}

double Mean(const std::vector<double>& values)
{
    if(values.empty())
        return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double Median(std::vector<double> values)
{
    if(values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if(values.size() % 2 == 0)
        return 0.5 * (values[mid - 1] + values[mid]);
    return values[mid];
}

double Clamp01(double value)
{
    return std::max(0.0, std::min(1.0, value));
}

double PointLineDistance(const cv::Point2f& p, const cv::Vec4f& line)
{
    const double x1 = line[0];
    const double y1 = line[1];
    const double x2 = line[2];
    const double y2 = line[3];
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    const double denom = std::sqrt(dx * dx + dy * dy);
    if(denom < 1e-6)
        return std::numeric_limits<double>::infinity();
    return std::fabs(dy * p.x - dx * p.y + x2 * y1 - y2 * x1) / denom;
}

} // namespace

FisheyeLineMatcher::FisheyeLineMatcher()
    : options_(Options())
{
}

FisheyeLineMatcher::FisheyeLineMatcher(const Options& options)
    : options_(options)
{
}

const char* FisheyeLineMatcher::ModeName(Mode mode)
{
    switch(mode)
    {
        case Mode::DescriptorOnly: return "M0-A";
        case Mode::DescriptorAngleLength: return "M0-B";
        case Mode::FullGeometry: return "M0-C";
        default: return "unknown";
    }
}

std::vector<cv::line_descriptor::KeyLine>
FisheyeLineMatcher::BuildKeyLines(const std::vector<FisheyeLineFrontend::ViewLine>& lines) const
{
    std::vector<cv::line_descriptor::KeyLine> keylines;
    keylines.reserve(lines.size());
    for(size_t i = 0; i < lines.size(); ++i)
    {
        const FisheyeLineFrontend::ViewLine& line = lines[i];
        cv::line_descriptor::KeyLine kl;
        kl.angle = static_cast<float>(line.virtual_angle_deg * kPi / 180.0);
        kl.class_id = static_cast<int>(i);
        kl.octave = 0;
        kl.pt = MidPoint(line.virtual_line);
        kl.response = static_cast<float>(line.response);
        kl.size = static_cast<float>(line.virtual_length_px);
        kl.startPointX = line.virtual_line[0];
        kl.startPointY = line.virtual_line[1];
        kl.endPointX = line.virtual_line[2];
        kl.endPointY = line.virtual_line[3];
        kl.sPointInOctaveX = line.virtual_line[0];
        kl.sPointInOctaveY = line.virtual_line[1];
        kl.ePointInOctaveX = line.virtual_line[2];
        kl.ePointInOctaveY = line.virtual_line[3];
        kl.lineLength = static_cast<float>(line.virtual_length_px);
        kl.numOfPixels = static_cast<int>(std::max(1.0, line.virtual_length_px));
        keylines.push_back(kl);
    }
    return keylines;
}

FisheyeLineMatcher::FrameFeatures
FisheyeLineMatcher::BuildFrameFeatures(int frame_id,
                                       double timestamp,
                                       const FisheyeLineFrontend::FrameResult& frontend_result,
                                       int orb_keypoints) const
{
    FrameFeatures features;
    features.frame_id = frame_id;
    features.timestamp = timestamp;
    features.orb_keypoints = orb_keypoints;
    features.views.resize(frontend_result.lines.size());

    cv::Ptr<cv::line_descriptor::BinaryDescriptor> descriptor =
        cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();

    for(size_t view_id = 0; view_id < frontend_result.lines.size(); ++view_id)
    {
        ViewFeatures& view = features.views[view_id];
        view.view_id = static_cast<int>(view_id);
        view.lines = frontend_result.lines[view_id];
        view.keylines = BuildKeyLines(view.lines);
        if(view_id < frontend_result.virtual_images.size() &&
           !view.keylines.empty() && !frontend_result.virtual_images[view_id].empty())
        {
            descriptor->compute(frontend_result.virtual_images[view_id], view.keylines, view.descriptors);
            std::vector<FisheyeLineFrontend::ViewLine> kept_lines;
            kept_lines.reserve(view.keylines.size());
            for(const cv::line_descriptor::KeyLine& keyline : view.keylines)
            {
                if(keyline.class_id >= 0 && keyline.class_id < static_cast<int>(view.lines.size()))
                    kept_lines.push_back(view.lines[keyline.class_id]);
            }
            if(static_cast<int>(kept_lines.size()) == view.descriptors.rows)
                view.lines.swap(kept_lines);
        }
    }
    return features;
}

double FisheyeLineMatcher::DescriptorDistance(const cv::Mat& desc1, int idx1,
                                              const cv::Mat& desc2, int idx2) const
{
    if(desc1.empty() || desc2.empty() || idx1 < 0 || idx2 < 0 ||
       idx1 >= desc1.rows || idx2 >= desc2.rows)
        return std::numeric_limits<double>::infinity();
    return cv::norm(desc1.row(idx1), desc2.row(idx2), cv::NORM_HAMMING);
}

double FisheyeLineMatcher::AngleDifferenceDeg(double a, double b) const
{
    double diff = std::fabs(a - b);
    while(diff > 180.0)
        diff -= 180.0;
    if(diff > 90.0)
        diff = 180.0 - diff;
    return std::fabs(diff);
}

double FisheyeLineMatcher::SymmetricLineDistance(const cv::Vec4f& a, const cv::Vec4f& b) const
{
    const cv::Point2f a1(a[0], a[1]);
    const cv::Point2f a2(a[2], a[3]);
    const cv::Point2f b1(b[0], b[1]);
    const cv::Point2f b2(b[2], b[3]);
    const double d = PointLineDistance(a1, b) + PointLineDistance(a2, b) +
                     PointLineDistance(b1, a) + PointLineDistance(b2, a);
    return 0.25 * d;
}

double FisheyeLineMatcher::ProjectionOverlapRatio(const cv::Vec4f& a, const cv::Vec4f& b) const
{
    const double ax = a[2] - a[0];
    const double ay = a[3] - a[1];
    const double len = std::sqrt(ax * ax + ay * ay);
    if(len < 1e-6)
        return 0.0;
    const double ux = ax / len;
    const double uy = ay / len;
    const cv::Point2f origin(a[0], a[1]);
    auto project = [&](const cv::Point2f& p)
    {
        return (p.x - origin.x) * ux + (p.y - origin.y) * uy;
    };

    const double a0 = 0.0;
    const double a1 = len;
    const double b0 = project(cv::Point2f(b[0], b[1]));
    const double b1 = project(cv::Point2f(b[2], b[3]));
    const double bmin = std::min(b0, b1);
    const double bmax = std::max(b0, b1);
    const double inter = std::max(0.0, std::min(a1, bmax) - std::max(a0, bmin));
    const double min_len = std::max(1e-6, std::min(LineLength(a), LineLength(b)));
    return Clamp01(inter / min_len);
}

double FisheyeLineMatcher::LengthScore(double ratio) const
{
    if(ratio <= 0.0)
        return 0.0;
    const double balanced = ratio < 1.0 ? ratio : 1.0 / ratio;
    return Clamp01((balanced - options_.min_length_ratio) / (1.0 - options_.min_length_ratio));
}

FisheyeLineMatcher::Match
FisheyeLineMatcher::EvaluateMatch(const FrameFeatures& first,
                                  const FrameFeatures& second,
                                  int view_id,
                                  int index_1,
                                  int index_2,
                                  double descriptor_distance,
                                  double ratio_value,
                                  bool mutual_valid,
                                  Mode mode) const
{
    Match match;
    match.frame_id_1 = first.frame_id;
    match.frame_id_2 = second.frame_id;
    match.timestamp_1 = first.timestamp;
    match.timestamp_2 = second.timestamp;
    match.view_id = view_id;
    match.index_1 = index_1;
    match.index_2 = index_2;
    match.descriptor_distance = descriptor_distance;
    match.ratio_value = ratio_value;
    match.mutual_valid = mutual_valid;

    if(view_id < 0 ||
       view_id >= static_cast<int>(first.views.size()) ||
       view_id >= static_cast<int>(second.views.size()))
    {
        match.accepted = false;
        match.rejection_reason = "view_id";
        return match;
    }

    const FisheyeLineFrontend::ViewLine& l1 = first.views[view_id].lines[index_1];
    const FisheyeLineFrontend::ViewLine& l2 = second.views[view_id].lines[index_2];
    match.line_id_1 = l1.line_id;
    match.line_id_2 = l2.line_id;
    if(l1.view_id != l2.view_id)
    {
        match.accepted = false;
        match.rejection_reason = "same_view";
        return match;
    }

    match.angle_difference_deg = AngleDifferenceDeg(l1.virtual_angle_deg, l2.virtual_angle_deg);
    match.length_ratio = l2.virtual_length_px > 1e-6 ? l1.virtual_length_px / l2.virtual_length_px : 0.0;
    const cv::Point2f m1 = MidPoint(l1.virtual_line);
    const cv::Point2f m2 = MidPoint(l2.virtual_line);
    match.midpoint_distance_px = cv::norm(m1 - m2);
    match.symmetric_line_distance_px = SymmetricLineDistance(l1.virtual_line, l2.virtual_line);
    match.overlap_ratio = ProjectionOverlapRatio(l1.virtual_line, l2.virtual_line);

    match.descriptor_score = Clamp01(1.0 - descriptor_distance / options_.max_descriptor_distance);
    match.angle_score = Clamp01(1.0 - match.angle_difference_deg / options_.max_angle_difference_deg);
    match.length_score = LengthScore(match.length_ratio);
    match.midpoint_score = Clamp01(1.0 - match.midpoint_distance_px / options_.max_midpoint_distance_px);
    match.overlap_score = Clamp01(match.overlap_ratio);
    match.final_match_score = Clamp01(0.30 * match.descriptor_score +
                                      0.20 * match.angle_score +
                                      0.15 * match.length_score +
                                      0.15 * match.midpoint_score +
                                      0.20 * match.overlap_score);

    match.accepted = true;
    match.rejection_reason = "accepted";
    if(descriptor_distance > options_.max_descriptor_distance)
    {
        match.accepted = false;
        match.rejection_reason = "descriptor_distance";
    }
    else if(ratio_value > options_.ratio_test_threshold)
    {
        match.accepted = false;
        match.rejection_reason = "ratio";
    }
    else if(options_.mutual_check && !mutual_valid)
    {
        match.accepted = false;
        match.rejection_reason = "mutual";
    }
    else if(mode != Mode::DescriptorOnly &&
            match.angle_difference_deg > options_.max_angle_difference_deg)
    {
        match.accepted = false;
        match.rejection_reason = "angle";
    }
    else if(mode != Mode::DescriptorOnly &&
            (match.length_ratio < options_.min_length_ratio ||
             match.length_ratio > options_.max_length_ratio))
    {
        match.accepted = false;
        match.rejection_reason = "length";
    }
    else if(mode == Mode::FullGeometry &&
            match.midpoint_distance_px > options_.max_midpoint_distance_px)
    {
        match.accepted = false;
        match.rejection_reason = "midpoint";
    }
    else if(mode == Mode::FullGeometry &&
            match.symmetric_line_distance_px > options_.max_symmetric_line_distance_px)
    {
        match.accepted = false;
        match.rejection_reason = "line_distance";
    }
    else if(mode == Mode::FullGeometry &&
            match.overlap_ratio < options_.min_overlap_ratio)
    {
        match.accepted = false;
        match.rejection_reason = "overlap";
    }

    return match;
}

FisheyeLineMatcher::PairResult
FisheyeLineMatcher::MatchPair(const FrameFeatures& first,
                              const FrameFeatures& second,
                              Mode mode) const
{
    PairResult result;
    result.frame_id_1 = first.frame_id;
    result.frame_id_2 = second.frame_id;
    result.timestamp_1 = first.timestamp;
    result.timestamp_2 = second.timestamp;

    const double matching_start = NowMs();
    std::vector<double> descriptor_distances;
    std::vector<double> accepted_distances;
    std::vector<double> accepted_angles;
    std::vector<double> accepted_midpoints;
    std::vector<double> accepted_overlaps;

    const size_t view_count = std::min(first.views.size(), second.views.size());
    for(size_t view_index = 0; view_index < view_count; ++view_index)
    {
        const int view_id = static_cast<int>(view_index);
        const ViewFeatures& v1 = first.views[view_index];
        const ViewFeatures& v2 = second.views[view_index];
        result.line_count_1 += static_cast<int>(v1.lines.size());
        result.line_count_2 += static_cast<int>(v2.lines.size());
        if(v1.descriptors.empty() || v2.descriptors.empty())
            continue;

        std::vector<int> reverse_best(v2.descriptors.rows, -1);
        std::vector<double> reverse_best_dist(v2.descriptors.rows, std::numeric_limits<double>::infinity());
        for(int j = 0; j < v2.descriptors.rows; ++j)
        {
            for(int i = 0; i < v1.descriptors.rows; ++i)
            {
                if(i >= static_cast<int>(v1.lines.size()) ||
                   j >= static_cast<int>(v2.lines.size()) ||
                   v1.lines[i].view_id != v2.lines[j].view_id)
                    continue;
                const double d = DescriptorDistance(v1.descriptors, i, v2.descriptors, j);
                if(d < reverse_best_dist[j])
                {
                    reverse_best_dist[j] = d;
                    reverse_best[j] = i;
                }
            }
        }

        for(int i = 0; i < v1.descriptors.rows; ++i)
        {
            double best = std::numeric_limits<double>::infinity();
            double second_best = std::numeric_limits<double>::infinity();
            int best_j = -1;
            for(int j = 0; j < v2.descriptors.rows; ++j)
            {
                if(i >= static_cast<int>(v1.lines.size()) ||
                   j >= static_cast<int>(v2.lines.size()) ||
                   v1.lines[i].view_id != v2.lines[j].view_id)
                    continue;
                const double d = DescriptorDistance(v1.descriptors, i, v2.descriptors, j);
                if(d < best)
                {
                    second_best = best;
                    best = d;
                    best_j = j;
                }
                else if(d < second_best)
                {
                    second_best = d;
                }
            }

            if(best_j < 0 || !std::isfinite(best))
                continue;

            ++result.raw_descriptor_matches;
            descriptor_distances.push_back(best);
            const double ratio = std::isfinite(second_best) && second_best > 1e-6 ? best / second_best : 1.0;
            const bool ratio_ok = ratio <= options_.ratio_test_threshold;
            if(ratio_ok)
                ++result.ratio_matches;
            const bool mutual = reverse_best[best_j] == i;
            if(ratio_ok && mutual)
                ++result.mutual_matches;

            Match match = EvaluateMatch(first, second, view_id, i, best_j, best, ratio, mutual, mode);
            if(match.rejection_reason != "descriptor_distance" &&
               match.rejection_reason != "ratio" &&
               match.rejection_reason != "mutual")
                ++result.geometry_matches;
            if(match.accepted)
            {
                ++result.final_matches;
                accepted_distances.push_back(match.descriptor_distance);
                accepted_angles.push_back(match.angle_difference_deg);
                accepted_midpoints.push_back(match.midpoint_distance_px);
                accepted_overlaps.push_back(match.overlap_ratio);
            }
            result.matches.push_back(match);
        }
    }

    result.matching_time_ms = NowMs() - matching_start;
    result.geometry_filter_time_ms = result.matching_time_ms;
    result.match_ratio_1 = result.line_count_1 > 0 ? static_cast<double>(result.final_matches) / result.line_count_1 : 0.0;
    result.match_ratio_2 = result.line_count_2 > 0 ? static_cast<double>(result.final_matches) / result.line_count_2 : 0.0;
    result.mean_descriptor_distance = Mean(accepted_distances.empty() ? descriptor_distances : accepted_distances);
    result.median_descriptor_distance = Median(accepted_distances.empty() ? descriptor_distances : accepted_distances);
    result.mean_angle_difference_deg = Mean(accepted_angles);
    result.mean_midpoint_displacement_px = Mean(accepted_midpoints);
    result.mean_overlap_ratio = Mean(accepted_overlaps);
    return result;
}

} // namespace StructureSLAM
