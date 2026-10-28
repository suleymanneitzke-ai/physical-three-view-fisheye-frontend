#include "FisheyeLineShadowMatcher.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace StructureSLAM
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

bool IsFinitePoint(const cv::Point2f& point)
{
    return std::isfinite(point.x) &&
           std::isfinite(point.y);
}

double PointDistance(
    const cv::Point2f& first,
    const cv::Point2f& second)
{
    const double dx =
        static_cast<double>(first.x) -
        static_cast<double>(second.x);

    const double dy =
        static_cast<double>(first.y) -
        static_cast<double>(second.y);

    return std::sqrt(dx * dx + dy * dy);
}

cv::Point2f Midpoint(
    const FisheyeLineObservation& observation)
{
    return cv::Point2f(
        0.5f * (
            observation.fisheye_start.x +
            observation.fisheye_end.x),
        0.5f * (
            observation.fisheye_start.y +
            observation.fisheye_end.y));
}

double Clamp(
    double value,
    double minimum,
    double maximum)
{
    return std::max(
        minimum,
        std::min(value, maximum));
}

std::vector<cv::Point2f> CollectValidSamples(
    const FisheyeLineObservation& observation)
{
    std::vector<cv::Point2f> samples;

    const std::size_t count =
        std::min(
            observation.fisheye_samples.size(),
            observation.sample_valid.size());

    samples.reserve(count);

    for(std::size_t i = 0; i < count; ++i)
    {
        if(observation.sample_valid[i] == 0)
            continue;

        const cv::Point2f& point =
            observation.fisheye_samples[i];

        if(IsFinitePoint(point))
            samples.push_back(point);
    }

    return samples;
}

double MeanNearestDistance(
    const std::vector<cv::Point2f>& source,
    const std::vector<cv::Point2f>& target)
{
    if(source.empty() || target.empty())
        return std::numeric_limits<double>::infinity();

    double sum = 0.0;

    for(const cv::Point2f& source_point : source)
    {
        double best =
            std::numeric_limits<double>::infinity();

        for(const cv::Point2f& target_point : target)
        {
            best = std::min(
                best,
                PointDistance(
                    source_point,
                    target_point));
        }

        sum += best;
    }

    return sum /
           static_cast<double>(source.size());
}

} // namespace

FisheyeLineShadowMatcher::FisheyeLineShadowMatcher()
    : options_(Options())
{
}

FisheyeLineShadowMatcher::FisheyeLineShadowMatcher(
    const Options& options)
    : options_(options)
{
}

FisheyeLineShadowMatcher::Result
FisheyeLineShadowMatcher::MatchAdjacentFrames(
    const std::vector<FisheyeLineObservation>& previous,
    const std::vector<FisheyeLineObservation>& current) const
{
    Result result;

    result.previous_count =
        static_cast<int>(previous.size());

    result.current_count =
        static_cast<int>(current.size());

    if(previous.empty() || current.empty())
        return result;

    const double infinity =
        std::numeric_limits<double>::infinity();

    std::vector<double> best_previous_cost(
        previous.size(),
        infinity);

    std::vector<int> best_previous_to_current(
        previous.size(),
        -1);

    std::vector<double> best_current_cost(
        current.size(),
        infinity);

    std::vector<int> best_current_to_previous(
        current.size(),
        -1);

    std::vector<std::vector<Candidate>> candidates(
        previous.size(),
        std::vector<Candidate>(current.size()));

    for(std::size_t previous_index = 0;
        previous_index < previous.size();
        ++previous_index)
    {
        for(std::size_t current_index = 0;
            current_index < current.size();
            ++current_index)
        {
            ++result.raw_candidate_count;

            Candidate candidate =
                EvaluateCandidate(
                    previous[previous_index],
                    current[current_index],
                    static_cast<int>(previous_index),
                    static_cast<int>(current_index));

            if(!candidate.valid)
                continue;

            ++result.gated_candidate_count;

            if(!std::isfinite(
                   candidate.match.total_cost))
            {
                ++result.nan_inf_count;
                continue;
            }

            candidates[previous_index][current_index] =
                candidate;

            if(candidate.match.total_cost <
               best_previous_cost[previous_index])
            {
                best_previous_cost[previous_index] =
                    candidate.match.total_cost;

                best_previous_to_current[previous_index] =
                    static_cast<int>(current_index);
            }

            if(candidate.match.total_cost <
               best_current_cost[current_index])
            {
                best_current_cost[current_index] =
                    candidate.match.total_cost;

                best_current_to_previous[current_index] =
                    static_cast<int>(previous_index);
            }
        }
    }

    std::set<int> used_previous;
    std::set<int> used_current;

    for(std::size_t previous_index = 0;
        previous_index < previous.size();
        ++previous_index)
    {
        const int current_index =
            best_previous_to_current[previous_index];

        if(current_index < 0)
            continue;

        if(best_current_to_previous[
               static_cast<std::size_t>(current_index)] !=
           static_cast<int>(previous_index))
        {
            continue;
        }

        const Candidate& candidate =
            candidates[previous_index][
                static_cast<std::size_t>(current_index)];

        if(!candidate.valid)
            continue;

        if(candidate.match.total_cost >
           options_.max_total_cost)
        {
            continue;
        }

        if(!used_previous.insert(
                static_cast<int>(previous_index)).second)
        {
            ++result.duplicate_previous_count;
            continue;
        }

        if(!used_current.insert(
                current_index).second)
        {
            ++result.duplicate_current_count;
            continue;
        }

        result.matches.push_back(
            candidate.match);
    }

    result.mutual_match_count =
        static_cast<int>(result.matches.size());

    return result;
}

FisheyeLineShadowMatcher::Candidate
FisheyeLineShadowMatcher::EvaluateCandidate(
    const FisheyeLineObservation& previous,
    const FisheyeLineObservation& current,
    int previous_index,
    int current_index) const
{
    Candidate candidate;

    if(options_.require_same_view &&
       previous.view_id != current.view_id)
    {
        return candidate;
    }

    if(previous.valid_sample_ratio <
           options_.min_valid_sample_ratio ||
       current.valid_sample_ratio <
           options_.min_valid_sample_ratio)
    {
        return candidate;
    }

    if(previous.virtual_length_px <
           options_.min_virtual_length_px ||
       current.virtual_length_px <
           options_.min_virtual_length_px)
    {
        return candidate;
    }

    if(!IsFinitePoint(previous.fisheye_start) ||
       !IsFinitePoint(previous.fisheye_end) ||
       !IsFinitePoint(current.fisheye_start) ||
       !IsFinitePoint(current.fisheye_end))
    {
        return candidate;
    }

    Match match;

    match.previous_index = previous_index;
    match.current_index = current_index;

    match.length_ratio =
        ComputeLengthRatio(
            previous,
            current);

    if(!std::isfinite(match.length_ratio) ||
       match.length_ratio <
           options_.min_length_ratio)
    {
        return candidate;
    }

    match.angle_deg =
        ComputeAngleDifferenceDeg(
            previous,
            current);

    match.midpoint_distance_px =
        ComputeMidpointDistance(
            previous,
            current);

    match.endpoint_distance_px =
        ComputeEndpointDistance(
            previous,
            current);

    match.sample_distance_px =
        ComputeSampleDistance(
            previous,
            current);

    if(!std::isfinite(match.angle_deg) ||
       !std::isfinite(match.midpoint_distance_px) ||
       !std::isfinite(match.endpoint_distance_px) ||
       !std::isfinite(match.sample_distance_px))
    {
        return candidate;
    }

    if(match.angle_deg >
           options_.max_angle_deg ||
       match.midpoint_distance_px >
           options_.max_midpoint_distance_px ||
       match.endpoint_distance_px >
           options_.max_endpoint_distance_px ||
       match.sample_distance_px >
           options_.max_sample_distance_px)
    {
        return candidate;
    }

    const double normalized_angle =
        match.angle_deg /
        options_.max_angle_deg;

    const double normalized_midpoint =
        match.midpoint_distance_px /
        options_.max_midpoint_distance_px;

    const double normalized_endpoint =
        match.endpoint_distance_px /
        options_.max_endpoint_distance_px;

    const double normalized_sample =
        match.sample_distance_px /
        options_.max_sample_distance_px;

    const double normalized_length =
        1.0 - match.length_ratio;

    match.total_cost =
        options_.angle_weight *
            normalized_angle +
        options_.midpoint_weight *
            normalized_midpoint +
        options_.endpoint_weight *
            normalized_endpoint +
        options_.sample_weight *
            normalized_sample +
        options_.length_weight *
            normalized_length;

    if(!std::isfinite(match.total_cost))
        return candidate;

    candidate.valid = true;
    candidate.match = match;

    return candidate;
}

double
FisheyeLineShadowMatcher::ComputeAngleDifferenceDeg(
    const FisheyeLineObservation& first,
    const FisheyeLineObservation& second)
{
    const double first_dx =
        static_cast<double>(
            first.fisheye_end.x -
            first.fisheye_start.x);

    const double first_dy =
        static_cast<double>(
            first.fisheye_end.y -
            first.fisheye_start.y);

    const double second_dx =
        static_cast<double>(
            second.fisheye_end.x -
            second.fisheye_start.x);

    const double second_dy =
        static_cast<double>(
            second.fisheye_end.y -
            second.fisheye_start.y);

    const double first_norm =
        std::sqrt(
            first_dx * first_dx +
            first_dy * first_dy);

    const double second_norm =
        std::sqrt(
            second_dx * second_dx +
            second_dy * second_dy);

    if(first_norm <= 1e-9 ||
       second_norm <= 1e-9)
    {
        return std::numeric_limits<double>::infinity();
    }

    double cosine =
        (
            first_dx * second_dx +
            first_dy * second_dy
        ) /
        (first_norm * second_norm);

    cosine = std::abs(
        Clamp(cosine, -1.0, 1.0));

    return std::acos(cosine) *
           180.0 / kPi;
}

double
FisheyeLineShadowMatcher::ComputeMidpointDistance(
    const FisheyeLineObservation& first,
    const FisheyeLineObservation& second)
{
    return PointDistance(
        Midpoint(first),
        Midpoint(second));
}

double
FisheyeLineShadowMatcher::ComputeEndpointDistance(
    const FisheyeLineObservation& first,
    const FisheyeLineObservation& second)
{
    const double direct =
        0.5 * (
            PointDistance(
                first.fisheye_start,
                second.fisheye_start) +
            PointDistance(
                first.fisheye_end,
                second.fisheye_end));

    const double reversed =
        0.5 * (
            PointDistance(
                first.fisheye_start,
                second.fisheye_end) +
            PointDistance(
                first.fisheye_end,
                second.fisheye_start));

    return std::min(direct, reversed);
}

double
FisheyeLineShadowMatcher::ComputeSampleDistance(
    const FisheyeLineObservation& first,
    const FisheyeLineObservation& second)
{
    const std::vector<cv::Point2f> first_samples =
        CollectValidSamples(first);

    const std::vector<cv::Point2f> second_samples =
        CollectValidSamples(second);

    if(first_samples.empty() ||
       second_samples.empty())
    {
        return std::numeric_limits<double>::infinity();
    }

    const double forward =
        MeanNearestDistance(
            first_samples,
            second_samples);

    const double backward =
        MeanNearestDistance(
            second_samples,
            first_samples);

    return 0.5 * (forward + backward);
}

double
FisheyeLineShadowMatcher::ComputeLengthRatio(
    const FisheyeLineObservation& first,
    const FisheyeLineObservation& second)
{
    const double minimum =
        std::min(
            first.virtual_length_px,
            second.virtual_length_px);

    const double maximum =
        std::max(
            first.virtual_length_px,
            second.virtual_length_px);

    if(maximum <= 1e-9)
        return 0.0;

    return minimum / maximum;
}

} // namespace StructureSLAM
