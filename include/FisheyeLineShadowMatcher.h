#ifndef STRUCTURE_SLAM_FISHEYE_LINE_SHADOW_MATCHER_H
#define STRUCTURE_SLAM_FISHEYE_LINE_SHADOW_MATCHER_H

#include "FisheyeLineObservation.h"

#include <cstddef>
#include <vector>

namespace StructureSLAM
{

class FisheyeLineShadowMatcher
{
public:
    struct Options
    {
        // Only match observations extracted from the same virtual view.
        bool require_same_view = true;

        double min_valid_sample_ratio = 0.60;
        double min_virtual_length_px = 15.0;
        double min_length_ratio = 0.50;

        double max_angle_deg = 15.0;
        double max_midpoint_distance_px = 50.0;
        double max_endpoint_distance_px = 50.0;
        double max_sample_distance_px = 15.0;

        // Normalized weighted matching cost.
        double angle_weight = 0.25;
        double midpoint_weight = 0.20;
        double endpoint_weight = 0.20;
        double sample_weight = 0.25;
        double length_weight = 0.10;

        double max_total_cost = 1.0;
    };

    struct Match
    {
        int previous_index = -1;
        int current_index = -1;

        double total_cost = 0.0;
        double angle_deg = 0.0;
        double midpoint_distance_px = 0.0;
        double endpoint_distance_px = 0.0;
        double sample_distance_px = 0.0;
        double length_ratio = 0.0;
    };

    struct Result
    {
        int previous_count = 0;
        int current_count = 0;

        int raw_candidate_count = 0;
        int gated_candidate_count = 0;
        int mutual_match_count = 0;

        int nan_inf_count = 0;
        int duplicate_previous_count = 0;
        int duplicate_current_count = 0;

        std::vector<Match> matches;
    };

    FisheyeLineShadowMatcher();

    explicit FisheyeLineShadowMatcher(
        const Options& options);

    Result MatchAdjacentFrames(
        const std::vector<FisheyeLineObservation>& previous,
        const std::vector<FisheyeLineObservation>& current) const;

    const Options& GetOptions() const
    {
        return options_;
    }

private:
    struct Candidate
    {
        bool valid = false;
        Match match;
    };

    Candidate EvaluateCandidate(
        const FisheyeLineObservation& previous,
        const FisheyeLineObservation& current,
        int previous_index,
        int current_index) const;

    static double ComputeAngleDifferenceDeg(
        const FisheyeLineObservation& first,
        const FisheyeLineObservation& second);

    static double ComputeMidpointDistance(
        const FisheyeLineObservation& first,
        const FisheyeLineObservation& second);

    static double ComputeEndpointDistance(
        const FisheyeLineObservation& first,
        const FisheyeLineObservation& second);

    static double ComputeSampleDistance(
        const FisheyeLineObservation& first,
        const FisheyeLineObservation& second);

    static double ComputeLengthRatio(
        const FisheyeLineObservation& first,
        const FisheyeLineObservation& second);

    Options options_;
};

} // namespace StructureSLAM

#endif
