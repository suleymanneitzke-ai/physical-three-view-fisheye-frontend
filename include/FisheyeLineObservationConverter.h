#ifndef STRUCTURE_SLAM_FISHEYE_LINE_OBSERVATION_CONVERTER_H
#define STRUCTURE_SLAM_FISHEYE_LINE_OBSERVATION_CONVERTER_H

#include "FisheyeLineFrontend.h"
#include "FisheyeLineObservation.h"

#include <utility>
#include <vector>

namespace StructureSLAM
{

inline std::vector<FisheyeLineObservation>
ConvertFisheyeLineObservations(
    const FisheyeLineFrontend::FrameResult& result)
{
    std::vector<FisheyeLineObservation> observations;

    observations.reserve(
        static_cast<std::size_t>(
            result.filtered_line_count_total));

    for(size_t view_id = 0; view_id < result.lines.size(); ++view_id)
    {
        for(const FisheyeLineFrontend::ViewLine& line :
            result.lines[view_id])
        {
            FisheyeLineObservation observation;

            observation.view_id = line.view_id;
            observation.line_id = line.line_id;
            observation.native_line_id = line.line_id;
            observation.duplicate_group_id =
                line.duplicate_group_id;

            observation.fisheye_start =
                line.fisheye_start;
            observation.fisheye_end =
                line.fisheye_end;
            observation.virtual_start =
                cv::Point2f(line.virtual_line[0],
                            line.virtual_line[1]);
            observation.virtual_end =
                cv::Point2f(line.virtual_line[2],
                            line.virtual_line[3]);

            observation.virtual_length_px =
                line.virtual_length_px;
            observation.virtual_angle_deg =
                line.virtual_angle_deg;
            observation.response =
                line.response;
            observation.width =
                line.width;
            observation.precision =
                line.precision;
            observation.valid_sample_ratio =
                line.valid_sample_ratio;

            observation.fisheye_samples =
                line.fisheye_samples;
            observation.virtual_samples =
                line.virtual_samples;
            observation.sample_valid =
                line.sample_valid;

            observations.push_back(
                std::move(observation));
        }
    }

    return observations;
}

} // namespace StructureSLAM

#endif
