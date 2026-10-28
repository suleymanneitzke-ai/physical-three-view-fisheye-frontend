#ifndef STRUCTURE_SLAM_PHYSICAL_LINE_OBSERVATION_BUILDER_H
#define STRUCTURE_SLAM_PHYSICAL_LINE_OBSERVATION_BUILDER_H

#include "FisheyeLineFrontend.h"
#include "PhysicalLineObservation.h"
#include <array>

namespace StructureSLAM {

class PhysicalLineObservationBuilder {
public:
    struct Stats {
        int v0=0,v1=0,v2=0,physical=0,single=0,two=0,three=0,ambiguous=0,conflict=0;
        long long tested40=0,accepted40=0,eligible80=0,skipped80=0,mask_queries=0;
    };
    struct Evidence {
        bool valid=false,common_visible=false;
        double line_angle_deg=0,line_offset_px=0,sample_median_px=0,sample_p90_px=0;
        double visible_overlap=0,plane_angle_deg=0;
        int valid_samples=0,mask_queries=0;
    };
    static std::vector<PhysicalLineObservation> Build(
        const FisheyeLineFrontend::FrameResult& frame,
        const std::vector<cv::Mat>& production_masks,
        unsigned long physical_frame_id,double timestamp,Stats* stats=0);
    static Evidence Compare(const FisheyeLineFrontend::ViewLine& a,
                            const FisheyeLineFrontend::ViewLine& b,
                            const cv::Mat& target_mask);
    static void AccumulateAndFlushAtExit(const Stats& stats);
};
}
#endif
