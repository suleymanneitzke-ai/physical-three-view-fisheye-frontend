#ifndef STRUCTURE_SLAM_PHYSICAL_LINE_OBSERVATION_WRAPPER_H
#define STRUCTURE_SLAM_PHYSICAL_LINE_OBSERVATION_WRAPPER_H

#include "FisheyeLineFrontend.h"
#include <opencv2/core.hpp>
#include <set>
#include <vector>

namespace StructureSLAM {

struct PhysicalLineGroup {
    enum Type { SINGLE_VIEW, TWO_VIEW_CONFIDENT, THREE_VIEW_CONFIDENT,
                MULTIPLE_POSSIBLE, CONFLICT };
    int group_id = -1;
    int frame_id = -1;
    Type type = SINGLE_VIEW;
    std::vector<std::pair<int,int> > members;
    cv::Vec3d physical_plane = cv::Vec3d(0,0,0);
};

struct SharedMapLineCandidate {
    enum Status { UNIQUE_SHARED_CANDIDATE, MULTIPLE_SHARED_CANDIDATES,
                  CROSS_VIEW_CANDIDATE_CONFLICT };
    Status status = CROSS_VIEW_CANDIDATE_CONFLICT;
    std::set<unsigned long> intersection;
    std::set<unsigned long> candidate_union;
};

class PhysicalLineObservationWrapper {
public:
    struct FrameStats {
        int observations = 0, single_view = 0, two_view = 0, three_view = 0;
        int multiple = 0, conflict = 0, checks40 = 0, checks80 = 0;
        int skipped80 = 0;
    };
    static std::vector<PhysicalLineGroup> Analyze(
        const FisheyeLineFrontend::FrameResult& frame,
        const std::vector<cv::Mat>& production_masks,
        int frame_id, FrameStats* stats = 0);
    static SharedMapLineCandidate IntersectCandidateSets(
        const std::vector<std::set<unsigned long> >& sets);
    static void AccumulateAndFlushAtExit(const FrameStats& stats);
};
}
#endif
