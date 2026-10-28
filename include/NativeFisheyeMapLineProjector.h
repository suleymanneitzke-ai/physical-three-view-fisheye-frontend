#ifndef STRUCTURE_SLAM_NATIVE_FISHEYE_MAPLINE_PROJECTOR_H
#define STRUCTURE_SLAM_NATIVE_FISHEYE_MAPLINE_PROJECTOR_H

#include "Frame.h"
#include "MapLine.h"
#include "FisheyeVirtualCameraGeometry.h"
#include "camera/KannalaBrandt8Camera.h"

#include <vector>

namespace StructureSLAM
{

class NativeFisheyeMapLineProjector
{
public:
    struct Options
    {
        double virtual_fx = NativeFisheyeVirtualFx();
        double virtual_fy = NativeFisheyeVirtualFy();
        double virtual_cx = NativeFisheyeVirtualCx();
        double virtual_cy = NativeFisheyeVirtualCy();
        int virtual_width = NativeFisheyeVirtualWidth();
        int virtual_height = NativeFisheyeVirtualHeight();
        double min_depth_m = 1e-6;
        double max_angle_difference_deg = 15.0;
        double min_overlap_ratio = 0.20;
        double max_descriptor_distance = 80.0;
        std::vector<double> virtual_yaw_deg{-60.0, -20.0, 20.0};
    };

    struct Result
    {
        int projected = 0;
        int matched = 0;
        int rejected_bad = 0;
        int rejected_geometry = 0;
        int rejected_temporal = 0;
        int rejected_descriptor = 0;
        int candidate_count = 0;
        int rejected_duplicate = 0;
        long long gate_input_pairs = 0;
        long long fail_input_invalid = 0;
        long long fail_invalid_yaw_or_pose = 0;
        long long fail_invalid_world_segment = 0;
        long long fail_invalid_camera_segment = 0;
        long long fail_depth = 0;
        long long fail_range = 0;
        long long fail_distance_invariance = 0;
        long long fail_fisheye_projection = 0;
        long long fail_fisheye_fov = 0;
        long long fail_virtual_depth = 0;
        long long fail_virtual_fov = 0;
        long long fail_angle = 0;
        long long fail_overlap = 0;
        long long fail_temporal = 0;
        long long fail_descriptor = 0;
        long long fail_duplicate = 0;
        std::vector<long long> view_input_pairs;
        std::vector<long long> view_projection_pass;
        std::vector<long long> view_angle_pass;
        std::vector<long long> view_overlap_pass;
        std::vector<long long> view_descriptor_pass;
        std::vector<long long> view_association;
        double angle_fail_sum = 0.0;
        double angle_pass_sum = 0.0;
        long long angle_fail_count = 0;
        long long angle_pass_count = 0;
        long long angle_fail_within_5pct = 0;
        long long angle_fail_within_10pct = 0;
        long long angle_fail_within_20pct = 0;
        double overlap_fail_sum = 0.0;
        double overlap_pass_sum = 0.0;
        long long overlap_fail_count = 0;
        long long overlap_pass_count = 0;
        long long overlap_fail_within_5pct = 0;
        long long overlap_fail_within_10pct = 0;
        long long overlap_fail_within_20pct = 0;
    };

    enum class ProjectionFailure
    {
        NONE = 0,
        INPUT_INVALID,
        INVALID_YAW_OR_POSE,
        INVALID_WORLD_SEGMENT,
        INVALID_CAMERA_SEGMENT,
        DEPTH,
        RANGE,
        DISTANCE_INVARIANCE,
        FISHEYE_PROJECTION,
        FISHEYE_FOV,
        VIRTUAL_DEPTH,
        VIRTUAL_FOV
    };

    explicit NativeFisheyeMapLineProjector(
        const KannalaBrandt8Camera::Parameters& camera_params);
    NativeFisheyeMapLineProjector(
        const KannalaBrandt8Camera::Parameters& camera_params,
        const Options& options);

    Result Associate(Frame& frame,
                     const std::vector<MapLine*>& local_maplines) const;
    Result AssociatePhysical(Frame& frame,
                             const std::vector<MapLine*>& local_maplines) const;

private:
    struct Projection
    {
        bool ok = false;
        cv::Point2f fisheye_start;
        cv::Point2f fisheye_end;
        cv::Point2f virtual_start;
        cv::Point2f virtual_end;
        double endpoint_depth_min = -1.0;
        double range_m = -1.0;
    };

    struct Candidate
    {
        size_t observation_idx = 0;
        MapLine* mapline = static_cast<MapLine*>(NULL);
        NativeFisheyeLineProjectionDiagnostic diagnostic;
    };

    MapLine* ResolveLiveMapLine(MapLine* pML) const;
    double ViewYawRad(int view_id) const;
    bool ValidViewId(int view_id) const;
    Result MakeResult() const;
    bool ProjectForObservation(const Frame& frame,
                               MapLine* pML,
                               const FisheyeLineObservation& observation,
                               Projection& projection,
                               ProjectionFailure* failure,
                               bool physical_mode = false) const;
    bool CompatibleTrack(MapLine* pML,
                         const FisheyeLineObservation& observation,
                         bool& has_mapline_track) const;
    double DescriptorDistance(MapLine* pML,
                              const FisheyeLineObservation& observation) const;
    NativeFisheyeLineProjectionDiagnostic BuildDiagnostic(
        MapLine* pML,
        const FisheyeLineObservation& observation,
        const Projection& projection,
        bool temporal_compatible,
        bool descriptor_compatible,
        double descriptor_distance) const;

    KannalaBrandt8Camera camera_;
    Options options_;
};

} // namespace StructureSLAM

#endif
