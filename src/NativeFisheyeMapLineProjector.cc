#include "NativeFisheyeMapLineProjector.h"

#include "FisheyeVirtualCameraGeometry.h"
#include "CandidateFingerprint.h"
#include "ReplaySnapshotWriter.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace StructureSLAM
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

bool FinitePoint(const cv::Point2f& p)
{
    return std::isfinite(p.x) && std::isfinite(p.y);
}

bool InImage(const cv::Point2f& p, const int width, const int height)
{
    return FinitePoint(p) && p.x >= 0.0f && p.y >= 0.0f &&
           p.x < static_cast<float>(width) &&
           p.y < static_cast<float>(height);
}

Eigen::Vector3d TransformPoint(const cv::Mat& Tcw,
                               const Eigen::Vector3d& Xw)
{
    if(Tcw.empty())
        return Xw;

    const bool is_float = Tcw.type() == CV_32F;
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    for(int r = 0; r < 3; ++r)
    {
        for(int c = 0; c < 3; ++c)
        {
            Rcw(r, c) = is_float ?
                static_cast<double>(Tcw.at<float>(r, c)) :
                Tcw.at<double>(r, c);
        }
        tcw(r) = is_float ?
            static_cast<double>(Tcw.at<float>(r, 3)) :
            Tcw.at<double>(r, 3);
    }
    return Rcw * Xw + tcw;
}

bool IsFiniteVector(const Eigen::Vector3d& v)
{
    return std::isfinite(v.x()) && std::isfinite(v.y()) &&
           std::isfinite(v.z());
}

double SegmentAngleDeg(const cv::Point2f& a, const cv::Point2f& b)
{
    double angle = std::atan2(static_cast<double>(b.y - a.y),
                              static_cast<double>(b.x - a.x)) *
                   180.0 / kPi;
    if(angle < 0.0)
        angle += 180.0;
    if(angle >= 180.0)
        angle -= 180.0;
    return angle;
}

double AngleDifferenceDeg(const double a, const double b)
{
    double d = std::fabs(a - b);
    while(d >= 180.0)
        d -= 180.0;
    if(d > 90.0)
        d = 180.0 - d;
    return d;
}

double ProjectionOverlapRatio(const cv::Point2f& pa,
                              const cv::Point2f& pb,
                              const cv::Point2f& oa,
                              const cv::Point2f& ob)
{
    const cv::Point2f axis = pb - pa;
    const double len = cv::norm(axis);
    if(len <= 1e-6)
        return 0.0;

    const cv::Point2f unit(axis.x / static_cast<float>(len),
                           axis.y / static_cast<float>(len));
    const auto project = [&](const cv::Point2f& p) {
        const cv::Point2f d = p - pa;
        return static_cast<double>(d.x * unit.x + d.y * unit.y);
    };

    double p0 = 0.0;
    double p1 = len;
    if(p0 > p1)
        std::swap(p0, p1);
    double o0 = project(oa);
    double o1 = project(ob);
    if(o0 > o1)
        std::swap(o0, o1);

    const double inter =
        std::max(0.0, std::min(p1, o1) - std::max(p0, o0));
    const double denom = std::max(1e-6, std::min(p1 - p0, o1 - o0));
    return inter / denom;
}

} // namespace

NativeFisheyeMapLineProjector::NativeFisheyeMapLineProjector(
    const KannalaBrandt8Camera::Parameters& camera_params)
    : camera_(camera_params), options_(Options())
{
}

NativeFisheyeMapLineProjector::NativeFisheyeMapLineProjector(
    const KannalaBrandt8Camera::Parameters& camera_params,
    const Options& options)
    : camera_(camera_params), options_(options)
{
}

bool NativeFisheyeMapLineProjector::ValidViewId(const int view_id) const
{
    return view_id >= 0 &&
           view_id < static_cast<int>(options_.virtual_yaw_deg.size());
}

double NativeFisheyeMapLineProjector::ViewYawRad(const int view_id) const
{
    if(!ValidViewId(view_id))
        return std::numeric_limits<double>::quiet_NaN();
    return options_.virtual_yaw_deg[view_id] * kPi / 180.0;
}

NativeFisheyeMapLineProjector::Result
NativeFisheyeMapLineProjector::MakeResult() const
{
    Result result;
    const size_t view_count = options_.virtual_yaw_deg.size();
    result.view_input_pairs.assign(view_count, 0);
    result.view_projection_pass.assign(view_count, 0);
    result.view_angle_pass.assign(view_count, 0);
    result.view_overlap_pass.assign(view_count, 0);
    result.view_descriptor_pass.assign(view_count, 0);
    result.view_association.assign(view_count, 0);
    return result;
}

MapLine* NativeFisheyeMapLineProjector::ResolveLiveMapLine(
    MapLine* pML) const
{
    std::set<MapLine*> seen;
    MapLine* current = pML;
    while(current && current->isBad())
    {
        if(seen.count(current))
            return static_cast<MapLine*>(NULL);
        seen.insert(current);
        current = current->GetReplaced();
    }
    return current && !current->isBad() ? current :
        static_cast<MapLine*>(NULL);
}

bool NativeFisheyeMapLineProjector::ProjectForObservation(
    const Frame& frame,
    MapLine* pML,
    const FisheyeLineObservation& observation,
    Projection& projection,
    ProjectionFailure* failure,
    const bool physical_mode) const
{
    projection = Projection();
    if(failure)
        *failure = ProjectionFailure::NONE;
    if(!pML || !ValidViewId(observation.view_id) || !camera_.valid())
    {
        if(failure)
            *failure = ProjectionFailure::INPUT_INVALID;
        return false;
    }

    const double yaw = ViewYawRad(observation.view_id);
    if(!std::isfinite(yaw) || frame.mTcw.empty())
    {
        if(failure)
            *failure = ProjectionFailure::INVALID_YAW_OR_POSE;
        return false;
    }

    const Vector6d world = pML->GetWorldPos();
    const Eigen::Vector3d sw(world(0), world(1), world(2));
    const Eigen::Vector3d ew(world(3), world(4), world(5));
    if(!IsFiniteVector(sw) || !IsFiniteVector(ew) ||
       (sw - ew).norm() <= 1e-9)
    {
        if(failure)
            *failure = ProjectionFailure::INVALID_WORLD_SEGMENT;
        return false;
    }

    const Eigen::Vector3d sc = TransformPoint(frame.mTcw, sw);
    const Eigen::Vector3d ec = TransformPoint(frame.mTcw, ew);
    if(!IsFiniteVector(sc) || !IsFiniteVector(ec))
    {
        if(failure)
            *failure = ProjectionFailure::INVALID_CAMERA_SEGMENT;
        return false;
    }

    projection.endpoint_depth_min = std::min(sc.z(), ec.z());
    // Physical KB8 visibility is determined by the calibrated projection,
    // not perspective z-positive cheirality. Virtual-view z is checked later.

    const Eigen::Vector3d mid = 0.5 * (sc + ec);
    projection.range_m = mid.norm();
    if(!std::isfinite(projection.range_m) || projection.range_m <= 0.0)
    {
        if(failure)
            *failure = ProjectionFailure::RANGE;
        return false;
    }

    const float min_dist = pML->GetMinDistanceInvariance();
    const float max_dist = pML->GetMaxDistanceInvariance();
    if(std::isfinite(min_dist) && std::isfinite(max_dist) &&
       min_dist > 0.0f && max_dist > min_dist)
    {
        if(projection.range_m < static_cast<double>(min_dist) ||
           projection.range_m > static_cast<double>(max_dist))
        {
            if(failure)
                *failure = ProjectionFailure::DISTANCE_INVARIANCE;
            return false;
        }
    }

    Eigen::Vector2d su;
    Eigen::Vector2d eu;
    if(!camera_.project(sc, su) || !camera_.project(ec, eu))
    {
        if(failure)
            *failure = ProjectionFailure::FISHEYE_PROJECTION;
        return false;
    }
    if(!camera_.isInImage(su) || !camera_.isInImage(eu))
    {
        if(failure)
            *failure = ProjectionFailure::FISHEYE_FOV;
        return false;
    }

    const Eigen::Matrix3d Rvc = RotationVcFromYaw(yaw);
    const Eigen::Vector3d sv = Rvc * sc;
    const Eigen::Vector3d ev = Rvc * ec;
    if(!physical_mode &&
       (sv.z() <= options_.min_depth_m || ev.z() <= options_.min_depth_m))
    {
        if(failure)
            *failure = ProjectionFailure::VIRTUAL_DEPTH;
        return false;
    }

    projection.fisheye_start =
        cv::Point2f(static_cast<float>(su.x()),
                    static_cast<float>(su.y()));
    projection.fisheye_end =
        cv::Point2f(static_cast<float>(eu.x()),
                    static_cast<float>(eu.y()));
    if(std::fabs(sv.z()) > options_.min_depth_m &&
       std::fabs(ev.z()) > options_.min_depth_m)
    {
        projection.virtual_start =
            cv::Point2f(static_cast<float>(options_.virtual_fx *
                                           sv.x() / sv.z() +
                                           options_.virtual_cx),
                        static_cast<float>(options_.virtual_fy *
                                           sv.y() / sv.z() +
                                           options_.virtual_cy));
        projection.virtual_end =
            cv::Point2f(static_cast<float>(options_.virtual_fx *
                                           ev.x() / ev.z() +
                                           options_.virtual_cx),
                        static_cast<float>(options_.virtual_fy *
                                           ev.y() / ev.z() +
                                           options_.virtual_cy));
    }

    if(!physical_mode &&
       (!InImage(projection.virtual_start, options_.virtual_width,
                options_.virtual_height) ||
        !InImage(projection.virtual_end, options_.virtual_width,
                 options_.virtual_height)))
    {
        if(failure)
            *failure = ProjectionFailure::VIRTUAL_FOV;
        return false;
    }

    projection.ok = true;
    return true;
}

namespace
{

void CountProjectionFailure(
    NativeFisheyeMapLineProjector::Result& result,
    const NativeFisheyeMapLineProjector::ProjectionFailure failure)
{
    switch(failure)
    {
    case NativeFisheyeMapLineProjector::ProjectionFailure::INPUT_INVALID:
        ++result.fail_input_invalid;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::INVALID_YAW_OR_POSE:
        ++result.fail_invalid_yaw_or_pose;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::INVALID_WORLD_SEGMENT:
        ++result.fail_invalid_world_segment;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::INVALID_CAMERA_SEGMENT:
        ++result.fail_invalid_camera_segment;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::DEPTH:
        ++result.fail_depth;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::RANGE:
        ++result.fail_range;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::DISTANCE_INVARIANCE:
        ++result.fail_distance_invariance;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::FISHEYE_PROJECTION:
        ++result.fail_fisheye_projection;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::FISHEYE_FOV:
        ++result.fail_fisheye_fov;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::VIRTUAL_DEPTH:
        ++result.fail_virtual_depth;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::VIRTUAL_FOV:
        ++result.fail_virtual_fov;
        break;
    case NativeFisheyeMapLineProjector::ProjectionFailure::NONE:
        break;
    }
}

} // namespace

bool NativeFisheyeMapLineProjector::CompatibleTrack(
    MapLine* pML,
    const FisheyeLineObservation& observation,
    bool& has_mapline_track) const
{
    has_mapline_track = false;
    if(!pML ||
       observation.temporal_track_id ==
           FisheyeLineObservation::kInvalidTrackId)
        return false;

    const std::map<KeyFrame*, size_t> native_obs =
        pML->GetNativeFisheyeObservations();
    for(std::map<KeyFrame*, size_t>::const_iterator it =
            native_obs.begin();
        it != native_obs.end(); ++it)
    {
        const FisheyeLineObservation stored =
            pML->GetNativeFisheyeObservation(it->first);
        if(stored.temporal_track_id !=
           FisheyeLineObservation::kInvalidTrackId)
        {
            has_mapline_track = true;
            if(stored.temporal_track_id == observation.temporal_track_id)
                return true;
        }
    }
    return false;
}

double NativeFisheyeMapLineProjector::DescriptorDistance(
    MapLine* pML,
    const FisheyeLineObservation& observation) const
{
    if(!pML || observation.lbd_descriptor.empty())
        return -1.0;
    const cv::Mat map_desc = pML->GetDescriptor();
    if(map_desc.empty() ||
       map_desc.cols != observation.lbd_descriptor.cols ||
       map_desc.type() != observation.lbd_descriptor.type())
        return -1.0;
    return static_cast<double>(
        cv::norm(map_desc, observation.lbd_descriptor, cv::NORM_HAMMING));
}

NativeFisheyeLineProjectionDiagnostic
NativeFisheyeMapLineProjector::BuildDiagnostic(
    MapLine* pML,
    const FisheyeLineObservation& observation,
    const Projection& projection,
    const bool temporal_compatible,
    const bool descriptor_compatible,
    const double descriptor_distance) const
{
    NativeFisheyeLineProjectionDiagnostic diagnostic;
    diagnostic.attempted = true;
    diagnostic.visible = projection.ok;
    diagnostic.temporal_track_compatible = temporal_compatible;
    diagnostic.descriptor_compatible = descriptor_compatible;
    diagnostic.mapline_id = pML ? pML->mnId :
        static_cast<long unsigned int>(-1);
    diagnostic.view_id = observation.view_id;
    diagnostic.projected_fisheye_start = projection.fisheye_start;
    diagnostic.projected_fisheye_end = projection.fisheye_end;
    diagnostic.projected_virtual_start = projection.virtual_start;
    diagnostic.projected_virtual_end = projection.virtual_end;
    diagnostic.endpoint_depth_min = projection.endpoint_depth_min;
    diagnostic.range_m = projection.range_m;
    diagnostic.descriptor_distance = descriptor_distance;

    if(projection.ok)
    {
        const double projected_angle =
            SegmentAngleDeg(projection.virtual_start,
                            projection.virtual_end);
        diagnostic.angle_difference_deg =
            AngleDifferenceDeg(projected_angle,
                               observation.virtual_angle_deg);
        diagnostic.overlap_ratio =
            ProjectionOverlapRatio(projection.virtual_start,
                                   projection.virtual_end,
                                   observation.virtual_start,
                                   observation.virtual_end);
        diagnostic.score =
            (temporal_compatible ? 1000.0 : 0.0) +
            (descriptor_compatible ? 100.0 : 0.0) +
            10.0 * diagnostic.overlap_ratio -
            diagnostic.angle_difference_deg;
    }
    return diagnostic;
}

NativeFisheyeMapLineProjector::Result
NativeFisheyeMapLineProjector::Associate(
    Frame& frame,
    const std::vector<MapLine*>& local_maplines) const
{
    Result result;
    if(frame.mvFisheyeLineObservations.empty() || local_maplines.empty())
        return result;

    // Diagnostic-only raw-state copy. The environment-disabled path returns
    // immediately and leaves the formal matcher data and ordering untouched.
    ReplaySnapshotWriter::Instance().Capture(
        frame, local_maplines, camera_.parameters());

    std::vector<NativeFisheyeLineProjectionDiagnostic> best_diagnostics(
        frame.mvFisheyeLineObservations.size(),
        NativeFisheyeLineProjectionDiagnostic());
    std::vector<Candidate> candidates;

    for(size_t ml_idx = 0; ml_idx < local_maplines.size(); ++ml_idx)
    {
        MapLine* live = ResolveLiveMapLine(local_maplines[ml_idx]);
        if(!live)
        {
            result.rejected_bad++;
            continue;
        }

        for(size_t obs_idx = 0;
            obs_idx < frame.mvFisheyeLineObservations.size(); ++obs_idx)
        {
            const FisheyeLineObservation& observation =
                frame.mvFisheyeLineObservations[obs_idx];
            if(ValidViewId(observation.view_id))
                result.view_input_pairs[observation.view_id]++;
            result.gate_input_pairs++;
            Projection projection;
            ProjectionFailure projection_failure = ProjectionFailure::NONE;
            const bool visible =
                ProjectForObservation(frame, live, observation, projection,
                                      &projection_failure);
            if(!visible)
            {
                result.rejected_geometry++;
                CountProjectionFailure(result, projection_failure);
                NativeFisheyeLineProjectionDiagnostic diagnostic;
                diagnostic.attempted = true;
                diagnostic.mapline_id = live->mnId;
                diagnostic.view_id = observation.view_id;
                if(!best_diagnostics[obs_idx].attempted)
                    best_diagnostics[obs_idx] = diagnostic;
                continue;
            }

            result.projected++;
            if(ValidViewId(observation.view_id))
                result.view_projection_pass[observation.view_id]++;
            bool has_mapline_track = false;
            const bool temporal_compatible =
                CompatibleTrack(live, observation, has_mapline_track);
            const double descriptor_distance =
                DescriptorDistance(live, observation);
            const bool descriptor_compatible =
                descriptor_distance >= 0.0 &&
                descriptor_distance <= options_.max_descriptor_distance;
            NativeFisheyeLineProjectionDiagnostic diagnostic =
                BuildDiagnostic(live, observation, projection,
                                temporal_compatible,
                                descriptor_compatible,
                                descriptor_distance);

            if(!best_diagnostics[obs_idx].attempted ||
               diagnostic.score > best_diagnostics[obs_idx].score)
                best_diagnostics[obs_idx] = diagnostic;

            if(diagnostic.angle_difference_deg >
                   options_.max_angle_difference_deg)
            {
                result.rejected_geometry++;
                result.fail_angle++;
                result.angle_fail_sum += diagnostic.angle_difference_deg;
                result.angle_fail_count++;
                if(diagnostic.angle_difference_deg <=
                   options_.max_angle_difference_deg * 1.05)
                    result.angle_fail_within_5pct++;
                if(diagnostic.angle_difference_deg <=
                   options_.max_angle_difference_deg * 1.10)
                    result.angle_fail_within_10pct++;
                if(diagnostic.angle_difference_deg <=
                   options_.max_angle_difference_deg * 1.20)
                    result.angle_fail_within_20pct++;
                continue;
            }
            result.angle_pass_sum += diagnostic.angle_difference_deg;
            result.angle_pass_count++;
            if(ValidViewId(observation.view_id))
                result.view_angle_pass[observation.view_id]++;

            if(diagnostic.overlap_ratio < options_.min_overlap_ratio)
            {
                result.rejected_geometry++;
                result.fail_overlap++;
                result.overlap_fail_sum += diagnostic.overlap_ratio;
                result.overlap_fail_count++;
                if(diagnostic.overlap_ratio >=
                   options_.min_overlap_ratio * 0.95)
                    result.overlap_fail_within_5pct++;
                if(diagnostic.overlap_ratio >=
                   options_.min_overlap_ratio * 0.90)
                    result.overlap_fail_within_10pct++;
                if(diagnostic.overlap_ratio >=
                   options_.min_overlap_ratio * 0.80)
                    result.overlap_fail_within_20pct++;
                continue;
            }
            result.overlap_pass_sum += diagnostic.overlap_ratio;
            result.overlap_pass_count++;
            if(ValidViewId(observation.view_id))
                result.view_overlap_pass[observation.view_id]++;

            if(has_mapline_track)
            {
                if(!temporal_compatible)
                {
                    result.rejected_temporal++;
                    result.fail_temporal++;
                    continue;
                }
            }
            else if(!descriptor_compatible)
            {
                result.rejected_descriptor++;
                result.fail_descriptor++;
                continue;
            }
            if(ValidViewId(observation.view_id))
                result.view_descriptor_pass[observation.view_id]++;

            Candidate candidate;
            candidate.observation_idx = obs_idx;
            candidate.mapline = live;
            candidate.diagnostic = diagnostic;
            candidates.push_back(candidate);
            result.candidate_count++;
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.diagnostic.score > b.diagnostic.score;
              });

    std::vector<ReplayCandidateKey> fingerprint_keys;
    if(ReplaySnapshotWriter::Instance().enabled())
    {
        fingerprint_keys.reserve(candidates.size());
        for(size_t i = 0; i < candidates.size(); ++i)
        {
            ReplayCandidateKey key;
            const FisheyeLineObservation& observation =
                frame.mvFisheyeLineObservations[candidates[i].observation_idx];
            key.view_id = observation.view_id;
            key.observation_id = observation.native_line_id >= 0 ?
                static_cast<std::uint64_t>(observation.native_line_id) :
                static_cast<std::uint64_t>(candidates[i].observation_idx);
            key.mapline_id = candidates[i].mapline->mnId;
            fingerprint_keys.push_back(key);
        }
        ReplaySnapshotWriter::Instance().RecordFingerprint(
            frame.mnId, FingerprintCandidates(fingerprint_keys));
    }

    std::set<size_t> used_observations;
    std::set<MapLine*> used_maplines;
    for(size_t i = 0; i < candidates.size(); ++i)
    {
        Candidate candidate = candidates[i];
        if(used_observations.count(candidate.observation_idx) ||
           used_maplines.count(candidate.mapline))
        {
            result.rejected_duplicate++;
            result.fail_duplicate++;
            continue;
        }

        candidate.diagnostic.matched = true;
        frame.SetNativeFisheyeLineMapLine(candidate.observation_idx,
                                          candidate.mapline);
        if(candidate.observation_idx < frame.mvbFisheyeLineOutlier.size())
            frame.mvbFisheyeLineOutlier[candidate.observation_idx] = false;
        frame.SetNativeFisheyeLineProjectionDiagnostic(
            candidate.observation_idx, candidate.diagnostic);
        used_observations.insert(candidate.observation_idx);
        used_maplines.insert(candidate.mapline);
        if(ValidViewId(candidate.diagnostic.view_id))
            result.view_association[candidate.diagnostic.view_id]++;
        candidate.mapline->MarkPhysicalAssociation(
            static_cast<long>(frame.mnId),
            -1,
            static_cast<long>(candidate.observation_idx),
            "NativeFisheyeMapLineProjector::AssociatePhysical");
        candidate.mapline->IncreaseVisible();
        candidate.mapline->IncreaseFound();
        candidate.mapline->mnLastFrameSeen = frame.mnId;
        result.matched++;
    }

    for(size_t obs_idx = 0; obs_idx < best_diagnostics.size(); ++obs_idx)
    {
        if(used_observations.count(obs_idx))
            continue;
        frame.SetNativeFisheyeLineProjectionDiagnostic(
            obs_idx, best_diagnostics[obs_idx]);
    }

    return result;
}

NativeFisheyeMapLineProjector::Result
NativeFisheyeMapLineProjector::AssociatePhysical(
    Frame& frame,
    const std::vector<MapLine*>& local_maplines) const
{
    Result result = MakeResult();
    if(frame.mvPhysicalLineObservations.empty() || local_maplines.empty())
        return result;

    if(frame.mvpPhysicalMapLines.size() !=
       frame.mvPhysicalLineObservations.size())
    {
        frame.mvpPhysicalMapLines.assign(
            frame.mvPhysicalLineObservations.size(),
            static_cast<MapLine*>(NULL));
    }
    if(frame.mvbPhysicalLineOutlier.size() !=
       frame.mvPhysicalLineObservations.size())
    {
        frame.mvbPhysicalLineOutlier.assign(
            frame.mvPhysicalLineObservations.size(), false);
    }
    frame.ResetPhysicalLineResidualState();

    std::vector<Candidate> candidates;
    for(size_t ml_idx = 0; ml_idx < local_maplines.size(); ++ml_idx)
    {
        MapLine* live = ResolveLiveMapLine(local_maplines[ml_idx]);
        if(!live)
        {
            result.rejected_bad++;
            continue;
        }

        for(size_t phys_idx = 0;
            phys_idx < frame.mvPhysicalLineObservations.size(); ++phys_idx)
        {
            const PhysicalLineObservation& physical =
                frame.mvPhysicalLineObservations[phys_idx];
            if(!physical.plane_valid || physical.members.empty())
            {
                ++result.fail_input_invalid;
                continue;
            }

            NativeFisheyeLineProjectionDiagnostic best_diag;
            int member_pass = 0;
            bool any_has_track = false;
            bool any_temporal = false;
            bool any_descriptor = false;

            for(size_t member_idx = 0;
                member_idx < physical.members.size(); ++member_idx)
            {
                const PhysicalLineObservationMember& member =
                    physical.members[member_idx];
                if(member.virtual_observation_index >=
                   frame.mvFisheyeLineObservations.size())
                {
                    ++result.fail_input_invalid;
                    continue;
                }

                const FisheyeLineObservation& observation =
                    frame.mvFisheyeLineObservations[
                        member.virtual_observation_index];
                if(ValidViewId(observation.view_id))
                    result.view_input_pairs[observation.view_id]++;
                result.gate_input_pairs++;

                Projection projection;
                ProjectionFailure projection_failure =
                    ProjectionFailure::NONE;
                const bool visible =
                    ProjectForObservation(frame, live, observation,
                                          projection,
                                          &projection_failure,
                                          true);
                if(!visible)
                {
                    result.rejected_geometry++;
                    CountProjectionFailure(result, projection_failure);
                    continue;
                }
                result.projected++;
                if(ValidViewId(observation.view_id))
                    result.view_projection_pass[observation.view_id]++;

                bool has_mapline_track = false;
                const bool temporal_compatible =
                    CompatibleTrack(live, observation, has_mapline_track);
                const double descriptor_distance =
                    DescriptorDistance(live, observation);
                const bool descriptor_compatible =
                    descriptor_distance >= 0.0 &&
                    descriptor_distance <= options_.max_descriptor_distance;
                NativeFisheyeLineProjectionDiagnostic diagnostic =
                    BuildDiagnostic(live, observation, projection,
                                    temporal_compatible,
                                    descriptor_compatible,
                                    descriptor_distance);
                if(!best_diag.attempted ||
                   diagnostic.score > best_diag.score)
                    best_diag = diagnostic;

                if(false && diagnostic.angle_difference_deg >
                   options_.max_angle_difference_deg)
                {
                    result.rejected_geometry++;
                    result.fail_angle++;
                    continue;
                }
                if(ValidViewId(observation.view_id))
                    result.view_angle_pass[observation.view_id]++;

                if(false && diagnostic.overlap_ratio < options_.min_overlap_ratio)
                {
                    result.rejected_geometry++;
                    result.fail_overlap++;
                    continue;
                }
                if(ValidViewId(observation.view_id))
                    result.view_overlap_pass[observation.view_id]++;

                any_has_track = any_has_track || has_mapline_track;
                any_temporal = any_temporal || temporal_compatible;
                any_descriptor = any_descriptor || descriptor_compatible;
                ++member_pass;
                if(ValidViewId(observation.view_id))
                    result.view_descriptor_pass[observation.view_id]++;
            }

            if(member_pass == 0)
                continue;
            // A temporal-track mismatch is a candidate-priority signal, not
            // a physical visibility failure.  Once native projection has
            // passed, descriptor and physical support/geometry remain valid
            // correspondence gates.  Keep identity in the score below but
            // do not hard-reject an otherwise valid candidate here.
            if(!any_has_track && !any_descriptor)
            {
                result.rejected_descriptor++;
                result.fail_descriptor++;
                continue;
            }

            Candidate candidate;
            candidate.observation_idx = phys_idx;
            candidate.mapline = live;
            candidate.diagnostic = best_diag;
            candidates.push_back(candidate);
            result.candidate_count++;
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.diagnostic.score > b.diagnostic.score;
              });

    std::set<size_t> used_physical_observations;
    std::set<MapLine*> used_maplines;
    for(size_t i = 0; i < candidates.size(); ++i)
    {
        Candidate candidate = candidates[i];
        if(used_physical_observations.count(candidate.observation_idx) ||
           used_maplines.count(candidate.mapline))
        {
            result.rejected_duplicate++;
            result.fail_duplicate++;
            continue;
        }

        candidate.diagnostic.matched = true;
        frame.SetPhysicalLineMapLine(candidate.observation_idx,
                                     candidate.mapline);
        if(candidate.observation_idx < frame.mvbPhysicalLineOutlier.size())
            frame.mvbPhysicalLineOutlier[candidate.observation_idx] = false;
        used_physical_observations.insert(candidate.observation_idx);
        used_maplines.insert(candidate.mapline);
        if(ValidViewId(candidate.diagnostic.view_id))
            result.view_association[candidate.diagnostic.view_id]++;
        candidate.mapline->IncreaseVisible();
        candidate.mapline->IncreaseFound();
        candidate.mapline->mnLastFrameSeen = frame.mnId;
        result.matched++;
    }

    return result;
}

} // namespace StructureSLAM
