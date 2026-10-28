#ifndef STRUCTURE_SLAM_NATIVE_LINE_IDENTITY_BRIDGE_H
#define STRUCTURE_SLAM_NATIVE_LINE_IDENTITY_BRIDGE_H

#include "FisheyeLineObservation.h"
#include "KeyFrame.h"
#include "MapLine.h"

#include <cstddef>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace StructureSLAM
{

// Bridges native temporal-LBD 2D tracks to formal MapLine identities.
//
// ID convention:
// - native_line_id is a Frame-local detector index.
// - temporal_track_id is a cross-frame native temporal-LBD identity.
// - mapline_id is MapLine::mnId and is valid only after a live MapLine exists.
// - virtual_view_id is FisheyeLineObservation::view_id.
//
// This class only builds the native observation graph. It does not create
// geometry, modify Tcw/Twc, run pose optimization, Local BA or MapLine BA.
class NativeLineIdentityBridge
{
public:
    struct Result
    {
        bool ok = false;
        std::string reason;
    };

    static Result Ok(const std::string& reason = "ok");
    static Result Fail(const std::string& reason);

    Result ObserveTrack(KeyFrame* pKF,
                        size_t observation_idx,
                        long unsigned int temporal_track_id);

    Result BindTrackToMapLine(long unsigned int temporal_track_id,
                              MapLine* pML);

    Result RegisterObservation(KeyFrame* pKF, size_t observation_idx);

    void MarkTrackLost(long unsigned int temporal_track_id);
    void MarkTrackReacquired(long unsigned int temporal_track_id);
    void OnMapLineReplaced(MapLine* pOld, MapLine* pNew);
    void CleanupBadMapLines();
    void OnKeyFrameErased(KeyFrame* pKF);

    MapLine* GetMapLineForTrack(
        long unsigned int temporal_track_id) const;

    bool CheckInvariants(std::string* report) const;

    size_t TrackCount() const;
    size_t PendingObservationCount() const;
    size_t RegisteredObservationCount() const;

private:
    struct ObservationKey
    {
        KeyFrame* pKF = static_cast<KeyFrame*>(NULL);
        size_t idx = 0;

        bool operator<(const ObservationKey& other) const
        {
            if(pKF != other.pKF)
                return pKF < other.pKF;
            return idx < other.idx;
        }
    };

    struct TrackRecord
    {
        MapLine* pMapLine = static_cast<MapLine*>(NULL);
        bool lost = false;
        std::set<ObservationKey> observations;
    };

    static bool IsInvalidTrackId(long unsigned int temporal_track_id);
    static MapLine* ResolveMapLine(MapLine* pML);

    Result RegisterObservationWithTrack(
        const ObservationKey& key,
        long unsigned int temporal_track_id,
        TrackRecord* record);

    std::map<long unsigned int, TrackRecord> mTracks;
    std::map<ObservationKey, long unsigned int> mObservationToTrack;
    std::set<ObservationKey> mRegisteredObservations;
};

} // namespace StructureSLAM

#endif
