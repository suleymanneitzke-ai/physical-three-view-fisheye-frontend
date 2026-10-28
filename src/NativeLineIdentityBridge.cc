#include "NativeLineIdentityBridge.h"

namespace StructureSLAM
{

NativeLineIdentityBridge::Result
NativeLineIdentityBridge::Ok(const std::string& reason)
{
    Result result;
    result.ok = true;
    result.reason = reason;
    return result;
}

NativeLineIdentityBridge::Result
NativeLineIdentityBridge::Fail(const std::string& reason)
{
    Result result;
    result.ok = false;
    result.reason = reason;
    return result;
}

bool NativeLineIdentityBridge::IsInvalidTrackId(
    const long unsigned int temporal_track_id)
{
    return temporal_track_id ==
           FisheyeLineObservation::kInvalidTrackId;
}

MapLine* NativeLineIdentityBridge::ResolveMapLine(MapLine* pML)
{
    std::set<MapLine*> visited;
    MapLine* current = pML;
    while(current)
    {
        if(!visited.insert(current).second)
            return static_cast<MapLine*>(NULL);

        if(!current->isBad())
            return current;

        current = current->GetReplaced();
    }
    return static_cast<MapLine*>(NULL);
}

NativeLineIdentityBridge::Result
NativeLineIdentityBridge::ObserveTrack(
    KeyFrame* pKF, const size_t observation_idx,
    const long unsigned int temporal_track_id)
{
    if(!pKF)
        return Fail("null_keyframe");
    if(IsInvalidTrackId(temporal_track_id))
        return Fail("invalid_temporal_track_id");

    const std::vector<FisheyeLineObservation>& observations =
        pKF->GetNativeFisheyeLineObservations();
    if(observation_idx >= observations.size())
        return Fail("observation_index_out_of_range");

    const FisheyeLineObservation& observation =
        observations[observation_idx];
    if(!IsInvalidTrackId(observation.temporal_track_id) &&
       observation.temporal_track_id != temporal_track_id)
    {
        return Fail("observation_track_id_mismatch");
    }

    const ObservationKey key = {pKF, observation_idx};
    const auto existing_observation =
        mObservationToTrack.find(key);
    if(existing_observation != mObservationToTrack.end() &&
       existing_observation->second != temporal_track_id)
    {
        return Fail("observation_already_bound_to_different_track");
    }

    TrackRecord& record = mTracks[temporal_track_id];
    record.observations.insert(key);
    record.lost = false;
    mObservationToTrack[key] = temporal_track_id;

    if(ResolveMapLine(record.pMapLine))
        return RegisterObservationWithTrack(
            key, temporal_track_id, &record);

    return Ok("pending_no_mapline");
}

NativeLineIdentityBridge::Result
NativeLineIdentityBridge::BindTrackToMapLine(
    const long unsigned int temporal_track_id, MapLine* pML)
{
    if(IsInvalidTrackId(temporal_track_id))
        return Fail("invalid_temporal_track_id");

    MapLine* live_mapline = ResolveMapLine(pML);
    if(!live_mapline)
        return Fail("null_or_bad_mapline");

    TrackRecord& record = mTracks[temporal_track_id];
    MapLine* existing = ResolveMapLine(record.pMapLine);
    if(existing && existing != live_mapline)
        return Fail("track_already_bound_to_different_mapline");

    record.pMapLine = live_mapline;
    record.lost = false;

    for(const ObservationKey& key : record.observations)
    {
        const Result result =
            RegisterObservationWithTrack(
                key, temporal_track_id, &record);
        if(!result.ok &&
           result.reason != "observation_already_registered")
        {
            return result;
        }
    }

    return Ok("track_bound_to_mapline");
}

NativeLineIdentityBridge::Result
NativeLineIdentityBridge::RegisterObservation(
    KeyFrame* pKF, const size_t observation_idx)
{
    const ObservationKey key = {pKF, observation_idx};
    const auto found = mObservationToTrack.find(key);
    if(found == mObservationToTrack.end())
        return Fail("observation_has_no_track");

    auto track_it = mTracks.find(found->second);
    if(track_it == mTracks.end())
        return Fail("track_record_missing");

    return RegisterObservationWithTrack(
        key, found->second, &track_it->second);
}

NativeLineIdentityBridge::Result
NativeLineIdentityBridge::RegisterObservationWithTrack(
    const ObservationKey& key,
    const long unsigned int temporal_track_id,
    TrackRecord* record)
{
    if(!record)
        return Fail("track_record_missing");
    if(!key.pKF)
        return Fail("null_keyframe");

    MapLine* live_mapline = ResolveMapLine(record->pMapLine);
    if(!live_mapline)
    {
        record->pMapLine = static_cast<MapLine*>(NULL);
        return Fail("track_has_no_live_mapline");
    }
    record->pMapLine = live_mapline;

    if(mRegisteredObservations.count(key))
        return Fail("observation_already_registered");

    const std::vector<FisheyeLineObservation>& observations =
        key.pKF->GetNativeFisheyeLineObservations();
    if(key.idx >= observations.size())
        return Fail("observation_index_out_of_range");

    const FisheyeLineObservation& observation = observations[key.idx];
    if(!IsInvalidTrackId(observation.temporal_track_id) &&
       observation.temporal_track_id != temporal_track_id)
    {
        return Fail("observation_track_id_mismatch");
    }

    MapLine* existing = key.pKF->GetNativeFisheyeMapLine(key.idx);
    existing = ResolveMapLine(existing);
    if(existing && existing != live_mapline)
        return Fail("observation_already_has_different_mapline");

    key.pKF->AddNativeFisheyeMapLine(live_mapline, key.idx);
    live_mapline->AddNativeFisheyeObservation(
        key.pKF, key.idx, observation);
    mRegisteredObservations.insert(key);

    return Ok("observation_registered");
}

void NativeLineIdentityBridge::MarkTrackLost(
    const long unsigned int temporal_track_id)
{
    auto it = mTracks.find(temporal_track_id);
    if(it != mTracks.end())
        it->second.lost = true;
}

void NativeLineIdentityBridge::MarkTrackReacquired(
    const long unsigned int temporal_track_id)
{
    auto it = mTracks.find(temporal_track_id);
    if(it != mTracks.end())
        it->second.lost = false;
}

void NativeLineIdentityBridge::OnMapLineReplaced(
    MapLine* pOld, MapLine* pNew)
{
    MapLine* live_new = ResolveMapLine(pNew);
    for(auto& track : mTracks)
    {
        if(track.second.pMapLine == pOld ||
           ResolveMapLine(track.second.pMapLine) == live_new)
        {
            track.second.pMapLine = live_new;
        }
    }
}

void NativeLineIdentityBridge::CleanupBadMapLines()
{
    for(auto& track : mTracks)
        track.second.pMapLine = ResolveMapLine(track.second.pMapLine);

    for(auto it = mRegisteredObservations.begin();
        it != mRegisteredObservations.end();)
    {
        MapLine* pML = static_cast<MapLine*>(NULL);
        if(it->pKF)
            pML = it->pKF->GetNativeFisheyeMapLine(it->idx);
        if(!ResolveMapLine(pML))
            it = mRegisteredObservations.erase(it);
        else
            ++it;
    }
}

void NativeLineIdentityBridge::OnKeyFrameErased(KeyFrame* pKF)
{
    if(!pKF)
        return;

    for(auto it = mObservationToTrack.begin();
        it != mObservationToTrack.end();)
    {
        if(it->first.pKF != pKF)
        {
            ++it;
            continue;
        }

        const ObservationKey key = it->first;
        const long unsigned int track_id = it->second;
        auto track_it = mTracks.find(track_id);
        if(track_it != mTracks.end())
            track_it->second.observations.erase(key);

        mRegisteredObservations.erase(key);
        it = mObservationToTrack.erase(it);
    }

    CleanupBadMapLines();
}

MapLine* NativeLineIdentityBridge::GetMapLineForTrack(
    const long unsigned int temporal_track_id) const
{
    auto it = mTracks.find(temporal_track_id);
    if(it == mTracks.end())
        return static_cast<MapLine*>(NULL);
    return ResolveMapLine(it->second.pMapLine);
}

bool NativeLineIdentityBridge::CheckInvariants(
    std::string* report) const
{
    bool ok = true;
    std::ostringstream stream;

    for(const auto& track : mTracks)
    {
        if(IsInvalidTrackId(track.first))
        {
            ok = false;
            stream << "invalid track record\n";
        }

        MapLine* live_mapline = ResolveMapLine(
            track.second.pMapLine);
        if(track.second.pMapLine && !live_mapline)
        {
            ok = false;
            stream << "track " << track.first
                   << " points to bad/dangling MapLine\n";
        }
    }

    for(const auto& observation : mObservationToTrack)
    {
        if(!observation.first.pKF)
        {
            ok = false;
            stream << "null KeyFrame observation key\n";
            continue;
        }

        const auto track_it = mTracks.find(observation.second);
        if(track_it == mTracks.end())
        {
            ok = false;
            stream << "observation references missing track\n";
            continue;
        }

        if(track_it->second.observations.count(
               observation.first) == 0)
        {
            ok = false;
            stream << "track missing reverse observation\n";
        }

        const std::vector<FisheyeLineObservation>& observations =
            observation.first.pKF->GetNativeFisheyeLineObservations();
        if(observation.first.idx >= observations.size())
        {
            ok = false;
            stream << "observation index out of range\n";
        }
    }

    for(const ObservationKey& key : mRegisteredObservations)
    {
        const auto obs_it = mObservationToTrack.find(key);
        if(obs_it == mObservationToTrack.end())
        {
            ok = false;
            stream << "registered observation has no track\n";
            continue;
        }

        MapLine* pML = key.pKF->GetNativeFisheyeMapLine(key.idx);
        pML = ResolveMapLine(pML);
        if(!pML)
        {
            ok = false;
            stream << "registered observation has no live MapLine\n";
            continue;
        }

        const auto native_obs = pML->GetNativeFisheyeObservations();
        const auto native_it = native_obs.find(key.pKF);
        if(native_it == native_obs.end() ||
           native_it->second != key.idx)
        {
            ok = false;
            stream << "MapLine missing reverse native observation\n";
        }
    }

    if(report)
        *report = ok ? "PASS" : stream.str();
    return ok;
}

size_t NativeLineIdentityBridge::TrackCount() const
{
    return mTracks.size();
}

size_t NativeLineIdentityBridge::PendingObservationCount() const
{
    size_t count = 0;
    for(const auto& observation : mObservationToTrack)
    {
        if(mRegisteredObservations.count(observation.first) == 0)
            ++count;
    }
    return count;
}

size_t NativeLineIdentityBridge::RegisteredObservationCount() const
{
    return mRegisteredObservations.size();
}

} // namespace StructureSLAM
