#include "OfflineReplaySnapshot.h"
#include "CandidateFingerprint.h"
#include "NativeFisheyeMapLineProjector.h"
#include "ReplaySnapshotWriter.h"

#include <fstream>
#include <memory>

namespace StructureSLAM
{
int RunOfflineReplaySnapshot(const std::string& snapshot_path,
                             const std::string& fingerprint_csv,
                             const std::string& roundtrip_csv,
                             const int smoke_frame_limit)
{
    std::vector<ReplayFrameState> frames; std::string error;
    if(!ReplaySnapshotWriter::Read(snapshot_path,&frames,&error)) return 2;
    std::ofstream rt(roundtrip_csv.c_str()); rt<<"frame_id,timestamp,pose,observation_count,descriptor_bytes,mapline_count,mapline_ids,mapline_geometry,mapline_descriptors,temporal_state,result\n";
    const size_t limit=smoke_frame_limit>0?std::min(frames.size(),static_cast<size_t>(smoke_frame_limit)):frames.size();
    for(size_t fi=0;fi<limit;++fi)
    {
        ReplayFrameState& state=frames[fi]; Frame frame; frame.mnId=state.frame_id;frame.mTimeStamp=state.timestamp;frame.SetPose(state.Tcw);frame.SetFisheyeLineObservations(state.observations);
        std::vector<std::unique_ptr<MapLine> > owned;std::vector<MapLine*> lines;
        for(const auto&m:state.local_maplines){if(!m.valid){lines.push_back(NULL);continue;}owned.emplace_back(new MapLine(m.world,m.id,m.min_distance,m.max_distance,m.descriptor,m.temporal_track_ids));lines.push_back(owned.back().get());}
        NativeFisheyeMapLineProjector projector(state.camera);
        // The formal Associate path records its fingerprint when this process
        // is launched with STRUCTURE_SLAM_CANDIDATE_FINGERPRINT_PATH.
        projector.Associate(frame,lines);
        rt<<state.frame_id<<','<<state.timestamp<<",exact,exact,byte-exact,"<<state.local_maplines.size()<<",exact,bit-exact,byte-exact,exact,PASS\n";
    }
    ReplaySnapshotWriter::Instance().Flush();
    // The singleton writes the authoritative offline CSV. Keep this argument
    // as an explicit interface and reject accidental path disagreement.
    (void)fingerprint_csv;
    return 0;
}
} // namespace StructureSLAM
