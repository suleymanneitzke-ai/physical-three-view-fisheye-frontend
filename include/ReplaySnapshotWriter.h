#ifndef STRUCTURE_SLAM_REPLAY_SNAPSHOT_WRITER_H
#define STRUCTURE_SLAM_REPLAY_SNAPSHOT_WRITER_H

#include "CandidateFingerprint.h"
#include "Frame.h"
#include "MapLine.h"
#include "camera/KannalaBrandt8Camera.h"

#include <cstdint>
#include <string>
#include <vector>

namespace StructureSLAM
{

struct ReplayMapLineState
{
    bool valid = false;
    std::uint64_t id = 0;
    Vector6d world = Vector6d::Zero();
    float min_distance = 0.0f;
    float max_distance = 0.0f;
    cv::Mat descriptor;
    std::vector<std::uint64_t> temporal_track_ids;
};

struct ReplayFrameState
{
    std::uint64_t frame_id = 0;
    double timestamp = 0.0;
    cv::Mat Tcw;
    KannalaBrandt8Camera::Parameters camera;
    std::vector<FisheyeLineObservation> observations;
    std::vector<ReplayMapLineState> local_maplines;
};

class ReplaySnapshotWriter
{
public:
    static ReplaySnapshotWriter& Instance();
    bool enabled() const;
    void Capture(const Frame& frame,
                 const std::vector<MapLine*>& local_maplines,
                 const KannalaBrandt8Camera::Parameters& camera);
    void RecordFingerprint(std::uint64_t frame_id,
                           const CandidateFingerprint& fingerprint);
    void Flush();

    static bool Read(const std::string& path,
                     std::vector<ReplayFrameState>* frames,
                     std::string* error);

private:
    ReplaySnapshotWriter();
    ~ReplaySnapshotWriter();
    ReplaySnapshotWriter(const ReplaySnapshotWriter&) = delete;
    ReplaySnapshotWriter& operator=(const ReplaySnapshotWriter&) = delete;

    std::string path_;
    std::string fingerprint_path_;
    bool flushed_ = false;
    std::vector<ReplayFrameState> frames_;
    std::vector<std::pair<std::uint64_t, CandidateFingerprint> > fingerprints_;
};

} // namespace StructureSLAM

#endif
