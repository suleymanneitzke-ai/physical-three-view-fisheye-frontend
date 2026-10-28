#ifndef STRUCTURE_SLAM_OFFLINE_REPLAY_SNAPSHOT_H
#define STRUCTURE_SLAM_OFFLINE_REPLAY_SNAPSHOT_H

#include <string>

namespace StructureSLAM
{
int RunOfflineReplaySnapshot(const std::string& snapshot_path,
                             const std::string& fingerprint_csv,
                             const std::string& roundtrip_csv,
                             int smoke_frame_limit);
}
#endif
