#ifndef STRUCTURE_SLAM_CANDIDATE_FINGERPRINT_H
#define STRUCTURE_SLAM_CANDIDATE_FINGERPRINT_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace StructureSLAM
{

struct ReplayCandidateKey
{
    int view_id = -1;
    std::uint64_t observation_id = 0;
    std::uint64_t mapline_id = 0;
};

struct CandidateFingerprint
{
    std::uint64_t count = 0;
    std::uint64_t ordered_hash = 1469598103934665603ULL;
    std::uint64_t set_hash = 0;
};

CandidateFingerprint FingerprintCandidates(
    const std::vector<ReplayCandidateKey>& candidates);

} // namespace StructureSLAM

#endif
