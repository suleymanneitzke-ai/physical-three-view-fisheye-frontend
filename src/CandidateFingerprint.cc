#include "CandidateFingerprint.h"

namespace StructureSLAM
{
namespace
{
std::uint64_t Mix(std::uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

void Fnv(std::uint64_t value, std::uint64_t* hash)
{
    for(int i = 0; i < 8; ++i)
    {
        *hash ^= static_cast<unsigned char>(value & 0xffU);
        *hash *= 1099511628211ULL;
        value >>= 8;
    }
}
} // namespace

CandidateFingerprint FingerprintCandidates(
    const std::vector<ReplayCandidateKey>& candidates)
{
    CandidateFingerprint result;
    result.count = candidates.size();
    for(size_t i = 0; i < candidates.size(); ++i)
    {
        const ReplayCandidateKey& key = candidates[i];
        Fnv(static_cast<std::uint64_t>(key.view_id), &result.ordered_hash);
        Fnv(key.observation_id, &result.ordered_hash);
        Fnv(key.mapline_id, &result.ordered_hash);
        const std::uint64_t tuple =
            Mix(static_cast<std::uint64_t>(key.view_id)) ^
            Mix(key.observation_id + 0x9e3779b97f4a7c15ULL) ^
            Mix(key.mapline_id + 0x243f6a8885a308d3ULL);
        result.set_hash += tuple;
    }
    return result;
}
} // namespace StructureSLAM
