#ifndef STRUCTURE_SLAM_KB8_COLMAP_EXPORTER_H
#define STRUCTURE_SLAM_KB8_COLMAP_EXPORTER_H

#include <string>

namespace StructureSLAM
{

class Map;

class Kb8ColmapExporter
{
public:
    static bool Export(const Map& map,
                       const std::string& output_dir,
                       std::string* error = 0);
};

} // namespace StructureSLAM

#endif
