#ifndef LINE_MAP_EXPORTER_H
#define LINE_MAP_EXPORTER_H

#include "FisheyeLineTriangulator.h"

#include <string>
#include <vector>

namespace StructureSLAM
{

class LineMapExporter
{
public:
    static bool WriteLineMapCsv(const std::string& path,
                                const std::vector<FisheyeLineTriangulator::Line3D>& lines);
    static bool WriteObj(const std::string& path,
                         const std::vector<FisheyeLineTriangulator::Line3D>& lines);
    static bool WritePlySamples(const std::string& path,
                                const std::vector<FisheyeLineTriangulator::Line3D>& lines,
                                int samples_per_line);
};

} // namespace StructureSLAM

#endif
