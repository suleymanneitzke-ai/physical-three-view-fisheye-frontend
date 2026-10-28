#include "mono_tum_vi_native_fisheye_plslam_runner.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace
{

std::string JoinFrames(int argc, char** argv, int start)
{
    std::ostringstream frames;
    bool first = true;
    for(int i = start; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if(arg == "--frame" && i + 1 < argc)
        {
            if(!first)
                frames << ",";
            frames << argv[++i];
            first = false;
        }
    }
    return frames.str();
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 6)
    {
        std::cerr
            << "Usage: export_kb8_map_points "
            << "vocabulary settings image_folder times_file output_dir "
            << "[max_frames] [--frame N ...]"
            << std::endl;
        return 1;
    }

    int max_frames = -1;
    int option_start = 6;
    if(argc >= 7 && std::string(argv[6]).find("--") != 0)
    {
        max_frames = std::atoi(argv[6]);
        option_start = 7;
    }

    setenv("STRUCTURE_SLAM_KB8_MAP_EXPORT_DIR", argv[5], 1);
    const std::string frames = JoinFrames(argc, argv, option_start);
    if(!frames.empty())
        setenv("STRUCTURE_SLAM_KB8_MAP_EXPORT_FRAMES", frames.c_str(), 1);

    return StructureSLAM::RunNativeFisheyePlslamSequence(
        argv[1],
        argv[2],
        argv[3],
        argv[4],
        argv[5],
        max_frames);
}
