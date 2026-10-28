#include "mono_tum_vi_native_fisheye_plslam_runner.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if(argc != 6 && argc != 7)
    {
        std::cerr
            << "Usage: export_kb8_to_colmap "
            << "vocabulary settings image_folder times_file output_dir "
            << "[max_frames]"
            << std::endl;
        return 1;
    }

    int max_frames = -1;
    if(argc == 7)
        max_frames = std::atoi(argv[6]);

    setenv("STRUCTURE_SLAM_KB8_COLMAP_EXPORT_DIR", argv[5], 1);

    return StructureSLAM::RunNativeFisheyePlslamSequence(
        argv[1],
        argv[2],
        argv[3],
        argv[4],
        argv[5],
        max_frames);
}
