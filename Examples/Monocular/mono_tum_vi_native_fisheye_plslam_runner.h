#ifndef STRUCTURE_SLAM_MONO_TUM_VI_NATIVE_FISHEYE_PLSLAM_RUNNER_H
#define STRUCTURE_SLAM_MONO_TUM_VI_NATIVE_FISHEYE_PLSLAM_RUNNER_H

#include <string>

namespace StructureSLAM
{

int RunNativeFisheyePlslamSequence(const std::string& vocabulary_path,
                                   const std::string& settings_path,
                                   const std::string& image_folder,
                                   const std::string& times_file,
                                   const std::string& output_dir,
                                   int max_frames);

} // namespace StructureSLAM

#endif
