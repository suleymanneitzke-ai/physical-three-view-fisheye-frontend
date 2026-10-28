#ifndef CORRECT_LOOP_HELPERS_H
#define CORRECT_LOOP_HELPERS_H

#include "KeyFrame.h"

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <Eigen/Core>
#include <map>
#include <set>
#include <vector>

namespace StructureSLAM
{
namespace CorrectLoopHelpers
{

cv::Mat Sim3ToSE3Tcw(const g2o::Sim3& Scw);

g2o::Sim3 BuildCorrectedConnectedSim3(const cv::Mat& Tiw,
                                      const cv::Mat& Twc,
                                      const g2o::Sim3& correctedScw);

Eigen::Vector3d CorrectMapPointWorldPosition(
    const g2o::Sim3& nonCorrectedSiw,
    const g2o::Sim3& correctedSiw,
    const Eigen::Vector3d& oldWorldPosition);

std::set<KeyFrame*> BuildLoopConnectionDifference(
    const std::set<KeyFrame*>& updatedConnections,
    const std::vector<KeyFrame*>& previousNeighbors,
    const std::vector<KeyFrame*>& currentConnectedKeyFrames);

cv::Mat PropagateGbaChildPose(const cv::Mat& childTcwBeforeGba,
                              const cv::Mat& parentTwcBeforeGba,
                              const cv::Mat& parentTcwGba);

cv::Mat CorrectGbaIndirectMapPointPosition(
    const cv::Mat& worldPositionBeforeGba,
    const cv::Mat& referenceTcwBeforeGba,
    const cv::Mat& referenceTwcAfterGba);

} // namespace CorrectLoopHelpers
} // namespace StructureSLAM

#endif // CORRECT_LOOP_HELPERS_H
