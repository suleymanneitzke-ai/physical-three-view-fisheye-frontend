#include "CorrectLoopHelpers.h"

#include "Converter.h"

namespace StructureSLAM
{
namespace CorrectLoopHelpers
{

cv::Mat Sim3ToSE3Tcw(const g2o::Sim3& Scw)
{
    Eigen::Matrix3d R = Scw.rotation().toRotationMatrix();
    Eigen::Vector3d t = Scw.translation();
    const double scale = Scw.scale();
    t *= (1.0 / scale);
    return Converter::toCvSE3(R, t);
}

g2o::Sim3 BuildCorrectedConnectedSim3(const cv::Mat& Tiw,
                                      const cv::Mat& Twc,
                                      const g2o::Sim3& correctedScw)
{
    const cv::Mat Tic = Tiw * Twc;
    const cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tic = Tic.rowRange(0, 3).col(3);
    const g2o::Sim3 Sic(Converter::toMatrix3d(Ric),
                        Converter::toVector3d(tic),
                        1.0);
    return Sic * correctedScw;
}

Eigen::Vector3d CorrectMapPointWorldPosition(
    const g2o::Sim3& nonCorrectedSiw,
    const g2o::Sim3& correctedSiw,
    const Eigen::Vector3d& oldWorldPosition)
{
    return correctedSiw.inverse().map(nonCorrectedSiw.map(oldWorldPosition));
}

std::set<KeyFrame*> BuildLoopConnectionDifference(
    const std::set<KeyFrame*>& updatedConnections,
    const std::vector<KeyFrame*>& previousNeighbors,
    const std::vector<KeyFrame*>& currentConnectedKeyFrames)
{
    std::set<KeyFrame*> loopConnections = updatedConnections;

    for(std::vector<KeyFrame*>::const_iterator it = previousNeighbors.begin(),
        end = previousNeighbors.end(); it != end; ++it)
    {
        loopConnections.erase(*it);
    }

    for(std::vector<KeyFrame*>::const_iterator it = currentConnectedKeyFrames.begin(),
        end = currentConnectedKeyFrames.end(); it != end; ++it)
    {
        loopConnections.erase(*it);
    }

    return loopConnections;
}

cv::Mat PropagateGbaChildPose(const cv::Mat& childTcwBeforeGba,
                              const cv::Mat& parentTwcBeforeGba,
                              const cv::Mat& parentTcwGba)
{
    const cv::Mat TchildParent = childTcwBeforeGba * parentTwcBeforeGba;
    return TchildParent * parentTcwGba;
}

cv::Mat CorrectGbaIndirectMapPointPosition(
    const cv::Mat& worldPositionBeforeGba,
    const cv::Mat& referenceTcwBeforeGba,
    const cv::Mat& referenceTwcAfterGba)
{
    const cv::Mat Rcw = referenceTcwBeforeGba.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = referenceTcwBeforeGba.rowRange(0, 3).col(3);
    const cv::Mat Xc = Rcw * worldPositionBeforeGba + tcw;

    const cv::Mat Rwc = referenceTwcAfterGba.rowRange(0, 3).colRange(0, 3);
    const cv::Mat twc = referenceTwcAfterGba.rowRange(0, 3).col(3);
    return Rwc * Xc + twc;
}

} // namespace CorrectLoopHelpers
} // namespace StructureSLAM
