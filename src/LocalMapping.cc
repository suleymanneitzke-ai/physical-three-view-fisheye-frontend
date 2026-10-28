/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/
/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LocalMapping.h"
#include "FisheyeLineTriangulator.h"
#include "LoopClosing.h"
#include "NativeLineIdentityBridge.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include<mutex>

namespace StructureSLAM
{
namespace
{

const char* kKb8LocalMappingCapabilityLifecycle =
    "KB8_LOCALMAPPING_LIFECYCLE_READY";
const char* kKb8LocalMappingCapabilityTriangulation =
    "KB8_NEW_MAPPOINT_TRIANGULATION_READY";
const char* kKb8LocalMappingCapabilityFusion =
    "KB8_MAPPOINT_FUSION_READY";
const char* kKb8LocalMappingCapabilityLocalBaPoint =
    "KB8_LOCAL_BA_POINT_EDGE_READY";
const char* kKb8LocalMappingCapabilityLocalBaLine =
    "KB8_LOCAL_BA_LINE_EDGE_READY";

bool Kb8LocalMappingLifecycleReady()
{
    return true;
}

bool Kb8NewMapPointTriangulationReady()
{
    return true;
}

bool Kb8MapPointFusionReady()
{
    return true;
}

bool Kb8LocalBaPointEdgeReady()
{
    return true;
}

const char* Kb8LocalBaLineEdgeState()
{
    return "EXISTING_OR_AUDIT";
}

std::mutex& Kb8LocalMappingLogMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::string Kb8LocalMappingLogPath()
{
    const char* path =
        std::getenv("STRUCTURE_SLAM_LOCALMAPPING_GUARD_LOG");
    return path ? std::string(path) : std::string();
}

std::string Kb8NewMapPointStatsPath()
{
    const char* path =
        std::getenv("STRUCTURE_SLAM_KB8_NEW_MAPPOINT_STATS");
    return path ? std::string(path) : std::string();
}

std::string Kb8NewMapPointCreatedPath()
{
    const char* path =
        std::getenv("STRUCTURE_SLAM_KB8_NEW_MAPPOINT_CREATED");
    return path ? std::string(path) : std::string();
}

bool FiniteScalar(double value)
{
    return std::isfinite(value);
}

bool FiniteVec3(const Eigen::Vector3d& value)
{
    return FiniteScalar(value.x()) &&
           FiniteScalar(value.y()) &&
           FiniteScalar(value.z());
}

bool FiniteVec2(const Eigen::Vector2d& value)
{
    return FiniteScalar(value.x()) &&
           FiniteScalar(value.y());
}

double ClampScalar(double value, double lo, double hi)
{
    return std::max(lo, std::min(hi, value));
}

double PlaneAngleDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
    constexpr double kPi = 3.14159265358979323846;
    const double denom = a.norm() * b.norm();
    if(denom <= 1e-12)
        return 0.0;
    const double c = ClampScalar(std::abs(a.dot(b) / denom), -1.0, 1.0);
    return std::acos(c) * 180.0 / kPi;
}

Eigen::Matrix3d CvMat33ToEigen(const cv::Mat& mat)
{
    Eigen::Matrix3d out = Eigen::Matrix3d::Identity();
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            out(r, c) = mat.at<float>(r, c);
    return out;
}

Eigen::Vector3d CvMat31ToEigen(const cv::Mat& mat)
{
    return Eigen::Vector3d(mat.at<float>(0),
                           mat.at<float>(1),
                           mat.at<float>(2));
}

cv::Mat EigenVec3ToCvMat31(const Eigen::Vector3d& value)
{
    return (cv::Mat_<float>(3, 1) << static_cast<float>(value.x()),
                                      static_cast<float>(value.y()),
                                      static_cast<float>(value.z()));
}

struct Kb8NewMapPointStats
{
    unsigned long current_keyframe_id = 0;
    unsigned long neighbor_keyframe_id = 0;
    int candidate_matches = 0;
    int valid_bearing_pairs = 0;
    int epipolar_gate_pass = 0;
    int triangulation_attempted = 0;
    int triangulation_finite = 0;
    int positive_or_valid_depth = 0;
    int parallax_pass = 0;
    int reprojection_pass_kf1 = 0;
    int reprojection_pass_kf2 = 0;
    int scale_consistency_pass = 0;
    int formal_mappoint_created = 0;
    int reject_invalid_bearing = 0;
    int reject_epipolar = 0;
    int reject_triangulation = 0;
    int reject_depth = 0;
    int reject_parallax = 0;
    int reject_reprojection = 0;
    int reject_scale = 0;
};

struct Kb8TriangulationResult
{
    bool ok = false;
    bool finite = false;
    bool positive_depth = false;
    bool parallax_pass = false;
    bool reprojection_pass_kf1 = false;
    bool reprojection_pass_kf2 = false;
    Eigen::Vector3d Pw = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d Pc1 = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d Pc2 = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    double parallax_deg = 0.0;
    double reprojection_error_kf1 = std::numeric_limits<double>::infinity();
    double reprojection_error_kf2 = std::numeric_limits<double>::infinity();
};

bool GetKb8Bearing(const KeyFrame* pKF,
                   size_t idx,
                   Eigen::Vector3d& bearing)
{
    bearing.setConstant(std::numeric_limits<double>::quiet_NaN());
    if(!pKF || idx >= pKF->mvPointUnitBearings.size() ||
       idx >= pKF->mvbPointBearingValid.size() ||
       !pKF->mvbPointBearingValid[idx])
    {
        return false;
    }
    bearing = pKF->mvPointUnitBearings[idx];
    const double n = bearing.norm();
    if(!FiniteVec3(bearing) || n <= 1e-12)
        return false;
    bearing /= n;
    return true;
}

bool TriangulateTwoKb8Bearings(const Eigen::Vector3d& C1,
                               const Eigen::Vector3d& d1,
                               const Eigen::Vector3d& C2,
                               const Eigen::Vector3d& d2,
                               Eigen::Vector3d& Pw)
{
    Pw.setConstant(std::numeric_limits<double>::quiet_NaN());
    const double a = d1.dot(d1);
    const double b = d1.dot(d2);
    const double c = d2.dot(d2);
    const Eigen::Vector3d w = C2 - C1;
    const double d = d1.dot(w);
    const double e = d2.dot(w);
    const double denom = a * c - b * b;
    if(!FiniteScalar(denom) || std::abs(denom) < 1e-12)
        return false;

    const double lambda1 = (d * c - b * e) / denom;
    const double lambda2 = (b * d - a * e) / denom;
    const Eigen::Vector3d P1 = C1 + lambda1 * d1;
    const Eigen::Vector3d P2 = C2 + lambda2 * d2;
    Pw = 0.5 * (P1 + P2);
    return FiniteVec3(Pw);
}

bool Kb8TriangulateAndValidate(
    KeyFrame* pKF1,
    KeyFrame* pKF2,
    size_t idx1,
    size_t idx2,
    const KannalaBrandt8Camera& camera1,
    const KannalaBrandt8Camera& camera2,
    Kb8TriangulationResult& out)
{
    out = Kb8TriangulationResult();
    Eigen::Vector3d b1;
    Eigen::Vector3d b2;
    if(!GetKb8Bearing(pKF1, idx1, b1) ||
       !GetKb8Bearing(pKF2, idx2, b2))
    {
        return false;
    }

    const Eigen::Matrix3d Rcw1 = CvMat33ToEigen(pKF1->GetRotation());
    const Eigen::Matrix3d Rwc1 = Rcw1.transpose();
    const Eigen::Vector3d tcw1 = CvMat31ToEigen(pKF1->GetTranslation());
    const Eigen::Vector3d C1 = CvMat31ToEigen(pKF1->GetCameraCenter());
    const Eigen::Matrix3d Rcw2 = CvMat33ToEigen(pKF2->GetRotation());
    const Eigen::Matrix3d Rwc2 = Rcw2.transpose();
    const Eigen::Vector3d tcw2 = CvMat31ToEigen(pKF2->GetTranslation());
    const Eigen::Vector3d C2 = CvMat31ToEigen(pKF2->GetCameraCenter());

    const Eigen::Vector3d ray1 = (Rwc1 * b1).normalized();
    const Eigen::Vector3d ray2 = (Rwc2 * b2).normalized();
    const double cos_parallax =
        ClampScalar(ray1.dot(ray2), -1.0, 1.0);
    out.parallax_deg =
        std::acos(cos_parallax) * 180.0 / 3.14159265358979323846;
    if(!(cos_parallax > 0.0 && cos_parallax < 0.9998))
        return false;
    out.parallax_pass = true;

    if(!TriangulateTwoKb8Bearings(C1, ray1, C2, ray2, out.Pw))
        return false;
    out.finite = FiniteVec3(out.Pw);
    if(!out.finite)
        return false;

    out.Pc1 = Rcw1 * out.Pw + tcw1;
    out.Pc2 = Rcw2 * out.Pw + tcw2;
    out.positive_depth =
        FiniteVec3(out.Pc1) && FiniteVec3(out.Pc2) &&
        out.Pc1.z() > pKF1->mKb8PointCameraParameters.min_positive_z &&
        out.Pc2.z() > pKF2->mKb8PointCameraParameters.min_positive_z;
    if(!out.positive_depth)
        return false;

    Eigen::Vector2d uv1_hat;
    Eigen::Vector2d uv2_hat;
    if(!camera1.project(out.Pc1, uv1_hat) ||
       !camera2.project(out.Pc2, uv2_hat) ||
       !FiniteVec2(uv1_hat) ||
       !FiniteVec2(uv2_hat))
    {
        return false;
    }

    const cv::KeyPoint& kp1 = pKF1->mvKeysUn[idx1];
    const cv::KeyPoint& kp2 = pKF2->mvKeysUn[idx2];
    const Eigen::Vector2d uv1(kp1.pt.x, kp1.pt.y);
    const Eigen::Vector2d uv2(kp2.pt.x, kp2.pt.y);
    out.reprojection_error_kf1 = (uv1_hat - uv1).norm();
    out.reprojection_error_kf2 = (uv2_hat - uv2).norm();

    const float sigma_square1 = pKF1->mvLevelSigma2[kp1.octave];
    const float sigma_square2 = pKF2->mvLevelSigma2[kp2.octave];
    out.reprojection_pass_kf1 =
        out.reprojection_error_kf1 * out.reprojection_error_kf1 <=
        5.991 * sigma_square1;
    out.reprojection_pass_kf2 =
        out.reprojection_error_kf2 * out.reprojection_error_kf2 <=
        5.991 * sigma_square2;
    out.ok = out.parallax_pass && out.finite && out.positive_depth &&
             out.reprojection_pass_kf1 && out.reprojection_pass_kf2;
    return out.ok;
}

bool Kb8EpipolarTriangulationGate(
    KeyFrame* pKF1,
    KeyFrame* pKF2,
    size_t idx1,
    size_t idx2,
    const KannalaBrandt8Camera& camera1,
    const KannalaBrandt8Camera& camera2,
    Kb8NewMapPointStats& stats)
{
    Eigen::Vector3d b1;
    Eigen::Vector3d b2;
    if(!GetKb8Bearing(pKF1, idx1, b1) ||
       !GetKb8Bearing(pKF2, idx2, b2))
    {
        ++stats.reject_invalid_bearing;
        return false;
    }
    ++stats.valid_bearing_pairs;

    Kb8TriangulationResult result;
    if(!Kb8TriangulateAndValidate(
           pKF1, pKF2, idx1, idx2, camera1, camera2, result))
    {
        ++stats.reject_epipolar;
        return false;
    }
    ++stats.epipolar_gate_pass;
    return true;
}

void ComputeThreeMaximaLocal(std::vector<int>* histo,
                             const int L,
                             int& ind1,
                             int& ind2,
                             int& ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;
    ind1 = -1;
    ind2 = -1;
    ind3 = -1;
    for(int i = 0; i < L; ++i)
    {
        const int s = static_cast<int>(histo[i].size());
        if(s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if(s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if(s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }
    if(max2 < 0.1f * static_cast<float>(max1))
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if(max3 < 0.1f * static_cast<float>(max1))
    {
        ind3 = -1;
    }
}

int SearchForKb8TriangulationMatches(
    KeyFrame* pKF1,
    KeyFrame* pKF2,
    const KannalaBrandt8Camera& camera1,
    const KannalaBrandt8Camera& camera2,
    std::vector<std::pair<size_t, size_t>>& vMatchedPairs,
    Kb8NewMapPointStats& stats)
{
    const DBoW2::FeatureVector& vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector& vFeatVec2 = pKF2->mFeatVec;

    int nmatches = 0;
    std::vector<bool> vbMatched2(pKF2->N, false);
    std::vector<int> vMatches12(pKF1->N, -1);
    std::vector<int> rotHist[ORBmatcher::HISTO_LENGTH];
    for(int i = 0; i < ORBmatcher::HISTO_LENGTH; ++i)
        rotHist[i].reserve(500);
    const float factor = 1.0f / ORBmatcher::HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    const DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    const DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1 = 0, iend1 = f1it->second.size();
                i1 < iend1; ++i1)
            {
                const size_t idx1 = f1it->second[i1];
                if(pKF1->GetMapPoint(idx1))
                    continue;

                const cv::KeyPoint& kp1 = pKF1->mvKeysUn[idx1];
                const cv::Mat& d1 = pKF1->mDescriptors.row(idx1);

                int bestDist = ORBmatcher::TH_LOW;
                int bestIdx2 = -1;
                for(size_t i2 = 0, iend2 = f2it->second.size();
                    i2 < iend2; ++i2)
                {
                    const size_t idx2 = f2it->second[i2];
                    if(vbMatched2[idx2] || pKF2->GetMapPoint(idx2))
                        continue;

                    const cv::Mat& d2 = pKF2->mDescriptors.row(idx2);
                    const int dist = ORBmatcher::DescriptorDistance(d1, d2);
                    if(dist > ORBmatcher::TH_LOW || dist > bestDist)
                        continue;
                    ++stats.candidate_matches;

                    if(!Kb8EpipolarTriangulationGate(
                           pKF1, pKF2, idx1, idx2, camera1, camera2, stats))
                    {
                        continue;
                    }

                    bestIdx2 = static_cast<int>(idx2);
                    bestDist = dist;
                }

                if(bestIdx2 >= 0)
                {
                    const cv::KeyPoint& kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1] = bestIdx2;
                    vbMatched2[bestIdx2] = true;
                    ++nmatches;

                    float rot = kp1.angle - kp2.angle;
                    if(rot < 0.0f)
                        rot += 360.0f;
                    int bin = static_cast<int>(std::round(rot * factor));
                    if(bin == ORBmatcher::HISTO_LENGTH)
                        bin = 0;
                    if(bin >= 0 && bin < ORBmatcher::HISTO_LENGTH)
                        rotHist[bin].push_back(static_cast<int>(idx1));
                }
            }
            ++f1it;
            ++f2it;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;
    ComputeThreeMaximaLocal(rotHist, ORBmatcher::HISTO_LENGTH,
                            ind1, ind2, ind3);
    for(int i = 0; i < ORBmatcher::HISTO_LENGTH; ++i)
    {
        if(i == ind1 || i == ind2 || i == ind3)
            continue;
        for(size_t j = 0, jend = rotHist[i].size(); j < jend; ++j)
        {
            const int idx = rotHist[i][j];
            if(idx >= 0 && idx < static_cast<int>(vMatches12.size()) &&
               vMatches12[idx] >= 0)
            {
                vbMatched2[vMatches12[idx]] = false;
                vMatches12[idx] = -1;
                --nmatches;
            }
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(std::max(0, nmatches));
    for(size_t i = 0, iend = vMatches12.size(); i < iend; ++i)
    {
        if(vMatches12[i] >= 0)
            vMatchedPairs.push_back(
                std::make_pair(i, static_cast<size_t>(vMatches12[i])));
    }
    return static_cast<int>(vMatchedPairs.size());
}

void AppendKb8NewMapPointStats(const Kb8NewMapPointStats& stats)
{
    const std::string path = Kb8NewMapPointStatsPath();
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(Kb8LocalMappingLogMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << stats.current_keyframe_id << ','
        << stats.neighbor_keyframe_id << ','
        << stats.candidate_matches << ','
        << stats.valid_bearing_pairs << ','
        << stats.epipolar_gate_pass << ','
        << stats.triangulation_attempted << ','
        << stats.triangulation_finite << ','
        << stats.positive_or_valid_depth << ','
        << stats.parallax_pass << ','
        << stats.reprojection_pass_kf1 << ','
        << stats.reprojection_pass_kf2 << ','
        << stats.scale_consistency_pass << ','
        << stats.formal_mappoint_created << ','
        << stats.reject_invalid_bearing << ','
        << stats.reject_epipolar << ','
        << stats.reject_triangulation << ','
        << stats.reject_depth << ','
        << stats.reject_parallax << ','
        << stats.reject_reprojection << ','
        << stats.reject_scale << '\n';
}

void AppendKb8CreatedMapPoint(KeyFrame* pKF1,
                              KeyFrame* pKF2,
                              size_t idx1,
                              size_t idx2,
                              const Kb8TriangulationResult& result)
{
    const std::string path = Kb8NewMapPointCreatedPath();
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(Kb8LocalMappingLogMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    const cv::KeyPoint& kp1 = pKF1->mvKeysUn[idx1];
    const cv::KeyPoint& kp2 = pKF2->mvKeysUn[idx2];
    out << std::setprecision(17)
        << pKF1->mnId << ','
        << pKF2->mnId << ','
        << idx1 << ','
        << idx2 << ','
        << result.parallax_deg << ','
        << result.reprojection_error_kf1 << ','
        << result.reprojection_error_kf2 << ','
        << kp1.octave << ','
        << kp2.octave << ','
        << result.Pw.x() << ','
        << result.Pw.y() << ','
        << result.Pw.z() << '\n';
}

FisheyeLineTriangulator::Pose MakeNativePose(KeyFrame* pKF)
{
    FisheyeLineTriangulator::Pose pose;
    pose.frame_id = static_cast<int>(pKF->mnFrameId);
    pose.timestamp = pKF->mTimeStamp;
    pose.Rcw = CvMat33ToEigen(pKF->GetRotation());
    pose.tcw = CvMat31ToEigen(pKF->GetTranslation());
    pose.Rwc = pose.Rcw.transpose();
    pose.twc = CvMat31ToEigen(pKF->GetCameraCenter());
    return pose;
}

FisheyeLineTriangulator::TrackObs MakeNativeTrackObs(
    const FisheyeLineObservation& observation)
{
    FisheyeLineTriangulator::TrackObs out;
    out.track_id = static_cast<int>(observation.temporal_track_id);
    out.frame_id = static_cast<int>(observation.frame_id);
    out.timestamp = observation.timestamp;
    out.view_id = observation.view_id;
    out.line_id = observation.line_id >= 0 ?
                  observation.line_id : observation.native_line_id;
    out.track_length = 2;
    out.match_score_to_previous =
        observation.response > 0.0 ? observation.response : 1.0;
    return out;
}

FisheyeLineTriangulator::LineInfo MakeNativeLineInfo(
    const FisheyeLineObservation& observation)
{
    FisheyeLineTriangulator::LineInfo out;
    out.virtual_x1 = observation.virtual_start.x;
    out.virtual_y1 = observation.virtual_start.y;
    out.virtual_x2 = observation.virtual_end.x;
    out.virtual_y2 = observation.virtual_end.y;
    out.virtual_length_px = observation.virtual_length_px;
    out.valid_sample_ratio = observation.valid_sample_ratio;
    return out;
}

std::vector<FisheyeLineTriangulator::Sample> MakeNativeSamples(
    const FisheyeLineObservation& observation)
{
    std::vector<FisheyeLineTriangulator::Sample> samples;
    const size_t count = std::min(observation.fisheye_samples.size(),
                                  observation.virtual_samples.size());
    samples.reserve(count + 2);
    for(size_t i = 0; i < count; ++i)
    {
        FisheyeLineTriangulator::Sample sample;
        sample.fisheye_u = observation.fisheye_samples[i].x;
        sample.fisheye_v = observation.fisheye_samples[i].y;
        sample.virtual_u = observation.virtual_samples[i].x;
        sample.virtual_v = observation.virtual_samples[i].y;
        sample.valid = i >= observation.sample_valid.size() ||
                       observation.sample_valid[i] != 0;
        samples.push_back(sample);
    }
    if(samples.empty())
    {
        FisheyeLineTriangulator::Sample start;
        start.fisheye_u = observation.fisheye_start.x;
        start.fisheye_v = observation.fisheye_start.y;
        start.virtual_u = observation.virtual_start.x;
        start.virtual_v = observation.virtual_start.y;
        start.valid = true;
        samples.push_back(start);

        FisheyeLineTriangulator::Sample end;
        end.fisheye_u = observation.fisheye_end.x;
        end.fisheye_v = observation.fisheye_end.y;
        end.virtual_u = observation.virtual_end.x;
        end.virtual_v = observation.virtual_end.y;
        end.valid = true;
        samples.push_back(end);
    }
    return samples;
}

bool PositiveDepthInKeyFrame(KeyFrame* pKF, const Eigen::Vector3d& Xw)
{
    const Eigen::Matrix3d Rcw = CvMat33ToEigen(pKF->GetRotation());
    const Eigen::Vector3d tcw = CvMat31ToEigen(pKF->GetTranslation());
    const Eigen::Vector3d Xc = Rcw * Xw + tcw;
    // Native CAM_B fisheye visibility is not a pinhole positive-z test.
    // The calibrated fisheye projection accepts bearings beyond 90 degrees;
    // finite support is checked by the reprojection sampler below.
    return FiniteVec3(Xc);
}

bool NativeObservationUsable(const FisheyeLineObservation& observation)
{
    return observation.temporal_track_id !=
               FisheyeLineObservation::kInvalidTrackId &&
           !observation.outlier &&
           observation.inlier;
}

} // namespace

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        // 告诉Tracking线程，LocalMapping正处于繁忙状态，LocalMapping线程处理的关键帧都是Tracking线程发过来的，
        // 在LocalMapping线程还没有处理完关键帧之前，Tracking线程最好不要发太快
        //cout<<"localmapping: setAccetkeyFrames"<<endl;
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        // 检查是否有在排队的关键帧
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            // 计算关键帧特征点的BoW映射，将关键帧插入地图
            ProcessNewKeyFrame();

            // Check recent MapPoints
            // VI-B recent map points culling
            // 剔除ProcessNewKeyFrame函数中引入的不合格的MapPoints  删除不好的MapLines
            thread threadCullPoint(&LocalMapping::MapPointCulling, this);
            thread threadCullLine(&LocalMapping::MapLineCulling, this);
            threadCullPoint.join();
            threadCullLine.join();

            const unsigned long map_points_after_lifecycle =
                mpMap ? mpMap->MapPointsInMap() : 0;
            const unsigned long map_lines_after_lifecycle =
                mpMap ? mpMap->MapLinesInMap() : 0;
            LogKb8LocalMappingGuardEvent(
                "ProcessNewKeyFrame+RecentCulling",
                false,
                "LIFECYCLE_ONLY",
                map_points_after_lifecycle,
                map_points_after_lifecycle,
                map_lines_after_lifecycle,
                map_lines_after_lifecycle);

            if(ShouldGuardKb8BeforeNewMapPointTriangulation())
            {
                LogKb8LocalMappingGuardEvent(
                    "BeforeCreateNewMapPoints",
                    true,
                    "KB8_NEW_MAPPOINT_TRIANGULATION_NOT_READY",
                    map_points_after_lifecycle,
                    map_points_after_lifecycle,
                    map_lines_after_lifecycle,
                    map_lines_after_lifecycle);
                mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
                ResetIfRequested();
                SetAcceptKeyFrames(true);
                continue;
            }


            // Triangulate new MapPoints
            // VI-C new map points creation
            // 相机运动过程中与相邻关键帧通过三角化恢复出一些MapPoints
//            ofstream file3("FeaturesCreationTime.txt", ios::app);
//            chrono::steady_clock::time_point t5 = chrono::steady_clock::now();

//            CreateNewMapPoints();
//            CreateNewMapLines2();   //采用极平面方式

            thread threadCreateP(&LocalMapping::CreateNewMapPoints, this);
            threadCreateP.join();
            // POINT_ONLY keyframes intentionally carry no legacy line
            // descriptors.  Do not enter the line triangulation path, whose
            // matcher assumes populated line descriptor containers.
            if(mpCurrentKeyFrame && mpCurrentKeyFrame->NL > 0)
            {
                thread threadCreateL(&LocalMapping::CreateNewMapLines2, this);
                threadCreateL.join();
            }
            //cout<<"localmapping:finish creating"<<endl;
//            chrono::steady_clock::time_point t6 = chrono::steady_clock::now();
//            chrono::duration<double> time_used3 = chrono::duration_cast<chrono::duration<double>>(t6-t5);
//            cout << "CreateMapPL time: " << time_used3.count() << endl;
//            file3 << time_used3.count() << endl;
//            file3.close();

            const unsigned long map_points_after_creation =
                mpMap ? mpMap->MapPointsInMap() : 0;
            const unsigned long map_lines_after_creation =
                mpMap ? mpMap->MapLinesInMap() : 0;
            if(ShouldGuardKb8BeforeMapPointFusion())
            {
                LogKb8LocalMappingGuardEvent(
                    "BeforeSearchInNeighbors",
                    true,
                    "KB8_MAPPOINT_FUSION_NOT_READY",
                    map_points_after_lifecycle,
                    map_points_after_creation,
                    map_lines_after_lifecycle,
                    map_lines_after_creation);
                mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
                ResetIfRequested();
                SetAcceptKeyFrames(true);
                continue;
            }

            // 已经处理完关键帧队列中最后一个关键帧
            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                // 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints,一级重复的MapLines
                //cout<<"localmapping: start searchIn Neighbors"<<endl;
                SearchInNeighbors();
                //SearchLineInNeighbors();
            }

            const unsigned long map_points_after_fusion =
                mpMap ? mpMap->MapPointsInMap() : 0;
            const unsigned long map_lines_after_fusion =
                mpMap ? mpMap->MapLinesInMap() : 0;
            if(ShouldGuardKb8BeforeLocalBA())
            {
                LogKb8LocalMappingGuardEvent(
                    "BeforeLocalBundleAdjustment",
                    true,
                    "KB8_LOCAL_BA_POINT_EDGE_NOT_READY",
                    map_points_after_creation,
                    map_points_after_fusion,
                    map_lines_after_creation,
                    map_lines_after_fusion);
                mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
                ResetIfRequested();
                SetAcceptKeyFrames(true);
                continue;
            }

            mbAbortBA = false;

            // 已经处理完队列中的最后一个关键帧，并且闭环检测没有请求停止LocalMapping
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // VI-D Local BA
                if(mpMap->KeyFramesInMap()>2)
                {
//                    ofstream file4("LocalBATime.txt", ios::app);
//                    chrono::steady_clock::time_point t7 = chrono::steady_clock::now();
                    cout<<"localmapping: start localBAwith line"<<endl;
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);     //包含线特征的局部BA
                    cout<<"localmapping: finish localBAwith line"<<endl;
//                    chrono::steady_clock::time_point t8 = chrono::steady_clock::now();
//                    chrono::duration<double> time_used4 = chrono::duration_cast<chrono::duration<double>>(t8-t7);
//                    cout << "LocalBA time: " << time_used4.count() << endl;
//                    file4 << time_used4.count() << endl;
//                    file4.close();
                }


                // Check redundant local Keyframes
                // VI-E local keyframes culling
                // 检测并剔除当前帧相邻的关键帧中冗余的关键帧
                // 剔除的标准：该关键帧的90%的MapPoints可以被其他关键帧观测到
                // trick：
                // Tracking中先把关键帧交给LocalMapping线程，并且在Tracking中InsertKeyFrame函数的条件比较松，交给LocalMapping线程的关键帧会比较密
                // 在这里再删除冗余的关键帧
//                ofstream file5("KFCullingTime.txt",ios::app);
//                chrono::steady_clock::time_point t9 = chrono::steady_clock::now();
                KeyFrameCulling();
//                chrono::steady_clock::time_point t10 = chrono::steady_clock::now();
//                chrono::duration<double> time_used5 = chrono::duration_cast<chrono::duration<double>>(t10-t9);
//                cout << "CullKF time: " << time_used5.count() << endl;
//                file5 << time_used5.count() << endl;
//                file5.close();
            }

            // 将当前帧插入到闭环检测队列中
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}

bool LocalMapping::CurrentKeyFrameUsesKb8() const
{
    return mpCurrentKeyFrame &&
           mpCurrentKeyFrame->mbUseKb8PointObservations;
}

bool LocalMapping::ShouldGuardKb8BeforeNewMapPointTriangulation() const
{
    return CurrentKeyFrameUsesKb8() &&
           Kb8LocalMappingLifecycleReady() &&
           !Kb8NewMapPointTriangulationReady();
}

bool LocalMapping::ShouldGuardKb8BeforeMapPointFusion() const
{
    return CurrentKeyFrameUsesKb8() &&
           Kb8LocalMappingLifecycleReady() &&
           Kb8NewMapPointTriangulationReady() &&
           !Kb8MapPointFusionReady();
}

bool LocalMapping::ShouldGuardKb8BeforeLocalBA() const
{
    return CurrentKeyFrameUsesKb8() &&
           Kb8LocalMappingLifecycleReady() &&
           Kb8NewMapPointTriangulationReady() &&
           Kb8MapPointFusionReady() &&
           !Kb8LocalBaPointEdgeReady();
}

void LocalMapping::LogKb8LocalMappingGuardEvent(
    const std::string& stage,
    const bool guard_triggered,
    const std::string& guard_reason,
    const unsigned long map_points_before,
    const unsigned long map_points_after,
    const unsigned long map_lines_before,
    const unsigned long map_lines_after,
    const int tracking_state) const
{
    if(!CurrentKeyFrameUsesKb8())
        return;

    const std::string path = Kb8LocalMappingLogPath();
    if(path.empty())
        return;

    std::lock_guard<std::mutex> lock(Kb8LocalMappingLogMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;

    out << mpCurrentKeyFrame->mnFrameId << ','
        << mpCurrentKeyFrame->mnId << ','
        << stage << ','
        << (guard_triggered ? 1 : 0) << ','
        << guard_reason << ','
        << map_points_before << ','
        << map_points_after << ','
        << map_lines_before << ','
        << map_lines_after << ','
        << tracking_state << ','
        << 1 << ','
        << kKb8LocalMappingCapabilityLifecycle << ":READY;"
        << kKb8LocalMappingCapabilityTriangulation
        << (Kb8NewMapPointTriangulationReady()
                ? ":READY;"
                : ":NOT_READY;")
        << kKb8LocalMappingCapabilityFusion
        << (Kb8MapPointFusionReady()
                ? ":READY;"
                : ":NOT_READY;")
        << kKb8LocalMappingCapabilityLocalBaPoint
        << (Kb8LocalBaPointEdgeReady()
                ? ":READY;"
                : ":NOT_READY;")
        << kKb8LocalMappingCapabilityLocalBaLine << ':'
        << Kb8LocalBaLineEdgeState()
        << '\n';
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

/**
 * @brief 处理队列中的关键帧
 *
 * --计算BoW，加速三角化新的MapPoint
 * --关联当前关键帧至MapPoints，并更新MapPoints的平均观测方向和观测距离范围
 * --插入关键帧，更新Covisibility图和Essential图
 */
void LocalMapping::ProcessNewKeyFrame()
{
    // step1：从缓冲队列中取出一帧关键帧
    // Tracking线程向LocalMapping中插入的关键帧在该队列中
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        // 从列表中获取一个等待被插入的关键帧
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    // step2: 计算该关键帧特征点的BoW映射关系
    mpCurrentKeyFrame->ComputeBoW();

    const vector<FisheyeLineObservation>& native_observations =
        mpCurrentKeyFrame->GetNativeFisheyeLineObservations();
    for(size_t i = 0; i < native_observations.size(); ++i)
    {
        const FisheyeLineObservation& observation =
            native_observations[i];
        if(observation.temporal_track_id ==
           FisheyeLineObservation::kInvalidTrackId)
        {
            continue;
        }
        mpMap->NativeLineBridge().ObserveTrack(
            mpCurrentKeyFrame, i, observation.temporal_track_id);
    }

    // Associate MapPoints to the new keyframe and update normal and descriptor
    // step3:跟踪局部地图过程中新匹配上的MapPoints和当前关键帧绑定
    // 在TrackLocalMap函数中将局部地图中的MapPoints与当前帧进行了匹配
    // 但没有对这些匹配上的MapPoints与当前帧进行关联
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                // 为当前帧在Tracking过程中跟踪到的MapPoints更新属性
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    // 添加观测
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    // 获得该点的平均观测方向和观测距离范围
                    pMP->UpdateNormalAndDepth();
                    // 加入关键帧后，更新3D点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    // 将双目或RGBD跟踪过程中新插入的MapPoints放入mlpRecentAddedMapPoints，等待检查
                    // CreateNewMapPoints函数中通过三角化也会生成MapPoints
                    // 这些MapPoints都会经过MapPointCulling函数的检验
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    /// 跟踪局部地图过程中新匹配上的MapLines,和当前关键帧进行绑定
    const vector<MapLine*> vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();

    for(size_t i=0; i<vpMapLineMatches.size(); i++)
    {
        MapLine* pML = vpMapLineMatches[i];
        if(pML)
        {
            if(!pML->isBad())
            {
                if(!pML->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pML->AddObservation(mpCurrentKeyFrame, i);  //添加观测
                    pML->UpdateAverageDir();    //更新观测方向
                    pML->ComputeDistinctiveDescriptors();
                } else
                {
                    mlpRecentAddedMapLines.push_back(pML);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    // step6；更新关键帧之间的连接关系，Covisibility Graph和Essential Graph
    mpCurrentKeyFrame->UpdateConnections(); //此处与MapPoint有关，暂时不修改

    // Insert Keyframe in Map
    // step5：将该关键帧插入到地图中
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

/**
 * @brief 剔除ProcessNewKeyFrame和CreateNewMapPoints函数中引入的质量不好的MapPoints
 */
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    // 遍历等待检查的MapPoints
    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            // step1：已经是坏点的MapPoint，直接从检查链表中剔除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            // step2:将不满足VI-B条件的MapPoint剔除
            // VI-B 条件1：
            // 跟踪到该MapPoint的Frame数量相比预计可观测到该MapPoint的Frame数的比例需要大于25%
            // IncreaseFound / IncreaseVisible < 25%，注意不一定是关键帧
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            // step3: 将不满足VI-B条件的MApPoint剔除
            // VI-B 条件2：从该点建立开始，到现在已经过了不小于2帧，但是观测到该点的关键帧数却不超过cnThObs，那么该点检验不合格
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            // step4：从建立该点开始，已经过了3帧，放弃对该MapPoint的检测
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::MapLineCulling()
{
    // Check Recent Added MapLines
    list<MapLine*>::iterator lit = mlpRecentAddedMapLines.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    // 遍历等待检查的MapLines
    while(lit!=mlpRecentAddedMapLines.end())
    {
        MapLine* pML = *lit;
        if(pML->isBad())
        {
            // step1: 将已经是坏的MapLine从检查链中删除
            lit = mlpRecentAddedMapLines.erase(lit);
        }
        else if(pML->GetFoundRatio()<0.6f)
        {
            pML->SetBadFlag();
            lit = mlpRecentAddedMapLines.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pML->mnFirstKFid)>=2 && pML->Observations()<=cnThObs)
        {
            pML->SetBadFlag();
            lit = mlpRecentAddedMapLines.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pML->mnFirstKFid)>=2)
            lit = mlpRecentAddedMapLines.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPointsKb8()
{
    int nn = 10;
    if(mbMonocular)
        nn = 20;

    const vector<KeyFrame*> vpNeighKFs =
        mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    const Eigen::Vector3d Cw1 =
        CvMat31ToEigen(mpCurrentKeyFrame->GetCameraCenter());

    const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;
    const KannalaBrandt8Camera camera1(
        mpCurrentKeyFrame->mKb8PointCameraParameters);
    if(!camera1.valid())
        return;

    for(size_t i = 0; i < vpNeighKFs.size(); ++i)
    {
        if(i > 0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];
        if(!pKF2 || pKF2->isBad() || !pKF2->mbUseKb8PointObservations)
            continue;

        Kb8NewMapPointStats stats;
        stats.current_keyframe_id = mpCurrentKeyFrame->mnId;
        stats.neighbor_keyframe_id = pKF2->mnId;

        const Eigen::Vector3d Cw2 =
            CvMat31ToEigen(pKF2->GetCameraCenter());
        const double baseline = (Cw2 - Cw1).norm();

        if(!mbMonocular)
        {
            if(baseline < pKF2->mb)
            {
                AppendKb8NewMapPointStats(stats);
                continue;
            }
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth =
                static_cast<float>(baseline) / medianDepthKF2;
            if(ratioBaselineDepth < 0.01f)
            {
                AppendKb8NewMapPointStats(stats);
                continue;
            }
        }

        const KannalaBrandt8Camera camera2(
            pKF2->mKb8PointCameraParameters);
        if(!camera2.valid())
        {
            AppendKb8NewMapPointStats(stats);
            continue;
        }

        vector<pair<size_t, size_t> > vMatchedIndices;
        SearchForKb8TriangulationMatches(
            mpCurrentKeyFrame, pKF2, camera1, camera2,
            vMatchedIndices, stats);

        const int nmatches = static_cast<int>(vMatchedIndices.size());
        const cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();
        const cv::Mat Ow2 = pKF2->GetCameraCenter();

        for(int ikp = 0; ikp < nmatches; ++ikp)
        {
            const int idx1 = static_cast<int>(vMatchedIndices[ikp].first);
            const int idx2 = static_cast<int>(vMatchedIndices[ikp].second);
            const cv::KeyPoint& kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const cv::KeyPoint& kp2 = pKF2->mvKeysUn[idx2];

            ++stats.triangulation_attempted;
            Kb8TriangulationResult result;
            if(!Kb8TriangulateAndValidate(
                   mpCurrentKeyFrame, pKF2, idx1, idx2,
                   camera1, camera2, result))
            {
                if(!result.parallax_pass)
                    ++stats.reject_parallax;
                else if(!result.finite)
                    ++stats.reject_triangulation;
                else if(!result.positive_depth)
                    ++stats.reject_depth;
                else
                    ++stats.reject_reprojection;
                continue;
            }

            ++stats.triangulation_finite;
            ++stats.positive_or_valid_depth;
            ++stats.parallax_pass;
            ++stats.reprojection_pass_kf1;
            ++stats.reprojection_pass_kf2;

            cv::Mat x3D = EigenVec3ToCvMat31(result.Pw);

            cv::Mat normal1 = x3D - Ow1;
            const float dist1 = cv::norm(normal1);
            cv::Mat normal2 = x3D - Ow2;
            const float dist2 = cv::norm(normal2);
            if(dist1 == 0.0f || dist2 == 0.0f)
            {
                ++stats.reject_scale;
                continue;
            }

            const float ratioDist = dist2 / dist1;
            const float ratioOctave =
                mpCurrentKeyFrame->mvScaleFactors[kp1.octave] /
                pKF2->mvScaleFactors[kp2.octave];
            if(ratioDist * ratioFactor < ratioOctave ||
               ratioDist > ratioOctave * ratioFactor)
            {
                ++stats.reject_scale;
                continue;
            }
            ++stats.scale_consistency_pass;

            MapPoint* pMP =
                new MapPoint(x3D, mpCurrentKeyFrame, mpMap);
            pMP->AddObservation(mpCurrentKeyFrame, idx1);
            pMP->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
            ++stats.formal_mappoint_created;
            AppendKb8CreatedMapPoint(
                mpCurrentKeyFrame, pKF2, idx1, idx2, result);
        }

        AppendKb8NewMapPointStats(stats);
    }
}

/**
 * @brief 相机运动过程中和共视程度比较高的关键帧通过三角化恢复出一些MapPoints
 */
void LocalMapping::CreateNewMapPoints()
{
    if(CurrentKeyFrameUsesKb8())
    {
        CreateNewMapPointsKb8();
        return;
    }

    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;

    // -step1: 在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻的vpNeighKFs
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    // 得到当前关键帧在世界坐标系中的坐标
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    // -step2：遍历相邻关键帧vpNeighKFs,根据对极约束寻找匹配对，并且三角化
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();  // 邻接关键帧在世界坐标系中的坐标
        cv::Mat vBaseline = Ow2-Ow1;    //基线向量，两个关键帧间的相机位移
        const float baseline = cv::norm(vBaseline); //基线长度，两个关键帧的位移长度

        // step3：判断两个关键帧之间的位移是不是足够长，如果太短就没必要计算
        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)   //如果是双目相机，关键帧间距太小时，不生成3D点
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);  //邻接关键帧的场景深度中值
            const float ratioBaselineDepth = baseline/medianDepthKF2;   //两个关键帧的位移长度与场景深度的比值

            if(ratioBaselineDepth<0.01) //如果特别远，则不考虑当前邻接的关键帧，不生成3D点
                continue;
        }

        // Compute Fundamental Matrix
        // -step4：根据两个关键帧的位姿计算它们之间的基本矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        // -step5:根据极线约束限制匹配时的搜索范围，进行特征点匹配
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        // step6：对每对匹配通过三角化生成3D点，和Triangulate函数差不多
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            // step6.1：取出匹配的特征点
            const int &idx1 = vMatchedIndices[ikp].first;   //匹配对在当前关键帧中的索引
            const int &idx2 = vMatchedIndices[ikp].second;  //匹配对在邻接关键帧中的索引

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            // 得到特征点在归一化平面上的坐标
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            // 将特征点坐标，世界坐标系下 观察光线
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            // 计算在世界坐标系下，两个坐标向量间的余弦值
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            // 三角化恢复3D点坐标
            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            // 检测生成的3D点是否在相机前方，两个关键帧都需要检验，可以参考《视觉SLAM十四讲》三角化那一节
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            // 在第一个关键帧中检测重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            //将三角化的点再转换到相机坐标系下，归一化平面上
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                // 将上面计算得到的相机坐标系下的坐标投影到像素平面上
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                // 计算重投影误差
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)    //5.991是基于卡方检验计算出的阈值，假设测量有一个像素的偏差
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            // 计算3D点在另一个关键帧下的重投影误差，和上面步骤一样
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)  ///7.8和上面的5.991联系？
                    continue;
            }

            //Check scale consistency
            // 检测尺度连续性
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);  //构造MapPoint

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::CreateNewMapLines1()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn=10;
    if(mbMonocular)
        nn=20;
    //step1：在当前关键帧的共视关键帧中找到共视成都最高的nn帧相邻帧vpNeighKFs
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    LSDmatcher lmatcher;

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    //得到当前关键帧在世界坐标系中的坐标
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    // step2: 遍历相邻关键帧vpNeighKFs
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        // 邻接的关键帧在世界坐标系中的坐标
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        // 基线向量，两个关键帧间的相机位移
        cv::Mat vBaseline = Ow2 - Ow1;
        // 基线长度
        const float baseline = cv::norm(vBaseline);

        // step3：判断相机运动的基线是不是足够长
        if(!mbMonocular)
        {
            // 如果是立体相机，关键帧间距太小时不生成3D点
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            // 邻接关键帧的场景深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            // baseline 与景深的比例
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            // 如果特别远（比例特别小），那么不考虑当前邻接的关键帧，不生成3D点
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        // step4：根据两个关键帧的位姿计算它们之间的基本矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fulfill epipolar constraint
        // step5：通过极线约束限制匹配时的搜索单位，进行特征点匹配
        vector<pair<size_t, size_t>> vMatchedIndices;
        lmatcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        // step6：对每对匹配通过三角化生成3D点
        const int nmatches = vMatchedIndices.size();
        for(int ikl=0; ikl<nmatches; ikl++)
        {
            // step6.1：取出匹配的特征线
            const int &idx1 = vMatchedIndices[ikl].first;
            const int &idx2 = vMatchedIndices[ikl].second;

            const KeyLine &keyline1 = mpCurrentKeyFrame->mvKeyLines[idx1];
            const KeyLine &keyline2 = pKF2->mvKeyLines[idx2];
            const Vector3d keyline2_function = pKF2->mvKeyLineFunctions[idx2];

            // 特征线段的中点
            Point2f midP1 = keyline1.pt;
            Point2f midP2 = keyline2.pt;
            // step6.2:将两个线段的中点反投影得到视差角
            // 特征线段的中点反投影
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (midP1.x - cx1)*invfx1, (midP1.y - cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (midP2.x - cx2)*invfx2, (midP2.y - cy2)*invfy2, 1.0);

            // 由相机坐标系转到世界坐标系，得到视差角余弦值
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));
//            cosParallaxRays = cosParallaxRays + 1;

            // step6.3：线段端点在两帧图像中的坐标
            cv::Mat StartC1, EndC1, StartC2, EndC2;
            StartC1 = (cv::Mat_<float>(3,1) << (keyline1.startPointX-cx1)*invfx1, (keyline1.startPointY-cy1)*invfy1, 1.0);
            EndC1 = (cv::Mat_<float>(3,1) << (keyline1.endPointX-cx1)*invfx1, (keyline1.endPointY-cy1)*invfy1, 1.0);
            StartC2 = (cv::Mat_<float>(3,1) << (keyline2.startPointX-cx2)*invfx2, (keyline2.startPointY-cy2)*invfy2, 1.0);
            EndC2 = (cv::Mat_<float>(3,1) << (keyline2.endPointX-cx2)*invfx2, (keyline2.endPointY-cy2)*invfy2, 1.0);

            // step6.4：三角化恢复线段的3D端点
            cv::Mat s3D, e3D;
            if(cosParallaxRays>0 && cosParallaxRays<0.9998)
            {
                cv::Mat A(4,4,CV_32F);
                A.row(0) = StartC1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = StartC1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = StartC2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = StartC2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w1, u1, vt1;
                cv::SVD::compute(A, w1, u1, vt1, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                s3D = vt1.row(3).t();

                if(s3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                s3D = s3D.rowRange(0,3)/s3D.at<float>(3);

                cv::Mat B(4,4,CV_32F);
                B.row(0) = EndC1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                B.row(1) = EndC1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                B.row(2) = EndC2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                B.row(3) = EndC2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w2, u2, vt2;
                cv::SVD::compute(B, w2, u2, vt2, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                e3D = vt2.row(3).t();

                if(e3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                e3D = e3D.rowRange(0,3)/e3D.at<float>(3);
            } else
                continue;

            cv::Mat s3Dt = s3D.t();
            cv::Mat e3Dt = e3D.t();

            // step6.5：检测生成的3D点是否在相机前方
            float SZC1 = Rcw1.row(2).dot(s3Dt)+tcw1.at<float>(2);   //起始点在C1下的Z坐标值
            if(SZC1<=0)
                continue;

            float SZC2 = Rcw2.row(2).dot(s3Dt)+tcw2.at<float>(2);   //起始点在C2下的Z坐标值
            if(SZC2<=0)
                continue;

            float EZC1 = Rcw1.row(2).dot(e3Dt)+tcw1.at<float>(2);   //终止点在C1下的Z坐标值
            if(EZC1<=0)
                continue;

            float EZC2 = Rcw2.row(2).dot(e3Dt)+tcw2.at<float>(2);   //终止点在C2下的Z坐标值
            if(EZC2<=0)
                continue;

            // step6.6：计算3D点在当前关键帧下的重投影误差
            const float &sigmaSquare2 = mpCurrentKeyFrame->mvLevelSigma2[keyline1.octave];
            // -1.该keyline在当前帧中所在直线系数
            Vector3d lC1 = mpCurrentKeyFrame->mvKeyLineFunctions[idx1];

            // -2.起始点在当前帧的重投影误差
            const float x1 = Rcw1.row(0).dot(s3Dt) + tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(s3Dt) + tcw1.at<float>(1);
            float e1 = lC1(0)*x1 + lC1(1)*y1 + lC1(2);

            // -3.终止点在当前帧的重投影误差
            const float x2 = Rcw1.row(0).dot(e3Dt) + tcw1.at<float>(0);
            const float y2 = Rcw1.row(1).dot(e3Dt) + tcw1.at<float>(1);
            float e2 = lC1(0)*x2 + lC1(1)*y2 + lC1(2);

            // -4.判断线段在当前帧的重投影误差是否符合阈值
            float eC1 = e1 + e2;
            if(eC1>7.8*sigmaSquare2)    ///Q:7.8是仿照CreateMapPoints()函数中的双目来的，这里需要重新计算
                continue;

            // step6.7：计算3D点在另一个关键帧下的重投影去查
            // -1.该keyline在pKF2中所在直线系数
            Vector3d lC2 = pKF2->mvKeyLineFunctions[idx2];

            // -2.起始点在当前帧的重投影误差
            const float x3 = Rcw2.row(0).dot(s3Dt) + tcw2.at<float>(0);
            const float y3 = Rcw2.row(1).dot(s3Dt) + tcw2.at<float>(1);
            float e3 = lC2(0)*x3 + lC2(1)*y3 + lC2(2);

            // -3.终止点在当前帧的重投影误差
            const float x4 = Rcw2.row(0).dot(e3Dt) + tcw2.at<float>(0);
            const float y4 = Rcw2.row(1).dot(e3Dt) + tcw2.at<float>(1);
            float e4 = lC2(0)*x4 + lC2(1)*y3 + lC2(2);

            // -4.判断线段在当前帧的重投影误差是否符合阈值
            float eC2 = e3 + e4;
            if(eC1>7.8*sigmaSquare2)
                continue;

            // step6.8:检测尺度连续性
            cv::Mat middle3D = 0.5*(s3D+e3D);
            // 世界坐标系下，线段3D中点与相机间的向量，方向由相机指向3D点
            cv::Mat normal1 = middle3D - Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = middle3D - Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            // ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioDist = dist2/dist1;
            // 金字塔尺度因子的比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[keyline1.octave]/pKF2->mvScaleFactors[keyline2.octave];
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // step6.9: 三角化成功，构造MapLine
//            cout << "三角化的线段的两个端点： " << "\n \t" << s3Dt << "\n \t" << e3Dt << endl;
//            cout << s3D.at<float>(0) << ", " << s3D.at<float>(1) << ", " << s3D.at<float>(2) << endl;
            Vector6d line3D;
            line3D << s3D.at<float>(0), s3D.at<float>(1), s3D.at<float>(2), e3D.at<float>(0), e3D.at<float>(1), e3D.at<float>(2);
            MapLine* pML = new MapLine(line3D, mpCurrentKeyFrame, mpMap);
            pML->SetCreationOrigin(
                MapLine::ORIGIN_LEGACY,
                mpCurrentKeyFrame ? static_cast<long>(mpCurrentKeyFrame->mnFrameId) : -1,
                mpCurrentKeyFrame ? static_cast<long>(mpCurrentKeyFrame->mnId) : -1,
                mpCurrentKeyFrame ? mpCurrentKeyFrame->mTimeStamp : -1.0,
                "LocalMapping::CreateNewMapLines");

            // step6.10：为该MapLine添加属性
            pML->AddObservation(mpCurrentKeyFrame, idx1);
            pML->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapLine(pML, idx1);
            pKF2->AddMapLine(pML, idx2);

            pML->ComputeDistinctiveDescriptors();
            pML->UpdateAverageDir();
            mpMap->AddMapLine(pML);

            // step6.11：将新产生的线特征放入检测队列，这些MapLines都会经过MapLineCulling函数的检验
            mlpRecentAddedMapLines.push_back(pML);

            nnew++;
        }
    }

}

/**
 * 采用极平面方式
 */
int LocalMapping::CreateNewNativeFisheyeMapLines(
    const vector<KeyFrame*>& vpNeighKFs)
{
    KannalaBrandt8Camera::Parameters camera_params;
    if(!mpMap || !mpMap->GetNativeFisheyeCameraParameters(camera_params))
        return 0;

    const vector<FisheyeLineObservation>& current_observations =
        mpCurrentKeyFrame->GetNativeFisheyeLineObservations();
    if(current_observations.empty())
        return 0;

    FisheyeLineTriangulator::Options options =
        FisheyeLineTriangulator::OptionsForConfig(
            FisheyeLineTriangulator::Config::Loose);
    options.min_valid_samples = 4;
    options.min_pose_observations = 2;
    options.min_baseline = 0.005;
    FisheyeLineTriangulator triangulator(camera_params, options);
    NativeLineIdentityBridge& bridge = mpMap->NativeLineBridge();

    int nnew = 0;
    for(size_t idx1 = 0; idx1 < current_observations.size(); ++idx1)
    {
        const FisheyeLineObservation& obs1 =
            current_observations[idx1];
        if(!NativeObservationUsable(obs1))
            continue;
        if(mpCurrentKeyFrame->GetNativeFisheyeMapLine(idx1))
            continue;
        if(bridge.GetMapLineForTrack(obs1.temporal_track_id))
            continue;

        for(KeyFrame* pKF2 : vpNeighKFs)
        {
            if(!pKF2 || pKF2 == mpCurrentKeyFrame || pKF2->isBad())
                continue;

            const vector<FisheyeLineObservation>& observations2 =
                pKF2->GetNativeFisheyeLineObservations();
            for(size_t idx2 = 0; idx2 < observations2.size(); ++idx2)
            {
                const FisheyeLineObservation& obs2 = observations2[idx2];
                if(!NativeObservationUsable(obs2))
                    continue;
                if(obs2.temporal_track_id != obs1.temporal_track_id)
                    continue;
                if(pKF2->GetNativeFisheyeMapLine(idx2))
                    continue;
                if(bridge.GetMapLineForTrack(obs1.temporal_track_id))
                    continue;
                const NativeLineIdentityBridge::Result observe1 =
                    bridge.ObserveTrack(mpCurrentKeyFrame, idx1,
                                        obs1.temporal_track_id);
                if(!observe1.ok &&
                   observe1.reason != "pending_no_mapline" &&
                   observe1.reason != "observation_already_registered")
                {
                    continue;
                }
                const NativeLineIdentityBridge::Result observe2 =
                    bridge.ObserveTrack(pKF2, idx2,
                                        obs1.temporal_track_id);
                if(!observe2.ok &&
                   observe2.reason != "pending_no_mapline" &&
                   observe2.reason != "observation_already_registered")
                {
                    continue;
                }

                const Eigen::Vector3d c1 =
                    CvMat31ToEigen(mpCurrentKeyFrame->GetCameraCenter());
                const Eigen::Vector3d c2 =
                    CvMat31ToEigen(pKF2->GetCameraCenter());
                const double baseline = (c2 - c1).norm();
                if(!FiniteScalar(baseline) ||
                   baseline < options.min_baseline)
                {
                    continue;
                }

                FisheyeLineTriangulator::PlaneObs plane1;
                FisheyeLineTriangulator::PlaneObs plane2;
                const vector<FisheyeLineTriangulator::Sample> samples1 =
                    MakeNativeSamples(obs1);
                const vector<FisheyeLineTriangulator::Sample> samples2 =
                    MakeNativeSamples(obs2);
                const FisheyeLineTriangulator::LineInfo line_info1 =
                    MakeNativeLineInfo(obs1);
                const FisheyeLineTriangulator::LineInfo line_info2 =
                    MakeNativeLineInfo(obs2);
                if(!triangulator.FitPlane(MakeNativeTrackObs(obs1),
                                          MakeNativePose(mpCurrentKeyFrame),
                                          samples1,
                                          &line_info1,
                                          plane1))
                {
                    continue;
                }
                if(!triangulator.FitPlane(MakeNativeTrackObs(obs2),
                                          MakeNativePose(pKF2),
                                          samples2,
                                          &line_info2,
                                          plane2))
                {
                    continue;
                }

                const double plane_angle =
                    PlaneAngleDeg(plane1.nw, plane2.nw);
                if(!FiniteScalar(plane_angle) ||
                   plane_angle < options.min_plane_angle_deg)
                {
                    continue;
                }

                Eigen::Vector3d p0 = Eigen::Vector3d::Zero();
                Eigen::Vector3d dir = Eigen::Vector3d::UnitX();
                double condition_number = 0.0;
                if(!triangulator.IntersectPlanes(plane1, plane2,
                                                 p0, dir,
                                                 condition_number))
                {
                    continue;
                }
                if(!FiniteVec3(p0) || !FiniteVec3(dir) ||
                   dir.norm() < 1e-12)
                {
                    continue;
                }
                dir.normalize();

                vector<FisheyeLineTriangulator::PlaneObs> planes;
                planes.push_back(plane1);
                planes.push_back(plane2);
                std::map<std::string,
                         vector<FisheyeLineTriangulator::Sample>> sample_map;
                sample_map[triangulator.ObservationKey(
                    plane1.obs.frame_id,
                    plane1.obs.view_id,
                    plane1.obs.line_id)] = samples1;
                sample_map[triangulator.ObservationKey(
                    plane2.obs.frame_id,
                    plane2.obs.view_id,
                    plane2.obs.line_id)] = samples2;

                FisheyeLineTriangulator::Line3D line;
                line.track_id = static_cast<int>(obs1.temporal_track_id);
                line.view_id = obs1.view_id;
                line.p0 = p0;
                line.dir = dir;
                line.moment = p0.cross(dir);
                line.condition_number = condition_number;
                line.baseline_m = baseline;
                line.plane_angle_deg = plane_angle;

                int support_count = 0;
                if(!triangulator.EstimateFiniteSegment(
                       line, planes, sample_map,
                       line.endpoint1, line.endpoint2,
                       line.length_m, support_count))
                {
                    continue;
                }
                line.finite_segment_valid = true;
                line.observation_count = 2;

                if(!FiniteVec3(line.endpoint1) ||
                   !FiniteVec3(line.endpoint2) ||
                   !(line.length_m > 1e-4))
                {
                    continue;
                }
                if(!PositiveDepthInKeyFrame(mpCurrentKeyFrame,
                                            line.endpoint1) ||
                   !PositiveDepthInKeyFrame(mpCurrentKeyFrame,
                                            line.endpoint2) ||
                   !PositiveDepthInKeyFrame(pKF2,
                                            line.endpoint1) ||
                   !PositiveDepthInKeyFrame(pKF2,
                                            line.endpoint2))
                {
                    continue;
                }

                const FisheyeLineTriangulator::ObsError error1 =
                    triangulator.ComputeReprojectionError(
                        -1, line, plane1, samples1);
                const FisheyeLineTriangulator::ObsError error2 =
                    triangulator.ComputeReprojectionError(
                        -1, line, plane2, samples2);
                if(!error1.inlier || !error2.inlier)
                    continue;

                Vector6d endpoints;
                endpoints << line.endpoint1.x(), line.endpoint1.y(),
                             line.endpoint1.z(), line.endpoint2.x(),
                             line.endpoint2.y(), line.endpoint2.z();

                MapLine* pML =
                    new MapLine(endpoints, mpCurrentKeyFrame, mpMap);
                pML->SetCreationOrigin(
                    MapLine::ORIGIN_PHYSICAL,
                    mpCurrentKeyFrame ? static_cast<long>(mpCurrentKeyFrame->mnFrameId) : -1,
                    mpCurrentKeyFrame ? static_cast<long>(mpCurrentKeyFrame->mnId) : -1,
                    mpCurrentKeyFrame ? mpCurrentKeyFrame->mTimeStamp : -1.0,
                    "LocalMapping::CreateNewNativeFisheyeMapLines");
                mpMap->AddMapLine(pML);
                pML->AddNativeFisheyeObservation(
                    mpCurrentKeyFrame, idx1, obs1);
                pML->AddNativeFisheyeObservation(
                    pKF2, idx2, obs2);
                mpCurrentKeyFrame->AddNativeFisheyeMapLine(pML, idx1);
                pKF2->AddNativeFisheyeMapLine(pML, idx2);
                const NativeLineIdentityBridge::Result bind_result =
                    bridge.BindTrackToMapLine(
                        obs1.temporal_track_id, pML);
                if(!bind_result.ok)
                {
                    pML->SetBadFlag();
                    continue;
                }
                pML->ComputeDistinctiveDescriptors();
                pML->UpdateAverageDir();

                mlpRecentAddedMapLines.push_back(pML);
                ++nnew;
                break;
            }
            if(bridge.GetMapLineForTrack(obs1.temporal_track_id))
                break;
        }
    }

    return nnew;
}

void LocalMapping::CreateNewMapLines2()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn=2;
    if(mbMonocular)
        nn=10;
    //step1：在当前关键帧的共视关键帧中找到共视成都最高的nn帧相邻帧vpNeighKFs
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    CreateNewNativeFisheyeMapLines(vpNeighKFs);

    // The legacy LSD/virtual-camera triangulator is a CAM_A-style detection
    // chart path.  It must not define CAM_B MapLine geometry or visibility.
    return;

    LSDmatcher lmatcher;    //建立线特征匹配

    // 获取当前帧的转换矩阵
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    //得到当前关键帧在世界坐标系中的坐标
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    //获取当前帧的相机内参
    const Mat &K1 = mpCurrentKeyFrame->mK;
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    // step2: 遍历相邻关键帧vpNeighKFs
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        // 邻接的关键帧在世界坐标系中的坐标
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        // 基线向量，两个关键帧间的相机位移
        cv::Mat vBaseline = Ow2 - Ow1;
        // 基线长度
        const float baseline = cv::norm(vBaseline);

        // step3：判断相机运动的基线是不是足够长
        if(!mbMonocular)
        {
            // 如果是立体相机，关键帧间距太小时不生成3D点
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            // 邻接关键帧的场景深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            // baseline 与景深的比例
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            // 如果特别远（比例特别小），那么不考虑当前邻接的关键帧，不生成3D点
            if(ratioBaselineDepth<0.1)
                continue;
        }

        // Compute Fundamental Matrix
        // step4：根据两个关键帧的位姿计算它们之间的基本矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fulfill epipolar constraint
        // step5：通过极线约束限制匹配时的搜索单位，进行特征点匹配
        vector<pair<size_t, size_t>> vMatchedIndices;
        lmatcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const Mat &K2 = pKF2->mK;
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each matched line Segment
        const int nmatches = vMatchedIndices.size();
        for(int ikl=0; ikl<nmatches; ikl++) {
            // step6.1：取出匹配的特征线
            const int &idx1 = vMatchedIndices[ikl].first;
            const int &idx2 = vMatchedIndices[ikl].second;

            const KeyLine &keyline1 = mpCurrentKeyFrame->mvKeyLines[idx1];
            const KeyLine &keyline2 = pKF2->mvKeyLines[idx2];
            const Vector3d keyline1_function = mpCurrentKeyFrame->mvKeyLineFunctions[idx1];
            const Vector3d keyline2_function = pKF2->mvKeyLineFunctions[idx2];
            const Mat klF1 = (Mat_<float>(3, 1) << keyline1_function(0),
                    keyline1_function(1),
                    keyline1_function(2));
            const Mat klF2 = (Mat_<float>(3, 1) << keyline2_function(0),
                    keyline2_function(1),
                    keyline2_function(2));

            // step6.2：线段在第一帧图像中的坐标
            cv::Mat StartC1, EndC1, MidC1;
            StartC1 = (cv::Mat_<float>(3, 1) << (keyline1.startPointX - cx1) * invfx1, (keyline1.startPointY - cy1) *
                                                                                       invfy1, 1.0);
            EndC1 = (cv::Mat_<float>(3, 1) << (keyline1.endPointX - cx1) * invfx1, (keyline1.endPointY - cy1) *
                                                                                   invfy1, 1.0);
            MidC1 = (StartC1 + EndC1) / 2;

            cv::Mat StartC2, EndC2, MidC2;
            StartC2 = (cv::Mat_<float>(3, 1) << (keyline2.startPointX - cx1) * invfx1, (keyline2.startPointY - cy1) *
                                                                                       invfy1, 1.0);
            EndC2 = (cv::Mat_<float>(3, 1) << (keyline2.endPointX - cx1) * invfx1, (keyline2.endPointY - cy1) *
                                                                                   invfy1, 1.0);
            MidC2 = (StartC2 + EndC2) / 2;

            cv::Mat ray1 = Rwc1 * MidC1;
            cv::Mat ray2 = Rwc2 * MidC2;
            // 计算在世界坐标系下，两个坐标向量间的余弦值
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays + 1;


            // step6.3：两帧图像的投影矩阵
            Mat M1 = K1 * Tcw1;
            Mat M2 = K2 * Tcw2;

            // step6.4：三角化恢复线段的3D端点
            cv::Mat s3D, e3D;

            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && cosParallaxRays < 0.98)
            {
                // 起始点
                cv::Mat A(4,4,CV_32F);
                A.row(0) = klF1.t()*M1;
                A.row(1) = klF2.t()*M2;
                A.row(2) = StartC1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(3) = StartC1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);

                cv::Mat w1, u1, vt1;
                cv::SVD::compute(A, w1, u1, vt1, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                s3D = vt1.row(3).t();
                if(s3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                s3D = s3D.rowRange(0,3)/s3D.at<float>(3);

                // 终止点
                cv::Mat B(4,4,CV_32F);
                B.row(0) = klF1.t()*M1;
                B.row(1) = klF2.t()*M2;
                B.row(2) = EndC1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                B.row(3) = EndC1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);

                cv::Mat w2, u2, vt2;
                cv::SVD::compute(B, w2, u2, vt2, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                e3D = vt2.row(3).t();

                if(e3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                e3D = e3D.rowRange(0,3)/e3D.at<float>(3);

            } else continue;


            cv::Mat s3Dt = s3D.t();
            cv::Mat e3Dt = e3D.t();

            // 判断起始点是否离两个相机中心太近
            // 邻接关键帧的场景深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            cv::Mat v1 = s3D - Ow1;
            float distance1 = cv::norm(v1);
            const float ratio1 = distance1/medianDepthKF2;
            if(ratio1 < 0.3)
                continue;

            cv::Mat v2 = s3D - Ow2;
            float distance2 = cv::norm(v2);
            const float ratio2 = distance2/medianDepthKF2;
            if(ratio2 < 0.3)
                continue;

            // 判断线段是否太长
            cv::Mat v3 = e3D - s3D;
            float distance3 = cv::norm(v3);
            const float ratio3 = distance3/medianDepthKF2;
            if(ratio3 > 0.61)
                continue;
//            cout << "ratio1 = " << ratio1 << endl << "ratio2 = " << ratio2 << endl;
//            cout << "distance1 = " << distance1 << endl;
//            cout << "distance2 = " << distance2 << endl;

            // 判断两个终止点离相机中心是否太近
            cv::Mat v4 = e3D - Ow1;
            float distance4 = cv::norm(v4);
            const float ratio4 = distance4/medianDepthKF2;
            if(ratio4 < 0.3)
                continue;
            cv::Mat v5 = e3D - Ow2;
            float distance5 = cv::norm(v5);
            const float ratio5 = distance5/medianDepthKF2;
            if(ratio5 < 0.3)
                continue;

            // step6.5：检测生成的3D点是否在相机前方
            float SZC1 = Rcw1.row(2).dot(s3Dt)+tcw1.at<float>(2);   //起始点在C1下的Z坐标值
            if(SZC1<=0)
                continue;

            float SZC2 = Rcw2.row(2).dot(s3Dt)+tcw2.at<float>(2);   //起始点在C2下的Z坐标值
            if(SZC2<=0)
                continue;

            float EZC1 = Rcw1.row(2).dot(e3Dt)+tcw1.at<float>(2);   //终止点在C1下的Z坐标值
            if(EZC1<=0)
                continue;

            float EZC2 = Rcw2.row(2).dot(e3Dt)+tcw2.at<float>(2);   //终止点在C2下的Z坐标值
            if(EZC2<=0)
                continue;

            // step6.9: 三角化成功，构造MapLine
            Vector6d line3D;
            line3D << s3D.at<float>(0), s3D.at<float>(1), s3D.at<float>(2), e3D.at<float>(0), e3D.at<float>(1), e3D.at<float>(2);
            MapLine* pML = new MapLine(line3D, mpCurrentKeyFrame, mpMap);
            pML->SetCreationOrigin(
                MapLine::ORIGIN_LEGACY,
                mpCurrentKeyFrame ? static_cast<long>(mpCurrentKeyFrame->mnFrameId) : -1,
                mpCurrentKeyFrame ? static_cast<long>(mpCurrentKeyFrame->mnId) : -1,
                mpCurrentKeyFrame ? mpCurrentKeyFrame->mTimeStamp : -1.0,
                "LocalMapping::CreateNewMapLines2");

            // step6.10：为该MapLine添加属性
            pML->AddObservation(mpCurrentKeyFrame, idx1);
            pML->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapLine(pML, idx1);
            pKF2->AddMapLine(pML, idx2);

            pML->ComputeDistinctiveDescriptors();
            pML->UpdateAverageDir();
            mpMap->AddMapLine(pML);

            // step6.11：将新产生的线特征放入检测队列，这些MapLines都会经过MapLineCulling函数的检验
            mlpRecentAddedMapLines.push_back(pML);

            nnew++;
        }

    }
}

/**
 * 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
 */
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if (mbMonocular)
        nn = 20;
    const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame *> vpTargetKFs;
    for (auto pKFi : vpNeighKFs) {
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame *> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (auto pKFi2 : vpSecondNeighKFs) {
            if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId ||
                pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (auto pKFi : vpTargetKFs) {
        matcher.Fuse(pKFi, vpMapPointMatches, 3.0,
                     mpCurrentKeyFrame->mnId,
                     "forward_current_to_neighbor");
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint *> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

    for (auto pKFi : vpTargetKFs) {
        vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for (auto pMP : vpMapPointsKFi) {
            if (!pMP)
                continue;
            if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates, 3.0,
                 mpCurrentKeyFrame->mnId,
                 "reverse_neighbor_to_current");


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (auto pMP : vpMapPointMatches) {
        if (pMP) {
            if (!pMP->isBad()) {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    LSDmatcher lineMatcher;
    vector<MapLine *> vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();
    for (auto pKFi : vpTargetKFs) {
        lineMatcher.Fuse(pKFi, vpMapLineMatches);
    }

    vector<MapLine *> vpLineFuseCandidates;
    vpLineFuseCandidates.reserve(vpTargetKFs.size() * vpMapLineMatches.size());

    for (auto pKFi : vpTargetKFs) {
        vector<MapLine *> vpMapLinesKFi = pKFi->GetMapLineMatches();

        for (auto pML : vpMapLinesKFi) {
            if (!pML)
                continue;

            if (pML->isBad() || pML->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;

            pML->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpLineFuseCandidates.push_back(pML);
        }
    }

    lineMatcher.Fuse(mpCurrentKeyFrame, vpLineFuseCandidates);

    // Update Lines
    vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();
    for (auto pML : vpMapLineMatches) {
        if (pML) {
            if (!pML->isBad()) {
                pML->ComputeDistinctiveDescriptors();
                pML->UpdateAverageDir();
            }
        }
    }

        // Update connections in covisibility graph
        mpCurrentKeyFrame->UpdateConnections();
}

/**
 * 检查并融合当前关键帧与相邻帧重复的MapLines
 */


cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();    // 点特征
        mlpRecentAddedMapLines.clear();     // 线特征
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
