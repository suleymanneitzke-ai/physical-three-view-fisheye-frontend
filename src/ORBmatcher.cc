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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "camera/KannalaBrandt8Camera.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include<stdint-gcc.h>

using namespace std;

namespace StructureSLAM
{

namespace
{

struct Kb8FusionStats
{
    unsigned long current_keyframe_id = 0;
    unsigned long target_keyframe_id = 0;
    std::string direction = "unknown";
    int input_mappoints = 0;
    int bad_mappoints_skipped = 0;
    int valid_camera_points = 0;
    int valid_kb8_projections = 0;
    int inside_image = 0;
    int distance_gate_pass = 0;
    int view_angle_pass = 0;
    int scale_gate_pass = 0;
    int feature_area_queries = 0;
    int feature_candidates = 0;
    int descriptor_compared = 0;
    int descriptor_pass = 0;
    int new_observations = 0;
    int replacement_candidates = 0;
    int replacements_performed = 0;
    int already_same_mappoint = 0;
    int fusion_fail_no_feature = 0;
    int fusion_fail_descriptor = 0;
    int fusion_fail_geometry = 0;
};

struct Kb8SearchBySim3Stats
{
    unsigned long keyframe1_id = 0;
    unsigned long keyframe2_id = 0;
    int map_points_kf1 = 0;
    int map_points_kf2 = 0;
    int kf1_to_kf2_attempts = 0;
    int kf2_to_kf1_attempts = 0;
    int valid_kb8_projections_kf2 = 0;
    int valid_kb8_projections_kf1 = 0;
    int inside_image_kf2 = 0;
    int inside_image_kf1 = 0;
    int feature_candidates_kf2 = 0;
    int feature_candidates_kf1 = 0;
    int descriptor_compared_kf2 = 0;
    int descriptor_compared_kf1 = 0;
    int tentative_matches_kf2 = 0;
    int tentative_matches_kf1 = 0;
    int bidirectional_matches = 0;
    int legacy_pinhole_projection_calls = 0;
};

struct Kb8LoopSearchByProjectionScwStats
{
    unsigned long target_keyframe_id = 0;
    int input_loop_mappoints = 0;
    int already_matched = 0;
    int projected_valid = 0;
    int inside_image = 0;
    int scale_gate_pass = 0;
    int descriptor_candidates = 0;
    int new_matches = 0;
    int rejected_invalid_depth = 0;
    int rejected_outside_image = 0;
    int rejected_scale = 0;
    int rejected_descriptor = 0;
    int kb8_projection_calls = 0;
    int legacy_pinhole_projection_calls = 0;
};

struct Kb8LoopFuseScwStats
{
    unsigned long target_keyframe_id = 0;
    int input_loop_mappoints = 0;
    int scw_projection_attempts = 0;
    int kb8_projection_calls = 0;
    int legacy_pinhole_projection_calls = 0;
    int valid_projection = 0;
    int inside_image = 0;
    int already_observed = 0;
    int scale_gate_pass = 0;
    int descriptor_candidates = 0;
    int matches = 0;
    int new_observations = 0;
    int replacements = 0;
    int rejected_bad_point = 0;
    int rejected_invalid_depth = 0;
    int rejected_outside_image = 0;
    int rejected_distance = 0;
    int rejected_view_angle = 0;
    int rejected_scale = 0;
    int rejected_descriptor = 0;
};

std::mutex& Kb8FusionLogMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::string EnvPath(const char* name)
{
    const char* path = std::getenv(name);
    return path ? std::string(path) : std::string();
}

void AppendKb8FusionStats(const Kb8FusionStats& stats)
{
    const std::string path =
        EnvPath("STRUCTURE_SLAM_KB8_FUSION_STATS");
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(Kb8FusionLogMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << stats.current_keyframe_id << ','
        << stats.target_keyframe_id << ','
        << stats.direction << ','
        << stats.input_mappoints << ','
        << stats.bad_mappoints_skipped << ','
        << stats.valid_camera_points << ','
        << stats.valid_kb8_projections << ','
        << stats.inside_image << ','
        << stats.distance_gate_pass << ','
        << stats.view_angle_pass << ','
        << stats.scale_gate_pass << ','
        << stats.feature_area_queries << ','
        << stats.feature_candidates << ','
        << stats.descriptor_compared << ','
        << stats.descriptor_pass << ','
        << stats.new_observations << ','
        << stats.replacement_candidates << ','
        << stats.replacements_performed << ','
        << stats.already_same_mappoint << ','
        << stats.fusion_fail_no_feature << ','
        << stats.fusion_fail_descriptor << ','
        << stats.fusion_fail_geometry << '\n';
}

void AppendKb8FusionReplacement(
    const unsigned long kept_id,
    const unsigned long replaced_id,
    const int kept_observations_before,
    const int replaced_observations_before,
    const int kept_observations_after,
    const int affected_keyframes)
{
    const std::string path =
        EnvPath("STRUCTURE_SLAM_KB8_FUSION_REPLACEMENTS");
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(Kb8FusionLogMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << kept_id << ','
        << replaced_id << ','
        << kept_observations_before << ','
        << replaced_observations_before << ','
        << kept_observations_after << ','
        << affected_keyframes << '\n';
}

void AppendKb8SearchBySim3Stats(const Kb8SearchBySim3Stats& stats)
{
    const std::string path =
        EnvPath("STRUCTURE_SLAM_KB8_SEARCHBYSIM3_STATS");
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(Kb8FusionLogMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << stats.keyframe1_id << ','
        << stats.keyframe2_id << ','
        << stats.map_points_kf1 << ','
        << stats.map_points_kf2 << ','
        << stats.kf1_to_kf2_attempts << ','
        << stats.kf2_to_kf1_attempts << ','
        << stats.valid_kb8_projections_kf2 << ','
        << stats.valid_kb8_projections_kf1 << ','
        << stats.inside_image_kf2 << ','
        << stats.inside_image_kf1 << ','
        << stats.feature_candidates_kf2 << ','
        << stats.feature_candidates_kf1 << ','
        << stats.descriptor_compared_kf2 << ','
        << stats.descriptor_compared_kf1 << ','
        << stats.tentative_matches_kf2 << ','
        << stats.tentative_matches_kf1 << ','
        << stats.bidirectional_matches << ','
        << stats.legacy_pinhole_projection_calls << '\n';
}

void AppendKb8LoopSearchByProjectionScwStats(
    const Kb8LoopSearchByProjectionScwStats& stats)
{
    const std::string path =
        EnvPath("STRUCTURE_SLAM_KB8_LOOP_SEARCH_BY_PROJECTION_SCW_STATS");
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(Kb8FusionLogMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << stats.target_keyframe_id << ','
        << stats.input_loop_mappoints << ','
        << stats.already_matched << ','
        << stats.projected_valid << ','
        << stats.inside_image << ','
        << stats.scale_gate_pass << ','
        << stats.descriptor_candidates << ','
        << stats.new_matches << ','
        << stats.rejected_invalid_depth << ','
        << stats.rejected_outside_image << ','
        << stats.rejected_scale << ','
        << stats.rejected_descriptor << ','
        << stats.kb8_projection_calls << ','
        << stats.legacy_pinhole_projection_calls << '\n';
}

void AppendKb8LoopFuseScwStats(const Kb8LoopFuseScwStats& stats)
{
    const std::string path =
        EnvPath("STRUCTURE_SLAM_KB8_LOOP_FUSE_SCW_STATS");
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(Kb8FusionLogMutex());
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    out << stats.target_keyframe_id << ','
        << stats.input_loop_mappoints << ','
        << stats.scw_projection_attempts << ','
        << stats.kb8_projection_calls << ','
        << stats.legacy_pinhole_projection_calls << ','
        << stats.valid_projection << ','
        << stats.inside_image << ','
        << stats.already_observed << ','
        << stats.scale_gate_pass << ','
        << stats.descriptor_candidates << ','
        << stats.matches << ','
        << stats.new_observations << ','
        << stats.replacements << ','
        << stats.rejected_bad_point << ','
        << stats.rejected_invalid_depth << ','
        << stats.rejected_outside_image << ','
        << stats.rejected_distance << ','
        << stats.rejected_view_angle << ','
        << stats.rejected_scale << ','
        << stats.rejected_descriptor << '\n';
}

bool ProjectCameraPointToKeyFrame(const KeyFrame& keyframe,
                                  const cv::Mat& p3Dc,
                                  const KannalaBrandt8Camera* kb8Camera,
                                  const float legacyFx,
                                  const float legacyFy,
                                  const float legacyCx,
                                  const float legacyCy,
                                  float& u,
                                  float& v,
                                  bool& usedLegacyPinhole)
{
    usedLegacyPinhole = false;
    const float z = p3Dc.at<float>(2);
    if(z < 0.0f)
        return false;

    if(keyframe.mbUseKb8PointObservations)
    {
        if(!kb8Camera)
            return false;
        Eigen::Vector2d uv;
        if(!kb8Camera->valid() ||
           !kb8Camera->project(Eigen::Vector3d(p3Dc.at<float>(0),
                                               p3Dc.at<float>(1),
                                               z),
                               uv) ||
           !std::isfinite(uv.x()) ||
           !std::isfinite(uv.y()))
        {
            return false;
        }
        u = static_cast<float>(uv.x());
        v = static_cast<float>(uv.y());
        return true;
    }

    const float invz = 1.0f / z;
    const float x = p3Dc.at<float>(0) * invz;
    const float y = p3Dc.at<float>(1) * invz;
    u = legacyFx * x + legacyCx;
    v = legacyFy * y + legacyCy;
    usedLegacyPinhole = true;
    return std::isfinite(u) && std::isfinite(v);
}

bool ProjectMapPointToFrame(const Frame& frame,
                            const cv::Mat& x3Dc,
                            float& u,
                            float& v)
{
    const float z = x3Dc.at<float>(2);
    if(!frame.HasKb8PointObservations() && z <= 0.0f)
        return false;

    if(frame.HasKb8PointObservations())
    {
        const KannalaBrandt8Camera camera(
            frame.mKb8PointCameraParameters);
        Eigen::Vector2d uv;
        if(!camera.valid() ||
           !camera.project(Eigen::Vector3d(x3Dc.at<float>(0),
                                           x3Dc.at<float>(1),
                                           z),
                           uv) ||
           !camera.isInImage(uv))
        {
            return false;
        }
        u = static_cast<float>(uv.x());
        v = static_cast<float>(uv.y());
        return true;
    }

    const float invz = 1.0f / z;
    u = frame.fx * x3Dc.at<float>(0) * invz + frame.cx;
    v = frame.fy * x3Dc.at<float>(1) * invz + frame.cy;

    if(u < frame.mnMinX || u > frame.mnMaxX)
        return false;
    if(v < frame.mnMinY || v > frame.mnMaxY)
        return false;
    return true;
}

} // namespace

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0;

    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        if(bFactor)
            r*=th;

        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty())
            continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

            if(F.mvuRight[idx]>0)
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx);

            const int dist = DescriptorDistance(MPdescriptor,d);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)
        {
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;

            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    //获得关键帧的mvMapPoint
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    //当前帧的mapPoint,个数与2d feature相同，此时全为空
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    //关键帧的 2D feature 和 当前帧的 2D feature
    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;                

                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;
                int bestIdxF =-1 ;
                int bestDist2=256;

                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if(vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                    const int dist =  DescriptorDistance(dKF,dF);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<=TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF]=pMP;

                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                        if(mbCheckOrientation)
                        {
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }

    //计算朝向，删除错误匹配
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Capability: KB8_LOOP_SEARCH_BY_PROJECTION_SCW_READY.
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const bool useKb8LoopSearchByProjectionScw =
        pKF->mbUseKb8PointObservations;
    const KannalaBrandt8Camera kb8Camera(
        pKF->mKb8PointCameraParameters);
    Kb8LoopSearchByProjectionScwStats stats;
    stats.target_keyframe_id = pKF->mnId;
    stats.input_loop_mappoints = static_cast<int>(vpPoints.size());

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
        {
            if(useKb8LoopSearchByProjectionScw &&
               spAlreadyFound.count(pMP))
            {
                ++stats.already_matched;
            }
            continue;
        }

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        float u = 0.0f;
        float v = 0.0f;
        bool usedLegacyPinhole = false;
        if(!ProjectCameraPointToKeyFrame(
               *pKF, p3Dc,
               useKb8LoopSearchByProjectionScw ? &kb8Camera : NULL,
               fx, fy, cx, cy,
               u, v, usedLegacyPinhole))
        {
            if(useKb8LoopSearchByProjectionScw)
                ++stats.rejected_invalid_depth;
            continue;
        }
        if(useKb8LoopSearchByProjectionScw)
        {
            ++stats.projected_valid;
            ++stats.kb8_projection_calls;
            if(usedLegacyPinhole)
                ++stats.legacy_pinhole_projection_calls;
        }

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
        {
            if(useKb8LoopSearchByProjectionScw)
                ++stats.rejected_outside_image;
            continue;
        }
        if(useKb8LoopSearchByProjectionScw)
            ++stats.inside_image;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
        {
            if(useKb8LoopSearchByProjectionScw)
                ++stats.rejected_scale;
            continue;
        }
        if(useKb8LoopSearchByProjectionScw)
            ++stats.scale_gate_pass;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
        if(useKb8LoopSearchByProjectionScw)
            stats.descriptor_candidates +=
                static_cast<int>(vIndices.size());

        if(vIndices.empty())
        {
            if(useKb8LoopSearchByProjectionScw)
                ++stats.rejected_descriptor;
            continue;
        }

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
            if(useKb8LoopSearchByProjectionScw)
                ++stats.new_matches;
        }
        else if(useKb8LoopSearchByProjectionScw)
        {
            ++stats.rejected_descriptor;
        }

    }

    if(useKb8LoopSearchByProjectionScw)
        AppendKb8LoopSearchByProjectionScwStats(stats);

    return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0)
            continue;

        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
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

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
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

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}

int ORBmatcher::Fuse(KeyFrame *pKF,
                     const vector<MapPoint *> &vpMapPoints,
                     const float th,
                     const long unsigned int currentKeyFrameIdForDiagnostics,
                     const std::string& directionForDiagnostics)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    const int nMPs = vpMapPoints.size();
    const bool useKb8Projection = pKF->mbUseKb8PointObservations;
    const KannalaBrandt8Camera kb8Camera(
        pKF->mKb8PointCameraParameters);
    Kb8FusionStats stats;
    stats.current_keyframe_id = currentKeyFrameIdForDiagnostics;
    stats.target_keyframe_id = pKF->mnId;
    stats.direction = directionForDiagnostics;
    stats.input_mappoints = nMPs;

    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
            continue;

        if(pMP->isBad())
        {
            if(useKb8Projection)
                ++stats.bad_mappoints_skipped;
            continue;
        }

        if(pMP->IsInKeyFrame(pKF))
        {
            if(useKb8Projection)
                ++stats.already_same_mappoint;
            continue;
        }

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive. Keep the legacy pinhole condition intact,
        // but use the KB8 camera contract for fisheye projection.
        const float z = p3Dc.at<float>(2);
        const bool invalidDepth =
            useKb8Projection
                ? (z <= pKF->mKb8PointCameraParameters.min_positive_z)
                : (z < 0.0f);
        if(invalidDepth)
        {
            if(useKb8Projection)
                ++stats.fusion_fail_geometry;
            continue;
        }
        if(useKb8Projection)
            ++stats.valid_camera_points;

        const float invz = 1/z;
        float u = 0.0f;
        float v = 0.0f;
        if(useKb8Projection)
        {
            Eigen::Vector2d uv;
            const bool projected =
                kb8Camera.valid() &&
                kb8Camera.project(
                    Eigen::Vector3d(p3Dc.at<float>(0),
                                    p3Dc.at<float>(1),
                                    p3Dc.at<float>(2)),
                    uv);
            if(!projected ||
               !std::isfinite(uv.x()) ||
               !std::isfinite(uv.y()))
            {
                ++stats.fusion_fail_geometry;
                continue;
            }
            ++stats.valid_kb8_projections;
            u = static_cast<float>(uv.x());
            v = static_cast<float>(uv.y());
        }
        else
        {
            const float x = p3Dc.at<float>(0)*invz;
            const float y = p3Dc.at<float>(1)*invz;
            u = fx*x+cx;
            v = fy*y+cy;
        }

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
        {
            if(useKb8Projection)
                ++stats.fusion_fail_geometry;
            continue;
        }
        if(useKb8Projection)
            ++stats.inside_image;

        const float ur = u-bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
        {
            if(useKb8Projection)
                ++stats.fusion_fail_geometry;
            continue;
        }
        if(useKb8Projection)
            ++stats.distance_gate_pass;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
        {
            if(useKb8Projection)
                ++stats.fusion_fail_geometry;
            continue;
        }
        if(useKb8Projection)
            ++stats.view_angle_pass;

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);
        if(useKb8Projection)
            ++stats.scale_gate_pass;

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        if(useKb8Projection)
            ++stats.feature_area_queries;
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
        if(useKb8Projection)
            stats.feature_candidates += static_cast<int>(vIndices.size());

        if(vIndices.empty())
        {
            if(useKb8Projection)
                ++stats.fusion_fail_no_feature;
            continue;
        }

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                    if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                        continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;

                    if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                        continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            if(useKb8Projection)
                ++stats.descriptor_compared;
            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            if(useKb8Projection)
                ++stats.descriptor_pass;
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(useKb8Projection)
                    ++stats.replacement_candidates;
                if(!pMPinKF->isBad())
                {
                    const int existingObsBefore = pMPinKF->Observations();
                    const int candidateObsBefore = pMP->Observations();
                    if(pMPinKF->Observations()>pMP->Observations())
                    {
                        pMP->Replace(pMPinKF);
                        if(useKb8Projection)
                        {
                            ++stats.replacements_performed;
                            AppendKb8FusionReplacement(
                                pMPinKF->mnId,
                                pMP->mnId,
                                existingObsBefore,
                                candidateObsBefore,
                                pMPinKF->Observations(),
                                candidateObsBefore);
                        }
                    }
                    else
                    {
                        pMPinKF->Replace(pMP);
                        if(useKb8Projection)
                        {
                            ++stats.replacements_performed;
                            AppendKb8FusionReplacement(
                                pMP->mnId,
                                pMPinKF->mnId,
                                candidateObsBefore,
                                existingObsBefore,
                                pMP->Observations(),
                                existingObsBefore);
                        }
                    }
                }
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
                if(useKb8Projection)
                    ++stats.new_observations;
            }
            nFused++;
        }
        else if(useKb8Projection)
        {
            ++stats.fusion_fail_descriptor;
        }
    }

    if(useKb8Projection)
        AppendKb8FusionStats(stats);

    return nFused;
}

int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Capability: KB8_LOOP_SEARCH_AND_FUSE_READY.
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const bool useKb8LoopFuseScw = pKF->mbUseKb8PointObservations;
    const KannalaBrandt8Camera kb8Camera(
        pKF->mKb8PointCameraParameters);
    Kb8LoopFuseScwStats stats;
    stats.target_keyframe_id = pKF->mnId;
    stats.input_loop_mappoints = static_cast<int>(vpPoints.size());

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad())
        {
            if(useKb8LoopFuseScw)
                ++stats.rejected_bad_point;
            continue;
        }

        if(spAlreadyFound.count(pMP))
        {
            if(useKb8LoopFuseScw)
                ++stats.already_observed;
            continue;
        }

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        if(useKb8LoopFuseScw)
            ++stats.scw_projection_attempts;
        float u = 0.0f;
        float v = 0.0f;
        bool usedLegacyPinhole = false;
        if(!ProjectCameraPointToKeyFrame(
               *pKF, p3Dc,
               useKb8LoopFuseScw ? &kb8Camera : NULL,
               fx, fy, cx, cy,
               u, v, usedLegacyPinhole))
        {
            if(useKb8LoopFuseScw)
                ++stats.rejected_invalid_depth;
            continue;
        }
        if(useKb8LoopFuseScw)
        {
            ++stats.valid_projection;
            ++stats.kb8_projection_calls;
            if(usedLegacyPinhole)
                ++stats.legacy_pinhole_projection_calls;
        }

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
        {
            if(useKb8LoopFuseScw)
                ++stats.rejected_outside_image;
            continue;
        }
        if(useKb8LoopFuseScw)
            ++stats.inside_image;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
        {
            if(useKb8LoopFuseScw)
            {
                ++stats.rejected_distance;
                ++stats.rejected_scale;
            }
            continue;
        }
        if(useKb8LoopFuseScw)
            ++stats.scale_gate_pass;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
        {
            if(useKb8LoopFuseScw)
                ++stats.rejected_view_angle;
            continue;
        }

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
        if(useKb8LoopFuseScw)
            stats.descriptor_candidates +=
                static_cast<int>(vIndices.size());

        if(vIndices.empty())
        {
            if(useKb8LoopFuseScw)
                ++stats.rejected_descriptor;
            continue;
        }

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    vpReplacePoint[iMP] = pMPinKF;
                    if(useKb8LoopFuseScw)
                        ++stats.replacements;
                }
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
                if(useKb8LoopFuseScw)
                    ++stats.new_observations;
            }
            nFused++;
            if(useKb8LoopFuseScw)
                ++stats.matches;
        }
        else if(useKb8LoopFuseScw)
        {
            ++stats.rejected_descriptor;
        }
    }

    if(useKb8LoopFuseScw)
        AppendKb8LoopFuseScwStats(stats);

    return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    // Capability: KB8_SEARCHBYSIM3_PROJECTION_READY.
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;
    const bool useKb8SearchBySim3 =
        pKF1->mbUseKb8PointObservations ||
        pKF2->mbUseKb8PointObservations;
    const KannalaBrandt8Camera kb8Camera1(
        pKF1->mKb8PointCameraParameters);
    const KannalaBrandt8Camera kb8Camera2(
        pKF2->mKb8PointCameraParameters);
    Kb8SearchBySim3Stats stats;
    stats.keyframe1_id = pKF1->mnId;
    stats.keyframe2_id = pKF2->mnId;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();
    stats.map_points_kf1 = N1;
    stats.map_points_kf2 = N2;

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        if(useKb8SearchBySim3)
            ++stats.kf1_to_kf2_attempts;
        float u = 0.0f;
        float v = 0.0f;
        bool usedLegacyPinhole = false;
        if(!ProjectCameraPointToKeyFrame(
               *pKF2, p3Dc2,
               pKF2->mbUseKb8PointObservations ? &kb8Camera2 : NULL,
               fx, fy, cx, cy,
               u, v, usedLegacyPinhole))
        {
            continue;
        }
        if(useKb8SearchBySim3 && usedLegacyPinhole)
            ++stats.legacy_pinhole_projection_calls;
        if(pKF2->mbUseKb8PointObservations)
            ++stats.valid_kb8_projections_kf2;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;
        if(pKF2->mbUseKb8PointObservations)
            ++stats.inside_image_kf2;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);
        if(pKF2->mbUseKb8PointObservations)
            stats.feature_candidates_kf2 +=
                static_cast<int>(vIndices.size());

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            if(pKF2->mbUseKb8PointObservations)
                ++stats.descriptor_compared_kf2;
            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
            if(pKF2->mbUseKb8PointObservations)
                ++stats.tentative_matches_kf2;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        if(useKb8SearchBySim3)
            ++stats.kf2_to_kf1_attempts;
        float u = 0.0f;
        float v = 0.0f;
        bool usedLegacyPinhole = false;
        if(!ProjectCameraPointToKeyFrame(
               *pKF1, p3Dc1,
               pKF1->mbUseKb8PointObservations ? &kb8Camera1 : NULL,
               fx, fy, cx, cy,
               u, v, usedLegacyPinhole))
        {
            continue;
        }
        if(useKb8SearchBySim3 && usedLegacyPinhole)
            ++stats.legacy_pinhole_projection_calls;
        if(pKF1->mbUseKb8PointObservations)
            ++stats.valid_kb8_projections_kf1;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;
        if(pKF1->mbUseKb8PointObservations)
            ++stats.inside_image_kf1;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);
        if(pKF1->mbUseKb8PointObservations)
            stats.feature_candidates_kf1 +=
                static_cast<int>(vIndices.size());

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            if(pKF1->mbUseKb8PointObservations)
                ++stats.descriptor_compared_kf1;
            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
            if(pKF1->mbUseKb8PointObservations)
                ++stats.tentative_matches_kf1;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }
    stats.bidirectional_matches = nFound;
    if(useKb8SearchBySim3)
        AppendKb8SearchBySim3Stats(stats);

    return nFound;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat twc = -Rcw.t()*tcw;

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat tlc = Rlw*twc+tlw;

    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

    for(int i=0; i<LastFrame.N; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                float u = 0.0f;
                float v = 0.0f;
                if(!ProjectMapPointToFrame(CurrentFrame, x3Dc, u, v))
                    continue;
                const float invzc = 1.0f / x3Dc.at<float>(2);

                int nLastOctave = LastFrame.mvKeys[i].octave;

                // Search in a window. Size depends on scale
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;

                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])  //这里的mvpMapPoints不应该是
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    if(CurrentFrame.mvuRight[i2]>0) //大于0 表示双目或RGBD
                    {
                        const float ur = u - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                float u = 0.0f;
                float v = 0.0f;
                if(!ProjectMapPointToFrame(CurrentFrame, x3Dc, u, v))
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
