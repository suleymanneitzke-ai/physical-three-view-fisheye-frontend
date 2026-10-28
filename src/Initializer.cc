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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include <opencv2/calib3d.hpp>

#include<thread>
#include <array>
#include <cstdlib>
#include <numeric>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
namespace StructureSLAM
{

namespace
{

cv::Mat SkewBearing(const Eigen::Vector3d &b)
{
    cv::Mat S = cv::Mat::zeros(3, 3, CV_32F);
    S.at<float>(0, 1) = -static_cast<float>(b.z());
    S.at<float>(0, 2) =  static_cast<float>(b.y());
    S.at<float>(1, 0) =  static_cast<float>(b.z());
    S.at<float>(1, 2) = -static_cast<float>(b.x());
    S.at<float>(2, 0) = -static_cast<float>(b.y());
    S.at<float>(2, 1) =  static_cast<float>(b.x());
    return S;
}

double Quantile(std::vector<double> values, double q)
{
    if(values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(),values.end());
    const size_t i=static_cast<size_t>(std::floor(
        std::max(0.0,std::min(1.0,q))*(values.size()-1)));
    return values[i];
}

} // namespace

Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();

    mvKeys1 = ReferenceFrame.mvKeysUn;
    mvKb8RawKeys1 = mvKeys1;
    mvKb8Bearings1 = ReferenceFrame.mvPointUnitBearings;
    mvbKb8BearingValid1 = ReferenceFrame.mvbPointBearingValid;
    if(ReferenceFrame.HasKb8PointObservations())
    {
        mKb8PointCameraParameters =
            ReferenceFrame.mKb8PointCameraParameters;
        mbHasKb8PointCameraParameters = true;
    }

    mvKeyLines1 = ReferenceFrame.mvKeylinesUn;  //自己添加的，传递线特征
    mvKeyLineFunctions1 = ReferenceFrame.mvKeyLineFunctions;    //自己添加的，传递线特征所在直线方程系数

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
    mLastFailureReason = "unknown";
    mLastModel = "none";
    mLastHomographyScore = 0.0f;
    mLastFundamentalScore = 0.0f;
    mLastHomographyRatio = 0.0f;
    mLastLegacySolverSuccess = false;
    mLastKb8ValidationSuccess = false;
    mLastFinalSuccess = false;
    ResetDecisionDiagnostics();
}

Initializer::Initializer(const Frame &ReferenceFrame,
                         const vector<cv::KeyPoint>& initializationKeys,
                         const cv::Mat& initializationK,
                         float sigma,
                         int iterations)
{
    mK = initializationK.clone();

    mvKeys1 = initializationKeys;
    mvKb8RawKeys1 = ReferenceFrame.mvKeysUn;
    mvKb8Bearings1 = ReferenceFrame.mvPointUnitBearings;
    mvbKb8BearingValid1 = ReferenceFrame.mvbPointBearingValid;
    mbReferenceKeysAreInitializationView = true;
    if(ReferenceFrame.HasKb8PointObservations())
    {
        mKb8PointCameraParameters =
            ReferenceFrame.mKb8PointCameraParameters;
        mbHasKb8PointCameraParameters = true;
    }

    mvKeyLines1 = ReferenceFrame.mvKeylinesUn;
    mvKeyLineFunctions1 = ReferenceFrame.mvKeyLineFunctions;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
    mLastFailureReason = "unknown";
    mLastModel = "none";
    mLastHomographyScore = 0.0f;
    mLastFundamentalScore = 0.0f;
    mLastHomographyRatio = 0.0f;
    mLastLegacySolverSuccess = false;
    mLastKb8ValidationSuccess = false;
    mLastFinalSuccess = false;
    ResetDecisionDiagnostics();
}

void Initializer::ResetDecisionDiagnostics()
{
    mLastCandidateCount = 0;
    mLastBestCandidateId = -1;
    mLastBestGoodPoints = 0;
    mLastSecondBestGoodPoints = 0;
    mLastBestPositiveDepthCam1 = 0;
    mLastBestPositiveDepthCam2 = 0;
    mLastBestBothPositive = 0;
    mLastBestTriangulated = 0;
    mLastBestReprojectionPass = 0;
    mLastBestMedianParallaxDeg = 0.0f;
    mLastRequiredMinGoodPoints = 0;
    mLastRequiredMinParallaxDeg = 0.0f;
    mLastAmbiguityRatioOrThreshold = 0.0f;
    mLastExactFailureBranch.clear();
    mLastPoseCandidateDiagnostics.clear();
    mPendingCheckModel.clear();
    mPendingCheckCandidateId = -1;
}

void Initializer::BeginCandidateDiagnostics(const std::string& model,
                                            int candidate_id)
{
    mPendingCheckModel = model;
    mPendingCheckCandidateId = candidate_id;
}

void Initializer::MarkAcceptedCandidate(const std::string& model,
                                        int candidate_id)
{
    for(size_t i = 0; i < mLastPoseCandidateDiagnostics.size(); ++i)
    {
        InitializationPoseCandidateDiagnostics& candidate =
            mLastPoseCandidateDiagnostics[i];
        if(candidate.model == model &&
           candidate.candidate_id == candidate_id)
        {
            candidate.accepted_by_check_rt = true;
            break;
        }
    }
}

void Initializer::UpdateBestCandidateDiagnostics()
{
    for(size_t i = 0; i < mLastPoseCandidateDiagnostics.size(); ++i)
    {
        const InitializationPoseCandidateDiagnostics& candidate =
            mLastPoseCandidateDiagnostics[i];
        if(candidate.candidate_id == mLastBestCandidateId &&
           candidate.model == mLastModel)
        {
            mLastBestPositiveDepthCam1 =
                candidate.positive_depth_cam1;
            mLastBestPositiveDepthCam2 =
                candidate.positive_depth_cam2;
            mLastBestBothPositive = candidate.both_positive;
            mLastBestTriangulated = candidate.triangulated;
            mLastBestReprojectionPass =
                candidate.reprojection_pass;
            mLastBestMedianParallaxDeg =
                static_cast<float>(candidate.median_parallax_deg);
            break;
        }
    }
}


// with points and lines
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated,
                             vector<pair<int, int>> &vLineMatches, vector<cv::Point3f> &vLineS3D, vector<cv::Point3f> &vLineE3D,
                             vector<bool> &vbLineTriangulated)
{
    mLastFailureReason = "unknown";
    mLastModel = "none";
    mLastHomographyScore = 0.0f;
    mLastFundamentalScore = 0.0f;
    mLastHomographyRatio = 0.0f;
    mLastLegacySolverSuccess = false;
    mLastKb8ValidationSuccess = false;
    mLastFinalSuccess = false;
    ResetDecisionDiagnostics();

    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    mvKeys2 = CurrentFrame.mvKeysUn;
    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());
    mvbMatched1.resize(mvKeys1.size());

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]>=0)
        {
            mvMatches12.push_back(make_pair(i, vMatches12[i]));
            mvbMatched1[i] = true;
        } else{
            mvbMatched1[i] = false;
        }
    }

    const int N = mvMatches12.size();
    if(N < 8)
    {
        mLastFailureReason = "not_enough_matches";
        return false;
    }
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    mvSets = vector<vector<size_t >>(mMaxIterations, vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    float SH, SF;
    cv::Mat H, F;

    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));
    threadH.join();
    threadF.join();
    float RH = SH/(SH+SF);
    mLastHomographyScore = SH;
    mLastFundamentalScore = SF;
    mLastHomographyRatio = RH;

    // Fill structures with current keylines and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    mvKeyLines2 = CurrentFrame.mvKeylinesUn;
    mvKeyLineFunctions2 = CurrentFrame.mvKeyLineFunctions;
    mvbLineMatched1.clear();
    mvLineMatches12.reserve(mvKeyLines2.size());
    mvbLineMatched1.resize(mvKeyLines1.size());

    vector<Match> lineMatches;

    //lineMatches.reserve(vLineMatches.size());
    for(size_t i=0, iend = vLineMatches.size(); i<iend; i++)
    {
        if(vLineMatches[i].first>=0&&vLineMatches[i].second>=0)
        {
            Match lmatch;
            lmatch.first = vLineMatches[i].first;
            lmatch.second = vLineMatches[i].second;
            lineMatches.push_back(lmatch);

        }
    }

    if(RH>0.40)
    {
        mLastModel = "homography";
        bool isReconstructH;
        isReconstructH = ReconstructH(vbMatchesInliersH,lineMatches,H,mK,R21,t21,vP3D,vLineS3D, vLineE3D,vbTriangulated,vbLineTriangulated,1.0,50);
        if(!isReconstructH && mLastFailureReason == "unknown")
            mLastFailureReason = "homography_failed";
        return isReconstructH;
    } else{
        mLastModel = "fundamental";
        bool isReconstructF;
        isReconstructF = ReconstructF(vbMatchesInliersF,lineMatches,F,mK,R21,t21,vP3D,vLineS3D, vLineE3D,vbTriangulated,vbLineTriangulated,1.0,50);
        if(!isReconstructF && mLastFailureReason == "unknown")
            mLastFailureReason = "fundamental_failed";
        return isReconstructF;
    }
}

bool Initializer::InitializeKb8Bearing(const Frame &CurrentFrame,
                                       const vector<int> &vMatches12,
                                       cv::Mat &R21,
                                       cv::Mat &t21,
                                       vector<cv::Point3f> &vP3D,
                                       vector<bool> &vbTriangulated)
{
    mLastFailureReason = "unknown";
    mLastModel = "kb8_bearing_essential";
    mLastHomographyScore = 0.0f;
    mLastFundamentalScore = 0.0f;
    mLastHomographyRatio = 0.0f;
    mLastLegacySolverSuccess = false;
    mLastKb8ValidationSuccess = false;
    mLastFinalSuccess = false;
    ResetDecisionDiagnostics();

    if(!CurrentFrame.HasKb8PointObservations())
    {
        mLastFailureReason = "invalid_bearing_current_frame_has_no_kb8";
        return false;
    }
    if(CurrentFrame.mvPointUnitBearings.size() != CurrentFrame.mvKeys.size())
    {
        mLastFailureReason =
            "invalid_bearing_current_bearing_size_mismatch";
        return false;
    }
    if(CurrentFrame.mvbPointBearingValid.size() != CurrentFrame.mvKeys.size())
    {
        mLastFailureReason =
            "invalid_bearing_current_valid_size_mismatch";
        return false;
    }
    if(!mbHasKb8PointCameraParameters)
    {
        mLastFailureReason = "invalid_bearing_reference_frame_has_no_kb8";
        return false;
    }
    if(mvKb8Bearings1.size() != mvKeys1.size())
    {
        mLastFailureReason =
            "invalid_bearing_reference_bearing_size_mismatch";
        return false;
    }
    if(mvbKb8BearingValid1.size() != mvKeys1.size())
    {
        mLastFailureReason =
            "invalid_bearing_reference_valid_size_mismatch";
        return false;
    }
    if(mvKb8RawKeys1.size() != mvKeys1.size())
    {
        mLastFailureReason =
            "invalid_bearing_reference_raw_size_mismatch";
        return false;
    }
    if(!Kb8ParametersConsistent(CurrentFrame))
    {
        mLastFailureReason =
            "invalid_bearing_inconsistent_kb8_parameters";
        return false;
    }

    const vector<cv::KeyPoint> initializationKeys1 = mvKeys1;
    const vector<cv::KeyPoint> rawKeys2 = CurrentFrame.mvKeysUn;
    mvKeys2 = rawKeys2;
    if(!mbReferenceKeysAreInitializationView)
        mvKb8RawKeys1 = mvKeys1;
    mvKb8RawKeys2 = rawKeys2;
    mvKb8Bearings2 = CurrentFrame.mvPointUnitBearings;
    mvbKb8BearingValid2 = CurrentFrame.mvbPointBearingValid;

    mvMatches12.clear();
    mvbMatched1.assign(mvKeys1.size(), false);
    int rejected_total = 0;
    int rejected_reference_invalid = 0;
    int rejected_current_invalid = 0;
    int rejected_reference_nonfinite = 0;
    int rejected_current_nonfinite = 0;
    int rejected_ref_index_oob = 0;
    int rejected_cur_index_oob = 0;
    int rejected_match_sentinel_invalid = 0;
    for(size_t i=0, iend=vMatches12.size(); i<iend; ++i)
    {
        const int idx2 = vMatches12[i];
        if(idx2 < 0)
        {
            ++rejected_match_sentinel_invalid;
            continue;
        }
        bool reject = false;
        if(i >= mvKeys1.size() || i >= mvKb8Bearings1.size() ||
           i >= mvbKb8BearingValid1.size())
        {
            reject = true;
            ++rejected_ref_index_oob;
        }
        if(static_cast<size_t>(idx2) >= mvKeys2.size() ||
           static_cast<size_t>(idx2) >= mvKb8Bearings2.size() ||
           static_cast<size_t>(idx2) >= mvbKb8BearingValid2.size())
        {
            reject = true;
            ++rejected_cur_index_oob;
        }
        if(!reject)
        {
            if(!mvbKb8BearingValid1[i])
            {
                reject = true;
                ++rejected_reference_invalid;
            }
            if(!mvbKb8BearingValid2[idx2])
            {
                reject = true;
                ++rejected_current_invalid;
            }
            if(!mvKb8Bearings1[i].allFinite())
            {
                reject = true;
                ++rejected_reference_nonfinite;
            }
            if(!mvKb8Bearings2[idx2].allFinite())
            {
                reject = true;
                ++rejected_current_nonfinite;
            }
        }
        if(reject)
        {
            ++rejected_total;
            continue;
        }
        mvMatches12.push_back(make_pair(static_cast<int>(i), idx2));
        mvbMatched1[i] = true;
    }

    const int N = static_cast<int>(mvMatches12.size());
    if(N < 8)
    {
        if(N == 0)
        {
            int current_valid = 0;
            int reference_valid = 0;
            for(size_t i = 0; i < mvbKb8BearingValid2.size(); ++i)
                if(mvbKb8BearingValid2[i])
                    ++current_valid;
            for(size_t i = 0; i < mvbKb8BearingValid1.size(); ++i)
                if(mvbKb8BearingValid1[i])
                    ++reference_valid;
            mLastFailureReason =
                "invalid_bearing_no_valid_matches_after_filter:"
                "current_valid=" + std::to_string(current_valid) +
                ":reference_valid=" + std::to_string(reference_valid) +
                ":total_initial_matches=" + std::to_string(vMatches12.size()) +
                ":rejected_total=" + std::to_string(rejected_total) +
                ":rejected_match_sentinel_invalid=" +
                    std::to_string(rejected_match_sentinel_invalid) +
                ":rej_ref_index_oob=" + std::to_string(rejected_ref_index_oob) +
                ":rej_cur_index_oob=" + std::to_string(rejected_cur_index_oob) +
                ":rej_ref_invalid=" + std::to_string(rejected_reference_invalid) +
                ":rej_cur_invalid=" + std::to_string(rejected_current_invalid) +
                ":rej_ref_nonfinite=" + std::to_string(rejected_reference_nonfinite) +
                ":rej_cur_nonfinite=" + std::to_string(rejected_current_nonfinite);
        }
        else
        {
            mLastFailureReason = "insufficient_matches";
        }
        mvKeys1 = initializationKeys1;
        return false;
    }

    vector<bool> vbMatchesInliers;
    cv::Mat E;
    if(!FindKb8Essential(vbMatchesInliers, E))
    {
        mLastFailureReason = "essential_estimation_failure";
        mvKeys1 = initializationKeys1;
        return false;
    }

    cv::Mat R1, R2, t;
    DecomposeE(E, R1, R2, t);
    cv::Mat t1 = t;
    cv::Mat t2 = -t;

    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
    float parallax1 = 0.0f, parallax2 = 0.0f, parallax3 = 0.0f, parallax4 = 0.0f;

    mLastModel = "kb8_bearing_essential";
    BeginCandidateDiagnostics(mLastModel, 0);
    const int nGood1 = CheckKb8RT(R1, t1, vbMatchesInliers, vP3D1, vbTriangulated1, parallax1);
    BeginCandidateDiagnostics(mLastModel, 1);
    const int nGood2 = CheckKb8RT(R2, t1, vbMatchesInliers, vP3D2, vbTriangulated2, parallax2);
    BeginCandidateDiagnostics(mLastModel, 2);
    const int nGood3 = CheckKb8RT(R1, t2, vbMatchesInliers, vP3D3, vbTriangulated3, parallax3);
    BeginCandidateDiagnostics(mLastModel, 3);
    const int nGood4 = CheckKb8RT(R2, t2, vbMatchesInliers, vP3D4, vbTriangulated4, parallax4);

    const int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));
    const int nEssentialInliers = static_cast<int>(
        std::count(vbMatchesInliers.begin(), vbMatchesInliers.end(), true));
    int min_good_floor = 50;
    if(const char* value = std::getenv("CAM_B_INIT_MIN_GOOD_POINTS"))
        min_good_floor = std::max(8, std::atoi(value));
    double min_parallax_deg = 1.0;
    if(const char* value = std::getenv("CAM_B_INIT_MIN_PARALLAX_DEG"))
        min_parallax_deg = std::max(0.01, std::atof(value));
    const int nMinGood =
        max(static_cast<int>(0.9f * nEssentialInliers), min_good_floor);
    int nsimilar = 0;
    if(nGood1 == maxGood) ++nsimilar;
    if(nGood2 == maxGood) ++nsimilar;
    if(nGood3 == maxGood) ++nsimilar;
    if(nGood4 == maxGood) ++nsimilar;

    mLastCandidateCount = 4;
    mLastBestGoodPoints = maxGood;
    mLastSecondBestGoodPoints = 0;
    mLastBestCandidateId = 0;
    const int good_counts[4] = {nGood1, nGood2, nGood3, nGood4};
    for(int i = 0; i < 4; ++i)
    {
        if(good_counts[i] == maxGood)
        {
            mLastBestCandidateId = i;
            break;
        }
    }
    for(int i = 0; i < 4; ++i)
    {
        if(i == mLastBestCandidateId)
            continue;
        mLastSecondBestGoodPoints =
            max(mLastSecondBestGoodPoints, good_counts[i]);
    }
    mLastRequiredMinGoodPoints = nMinGood;
    mLastRequiredMinParallaxDeg =
        static_cast<float>(min_parallax_deg);
    mLastAmbiguityRatioOrThreshold = 1.0f;

    UpdateBestCandidateDiagnostics();

    if(maxGood < nMinGood || nsimilar > 1)
    {
        mLastExactFailureBranch =
            (maxGood < nMinGood) ?
            "Kb8Essential:maxGood<nMinGood" :
            "Kb8Essential:directed_positive_lambda_exact_tie";
        mLastFailureReason = "recover_pose_failed";
        mvKeys1 = initializationKeys1;
        return false;
    }

    bool initialized = false;
    if(mLastBestCandidateId == 0)
    {
        if(parallax1 > min_parallax_deg)
        {
            MarkAcceptedCandidate(mLastModel, 0);
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;
            R1.copyTo(R21);
            t1.copyTo(t21);
            initialized = true;
        }
    }
    else if(mLastBestCandidateId == 1)
    {
        if(parallax2 > min_parallax_deg)
        {
            MarkAcceptedCandidate(mLastModel, 1);
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;
            R2.copyTo(R21);
            t1.copyTo(t21);
            initialized = true;
        }
    }
    else if(mLastBestCandidateId == 2)
    {
        if(parallax3 > min_parallax_deg)
        {
            MarkAcceptedCandidate(mLastModel, 2);
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;
            R1.copyTo(R21);
            t2.copyTo(t21);
            initialized = true;
        }
    }
    else if(mLastBestCandidateId == 3)
    {
        if(parallax4 > min_parallax_deg)
        {
            MarkAcceptedCandidate(mLastModel, 3);
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;
            R2.copyTo(R21);
            t2.copyTo(t21);
            initialized = true;
        }
    }

    if(!initialized)
    {
        mLastExactFailureBranch =
            "Kb8Essential:bestParallax<=minParallax";
        mLastFailureReason = "parallax_failed";
        mvKeys1 = initializationKeys1;
        return false;
    }

    mLastLegacySolverSuccess = initialized;
    mLastKb8ValidationSuccess = initialized;
    if(initialized)
    {
        int triangulated = 0;
        for(size_t i=0; i<vbTriangulated.size(); ++i)
            if(vbTriangulated[i])
                ++triangulated;
        int min_reprojection_floor = 50;
        if(const char* value =
               std::getenv("CAM_B_INIT_MIN_REPROJECTION_POINTS"))
        {
            min_reprojection_floor = std::max(8, std::atoi(value));
        }
        const int minValidated = max(static_cast<int>(0.9f*triangulated),
                                     min_reprojection_floor);
        if(!ValidateKb8Reprojection(R21, t21, vP3D, vbTriangulated,
                                    minValidated))
        {
            initialized = false;
            mLastKb8ValidationSuccess = false;
            mLastExactFailureBranch =
                "InitializeKb8Bearing:ValidateKb8Reprojection<minValidated";
            mLastFailureReason = "kb8_reprojection_validation_failed";
        }
    }

    mLastFinalSuccess = initialized;
    mvKeys1 = initializationKeys1;
    return initialized;
}

void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    const int N = mvMatches12.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();

        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();
    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        F21i = T2t*Fn*T1;

        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}

cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}


bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, vector<Match> &vLineMatchesF,cv::Mat &F21, cv::Mat &K,
                               cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<Point3f> &vLineS3D,
                               vector<Point3f> &vLineE3D, vector<bool> &vbTriangulated,vector<bool> &vbLineTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;



    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;  //从F矩阵计算E矩阵

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    DecomposeE(E21,R1,R2,t);

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    BeginCandidateDiagnostics(mLastModel, 0);
    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    BeginCandidateDiagnostics(mLastModel, 1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    BeginCandidateDiagnostics(mLastModel, 2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    BeginCandidateDiagnostics(mLastModel, 3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));
    mLastCandidateCount = 4;
    mLastBestGoodPoints = maxGood;
    mLastRequiredMinParallaxDeg = minParallax;
    mLastAmbiguityRatioOrThreshold = 0.7f;

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);
    mLastRequiredMinGoodPoints = nMinGood;

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;
    int good_counts[4] = {nGood1, nGood2, nGood3, nGood4};
    for(int i = 0; i < 4; ++i)
    {
        if(good_counts[i] == maxGood)
        {
            mLastBestCandidateId = i;
            break;
        }
    }
    mLastSecondBestGoodPoints = 0;
    for(int i = 0; i < 4; ++i)
    {
        if(i == mLastBestCandidateId)
            continue;
        mLastSecondBestGoodPoints =
            max(mLastSecondBestGoodPoints, good_counts[i]);
    }
    UpdateBestCandidateDiagnostics();

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {
        mLastExactFailureBranch =
            (maxGood < nMinGood) ?
            "ReconstructF:maxGood<nMinGood" :
            "ReconstructF:nsimilar>1";
        mLastFailureReason = "recover_pose_failed";
        return false;
    }

    // If best reconstruction has enough parallax initialize
    // 根据特征点三角化的特征点数量，以及视差角选择最优的Rt，此时就可以利用Rt三角化线特征了
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            MarkAcceptedCandidate(mLastModel, 0);
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            ReconstructLine(vLineMatchesF, mK, R21, t21, mvKeyLineFunctions1, mvKeyLineFunctions2, vLineS3D, vLineE3D, vbLineTriangulated);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            MarkAcceptedCandidate(mLastModel, 1);
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            ReconstructLine(vLineMatchesF, mK, R21, t21, mvKeyLineFunctions1, mvKeyLineFunctions2, vLineS3D, vLineE3D, vbLineTriangulated);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            MarkAcceptedCandidate(mLastModel, 2);
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            ReconstructLine(vLineMatchesF, mK, R21, t21, mvKeyLineFunctions1, mvKeyLineFunctions2, vLineS3D, vLineE3D, vbLineTriangulated);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            MarkAcceptedCandidate(mLastModel, 3);
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            ReconstructLine(vLineMatchesF, mK, R21, t21, mvKeyLineFunctions1, mvKeyLineFunctions2, vLineS3D, vLineE3D, vbLineTriangulated);
            return true;
        }
    }

    if(mLastFailureReason == "unknown")
    {
        mLastExactFailureBranch = "ReconstructF:bestParallax<=minParallax";
        mLastFailureReason = "parallax_failed";
    }
    return false;
}


bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers,vector<Match> &vLineMatchesH, cv::Mat &H21, cv::Mat &K,
                               cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<Point3f> &vLineS3D,
                               vector<Point3f> &vLineE3D,vector<bool> &vbTriangulated,vector<bool> &vbLineTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    // step1：因为特征点是图像坐标系，所以将H矩阵由相机坐标系转换到图像坐标系下
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    // SVD分解的正常情况是特征值降序排列
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        mLastFailureReason = "homography_failed";
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    // 下面类似于RANSAC算法，对8组R t检测，并进行3D特征点计算
    int bestGood = 0;
    int secondBestGood = 0;
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        BeginCandidateDiagnostics(mLastModel, static_cast<int>(i));
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }
    mLastCandidateCount = 8;
    mLastBestCandidateId = bestSolutionIdx;
    mLastBestGoodPoints = bestGood;
    mLastSecondBestGoodPoints = secondBestGood;
    mLastRequiredMinGoodPoints =
        max(static_cast<int>(0.9f*N), minTriangulated);
    mLastRequiredMinParallaxDeg = minParallax;
    mLastAmbiguityRatioOrThreshold = 0.75f;
    UpdateBestCandidateDiagnostics();

    // 如果次优结果数量小于最优结果数量的0.75，则为最优结果
    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        MarkAcceptedCandidate(mLastModel, bestSolutionIdx);
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        cout << "in Initializer::ReconstructH, " << endl;
        cout << "\t R21 = " << R21 << endl;
        cout << "\t t21 = " << t21 << endl;
        ReconstructLine(vLineMatchesH, mK, R21, t21, mvKeyLineFunctions1, mvKeyLineFunctions2, vLineS3D, vLineE3D, vbLineTriangulated);    //三角化建立线特征的3D端点
        return true;



    }

    if(mLastFailureReason == "unknown")
    {
        if(!(secondBestGood < 0.75f * bestGood))
            mLastExactFailureBranch =
                "ReconstructH:secondBestGood>=0.75*bestGood";
        else if(!(bestParallax >= minParallax))
            mLastExactFailureBranch =
                "ReconstructH:bestParallax<minParallax";
        else if(!(bestGood > minTriangulated))
            mLastExactFailureBranch =
                "ReconstructH:bestGood<=minTriangulated";
        else
            mLastExactFailureBranch =
                "ReconstructH:bestGood<=0.9*N";
        mLastFailureReason = "parallax_failed";
    }
    return false;
}


void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}


int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                         const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                         const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);

    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    // step1：得到相机第一帧的投影矩阵
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    // 第一帧的相机光心世界坐标系
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    // step2:得到第二帧的投影矩阵
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    //cout<<"the second frame"<<R<<", "<<t<<", "<<vMatches12.size()<<","<<endl;
    P2 = K*P2;

    // 第二帧的相机光心世界坐标系
    cv::Mat O2 = -R.t()*t;

    int nGood=0;
    InitializationPoseCandidateDiagnostics candidate_diagnostics;
    candidate_diagnostics.model = mPendingCheckModel;
    candidate_diagnostics.candidate_id = mPendingCheckCandidateId;


    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        // step3：利用三角化法恢复出三维点 p3dC1
        Triangulate(kp1,kp2,P1,P2,p3dC1);
        //cout<<"p3dC1:"<<p3dC1<<endl;
        // step4：判断三角化的特征点的三个坐标是否有效
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        // step5：计算三维特征点与两帧的相机光心之间的夹角余弦
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        cv::Mat p3dC2 = R*p3dC1+t;
        const bool positive1 = p3dC1.at<float>(2) > 0;
        const bool positive2 = p3dC2.at<float>(2) > 0;
        if(positive1)
            candidate_diagnostics.positive_depth_cam1++;
        if(positive2)
            candidate_diagnostics.positive_depth_cam2++;

        // step6：判断3D点是否在两个相机前方
        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // step6.1:3D点深度为负，在第一个相机后方，淘汰
        if(!positive1)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // step6.2:3D点深度为负，在第二个相机后方，淘汰
        if(!positive2)
            continue;
        candidate_diagnostics.both_positive++;

        // step7：计算重投影误差
        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;
        candidate_diagnostics.reprojection_pass++;

        // step8：统计经过检验的3D点个数，记录3D点视差角
        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;


        vbGood[vMatches12[i].first]=true;
    }

    // step8：得到3D点中较大的视差角
    if(nGood>0)
    {
        // 从小到大排序
        sort(vCosParallax.begin(),vCosParallax.end());

        // 技巧：排序后并没有取最大的视差角
        // 取适当大的
        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    candidate_diagnostics.triangulated = nGood;
    candidate_diagnostics.median_parallax_deg = parallax;
    mLastPoseCandidateDiagnostics.push_back(candidate_diagnostics);

    return nGood;
}


void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}


void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

bool Initializer::BearingToNormalizedPoint(const Eigen::Vector3d &bearing,
                                           cv::Point2f &point) const
{
    if(!bearing.allFinite())
        return false;
    const double norm = bearing.norm();
    if(!std::isfinite(norm) || norm <= 0.0 ||
       std::abs(norm - 1.0) > 1e-6 ||
       std::abs(bearing.z()) < 1e-9)
    {
        return false;
    }
    point.x = static_cast<float>(bearing.x() / bearing.z());
    point.y = static_cast<float>(bearing.y() / bearing.z());
    return std::isfinite(point.x) && std::isfinite(point.y);
}

bool Initializer::Kb8ParametersConsistent(const Frame& CurrentFrame) const
{
    if(!mbHasKb8PointCameraParameters ||
       !CurrentFrame.HasKb8PointObservations())
    {
        return false;
    }
    const KannalaBrandt8Camera::Parameters& rhs =
        CurrentFrame.mKb8PointCameraParameters;
    const double eps = 1e-9;
    return std::abs(mKb8PointCameraParameters.fx - rhs.fx) <= eps &&
           std::abs(mKb8PointCameraParameters.fy - rhs.fy) <= eps &&
           std::abs(mKb8PointCameraParameters.cx - rhs.cx) <= eps &&
           std::abs(mKb8PointCameraParameters.cy - rhs.cy) <= eps &&
           std::abs(mKb8PointCameraParameters.k1 - rhs.k1) <= eps &&
           std::abs(mKb8PointCameraParameters.k2 - rhs.k2) <= eps &&
           std::abs(mKb8PointCameraParameters.k3 - rhs.k3) <= eps &&
           std::abs(mKb8PointCameraParameters.k4 - rhs.k4) <= eps &&
           mKb8PointCameraParameters.width == rhs.width &&
           mKb8PointCameraParameters.height == rhs.height;
}

bool Initializer::RectifyKb8Keypoints(
        const vector<cv::KeyPoint>& rawKeys,
        vector<cv::KeyPoint>& rectifiedKeys) const
{
    rectifiedKeys = rawKeys;
    if(!mbHasKb8PointCameraParameters)
        return false;
    if(rawKeys.empty())
        return true;

    vector<cv::Point2f> points(rawKeys.size());
    for(size_t i=0; i<rawKeys.size(); ++i)
    {
        points[i] = rawKeys[i].pt;
        if(!std::isfinite(points[i].x) || !std::isfinite(points[i].y))
            return false;
    }

    const KannalaBrandt8Camera::Parameters& p =
        mKb8PointCameraParameters;
    cv::Mat K = (cv::Mat_<float>(3,3) <<
        static_cast<float>(p.fx), 0.0f, static_cast<float>(p.cx),
        0.0f, static_cast<float>(p.fy), static_cast<float>(p.cy),
        0.0f, 0.0f, 1.0f);
    cv::Mat D = (cv::Mat_<float>(4,1) <<
        static_cast<float>(p.k1), static_cast<float>(p.k2),
        static_cast<float>(p.k3), static_cast<float>(p.k4));
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);

    try
    {
        cv::fisheye::undistortPoints(points, points, K, D, R, K);
    }
    catch(const cv::Exception&)
    {
        return false;
    }

    for(size_t i=0; i<points.size(); ++i)
    {
        if(!std::isfinite(points[i].x) || !std::isfinite(points[i].y))
            return false;
        rectifiedKeys[i].pt = points[i];
    }
    return true;
}

bool Initializer::ValidateKb8Reprojection(
        const cv::Mat& R21,
        const cv::Mat& t21,
        const vector<cv::Point3f>& vP3D,
        const vector<bool>& vbTriangulated,
        const int minValidated) const
{
    if(!mbHasKb8PointCameraParameters || R21.empty() || t21.empty())
        return false;
    const KannalaBrandt8Camera camera(mKb8PointCameraParameters);
    if(!camera.valid())
        return false;

    int validated = 0;
    const double th2 = 4.0 * static_cast<double>(mSigma2);
    for(size_t i=0; i<mvMatches12.size(); ++i)
    {
        const int idx1 = mvMatches12[i].first;
        const int idx2 = mvMatches12[i].second;
        if(idx1 < 0 || idx2 < 0 ||
           static_cast<size_t>(idx1) >= vbTriangulated.size() ||
           static_cast<size_t>(idx1) >= vP3D.size() ||
           !vbTriangulated[idx1] ||
           static_cast<size_t>(idx1) >= mvKb8Bearings1.size() ||
           static_cast<size_t>(idx2) >= mvKb8Bearings2.size())
        {
            continue;
        }

        const cv::Point3f& X = vP3D[idx1];
        cv::Mat X1 = (cv::Mat_<float>(3,1) << X.x, X.y, X.z);
        cv::Mat X2 = R21 * X1 + t21;
        Eigen::Vector2d uv1;
        Eigen::Vector2d uv2;
        if(!camera.project(Eigen::Vector3d(X1.at<float>(0),
                                           X1.at<float>(1),
                                           X1.at<float>(2)), uv1) ||
           !camera.project(Eigen::Vector3d(X2.at<float>(0),
                                           X2.at<float>(1),
                                           X2.at<float>(2)), uv2))
        {
            continue;
        }
        if(static_cast<size_t>(idx1) >= mvKb8RawKeys1.size() ||
           static_cast<size_t>(idx2) >= mvKb8RawKeys2.size())
        {
            continue;
        }
        const cv::Point2f raw1 = mvKb8RawKeys1[idx1].pt;
        const cv::Point2f raw2 = mvKb8RawKeys2[idx2].pt;
        const double e1 =
            (uv1.x() - raw1.x) * (uv1.x() - raw1.x) +
            (uv1.y() - raw1.y) * (uv1.y() - raw1.y);
        const double e2 =
            (uv2.x() - raw2.x) * (uv2.x() - raw2.x) +
            (uv2.y() - raw2.y) * (uv2.y() - raw2.y);
        if(std::isfinite(e1) && std::isfinite(e2) &&
           e1 <= th2 && e2 <= th2)
        {
            ++validated;
        }
    }
    return validated >= minValidated;
}

bool Initializer::FindKb8Essential(vector<bool> &vbMatchesInliers,
                                   cv::Mat &E21)
{
    const int N = static_cast<int>(mvMatches12.size());
    vbMatchesInliers.assign(N, false);
    E21 = cv::Mat();
    if(N < 8)
        return false;

    double threshold_deg = 0.5;
    if(const char* value = std::getenv("CAM_B_SPHERICAL_THRESHOLD_DEG"))
        threshold_deg = std::max(0.01, std::atof(value));
    const double threshold = threshold_deg * CV_PI / 180.0;

    auto fit = [&](const vector<int>& ids, cv::Mat& E) -> bool {
        if(ids.size() < 8) return false;
        cv::Mat A(static_cast<int>(ids.size()), 9, CV_64F);
        for(size_t row=0; row<ids.size(); ++row)
        {
            const Match& m = mvMatches12[ids[row]];
            const Eigen::Vector3d a = mvKb8Bearings1[m.first].normalized();
            const Eigen::Vector3d b = mvKb8Bearings2[m.second].normalized();
            int k=0;
            for(int r=0;r<3;++r) for(int c=0;c<3;++c)
                A.at<double>(static_cast<int>(row),k++) = b[r]*a[c];
        }
        cv::Mat w,u,vt;
        cv::SVD::compute(A,w,u,vt,cv::SVD::FULL_UV);
        if(vt.rows < 9) return false;
        cv::Mat raw = vt.row(8).reshape(1,3).clone();
        cv::SVD::compute(raw,w,u,vt,cv::SVD::FULL_UV);
        if(cv::determinant(u*vt)<0) u.col(2) *= -1.0;
        const double s=0.5*(w.at<double>(0)+w.at<double>(1));
        E=u*cv::Mat::diag((cv::Mat_<double>(3,1)<<s,s,0.0))*vt;
        return cv::checkRange(E);
    };
    auto residual = [&](const cv::Mat& E, int i) -> double {
        const Match& m=mvMatches12[i];
        const Eigen::Vector3d a=mvKb8Bearings1[m.first].normalized();
        const Eigen::Vector3d b=mvKb8Bearings2[m.second].normalized();
        cv::Mat ac=(cv::Mat_<double>(3,1)<<a.x(),a.y(),a.z());
        cv::Mat bc=(cv::Mat_<double>(3,1)<<b.x(),b.y(),b.z());
        cv::Mat n2=E*ac, n1=E.t()*bc;
        const double d2=cv::norm(n2), d1=cv::norm(n1);
        if(d1<1e-12 || d2<1e-12) return CV_PI;
        const double q2=std::min(1.0,std::abs(bc.dot(n2))/d2);
        const double q1=std::min(1.0,std::abs(ac.dot(n1))/d1);
        const double e2=std::asin(q2), e1=std::asin(q1);
        return std::sqrt(0.5*(e1*e1+e2*e2));
    };

    std::mt19937 rng(0x43414d42u);
    vector<int> pool(N); std::iota(pool.begin(),pool.end(),0);
    vector<int> best; cv::Mat bestE; double best_cost=1e100;
    for(int it=0; it<mMaxIterations; ++it)
    {
        std::shuffle(pool.begin(),pool.end(),rng);
        vector<int> sample(pool.begin(),pool.begin()+8); cv::Mat E;
        if(!fit(sample,E)) continue;
        vector<int> inliers; double cost=0.0;
        for(int i=0;i<N;++i) { const double e=residual(E,i); if(e<=threshold){inliers.push_back(i);cost+=e;}}
        if(inliers.size()>best.size() || (inliers.size()==best.size() && cost<best_cost))
        { best.swap(inliers); bestE=E.clone(); best_cost=cost; }
    }
    if(best.size()<8 || !fit(best,bestE)) return false;
    best.clear();
    for(int i=0;i<N;++i) if(residual(bestE,i)<=threshold) best.push_back(i);
    if(best.size()<8 || !fit(best,bestE)) return false;
    for(int i:best) vbMatchesInliers[i]=true;
    bestE.convertTo(E21,CV_32F);
    return true;
}

void Initializer::TriangulateBearings(const Eigen::Vector3d &b1d,
                                      const Eigen::Vector3d &b2d,
                                      const cv::Mat &R,
                                      const cv::Mat &t,
                                      cv::Mat &x3D,
                                      double &lambda1,
                                      double &lambda2,
                                      double &rayMiss,
                                      double &condition)
{
    lambda1 = lambda2 = rayMiss = condition =
        std::numeric_limits<double>::quiet_NaN();
    if(!b1d.allFinite() || !b2d.allFinite())
    {
        x3D = cv::Mat();
        return;
    }

    Eigen::Matrix3d Re;
    Eigen::Vector3d te;
    for(int r=0;r<3;++r) {
        te[r] = t.at<float>(r);
        for(int c=0;c<3;++c) Re(r,c)=R.at<float>(r,c);
    }
    const Eigen::Vector3d b1=b1d.normalized(), b2=b2d.normalized();
    Eigen::Matrix<double,3,2> A;
    A.col(0)=Re*b1;
    A.col(1)=-b2;
    Eigen::JacobiSVD<Eigen::Matrix<double,3,2> > svd(
        A,Eigen::ComputeFullU|Eigen::ComputeFullV);
    if(svd.singularValues()(1) <= 1e-12)
    {
        x3D = cv::Mat();
        return;
    }
    const Eigen::Vector2d lambda=svd.solve(-te);
    lambda1=lambda[0]; lambda2=lambda[1];
    condition=svd.singularValues()(0)/svd.singularValues()(1);
    const Eigen::Vector3d X1=lambda1*b1;
    rayMiss=(lambda2*b2-(Re*X1+te)).norm();
    if(!X1.allFinite() || !std::isfinite(rayMiss) || !std::isfinite(condition))
    {
        x3D=cv::Mat(); return;
    }
    x3D=(cv::Mat_<float>(3,1)<<static_cast<float>(X1.x()),
          static_cast<float>(X1.y()),static_cast<float>(X1.z()));
}

int Initializer::CheckKb8RT(const cv::Mat &R,
                            const cv::Mat &t,
                            const vector<bool> &vbMatchesInliers,
                            vector<cv::Point3f> &vP3D,
                            vector<bool> &vbGood,
                            float &parallax)
{
    vbGood.assign(mvKeys1.size(), false);
    vP3D.assign(mvKeys1.size(), cv::Point3f());
    std::vector<double> parallaxes, misses, angular_errors, conditions;
    InitializationPoseCandidateDiagnostics candidate_diagnostics;
    candidate_diagnostics.model = mPendingCheckModel;
    candidate_diagnostics.candidate_id = mPendingCheckCandidateId;
    Eigen::Matrix3d Re;
    for(int r=0;r<3;++r) {
        candidate_diagnostics.t21[r]=t.at<float>(r);
        for(int c=0;c<3;++c) {
            Re(r,c)=R.at<float>(r,c);
            candidate_diagnostics.R21[3*r+c]=Re(r,c);
        }
    }
    int nGood = 0;
    for(size_t i=0, iend=mvMatches12.size(); i<iend; ++i)
    {
        if(i >= vbMatchesInliers.size() || !vbMatchesInliers[i])
            continue;
        ++candidate_diagnostics.valid_match_count;
        const int idx1 = mvMatches12[i].first;
        const int idx2 = mvMatches12[i].second;
        const Eigen::Vector3d &b1 = mvKb8Bearings1[idx1];
        const Eigen::Vector3d &b2 = mvKb8Bearings2[idx2];
        const double theta1=std::acos(std::max(-1.0,std::min(1.0,
            b1.normalized().z())))*180.0/CV_PI;
        const int bin=theta1<30.0?0:(theta1<60.0?1:(theta1<90.0?2:3));
        ++candidate_diagnostics.angular_bin_total[bin];

        cv::Mat p3dC1; double lambda1,lambda2,ray_miss,condition;
        TriangulateBearings(b1,b2,R,t,p3dC1,lambda1,lambda2,
                            ray_miss,condition);
        if(p3dC1.empty() ||
           !std::isfinite(p3dC1.at<float>(0)) ||
           !std::isfinite(p3dC1.at<float>(1)) ||
           !std::isfinite(p3dC1.at<float>(2)))
        {
            continue;
        }
        misses.push_back(ray_miss);
        conditions.push_back(condition);
        if(lambda1<=0.0) ++candidate_diagnostics.negative_lambda1_count;
        if(lambda2<=0.0) ++candidate_diagnostics.negative_lambda2_count;
        if(lambda1<=0.0 || lambda2<=0.0) continue;
        ++candidate_diagnostics.positive_lambda_count;
        ++candidate_diagnostics.angular_bin_positive[bin];

        cv::Mat b1cv = (cv::Mat_<float>(3,1)
            << static_cast<float>(b1.x()),
               static_cast<float>(b1.y()),
               static_cast<float>(b1.z()));
        cv::Mat b2cv = (cv::Mat_<float>(3,1)
            << static_cast<float>(b2.x()),
               static_cast<float>(b2.y()),
               static_cast<float>(b2.z()));
        cv::Mat p3dC2 = R * p3dC1 + t;
        candidate_diagnostics.positive_depth_cam1++;
        candidate_diagnostics.positive_depth_cam2++;
        candidate_diagnostics.both_positive++;
        const Eigen::Vector3d b2_in_1=Re.transpose()*b2.normalized();
        const double cosParallax=std::max(-1.0,std::min(1.0,
            b1.normalized().dot(b2_in_1)));
        parallaxes.push_back(std::acos(cosParallax)*180.0/CV_PI);

        cv::Mat p3dC1n = p3dC1 / cv::norm(p3dC1);
        cv::Mat p3dC2n = p3dC2 / cv::norm(p3dC2);
        const float cosErr1 =
            std::max(-1.0f, std::min(1.0f,
                static_cast<float>(p3dC1n.dot(b1cv))));
        const float cosErr2 =
            std::max(-1.0f, std::min(1.0f,
                static_cast<float>(p3dC2n.dot(b2cv))));
        const double max_error=std::max(std::acos(cosErr1),std::acos(cosErr2));
        angular_errors.push_back(max_error*180.0/CV_PI);
        const float maxAngularErrorRad = 0.02f;
        if(max_error > maxAngularErrorRad)
        {
            continue;
        }
        candidate_diagnostics.reprojection_pass++;

        vP3D[idx1] = cv::Point3f(p3dC1.at<float>(0),
                                 p3dC1.at<float>(1),
                                 p3dC1.at<float>(2));
        vbGood[idx1] = true;
        ++nGood;
    }

    parallax=parallaxes.empty()?0.0f:static_cast<float>(Quantile(parallaxes,0.5));

    candidate_diagnostics.triangulated = nGood;
    candidate_diagnostics.median_parallax_deg = parallax;
    candidate_diagnostics.positive_lambda_ratio =
        candidate_diagnostics.valid_match_count>0 ?
        static_cast<double>(candidate_diagnostics.positive_lambda_count)/
        candidate_diagnostics.valid_match_count : 0.0;
    candidate_diagnostics.ray_miss_median=Quantile(misses,0.5);
    candidate_diagnostics.ray_miss_p95=Quantile(misses,0.95);
    candidate_diagnostics.angular_error_median_deg=Quantile(angular_errors,0.5);
    candidate_diagnostics.angular_error_p95_deg=Quantile(angular_errors,0.95);
    candidate_diagnostics.triangulation_condition_median=Quantile(conditions,0.5);
    candidate_diagnostics.triangulation_condition_p95=Quantile(conditions,0.95);
    mLastPoseCandidateDiagnostics.push_back(candidate_diagnostics);

    return candidate_diagnostics.positive_lambda_count;
}


void Initializer::LineTriangulate(const KeyLine &kl1, const KeyLine &kl2, const cv::Mat &P1, const cv::Mat &P2,
                          const Vector3d &klf1, const Vector3d &klf2,
                          cv::Mat &LineStart3D, cv::Mat &LineEnd3D)
{
    const float &fx1 = P1.at<float>(0,0);
    const float &fy1 = P1.at<float>(1,1);
    const float &cx1 = P1.at<float>(0,2);
    const float &cy1 = P1.at<float>(1,2);
    const float &invfx1 = 1.0/P1.at<float>(0,0);
    const float &invfy1 = 1.0/P1.at<float>(1,1);
    cv::Mat K=cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = P1.at<float>(0,0);
    K.at<float>(1,1) = P1.at<float>(1,1);
    K.at<float>(0,2)= P1.at<float>(0,2);
    K.at<float>(1,2) = P1.at<float>(1,2);


    Mat lineF1 = (Mat_<float>(3,1) << klf1(0), klf1(1), klf1(2));
    Mat lineF2 = (Mat_<float>(3,1) << klf2(0), klf2(1), klf2(2));

    cv::Mat e = cv::Mat::eye(4,4,CV_32F);
    cv::Mat Tw(3,4,CV_32F);
    e.rowRange(0,3).colRange(0,3).copyTo(Tw.colRange(0,3));
    e.rowRange(0,3).col(3).copyTo(Tw.col(3));
    cv::Mat p1=K*Tw;
    cv::Mat StartC1, EndC1;
    StartC1 = (cv::Mat_<float>(3,1) << (kl1.startPointX-cx1)*invfx1, (kl1.startPointY-cy1)*invfy1, 1.0);
    EndC1 = (cv::Mat_<float>(3,1) << (kl1.endPointX-cx1)*invfx1, (kl1.endPointY-cy1)*invfy1, 1.0);

    cv::Mat A(4, 4, CV_32F);
    A.row(0) = lineF1.t()*p1;
    A.row(1) = lineF2.t()*P2;
    A.row(2) = StartC1.at<float>(0) * Tw.row(2) - Tw.row(0);
    A.row(3) = StartC1.at<float>(1) * Tw.row(2) - Tw.row(1);

    cv::Mat u1, w1, vt1;
    cv::SVD::compute(A, w1, u1, vt1, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    LineStart3D = vt1.row(3).t();

    LineStart3D = LineStart3D.rowRange(0,3)/LineStart3D.at<float>(3);
    cout<<LineStart3D<<endl;
    // 终止点
    cv::Mat B(4, 4, CV_32F);
    B.row(0) = lineF1.t()*p1;
    B.row(1) = lineF2.t()*P2;
    B.row(2) = EndC1.at<float>(0) *Tw.row(2) -Tw.row(0);
    B.row(3) = EndC1.at<float>(1) * Tw.row(2) - Tw.row(1);

    cv::Mat u2, w2, vt2;
    cv::SVD::compute(B, w2, u2, vt2, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    LineEnd3D = vt2.row(3).t();
    LineEnd3D = LineEnd3D.rowRange(0,3)/LineEnd3D.at<float>(3);
}



void Initializer::ReconstructLine(vector<Match> &vLineMatches, cv::Mat &K, cv::Mat &R21, cv::Mat &t21,
                                  vector<Vector3d> &vKeyLineFunctions1, vector<Vector3d> &vKeyLineFunctions2,
                                  vector<Point3f> &vLineS3D, vector<Point3f> &vLineE3D, vector<bool> &vbLineTriangulated)
{
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vLineS3D.resize(vLineMatches.size(), Point3f(0, 0, 0));
    vLineE3D.resize(vLineMatches.size(), Point3f(0, 0, 0));
    vbLineTriangulated.resize(vLineMatches.size(), true);

    // 得到第一个相机的投影矩阵，并以第一个相机的光心作为世界坐标系
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // 得到第二个相机的投影矩阵，并以第二个相机的光心作为世界坐标系
    cv::Mat P2(3, 4, CV_32F);
    R21.copyTo(P2.rowRange(0,3).colRange(0,3));
    t21.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;
    cv::Mat O2 = -R21.t()*t21;

    vector<double> errorC1, errorC2;
    errorC1.resize(vLineMatches.size(), 0);
    errorC2.resize(vLineMatches.size(), 0);

    for(size_t i=0, iend = vLineMatches.size(); i<iend; i++)
    {
        // 匹配的两条特征线
        const KeyLine &kl1 = mvKeyLines1[vLineMatches[i].first];
        const KeyLine &kl2 = mvKeyLines2[vLineMatches[i].second];
        cv::Mat L3dSC1; //线特征的3D起始点在camera1下的坐标
        cv::Mat L3dEC1; //线特征的3D终止点在camera1下的坐标
        Vector3d &klf1 = mvKeyLineFunctions1[vLineMatches[i].first];    //第一帧的特征线段所在直线系数
        Vector3d &klf2 = mvKeyLineFunctions2[vLineMatches[i].second];   //第二帧的特征线段所在直线系数


//        LineTriangulate(kl1, kl2, P1, P2, L3dSC1, L3dEC1);
        LineTriangulate(kl1, kl2, P1, P2, klf1, klf2, L3dSC1, L3dEC1);

        if(!isfinite(L3dSC1.at<float>(0)) || !isfinite(L3dSC1.at<float>(1)) || !isfinite(L3dSC1.at<float>(2)) ||
           !isfinite(L3dEC1.at<float>(0)) || !isfinite(L3dEC1.at<float>(1)) || !isfinite(L3dEC1.at<float>(2)) )
        {
            continue;
        }

        vLineS3D[i] = cv::Point3f(L3dSC1.at<float>(0), L3dSC1.at<float>(1), L3dSC1.at<float>(2));
        vLineE3D[i] = cv::Point3f(L3dEC1.at<float>(0), L3dEC1.at<float>(1), L3dEC1.at<float>(2));

        // 线特征的两个端点在camera2下的坐标
        cv::Mat L3dSC2 = R21*L3dSC1 + t21;
        cv::Mat L3dEC2 = R21*L3dEC1 + t21;

        //***********根据线特征的重投影误差****************
        // step1: 计算起始点和终止点在第一帧上的重投影误差
        // 1.1：计算线段所在的直线方程
        Vector3d l_obs1 = vKeyLineFunctions1[vLineMatches[i].first];

        // 1.2:计算起始点和终止点的3D坐标反投影至图像平面上的坐标
        float im1Startx, im1Starty;
        float invZ1start = 1.0/L3dSC1.at<float>(2);
        im1Startx = fx*L3dSC1.at<float>(0)*invZ1start+cx;
        im1Starty = fy*L3dSC1.at<float>(1)*invZ1start+cy;

        float im1Endx, im1Endy;
        float invZ1end = 1.0/L3dEC1.at<float>(2);
        im1Endx = fx*L3dEC1.at<float>(0)*invZ1end+cx;
        im1Endy = fx*L3dEC1.at<float>(1)*invZ1end+cy;

        // 1.3：计算起始点和终止点的2D反投影坐标到直线的距离，误差为距离之和
        Vector2d err_l1;
        err_l1(0) = l_obs1(0)*im1Startx + l_obs1(1)*im1Starty + l_obs1(2);
        err_l1(1) = l_obs1(0)*im1Endx + l_obs1(1)*im1Endy + l_obs1(2);
        double err1 = err_l1.norm();
        errorC1[i] = err1;

        // step2：计算起始点和终止点在第二帧上的重投影误差
        // 2.1：计算第二帧的线段所在直线的方程
        Vector3d l_obs2 = vKeyLineFunctions2[vLineMatches[i].second];

        // 2.2：计算起始点和终止点的3D坐标反投影至图像平面上的坐标
        float im2Startx, im2Starty;
        float invZ2start = 1.0/L3dSC2.at<float>(2);
        im2Startx = fx*L3dSC2.at<float>(0)*invZ2start+cx;
        im2Starty = fy*L3dSC2.at<float>(1)*invZ2start+cy;

        float im2Endx, im2Endy;
        float invZ2end = 1.0/L3dEC2.at<float>(2);
        im2Endx = fx*L3dEC2.at<float>(0)*invZ2end+cx;
        im2Endy = fy*L3dEC2.at<float>(1)*invZ2end+cy;

        // 2.3：计算起始点和终止点的2D反投影坐标到直线的距离，误差为距离之和
        Vector2d err_l2;
        err_l2(0) = l_obs2(0)*im2Startx + l_obs2(1)*im2Starty + l_obs2(2);
        err_l2(1) = l_obs2(0)*im2Endx + l_obs2(1)*im2Endy + l_obs2(2);
        double err2 = err_l2.norm();
        errorC2[i] = err2;
    }

    //**********去除线特征集中的异常值**********
    double inlier_th_l1 = 2.0 * vector_mad(errorC1);
    double inlier_th_l2 = 2.0 * vector_mad(errorC2);
    for(int i=0; i<vLineS3D.size(); i++)
    {
        if(errorC1[i]>inlier_th_l1 || errorC2[i]>inlier_th_l2)
        {
            vbLineTriangulated[i] = false;
        }
    }
}

} //namespace StructureSLAM
