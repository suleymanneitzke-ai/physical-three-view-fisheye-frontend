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

#include "LoopClosing.h"

#include "CorrectLoopHelpers.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include<mutex>
#include <sstream>
#include<thread>


namespace StructureSLAM
{

namespace
{
const char* kKb8LoopTriggeredGbaCapability =
    "KB8_LOOP_TRIGGERED_GBA_READY";
const char* kKb8LoopClosingRuntimeCapability =
    "KB8_LOOPCLOSING_RUNTIME_NOT_READY";
const char* kKb8RelocalizationCapability =
    "KB8_RELOCALIZATION_NOT_READY";

std::mutex gLoopClosingRuntimeDiagnosticsMutex;

long long NowMicros()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string CurrentThreadIdString()
{
    std::ostringstream os;
    os << std::this_thread::get_id();
    return os.str();
}

void AppendDiagnosticLine(const char* env_name, const std::string& line)
{
    const char* path = std::getenv(env_name);
    if(!path || !path[0])
        return;

    std::unique_lock<std::mutex> lock(gLoopClosingRuntimeDiagnosticsMutex);
    std::ofstream out(path, std::ios::app);
    if(out.is_open())
        out << line << '\n';
}

void LogLoopClosingThreadEvent(const std::string& event,
                               const KeyFrame* pKF,
                               const std::string& state)
{
    std::ostringstream os;
    os << NowMicros() << ','
       << CurrentThreadIdString() << ','
       << event << ','
       << (pKF ? static_cast<long long>(pKF->mnId) : -1) << ','
       << state;
    AppendDiagnosticLine("KB8_LOOPCLOSING_THREAD_EVENTS_CSV", os.str());
}

std::string CompactPoseString(const cv::Mat& Tcw)
{
    if(Tcw.empty())
        return "\"\"";

    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(6);
    os << '"';
    for(int r = 0; r < Tcw.rows; ++r)
    {
        for(int c = 0; c < Tcw.cols; ++c)
        {
            if(r != 0 || c != 0)
                os << ' ';
            os << Tcw.at<float>(r, c);
        }
    }
    os << '"';
    return os.str();
}

void LogNaturalKeyFrameGeneration(KeyFrame* pKF)
{
    if(!pKF)
        return;

    std::ostringstream os;
    os << NowMicros() << ','
       << static_cast<long long>(pKF->mnFrameId) << ','
       << static_cast<long long>(pKF->mnId) << ','
       << CompactPoseString(pKF->GetPose()) << ','
       << pKF->TrackedMapPoints(0);
    AppendDiagnosticLine("KB8_NATURAL_KF_GENERATION_CSV", os.str());
}

void LogNaturalLoopDetection(const KeyFrame* pCurrentKF,
                             const KeyFrame* pCandidateKF,
                             const float bow_score,
                             const int candidate_count,
                             const bool accepted)
{
    std::ostringstream os;
    os << (pCurrentKF ? static_cast<long long>(pCurrentKF->mnId) : -1) << ','
       << (pCandidateKF ? static_cast<long long>(pCandidateKF->mnId) : -1)
       << ','
       << bow_score << ','
       << candidate_count << ','
       << (accepted ? 1 : 0);
    AppendDiagnosticLine("KB8_NATURAL_LOOP_DETECTION_CSV", os.str());
}

void LogNaturalLoopCallChain(const std::string& event,
                             const std::string& result)
{
    std::ostringstream os;
    os << event << ','
       << NowMicros() << ','
       << CurrentThreadIdString() << ','
       << result;
    AppendDiagnosticLine("KB8_NATURAL_LOOP_CALL_CHAIN_CSV", os.str());
}

void LogCandidateInjectionEvent(const std::string& event,
                                const KeyFrame* pCurrentKF,
                                const KeyFrame* pMatchedKF,
                                const std::string& result)
{
    std::ostringstream os;
    os << NowMicros() << ','
       << CurrentThreadIdString() << ','
       << event << ','
       << (pCurrentKF ? static_cast<long long>(pCurrentKF->mnId) : -1) << ','
       << (pMatchedKF ? static_cast<long long>(pMatchedKF->mnId) : -1) << ','
       << result;
    AppendDiagnosticLine("KB8_LOOPCLOSING_CANDIDATE_INJECTION_CSV", os.str());
}

void LogComputeSim3Runtime(const KeyFrame* pCurrentKF,
                           const KeyFrame* pMatchedKF,
                           const int initial_matches,
                           const int map_points_current,
                           const int map_points_loop,
                           const bool sim3_solver_called,
                           const bool optimize_sim3_called,
                           const bool accepted,
                           const std::string& failure_reason)
{
    std::ostringstream os;
    os << (pCurrentKF ? static_cast<long long>(pCurrentKF->mnId) : -1) << ','
       << (pMatchedKF ? static_cast<long long>(pMatchedKF->mnId) : -1) << ','
       << initial_matches << ','
       << map_points_current << ','
       << map_points_loop << ','
       << (sim3_solver_called ? 1 : 0) << ','
       << (optimize_sim3_called ? 1 : 0) << ','
       << (accepted ? 1 : 0) << ','
       << failure_reason;
    AppendDiagnosticLine("KB8_LOOPCLOSING_COMPUTE_SIM3_CSV", os.str());
}

void LogCorrectLoopLocalMappingEvent(const std::string& event,
                                     LocalMapping* pLocalMapper,
                                     const bool map_write_allowed,
                                     const std::string& result)
{
    std::ostringstream os;
    os << event << ','
       << NowMicros() << ','
       << CurrentThreadIdString() << ','
       << (pLocalMapper && pLocalMapper->isStopped() ? "STOPPED" : "RUNNING")
       << ','
       << (map_write_allowed ? 1 : 0) << ','
       << result;
    AppendDiagnosticLine("KB8_CORRECTLOOP_LOCALMAPPING_CSV", os.str());
}

void LogCorrectLoopCallChain(const std::string& function,
                             const KeyFrame* pCurrentKF,
                             const KeyFrame* pMatchedKF,
                             const std::string& state_before,
                             const std::string& state_after,
                             const std::string& result)
{
    std::ostringstream os;
    os << function << ','
       << CurrentThreadIdString() << ','
       << "LoopClosing::Run" << ','
       << NowMicros() << ','
       << (pCurrentKF ? static_cast<long long>(pCurrentKF->mnId) : -1) << ','
       << (pMatchedKF ? static_cast<long long>(pMatchedKF->mnId) : -1) << ','
       << state_before << ','
       << state_after << ','
       << "KB8" << ','
       << result;
    AppendDiagnosticLine("KB8_CORRECTLOOP_CALL_CHAIN_CSV", os.str());
}

void LogCorrectLoopMutexStage(const std::string& stage,
                              const bool mutex_acquired,
                              const int write_count,
                              const bool commit,
                              const bool partial_commit)
{
    std::ostringstream os;
    os << stage << ','
       << (mutex_acquired ? 1 : 0) << ','
       << write_count << ','
       << (commit ? 1 : 0) << ','
       << (partial_commit ? 1 : 0);
    AppendDiagnosticLine("KB8_CORRECTLOOP_MUTEX_CSV", os.str());
}

void LogSearchAndFuseCall(const std::string& function,
                          const KeyFrame* pCurrentKF,
                          const KeyFrame* pTargetKF,
                          const std::string& direction,
                          const std::string& result)
{
    std::ostringstream os;
    os << NowMicros() << ','
       << CurrentThreadIdString() << ','
       << (pCurrentKF ? static_cast<long long>(pCurrentKF->mnId) : -1)
       << ','
       << (pTargetKF ? static_cast<long long>(pTargetKF->mnId) : -1)
       << ','
       << function << ','
       << direction << ','
       << result;
    AppendDiagnosticLine("KB8_SEARCHANDFUSE_CALL_CHAIN_CSV", os.str());
}

void LogSearchAndFuseThreadCoordination(const std::string& event,
                                        LocalMapping* pLocalMapper,
                                        const std::string& result)
{
    std::ostringstream os;
    os << event << ','
       << NowMicros() << ','
       << CurrentThreadIdString() << ','
       << (pLocalMapper && pLocalMapper->isStopped() ? "STOPPED" : "RUNNING")
       << ','
       << result;
    AppendDiagnosticLine("KB8_SEARCHANDFUSE_THREAD_COORDINATION_CSV", os.str());
}

void LogSearchAndFuseReplacement(const MapPoint* pOld,
                                 const MapPoint* pNew,
                                 const std::string& reason)
{
    std::ostringstream os;
    os << (pOld ? static_cast<long long>(pOld->mnId) : -1) << ','
       << (pNew ? static_cast<long long>(pNew->mnId) : -1) << ','
       << reason << ','
       << NowMicros();
    AppendDiagnosticLine("KB8_SEARCHANDFUSE_REPLACE_CHAIN_CSV", os.str());
}

void LogEssentialGraphThreadCoordination(const std::string& event,
                                         LocalMapping* pLocalMapper,
                                         const std::string& result)
{
    std::ostringstream os;
    os << event << ','
       << NowMicros() << ','
       << CurrentThreadIdString() << ','
       << (pLocalMapper && pLocalMapper->isStopped() ? "STOPPED" : "RUNNING")
       << ','
       << result;
    AppendDiagnosticLine("KB8_ESSENTIAL_GRAPH_THREAD_COORDINATION_CSV",
                         os.str());
}

void LogLoopGbaRuntimeEvent(const std::string& event,
                            Map* pMap,
                            const std::string& result)
{
    std::ostringstream os;
    os << NowMicros() << ','
       << CurrentThreadIdString() << ','
       << event << ','
       << (pMap ? static_cast<long long>(pMap->KeyFramesInMap()) : -1)
       << ','
       << (pMap ? static_cast<long long>(pMap->MapPointsInMap()) : -1)
       << ','
       << result;
    AppendDiagnosticLine("KB8_LOOP_GBA_RUNTIME_CSV", os.str());
}

void LogLoopGbaThreadLifecycle(const std::string& thread_name,
                               const std::string& event,
                               const std::string& state)
{
    std::ostringstream os;
    os << thread_name << ','
       << event << ','
       << NowMicros() << ','
       << state;
    AppendDiagnosticLine("KB8_LOOP_GBA_THREAD_LIFECYCLE_CSV", os.str());
}
}

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
    mpInjectedCurrentKFForTesting = NULL;
    mpInjectedMatchedKFForTesting = NULL;
    mbStopAfterComputeSim3ForTesting = false;
    mbStopAfterCorrectLoopMapCorrectionForTesting = false;
    mbStopAfterSearchAndFuseForTesting = false;
    mbStopAfterEssentialGraphForTesting = false;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;
    LogLoopClosingThreadEvent("LOOP_THREAD_START", NULL, "RUN_ENTER");

    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            if(DetectLoop())
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               if(ComputeSim3())
               {
                   {
                       unique_lock<mutex> lock(mMutexLoopCandidateInjectionForTesting);
                       if(mbStopAfterComputeSim3ForTesting)
                       {
                           LogLoopClosingThreadEvent(
                               "STOP_AFTER_COMPUTE_SIM3",
                               mpCurrentKF,
                               "CORRECTLOOP_SKIPPED_FOR_TEST");
                           RequestFinish();
                           continue;
                       }
                   }

                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();
               }
            }
        }       

        ResetIfRequested();

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
    LogLoopClosingThreadEvent("LOOP_THREAD_FINISH", NULL, "RUN_EXIT");
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    {
        unique_lock<mutex> lock(mMutexFinish);
        if(mbFinishRequested || mbFinished)
        {
            LogLoopClosingThreadEvent(
                "INSERT_KEYFRAME_REJECTED_AFTER_FINISH", pKF,
                "FINISH_REQUESTED");
            return;
        }
    }

    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF && pKF->mnId!=0)
    {
        mlpLoopKeyFrameQueue.push_back(pKF);
        LogLoopClosingThreadEvent("INSERT_KEYFRAME", pKF, "QUEUED");
        LogNaturalKeyFrameGeneration(pKF);
    }
}

void LoopClosing::InjectLoopCandidateForTesting(KeyFrame* pCurrentKF,
                                                KeyFrame* pMatchedKF)
{
    unique_lock<mutex> lock(mMutexLoopCandidateInjectionForTesting);
    mpInjectedCurrentKFForTesting = pCurrentKF;
    mpInjectedMatchedKFForTesting = pMatchedKF;
    LogCandidateInjectionEvent(
        "INJECT_LOOP_CANDIDATE", pCurrentKF, pMatchedKF,
        (pCurrentKF && pMatchedKF) ? "READY" : "INVALID");
}

void LoopClosing::RequestStopAfterComputeSim3ForTesting()
{
    unique_lock<mutex> lock(mMutexLoopCandidateInjectionForTesting);
    mbStopAfterComputeSim3ForTesting = true;
}

void LoopClosing::RequestStopAfterCorrectLoopMapCorrectionForTesting()
{
    unique_lock<mutex> lock(mMutexLoopCandidateInjectionForTesting);
    mbStopAfterCorrectLoopMapCorrectionForTesting = true;
}

void LoopClosing::RequestStopAfterSearchAndFuseForTesting()
{
    unique_lock<mutex> lock(mMutexLoopCandidateInjectionForTesting);
    mbStopAfterSearchAndFuseForTesting = true;
}

void LoopClosing::RequestStopAfterEssentialGraphForTesting()
{
    unique_lock<mutex> lock(mMutexLoopCandidateInjectionForTesting);
    mbStopAfterEssentialGraphForTesting = true;
}

bool LoopClosing::JoinGBAThread()
{
    thread* pThread = NULL;
    {
        unique_lock<mutex> lock(mMutexGBA);
        pThread = mpThreadGBA;
        mpThreadGBA = NULL;
    }

    if(!pThread)
        return false;

    LogLoopGbaThreadLifecycle("GBA", "JOIN_BEGIN", "JOINABLE");
    if(pThread->joinable())
        pThread->join();
    LogLoopGbaThreadLifecycle("GBA", "JOIN_END", "JOINED");
    delete pThread;
    return true;
}

bool LoopClosing::JoinGBAForTesting()
{
    return JoinGBAThread();
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosing::DetectLoop()
{
    LogNaturalLoopCallChain("DETECT_LOOP_CALL", "BEGIN");
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
        LogLoopClosingThreadEvent("POP_KEYFRAME", mpCurrentKF, "DEQUEUED");
    }

    {
        unique_lock<mutex> lock(mMutexLoopCandidateInjectionForTesting);
        if(mpInjectedMatchedKFForTesting &&
           (!mpInjectedCurrentKFForTesting ||
            mpInjectedCurrentKFForTesting == mpCurrentKF))
        {
            mvpEnoughConsistentCandidates.clear();
            mvpEnoughConsistentCandidates.push_back(
                mpInjectedMatchedKFForTesting);
            LogLoopClosingThreadEvent(
                "CANDIDATE_AVAILABLE", mpCurrentKF, "INJECTED");
            LogCandidateInjectionEvent(
                "CANDIDATE_CONSUMED", mpCurrentKF,
                mpInjectedMatchedKFForTesting, "DETECTLOOP_TRUE");
            mpInjectedCurrentKFForTesting = NULL;
            mpInjectedMatchedKFForTesting = NULL;
            return true;
        }
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        LogNaturalLoopDetection(mpCurrentKF, NULL, 0.0f, 0, false);
        LogNaturalLoopCallChain("DETECT_LOOP_FAIL", "TOO_EARLY");
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        LogNaturalLoopDetection(mpCurrentKF, NULL, minScore, 0, false);
        LogNaturalLoopCallChain("DETECT_LOOP_FAIL", "NO_DATABASE_CANDIDATE");
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit))
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            if(bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    for(size_t i = 0, iend = vpCandidateKFs.size(); i < iend; ++i)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];
        const bool accepted =
            std::find(mvpEnoughConsistentCandidates.begin(),
                      mvpEnoughConsistentCandidates.end(),
                      pCandidateKF) != mvpEnoughConsistentCandidates.end();
        LogNaturalLoopDetection(mpCurrentKF,
                                pCandidateKF,
                                pCandidateKF ? pCandidateKF->mLoopScore : 0.0f,
                                static_cast<int>(vpCandidateKFs.size()),
                                accepted);
    }

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        LogNaturalLoopCallChain("DETECT_LOOP_FAIL", "CONSISTENCY_REJECTED");
        return false;
    }
    else
    {
        LogLoopClosingThreadEvent(
            "CANDIDATE_AVAILABLE", mpCurrentKF, "DATABASE");
        LogNaturalLoopCallChain("DETECT_LOOP_SUCCESS", "DATABASE");
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}

bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3
    cout<<"loop computeSim3"<<endl;
    LogLoopClosingThreadEvent("COMPUTE_SIM3_BEGIN", mpCurrentKF, "ENTER");
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();
    int best_initial_matches = 0;
    bool sim3_solver_called = false;
    bool optimize_sim3_called = false;
    KeyFrame* pBestCandidateForDiagnostics =
        nInitialCandidates > 0 ? mvpEnoughConsistentCandidates[0] : NULL;

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);
    LSDmatcher lmatcher;
    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<vector<MapLine*>> vvpMaoLineMatches;
    vvpMaoLineMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);
        if(nmatches > best_initial_matches)
            best_initial_matches = nmatches;

        int nlmatches=0;
        if(nmatches<20&&nlmatches<6)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
            sim3_solver_called = true;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty())
            {
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                optimize_sim3_called = true;
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)
                {
                    bMatch = true;
                    mpMatchedKF = pKF;
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        LogComputeSim3Runtime(
            mpCurrentKF, pBestCandidateForDiagnostics, best_initial_matches,
            mpCurrentKF ? static_cast<int>(mpCurrentKF->GetMapPointMatches().size()) : 0,
            pBestCandidateForDiagnostics ? static_cast<int>(pBestCandidateForDiagnostics->GetMapPointMatches().size()) : 0,
            sim3_solver_called, optimize_sim3_called, false,
            sim3_solver_called ? "RANSAC_OR_OPTIMIZE_FAILED" :
                                 "INSUFFICIENT_INITIAL_MATCHES");
        LogLoopClosingThreadEvent("COMPUTE_SIM3_END", mpCurrentKF, "FAILED");
        LogNaturalLoopCallChain("COMPUTE_SIM3_FAIL",
                                sim3_solver_called ? "RANSAC_OR_OPTIMIZE_FAILED" :
                                                     "INSUFFICIENT_INITIAL_MATCHES");
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        LogComputeSim3Runtime(
            mpCurrentKF, mpMatchedKF, best_initial_matches,
            mpCurrentKF ? static_cast<int>(mpCurrentKF->GetMapPointMatches().size()) : 0,
            static_cast<int>(mvpLoopMapPoints.size()),
            sim3_solver_called, optimize_sim3_called, true, "ACCEPTED");
        LogLoopClosingThreadEvent("COMPUTE_SIM3_END", mpCurrentKF, "ACCEPTED");
        LogNaturalLoopCallChain("COMPUTE_SIM3_SUCCESS", "ACCEPTED");
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        LogComputeSim3Runtime(
            mpCurrentKF, mpMatchedKF, best_initial_matches,
            mpCurrentKF ? static_cast<int>(mpCurrentKF->GetMapPointMatches().size()) : 0,
            static_cast<int>(mvpLoopMapPoints.size()),
            sim3_solver_called, optimize_sim3_called, false,
            "FINAL_PROJECTION_MATCHES_BELOW_THRESHOLD");
        LogLoopClosingThreadEvent("COMPUTE_SIM3_END", mpCurrentKF, "FAILED");
        LogNaturalLoopCallChain("COMPUTE_SIM3_FAIL",
                                "FINAL_PROJECTION_MATCHES_BELOW_THRESHOLD");
        return false;
    }

}

void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;
    LogCorrectLoopCallChain(
        "LoopClosing::CorrectLoop", mpCurrentKF, mpMatchedKF,
        "COMPUTE_SIM3_ACCEPTED", "CORRECTLOOP_BEGIN", "ENTER");
    LogNaturalLoopCallChain("CORRECT_LOOP_BEGIN", "ENTER");

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    LogCorrectLoopLocalMappingEvent(
        "REQUEST_STOP", mpLocalMapper, false, "BEGIN");
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }
    LogCorrectLoopLocalMappingEvent(
        "STOP_CONFIRMED", mpLocalMapper, false, "PASS");
    LogCorrectLoopLocalMappingEvent(
        "CORRECTLOOP_BEGIN", mpLocalMapper, false, "PASS");

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();

    int corrected_keyframe_writes = 0;
    int corrected_mappoint_writes = 0;
    int replacement_or_observation_writes = 0;

    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        LogCorrectLoopMutexStage(
            "MAP_CORRECTION_BEGIN", true, 0, false, false);

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();

            if(pKFi!=mpCurrentKF)
            {
                g2o::Sim3 g2oCorrectedSiw =
                    CorrectLoopHelpers::BuildCorrectedConnectedSim3(
                        Tiw, Twc, mg2oScw);
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw =
                    CorrectLoopHelpers::CorrectMapPointWorldPosition(
                        g2oSiw, g2oCorrectedSiw, eigP3Dw);

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
                corrected_mappoint_writes++;
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            cv::Mat correctedTiw =
                CorrectLoopHelpers::Sim3ToSE3Tcw(g2oCorrectedSiw);

            pKFi->SetPose(correctedTiw);
            corrected_keyframe_writes++;

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                {
                    pCurMP->Replace(pLoopMP);
                    replacement_or_observation_writes++;
                }
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                    replacement_or_observation_writes++;
                }
            }
        }

        LogCorrectLoopMutexStage(
            "KEYFRAME_POSE_UPDATE", true, corrected_keyframe_writes,
            true, false);
        LogCorrectLoopMutexStage(
            "MAPPOINT_UPDATE", true, corrected_mappoint_writes,
            true, false);
        LogCorrectLoopMutexStage(
            "MATCHED_MAPPOINT_REPLACEMENT_OR_OBSERVATION", true,
            replacement_or_observation_writes, true, false);
        LogCorrectLoopMutexStage(
            "MAP_CORRECTION_END", true,
            corrected_keyframe_writes + corrected_mappoint_writes +
                replacement_or_observation_writes,
            true, false);

    }

    {
        unique_lock<mutex> lock(mMutexLoopCandidateInjectionForTesting);
        if(mbStopAfterCorrectLoopMapCorrectionForTesting)
        {
            LogCorrectLoopLocalMappingEvent(
                "CORRECTLOOP_END", mpLocalMapper, false,
                "MAP_CORRECTION_ONLY_FOR_TEST");
            mpLocalMapper->Release();
            LogCorrectLoopLocalMappingEvent(
                "RELEASE", mpLocalMapper, true, "PASS");
            LogCorrectLoopCallChain(
                "LoopClosing::CorrectLoop", mpCurrentKF, mpMatchedKF,
                "MAP_CORRECTION_DONE", "SEARCHANDFUSE_SKIPPED_FOR_TEST",
                "PASS");
            RequestFinish();
            return;
        }
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);
    LogNaturalLoopCallChain("SEARCH_AND_FUSE_SUCCESS", "PASS");

    {
        unique_lock<mutex> lock(mMutexLoopCandidateInjectionForTesting);
        if(mbStopAfterSearchAndFuseForTesting)
        {
            LogSearchAndFuseThreadCoordination(
                "SEARCHANDFUSE_END", mpLocalMapper,
                "ESSENTIAL_GRAPH_SKIPPED_FOR_TEST");
            mpLocalMapper->Release();
            LogSearchAndFuseThreadCoordination(
                "RELEASE", mpLocalMapper, "PASS");
            LogCorrectLoopCallChain(
                "LoopClosing::CorrectLoop", mpCurrentKF, mpMatchedKF,
                "SEARCHANDFUSE_DONE", "ESSENTIAL_GRAPH_SKIPPED_FOR_TEST",
                "PASS");
            RequestFinish();
            return;
        }
    }


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi] =
            CorrectLoopHelpers::BuildLoopConnectionDifference(
                pKFi->GetConnectedKeyFrames(),
                vpPreviousNeighbors,
                mvpCurrentConnectedKFs);
    }

    // Optimize graph
    LogEssentialGraphThreadCoordination(
        "OPTIMIZE_ESSENTIAL_GRAPH_BEGIN", mpLocalMapper, "BEGIN");
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);
    LogEssentialGraphThreadCoordination(
        "OPTIMIZE_ESSENTIAL_GRAPH_END", mpLocalMapper, "PASS");
    LogNaturalLoopCallChain("ESSENTIAL_GRAPH_SUCCESS", "PASS");

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);
    LogEssentialGraphThreadCoordination(
        "LOOP_EDGE_ADDED", mpLocalMapper, "PASS");

    {
        unique_lock<mutex> lock(mMutexLoopCandidateInjectionForTesting);
        if(mbStopAfterEssentialGraphForTesting)
        {
            LogEssentialGraphThreadCoordination(
                "LOOP_EDGE_ADDED", mpLocalMapper, "GBA_SKIPPED_FOR_TEST");
            mpLocalMapper->Release();
            LogEssentialGraphThreadCoordination(
                "RELEASE", mpLocalMapper, "PASS");
            LogCorrectLoopCallChain(
                "LoopClosing::CorrectLoop", mpCurrentKF, mpMatchedKF,
                "ESSENTIAL_GRAPH_DONE", "GBA_SKIPPED_FOR_TEST", "PASS");
            RequestFinish();
            return;
        }
    }

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    LogLoopGbaRuntimeEvent("LOOP_GBA_BEGIN", mpMap, "CREATE_GBA_THREAD");
    LogLoopGbaThreadLifecycle("GBA", "CREATE_GBA_THREAD", "STARTING");
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);
    LogLoopGbaThreadLifecycle("GBA", "THREAD_CREATED", "RUNNING");
    LogNaturalLoopCallChain("LOOP_GBA_TRIGGERED", "THREAD_CREATED");

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId;   
    LogNaturalLoopCallChain("CORRECT_LOOP_SUCCESS", "PASS");
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    LogSearchAndFuseCall(
        "LoopClosing::SearchAndFuse", mpCurrentKF, NULL,
        "all_corrected_connected_keyframes", "BEGIN");
    LogSearchAndFuseThreadCoordination(
        "SEARCHANDFUSE_BEGIN", mpLocalMapper, "BEGIN");
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        LogSearchAndFuseCall(
            "ORBmatcher::Fuse", mpCurrentKF, pKF,
            "loop_mappoints_to_corrected_keyframe", "BEGIN");
        const int fused = matcher.Fuse(
            pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);
        LogSearchAndFuseCall(
            "ORBmatcher::Fuse", mpCurrentKF, pKF,
            "loop_mappoints_to_corrected_keyframe",
            fused > 0 ? "FUSED_OR_MATCHED" : "NO_MATCH");

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                LogSearchAndFuseReplacement(
                    pRep, mvpLoopMapPoints[i],
                    "SEARCHANDFUSE_FUSE_SCW_REPLACEMENT");
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
    LogSearchAndFuseCall(
        "LoopClosing::SearchAndFuse", mpCurrentKF, NULL,
        "all_corrected_connected_keyframes", "END");
}


void LoopClosing::RequestReset()
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
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;
    LogLoopGbaRuntimeEvent("GLOBAL_BA_START", mpMap, "BEGIN");
    LogLoopGbaThreadLifecycle("GBA", "RUN_BEGIN", "RUNNING");

    int idx =  mnFullBAIdx;
//    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);   //原本只要点特征的BA
    LogLoopGbaRuntimeEvent("OPTIMIZER_RUNNING", mpMap, "CALL_GLOBAL_BA");
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,false,&mbStopGBA,nLoopKF,false);  //包含线特征的BA
    LogLoopGbaRuntimeEvent("GLOBAL_BA_FINISH", mpMap, "OPTIMIZER_RETURNED");
    LogNaturalLoopCallChain("LOOP_GBA_SUCCESS", "OPTIMIZER_RETURNED");

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
            LogLoopGbaRuntimeEvent("GLOBAL_BA_COMMIT_BEGIN", mpMap, "MUTEX_HELD");

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        pChild->mTcwGBA =
                            CorrectLoopHelpers::PropagateGbaChildPose(
                                pChild->GetPose(), Twc, pKF->mTcwGBA);
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    pMP->SetWorldPos(
                        CorrectLoopHelpers::CorrectGbaIndirectMapPointPosition(
                            pMP->GetWorldPos(),
                            pRefKF->mTcwBefGBA,
                            pRefKF->GetPoseInverse()));
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
            LogLoopGbaRuntimeEvent("GLOBAL_BA_COMMIT_END", mpMap, "PASS");
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
        LogLoopGbaThreadLifecycle("GBA", "FLAGS_CLEARED", "FINISHED");
    }
    LogLoopGbaThreadLifecycle("GBA", "RUN_END", "FINISHED");
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
