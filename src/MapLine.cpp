/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/
#include "MapLine.h"
#include <mutex>
#include <include/LSDmatcher.h>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace StructureSLAM
{
    mutex MapLine::mGlobalMutex;
    long unsigned int MapLine::nNextId=0;

    namespace
    {
        struct ProvenanceRecord
        {
            unsigned long id = 0;
            int creation_origin = MapLine::ORIGIN_UNKNOWN;
            long creation_frame = -1;
            long creation_keyframe = -1;
            double creation_timestamp = -1.0;
            int physical_touch_count = 0;
            int legacy_touch_count = 0;
            int physical_association_count = 0;
            int legacy_fusion_count = 0;
            int replacement_count = 0;
            long replaced_by_id = -1;
            bool final_alive = true;
            int optimizer_use_count = 0;
            int residual_created_count = 0;
            int residual_accepted_count = 0;
            int observation_count = 0;
            long last_frame = -1;
        };

        std::mutex gProvenanceMutex;
        std::ofstream gEventLog;
        bool gLogInitialized = false;
        bool gFinalWritten = false;
        unsigned long gEventIndex = 0;
        std::string gProvenanceDir;
        std::map<unsigned long, ProvenanceRecord> gRecords;

        std::string EscapeCsv(const std::string& value)
        {
            bool needs_quotes = false;
            for(size_t i = 0; i < value.size(); ++i)
            {
                if(value[i] == ',' || value[i] == '"' ||
                   value[i] == '\n' || value[i] == '\r')
                {
                    needs_quotes = true;
                    break;
                }
            }
            if(!needs_quotes)
                return value;
            std::string escaped = "\"";
            for(size_t i = 0; i < value.size(); ++i)
            {
                if(value[i] == '"')
                    escaped += "\"\"";
                else
                    escaped += value[i];
            }
            escaped += "\"";
            return escaped;
        }

        const char* OriginNameFromInt(const int origin)
        {
            switch(origin)
            {
                case MapLine::ORIGIN_PHYSICAL:
                    return "PHYSICAL";
                case MapLine::ORIGIN_LEGACY:
                    return "LEGACY";
                default:
                    return "UNKNOWN";
            }
        }

        void WriteFinalSummaryLocked();

        void EnsureProvenanceLogLocked()
        {
            if(gLogInitialized)
                return;
            gLogInitialized = true;
            const char* dir = std::getenv("T265_MAPLINE_PROVENANCE_DIR");
            if(!dir || std::string(dir).empty())
                return;
            gProvenanceDir = dir;
            gEventLog.open(gProvenanceDir + "/mapline_provenance_events.csv",
                           std::ios::out | std::ios::trunc);
            if(!gEventLog.is_open())
                return;
            gEventLog << "event_index,timestamp,frame_id,keyframe_id,event_type,"
                         "mapline_id,secondary_mapline_id,origin,secondary_origin,"
                         "physical_obs_id,result,reason\n";
            std::atexit([]() {
                std::lock_guard<std::mutex> lock(gProvenanceMutex);
                WriteFinalSummaryLocked();
                if(gEventLog.is_open())
                    gEventLog.close();
            });
        }

        void SyncRecordLocked(const MapLine* pML)
        {
            if(!pML)
                return;
            ProvenanceRecord& rec = gRecords[pML->mnId];
            rec.id = pML->mnId;
            rec.creation_origin = pML->creation_origin;
            rec.creation_frame = pML->creation_frame;
            rec.creation_keyframe = pML->creation_keyframe;
            rec.creation_timestamp = pML->creation_timestamp;
            rec.physical_touch_count = pML->physical_touch_count;
            rec.legacy_touch_count = pML->legacy_touch_count;
            rec.physical_association_count = pML->physical_association_count;
            rec.legacy_fusion_count = pML->legacy_fusion_count;
            rec.replacement_count = pML->replacement_count;
            rec.replaced_by_id = pML->replaced_by_id;
            rec.final_alive = !pML->mbBad;
            rec.optimizer_use_count = pML->optimizer_use_count;
            rec.residual_created_count = pML->residual_created_count;
            rec.residual_accepted_count = pML->residual_accepted_count;
            rec.observation_count = pML->nObs;
            rec.last_frame = pML->provenance_last_frame_seen;
        }

        void LogEventLocked(const std::string& event_type,
                            const MapLine* pML,
                            const MapLine* secondary,
                            const long frame_id,
                            const long keyframe_id,
                            const long physical_obs_id,
                            const std::string& result,
                            const std::string& reason)
        {
            EnsureProvenanceLogLocked();
            if(pML)
                SyncRecordLocked(pML);
            if(secondary)
                SyncRecordLocked(secondary);
            if(!gEventLog.is_open())
                return;
            const long primary_id = pML ? static_cast<long>(pML->mnId) : -1;
            const long secondary_id =
                secondary ? static_cast<long>(secondary->mnId) : -1;
            const int origin =
                pML ? static_cast<int>(pML->creation_origin)
                    : MapLine::ORIGIN_UNKNOWN;
            const int secondary_origin =
                secondary ? static_cast<int>(secondary->creation_origin)
                          : MapLine::ORIGIN_UNKNOWN;
            const double timestamp =
                pML ? pML->creation_timestamp
                    : (secondary ? secondary->creation_timestamp : -1.0);
            gEventLog << gEventIndex++ << ','
                      << timestamp << ','
                      << frame_id << ','
                      << keyframe_id << ','
                      << EscapeCsv(event_type) << ','
                      << primary_id << ','
                      << secondary_id << ','
                      << OriginNameFromInt(origin) << ','
                      << OriginNameFromInt(secondary_origin) << ','
                      << physical_obs_id << ','
                      << EscapeCsv(result) << ','
                      << EscapeCsv(reason) << '\n';
        }

        std::string ClassForRecord(const ProvenanceRecord& rec)
        {
            const bool cross_replaced =
                rec.replaced_by_id >= 0 &&
                gRecords.count(static_cast<unsigned long>(rec.replaced_by_id)) &&
                gRecords[static_cast<unsigned long>(rec.replaced_by_id)]
                        .creation_origin != rec.creation_origin;
            if(cross_replaced)
                return "MIXED_REPLACEMENT";
            if(rec.creation_origin == MapLine::ORIGIN_PHYSICAL)
            {
                if(rec.legacy_touch_count > 0)
                    return "PHYSICAL_CREATED_LEGACY_TOUCHED";
                return "PHYSICAL_ONLY";
            }
            if(rec.creation_origin == MapLine::ORIGIN_LEGACY)
            {
                if(rec.physical_association_count > 0 ||
                   rec.optimizer_use_count > 0)
                    return "LEGACY_CREATED_PHYSICAL_USED";
                return "LEGACY_ONLY";
            }
            return "UNKNOWN";
        }

        void WriteFinalSummaryLocked()
        {
            EnsureProvenanceLogLocked();
            if(gFinalWritten || gProvenanceDir.empty())
                return;
            gFinalWritten = true;
            std::ofstream final_log(
                gProvenanceDir + "/mapline_provenance_final.csv",
                std::ios::out | std::ios::trunc);
            if(!final_log.is_open())
                return;
            final_log << "mapline_id,creation_origin,creation_frame,"
                         "creation_keyframe,physical_touch_count,"
                         "legacy_touch_count,physical_association_count,"
                         "legacy_fusion_count,replacement_count,replaced_by,"
                         "final_alive,optimizer_use_count,"
                         "residual_created_count,residual_accepted_count,"
                         "observation_count,lifetime,provenance_class\n";
            for(std::map<unsigned long, ProvenanceRecord>::const_iterator it =
                    gRecords.begin(); it != gRecords.end(); ++it)
            {
                const ProvenanceRecord& rec = it->second;
                const long lifetime =
                    rec.last_frame >= 0 && rec.creation_frame >= 0
                        ? rec.last_frame - rec.creation_frame
                        : -1;
                final_log << rec.id << ','
                          << OriginNameFromInt(rec.creation_origin) << ','
                          << rec.creation_frame << ','
                          << rec.creation_keyframe << ','
                          << rec.physical_touch_count << ','
                          << rec.legacy_touch_count << ','
                          << rec.physical_association_count << ','
                          << rec.legacy_fusion_count << ','
                          << rec.replacement_count << ','
                          << rec.replaced_by_id << ','
                          << (rec.final_alive ? 1 : 0) << ','
                          << rec.optimizer_use_count << ','
                          << rec.residual_created_count << ','
                          << rec.residual_accepted_count << ','
                          << rec.observation_count << ','
                          << lifetime << ','
                          << ClassForRecord(rec) << '\n';
            }
        }
    }

    MapLine::MapLine(Vector6d &Pos, KeyFrame *pRefKF, Map *pMap):
            mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0),
            creation_origin(ORIGIN_UNKNOWN), creation_frame(pRefKF ? static_cast<long>(pRefKF->mnFrameId) : -1),
            creation_keyframe(pRefKF ? static_cast<long>(pRefKF->mnId) : -1),
            creation_timestamp(pRefKF ? pRefKF->mTimeStamp : -1.0),
            physical_touch_count(0), legacy_touch_count(0), physical_association_count(0),
            legacy_fusion_count(0), replacement_count(0), optimizer_use_count(0),
            residual_created_count(0), residual_accepted_count(0), replaced_by_id(-1),
            provenance_last_frame_seen(pRefKF ? static_cast<long>(pRefKF->mnFrameId) : -1),
            mnTrackReferenceForFrame(0),
            mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0), mnCorrectedByKF(0),
            mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
            mpReplaced(static_cast<MapLine*>(NULL)), mpMap(pMap)
    {
        mWorldPos = Pos;

        mNormalVector << 0, 0, 0;

        // MapLines can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
        {
            std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
            SyncRecordLocked(this);
        }
    }

    MapLine::MapLine(Vector6d &Pos, Map *pMap, Frame *pFrame, const int &idxF):
            mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0),
            creation_origin(ORIGIN_UNKNOWN), creation_frame(pFrame ? static_cast<long>(pFrame->mnId) : -1),
            creation_keyframe(-1),
            creation_timestamp(pFrame ? pFrame->mTimeStamp : -1.0),
            physical_touch_count(0), legacy_touch_count(0), physical_association_count(0),
            legacy_fusion_count(0), replacement_count(0), optimizer_use_count(0),
            residual_created_count(0), residual_accepted_count(0), replaced_by_id(-1),
            provenance_last_frame_seen(pFrame ? static_cast<long>(pFrame->mnId) : -1),
            mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
            mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0), mnCorrectedByKF(0),
            mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
            mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
    {
        mWorldPos = Pos;
        Mat Ow = pFrame->GetCameraCenter();
        Vector3d OW;
        OW << Ow.at<double>(0), Ow.at<double>(1), Ow.at<double>(2);
        mStart3D = Pos.head(3);
        mEnd3D = Pos.tail(3);
        Vector3d midPoint = 0.5*(mStart3D + mEnd3D);
        mNormalVector = midPoint - OW;
//    mNormalVector = mNormalVector/mNormalVector.norm();  //世界坐标系下，相机到3D点的单位向量
        mNormalVector.normalize();

        Vector3d PC = midPoint - OW;
        const float dist = PC.norm();   //相机光心到线段中点的距离

        const int level = pFrame->mvKeylinesUn[idxF].octave;
        const float levelScaleFactor = pFrame->mvScaleFactors[level];
        const int nLevels = pFrame->mnScaleLevels;

        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

        pFrame->mLdesc.row(idxF).copyTo(mLDescriptor);

        // MapLines can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexLineCreation);
        mnId = nNextId++;
        {
            std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
            SyncRecordLocked(this);
        }
    }

    MapLine::MapLine(const Vector6d& Pos, const long unsigned int replay_id,
                     const float min_distance, const float max_distance,
                     const cv::Mat& descriptor,
                     const std::vector<long unsigned int>& temporal_track_ids):
            mnId(replay_id), mnFirstKFid(-1), mnFirstFrame(-1), nObs(0),
            creation_origin(ORIGIN_UNKNOWN), creation_frame(-1), creation_keyframe(-1),
            creation_timestamp(-1.0), physical_touch_count(0), legacy_touch_count(0),
            physical_association_count(0), legacy_fusion_count(0), replacement_count(0),
            optimizer_use_count(0), residual_created_count(0), residual_accepted_count(0),
            replaced_by_id(-1), provenance_last_frame_seen(-1),
            mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0),
            mnFuseCandidateForKF(0), mnLoopLineForKF(0), mnCorrectedByKF(0),
            mnCorrectedReference(0), mnBAGlobalForKF(0),
            mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1), mnFound(1),
            mbBad(false), mpReplaced(static_cast<MapLine*>(NULL)), mpMap(NULL)
    {
        mWorldPos = Pos;
        mStart3D = Pos.head(3);
        mEnd3D = Pos.tail(3);
        mNormalVector.setZero();
        mfMinDistance = min_distance;
        mfMaxDistance = max_distance;
        mLDescriptor = descriptor.clone();
        for(size_t i = 0; i < temporal_track_ids.size(); ++i)
        {
            KeyFrame* token = reinterpret_cast<KeyFrame*>(i + 1);
            FisheyeLineObservation observation;
            observation.temporal_track_id = temporal_track_ids[i];
            observation.mapline_id = replay_id;
            mNativeFisheyeObservations[token] = i;
            mNativeFisheyeObservationData[token] = observation;
        }
        nObs = static_cast<int>(temporal_track_ids.size());
        {
            std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
            SyncRecordLocked(this);
        }
    }


    void MapLine::SetWorldPos(const Vector6d &Pos)
    {
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        mWorldPos = Pos;
    }

    Vector6d MapLine::GetWorldPos()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos;
    }

    Vector3d MapLine::GetNormal()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mNormalVector;
    }

    KeyFrame* MapLine::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    void MapLine::AddObservation(KeyFrame *pKF, size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return;

        mObservations[pKF]=idx;
        if(pKF)
            provenance_last_frame_seen =
                static_cast<long>(pKF->mnFrameId);

        nObs++;
        {
            std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
            SyncRecordLocked(this);
        }
    }

    void MapLine::AddNativeFisheyeObservation(
        KeyFrame *pKF, size_t idx,
        const FisheyeLineObservation& observation)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mNativeFisheyeObservations.count(pKF))
            return;

        FisheyeLineObservation stored = observation;
        stored.keyframe_id = pKF ? pKF->mnId : stored.keyframe_id;
        stored.mapline_id = mnId;
        mNativeFisheyeObservations[pKF] = idx;
        mNativeFisheyeObservationData[pKF] = stored;
        if(pKF)
            provenance_last_frame_seen =
                static_cast<long>(pKF->mnFrameId);
        nObs++;
        {
            std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
            SyncRecordLocked(this);
        }
    }

    void MapLine::EraseNativeFisheyeObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        size_t idx = 0;
        bool had_observation = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if(mNativeFisheyeObservations.count(pKF))
            {
                idx = mNativeFisheyeObservations[pKF];
                had_observation = true;
                mNativeFisheyeObservations.erase(pKF);
                mNativeFisheyeObservationData.erase(pKF);
                nObs--;
                if(nObs<=2)
                    bBad=true;
            }
        }

        if(had_observation && pKF)
            pKF->EraseNativeFisheyeMapLineMatch(idx);
        if(had_observation)
        {
            std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
            SyncRecordLocked(this);
        }

        if(bBad)
            SetBadFlag();
    }

    void MapLine::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if(mObservations.count(pKF))
            {
                mObservations.erase(pKF);
                nObs--;

                if(mpRefKF==pKF)
                    mpRefKF=mObservations.begin()->first;

                if(nObs<=2)
                    bBad=true;
            }
        }

        {
            std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
            SyncRecordLocked(this);
        }

        if(bBad)
            SetBadFlag();
    }

    map<KeyFrame*, size_t> MapLine::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    map<KeyFrame*, size_t> MapLine::GetNativeFisheyeObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mNativeFisheyeObservations;
    }

    FisheyeLineObservation MapLine::GetNativeFisheyeObservation(
        KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mNativeFisheyeObservationData.count(pKF))
            return mNativeFisheyeObservationData[pKF];
        return FisheyeLineObservation();
    }

    int MapLine::NativeFisheyeObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<int>(mNativeFisheyeObservations.size());
    }

    int MapLine::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    int MapLine::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    bool MapLine::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }

    void MapLine::SetBadFlag()
    {
        map<KeyFrame*, size_t> obs;
        map<KeyFrame*, size_t> native_obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad=true;
            obs = mObservations;
            native_obs = mNativeFisheyeObservations;
            mObservations.clear();
            mNativeFisheyeObservations.clear();
            mNativeFisheyeObservationData.clear();
        }
        {
            std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
            SyncRecordLocked(this);
            gRecords[mnId].final_alive = false;
            LogEventLocked("SET_BAD", this, NULL, -1, -1, -1, "BAD", "");
        }

        for(auto & ob : obs)
        {
            KeyFrame* pKF = ob.first;
            pKF->EraseMapLineMatch(ob.second);
        }
        for(auto & ob : native_obs)
        {
            KeyFrame* pKF = ob.first;
            if(pKF)
                pKF->EraseNativeFisheyeMapLineMatch(ob.second);
        }

        mpMap->EraseMapLine(this);
    }

    bool MapLine::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }

    void MapLine::Replace(MapLine *pML)
    {
        if(pML->mnId==this->mnId)
            return;

        int nvisible, nfound;
        map<KeyFrame*, size_t> obs;
        map<KeyFrame*, size_t> native_obs;
        map<KeyFrame*, FisheyeLineObservation> native_data;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            obs=mObservations;
            native_obs=mNativeFisheyeObservations;
            native_data=mNativeFisheyeObservationData;
            mObservations.clear();
            mNativeFisheyeObservations.clear();
            mNativeFisheyeObservationData.clear();
            mbBad=true;
            nvisible=mnVisible;
            nfound = mnFound;
            mpReplaced = pML;
            replaced_by_id = static_cast<long>(pML->mnId);
            replacement_count++;
            if(pML)
                pML->replacement_count++;
        }
        {
            std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
            SyncRecordLocked(this);
            SyncRecordLocked(pML);
            gRecords[mnId].final_alive = false;
            gRecords[mnId].replaced_by_id = static_cast<long>(pML->mnId);
            LogEventLocked("REPLACE", this, pML, -1, -1, -1, "SUCCESS", "");
        }

        for(map<KeyFrame*, size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;

            if(!pML->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapLineMatch(mit->second, pML);
                pML->AddObservation(pKF, mit->second);
            } else{
                pKF->EraseMapLineMatch(mit->second);
            }
        }

        for(map<KeyFrame*, size_t>::iterator mit=native_obs.begin(),
                mend=native_obs.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;
            const size_t idx = mit->second;
            if(!pML->mNativeFisheyeObservations.count(pKF))
            {
                pKF->ReplaceNativeFisheyeMapLineMatch(idx, pML);
                pML->AddNativeFisheyeObservation(pKF, idx, native_data[pKF]);
            }
            else
            {
                pKF->EraseNativeFisheyeMapLineMatch(idx);
            }
        }

        pML->IncreaseFound(nfound);
        pML->IncreaseVisible(nvisible);
        pML->ComputeDistinctiveDescriptors();

        mpMap->EraseMapLine(this);
    }

    MapLine* MapLine::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    const char* MapLine::OriginName(MapLineOrigin origin)
    {
        return OriginNameFromInt(static_cast<int>(origin));
    }

    void MapLine::SetCreationOrigin(
        MapLineOrigin origin,
        long frame_id,
        long keyframe_id,
        double timestamp,
        const std::string& function_name)
    {
        creation_origin = origin;
        creation_frame = frame_id;
        creation_keyframe = keyframe_id;
        creation_timestamp = timestamp;
        provenance_last_frame_seen = frame_id;
        std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
        SyncRecordLocked(this);
        LogEventLocked("CREATE", this, NULL, frame_id, keyframe_id, -1,
                       OriginName(origin), function_name);
    }

    void MapLine::MarkPhysicalAssociation(
        long frame_id,
        long keyframe_id,
        long physical_obs_id,
        const std::string& function_name)
    {
        physical_touch_count++;
        physical_association_count++;
        if(frame_id >= 0)
            provenance_last_frame_seen = frame_id;
        if(frame_id >= 0)
            mnLastFrameSeen = static_cast<unsigned long>(frame_id);
        std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
        SyncRecordLocked(this);
        LogEventLocked("PHYSICAL_ASSOCIATION", this, NULL, frame_id,
                       keyframe_id, physical_obs_id, "SUCCESS",
                       function_name);
    }

    void MapLine::MarkLegacyFusionTouch(
        long frame_id,
        long keyframe_id,
        MapLine* secondary,
        const std::string& event_type,
        const std::string& result,
        const std::string& reason)
    {
        legacy_touch_count++;
        legacy_fusion_count++;
        if(frame_id >= 0)
            provenance_last_frame_seen = frame_id;
        std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
        SyncRecordLocked(this);
        SyncRecordLocked(secondary);
        LogEventLocked(event_type, this, secondary, frame_id, keyframe_id,
                       -1, result, reason);
    }

    void MapLine::MarkOptimizerResidualCreated(
        long frame_id,
        long physical_obs_id)
    {
        physical_touch_count++;
        optimizer_use_count++;
        residual_created_count++;
        if(frame_id >= 0)
            provenance_last_frame_seen = frame_id;
        if(frame_id >= 0)
            mnLastFrameSeen = static_cast<unsigned long>(frame_id);
        std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
        SyncRecordLocked(this);
        LogEventLocked("FORMAL_RESIDUAL_CREATE", this, NULL, frame_id, -1,
                       physical_obs_id, "CREATED", "");
    }

    void MapLine::MarkOptimizerResidualAccepted(
        long frame_id,
        long physical_obs_id)
    {
        residual_accepted_count++;
        if(frame_id >= 0)
            provenance_last_frame_seen = frame_id;
        if(frame_id >= 0)
            mnLastFrameSeen = static_cast<unsigned long>(frame_id);
        std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
        SyncRecordLocked(this);
        LogEventLocked("FORMAL_RESIDUAL_ACCEPT", this, NULL, frame_id, -1,
                       physical_obs_id, "ACCEPT", "");
    }

    void MapLine::MarkOptimizerResidualRejected(
        long frame_id,
        long physical_obs_id,
        const std::string& reason)
    {
        if(frame_id >= 0)
            provenance_last_frame_seen = frame_id;
        std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
        SyncRecordLocked(this);
        LogEventLocked("FORMAL_RESIDUAL_REJECT", this, NULL, frame_id, -1,
                       physical_obs_id, "REJECT", reason);
    }

    void MapLine::LogFusionAttempt(
        const std::string& event_type,
        MapLine* source,
        MapLine* target,
        long frame_id,
        long keyframe_id,
        const std::string& result,
        const std::string& reason)
    {
        std::lock_guard<std::mutex> prov_lock(gProvenanceMutex);
        SyncRecordLocked(source);
        SyncRecordLocked(target);
        LogEventLocked(event_type, source, target, frame_id, keyframe_id,
                       -1, result, reason);
    }

    void MapLine::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible+=n;
    }

    void MapLine::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound+=n;
    }

    float MapLine::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound/mnVisible);
    }

    void MapLine::ComputeDistinctiveDescriptors()
    {
        // Retrieve all observed descriptors
        vector<Mat> vDescriptors;

        map<KeyFrame*, size_t> observations;
        map<KeyFrame*, FisheyeLineObservation> native_observations;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            if(mbBad)
                return ;
            observations = mObservations;
            native_observations = mNativeFisheyeObservationData;
        }

        if(observations.empty() && native_observations.empty())
            return;

        vDescriptors.reserve(observations.size() +
                             native_observations.size());

        // 遍历观测到3d特征线的所有关键帧，获得LBD描述子，并插入到vDescriptors中
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;

            if(!pKF->isBad())
                vDescriptors.push_back(pKF->mLineDescriptors.row(mit->second));
        }

        if(vDescriptors.empty())
        {
            for(map<KeyFrame*, FisheyeLineObservation>::iterator
                    mit = native_observations.begin(),
                    mend = native_observations.end();
                mit != mend; mit++)
            {
                KeyFrame* pKF = mit->first;
                if(pKF && !pKF->isBad() &&
                   !mit->second.lbd_descriptor.empty())
                {
                    vDescriptors.push_back(
                        mit->second.lbd_descriptor.clone());
                }
            }
        }

        if(vDescriptors.empty())
            return;

        // 计算这些描述子两两之间的距离
        const size_t NL=vDescriptors.size();

        //float Distances[N][N]
        std::vector<std::vector<float>> Distances;
        Distances.resize(NL, vector<float>(NL, 0));
        for(size_t i=0; i<NL; i++)
        {
            Distances[i][i]=0;
            for(size_t j=0; j<NL; j++)
            {
//                int distij = LSDmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                int distij = norm(vDescriptors[i], vDescriptors[j], NORM_HAMMING);
                Distances[i][j]=distij;
                Distances[j][i]=distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for(size_t i=0; i<NL; i++)
        {
            vector<int> vDists(Distances[i].begin(), Distances[i].end());
            sort(vDists.begin(), vDists.end());

            //获得中值
            int median = vDists[0.5*(NL-1)];

            //寻找最小的中值
            if(median<BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }
        {
            unique_lock<mutex> lock(mMutexFeatures);
            mLDescriptor = vDescriptors[BestIdx].clone();
        }

    }

    Mat MapLine::GetDescriptor()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mLDescriptor.clone();
    }

    void MapLine::UpdateAverageDir()
    {
        map<KeyFrame*, size_t> observations;
        map<KeyFrame*, size_t> native_observations;
        KeyFrame* pRefKF;
        Vector6d Pos;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            if(mbBad)
                return;
            observations = mObservations;
            native_observations = mNativeFisheyeObservations;
            pRefKF = mpRefKF;
            Pos = mWorldPos;
        }

        if(observations.empty() && native_observations.empty())
            return;

        Vector3d normal(0, 0, 0);
        int n=0;
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;
            Mat Owi = pKF->GetCameraCenter();
            Vector3d OWi(Owi.at<float>(0), Owi.at<float>(1), Owi.at<float>(2));
            Vector3d middlePos = 0.5*(mWorldPos.head(3)+mWorldPos.tail(3));
            Vector3d normali = middlePos - OWi;
            normal = normal + normali/normali.norm();
            n++;
        }

        if(observations.empty())
        {
            for(map<KeyFrame*,size_t>::iterator
                    mit=native_observations.begin(),
                    mend=native_observations.end();
                mit!=mend; mit++)
            {
                KeyFrame* pKF = mit->first;
                if(!pKF || pKF->isBad())
                    continue;
                Mat Owi = pKF->GetCameraCenter();
                Vector3d OWi(Owi.at<float>(0), Owi.at<float>(1),
                              Owi.at<float>(2));
                Vector3d middlePos =
                    0.5*(mWorldPos.head(3)+mWorldPos.tail(3));
                Vector3d normali = middlePos - OWi;
                if(normali.norm() <= 1e-12)
                    continue;
                normal = normal + normali/normali.norm();
                n++;
            }
        }

        if(n == 0)
            return;

        cv::Mat SP = (Mat_<float>(3,1) << Pos(0), Pos(1), Pos(2));
        cv::Mat EP = (Mat_<float>(3,1) << Pos(3), Pos(4), Pos(5));
        cv::Mat MP = 0.5*(SP+EP);

        cv::Mat CM = MP - pRefKF->GetCameraCenter();
        const float dist = cv::norm(CM);
        float levelScaleFactor = 1.0f;
        if(!observations.empty() && observations.count(pRefKF) &&
           observations[pRefKF] < pRefKF->mvKeyLines.size())
        {
            const int level = pRefKF->mvKeyLines[observations[pRefKF]].octave;
            levelScaleFactor = pRefKF->mvScaleFactors[level];
        }
        const int nLevels = pRefKF->mnScaleLevels;

        {
            unique_lock<mutex> lock3(mMutexPos);
            mfMaxDistance = dist*levelScaleFactor;
            mfMinDistance = nLevels > 0 ?
                mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1] :
                mfMaxDistance;
            mNormalVector = normal/n;
        }
    }

    float MapLine::GetMinDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 0.8f*mfMinDistance;
    }

    float MapLine::GetMaxDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 1.2f*mfMaxDistance;
    }

    int MapLine::PredictScale(const float &currentDist, const float &logScaleFactor)
    {
        float ratio;
        {
            unique_lock<mutex> lock3(mMutexPos);
            ratio = mfMaxDistance/currentDist;
        }

        return ceil(log(ratio)/logScaleFactor);
    }
}
