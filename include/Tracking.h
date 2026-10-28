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

#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"

#include "pointcloudmapping.h"
#include <mutex>

#include "auxiliar.h"
#include "ExtractLineSegment.h"
#include "MapLine.h"
#include "LSDmatcher.h"
#include "CornerFrameDatebase.h"
#include "FisheyeLineFrontend.h"
#include "FisheyeLineObservationConverter.h"
#include "FisheyeLineShadowMatcher.h"
#include "NativeLoopFrameDiagnostics.h"

#include <memory>

class PointCloudMapping;

namespace StructureSLAM
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    // TO DO
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,Map* pCornerMap,
             KeyFrameDatabase* pKFDB, CornerFrameDatebase *pCFDB,KeyFrameDatabase* pCKFDB, const string &strSettingPath, const int sensor);

    
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageMonocularWithNormal(const cv::Mat &im,const cv::Mat &normal,  const double &timestamp);

    cv::Mat GrabImageMonocularWithPL(const cv::Mat &im,  const double &timestamp);


    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);
    StructureSLAM::NativeLoopFrameDiagnostics
    GetNativeLoopFrameDiagnostics() const;

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImRGB;
    cv::Mat mImGray;
    cv::Mat mNormal;
    cv::Mat mImDepth;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    vector<pair<int, int>> mvLineMatches;
    vector<cv::Point3f> mvLineS3D;   //初始化时线段起始点的3D位置
    vector<cv::Point3f> mvLineE3D;   //初始化时线段终止点的3D位置
    //vector<bool> mvbLineTriangulated;   //匹配的线特征是否能够三角化
    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;
    cv::Mat Rotation_cm;
    cv::Mat mRotation_wc;
    cv::Mat mRotation_cw;
    cv::Mat mLastRcm;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

protected:

    // Main tracking function under the assumption of Manhattan World.
    void TrackWithPL();

    void MonocularInitializationWithNormal();
    void CreateInitialMapMonoWithPL();
    void MonocularInitialization();
    void CreateInitialMapMonocular();
    cv::Mat SeekManhattanFrame(vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection);



    ResultOfMS ProjectSN2MF(int a,const cv::Mat &R_mc,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection,const int numOfSN);
    axiSNV ProjectSN2Conic(int a,const cv::Mat &R_mc,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection);
    cv::Mat TrackManhattanFrame(cv::Mat &mLastRcm,vector<SurfaceNormal> &vSurfaceNormal,vector<FrameLine> &vVanishingDirection);

    sMS MeanShift(vector<cv::Point2d> & v2D);
    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModelWithLine();
    bool TranslationWithMotionModel(cv::Mat &relatedPose);
    bool TrackWithMotionModel();

    bool Relocalization();
    bool LocalizationforPL();//designed for tracking based on points and lines

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalLines();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    bool TrackLocalMapWithLines();
    void SearchLocalPoints();
    void SearchLocalLines();


    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    std::vector<KeyFrame*> mvpCornerFrames;
    std::vector<MapPoint*> mvpCornerMapPoints;

    KeyFrameDatabase* mpCornerKeyFrameDB;
    CornerFrameDatebase*mpCornerFrameDB;
    std::queue<Frame*> mqCornerFrame;
    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    std::vector<MapLine*> mvpLocalMapLines;

    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;
    Map * mpCornerMap;
    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;
    bool mbUseKb8PointObservations;
    KannalaBrandt8Camera::Parameters mKb8PointCameraParameters;

    //自己添加的，两个用于纠正畸变的映射矩阵
    Mat mUndistX, mUndistY;
    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;
    int mnLineMatchesInliers;   //线特征

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;
    list<MapLine*>mlpTemporalLines;
    shared_ptr<PointCloudMapping>  mpPointCloudMapping;

    enum PointLineMode
    {
        POINT_ONLY = 0,
        LEGACY_PL = 1,
        PHYSICAL_ANGULAR_PL = 2,
        PHYSICAL_PIXEL_PL = 3,
        PHYSICAL_PIXEL_COV_PL = 4,
        PHYSICAL_PL = 5
    };

    // Native fisheye line frontend switches. Both are false by default so the
    // point/legacy baseline is unchanged unless settings explicitly enable
    // them. EnableNativeFisheyeLines allows Frame/KeyFrame native observation
    // storage; EnablePointLineTracking is reserved for the native residual
    // optimizer path and must not fall back to legacy pinhole line edges.
    bool mbEnableNativeFisheyeLines;
    bool mbEnablePointLineTracking;
    PointLineMode mPointLineMode;

    // P9-C2b monocular fisheye frontend. In native mode it populates the
    // formal native observation container; the old shadow/replay tools remain
    // separate regression utilities.
    bool mbEnableFisheyeLineShadow;
    shared_ptr<FisheyeLineFrontend> mpFisheyeLineFrontend;

    // P9-C3b shadow-only adjacent-frame line association.
    // These members do not modify the legacy line/map/optimizer path.
    shared_ptr<FisheyeLineShadowMatcher>
        mpFisheyeLineShadowMatcher;

    std::vector<FisheyeLineObservation>
        mvPreviousFisheyeLineObservations;

    FisheyeLineShadowMatcher::Result
        mLastFisheyeLineShadowMatchResult;

    bool mbHasPreviousFisheyeLineObservations;
    long unsigned int mnPreviousFisheyeLineFrameId;

    // P9-C3c shadow-only multi-frame track identities.
    // The index of each ID is aligned with the corresponding
    // fisheye observation vector.
    std::vector<long unsigned int>
        mvPreviousFisheyeLineTrackIds;

    // P9-C3e: retain t-2 observations for one-missing-frame
    // reconnection diagnosis. These remain shadow-only.
    std::vector<FisheyeLineObservation>
        mvTwoFramesAgoFisheyeLineObservations;

    std::vector<long unsigned int>
        mvTwoFramesAgoFisheyeLineTrackIds;

    bool mbHasTwoFramesAgoFisheyeLineObservations;
    long unsigned int mnTwoFramesAgoFisheyeLineFrameId;

    std::vector<long unsigned int>
        mvCurrentFisheyeLineTrackIds;

    // Track IDs are allocated monotonically from zero, so this
    // vector can be indexed directly by track ID.
    std::vector<unsigned int>
        mvFisheyeLineTrackLengths;

    // Number of deliberately bridged missing frames for every
    // shadow-only track. P9-C3f currently permits only gap=1.
    std::vector<unsigned int>
        mvFisheyeLineTrackMissingFrameCounts;

    // First and most recent frame observed for every track ID.
    std::vector<long unsigned int>
        mvFisheyeLineTrackStartFrames;

    std::vector<long unsigned int>
        mvFisheyeLineTrackLastFrames;

    long unsigned int mnNextFisheyeLineTrackId;
    int mnConfiguredNativeVirtualViews;
    int mnMonocularInitializationMinMatches;
    int mnMonocularInitializationSearchWindow;
    int mnMonocularInitializationMaxFrameGap;
    StructureSLAM::NativeLoopFrameDiagnostics
        mLastNativeLoopFrameDiagnostics;

};

} //namespace StructureSLAM

#endif // TRACKING_H
