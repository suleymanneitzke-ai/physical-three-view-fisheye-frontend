#include "ReplaySnapshotWriter.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <set>

namespace StructureSLAM
{
namespace
{
const char kMagic[8] = {'T','2','6','5','R','S','V','1'};

template<typename T> void WriteValue(std::ostream& out, const T& value)
{ out.write(reinterpret_cast<const char*>(&value), sizeof(T)); }
template<typename T> bool ReadValue(std::istream& in, T* value)
{ return static_cast<bool>(in.read(reinterpret_cast<char*>(value), sizeof(T))); }

void WriteMat(std::ostream& out, const cv::Mat& mat)
{
    const std::int32_t rows = mat.rows, cols = mat.cols, type = mat.type();
    WriteValue(out, rows); WriteValue(out, cols); WriteValue(out, type);
    const std::uint64_t bytes = mat.empty() ? 0 : mat.total() * mat.elemSize();
    WriteValue(out, bytes);
    if(bytes)
    {
        cv::Mat continuous = mat.isContinuous() ? mat : mat.clone();
        out.write(reinterpret_cast<const char*>(continuous.data), bytes);
    }
}

bool ReadMat(std::istream& in, cv::Mat* mat)
{
    std::int32_t rows = 0, cols = 0, type = 0; std::uint64_t bytes = 0;
    if(!ReadValue(in,&rows)||!ReadValue(in,&cols)||!ReadValue(in,&type)||!ReadValue(in,&bytes)) return false;
    if(rows < 0 || cols < 0 || bytes > (1ULL<<32)) return false;
    if(!bytes) { mat->release(); return true; }
    *mat = cv::Mat(rows, cols, type);
    if(mat->total()*mat->elemSize()!=bytes) return false;
    return static_cast<bool>(in.read(reinterpret_cast<char*>(mat->data), bytes));
}

void WritePoints(std::ostream& out, const std::vector<cv::Point2f>& points)
{
    const std::uint64_t n=points.size(); WriteValue(out,n);
    for(size_t i=0;i<points.size();++i){WriteValue(out,points[i].x);WriteValue(out,points[i].y);}
}
bool ReadPoints(std::istream& in, std::vector<cv::Point2f>* points)
{
    std::uint64_t n=0; if(!ReadValue(in,&n)||n>1000000) return false; points->resize(n);
    for(size_t i=0;i<n;++i) if(!ReadValue(in,&(*points)[i].x)||!ReadValue(in,&(*points)[i].y)) return false;
    return true;
}

void WriteObservation(std::ostream& out, const FisheyeLineObservation& o)
{
    WriteValue(out,o.native_line_id); WriteValue(out,o.temporal_track_id); WriteValue(out,o.mapline_id);
    WriteValue(out,o.frame_id); WriteValue(out,o.keyframe_id); WriteValue(out,o.timestamp);
    WriteValue(out,o.view_id); WriteValue(out,o.line_id); WriteValue(out,o.duplicate_group_id);
    WriteValue(out,o.fisheye_start.x); WriteValue(out,o.fisheye_start.y); WriteValue(out,o.fisheye_end.x); WriteValue(out,o.fisheye_end.y);
    WriteValue(out,o.virtual_start.x); WriteValue(out,o.virtual_start.y); WriteValue(out,o.virtual_end.x); WriteValue(out,o.virtual_end.y);
    WriteValue(out,o.virtual_length_px); WriteValue(out,o.virtual_angle_deg); WriteValue(out,o.response); WriteValue(out,o.width);
    WriteValue(out,o.precision); WriteValue(out,o.sigma_px); WriteValue(out,o.information); WriteValue(out,o.valid_sample_ratio);
    const std::uint8_t flags=(o.inlier?1:0)|(o.outlier?2:0)|(o.descriptor_valid?4:0); WriteValue(out,flags);
    WritePoints(out,o.fisheye_samples); WritePoints(out,o.virtual_samples);
    const std::uint64_t n=o.sample_valid.size(); WriteValue(out,n); if(n) out.write(reinterpret_cast<const char*>(o.sample_valid.data()),n);
    WriteMat(out,o.lbd_descriptor);
}

bool ReadObservation(std::istream& in, FisheyeLineObservation* o)
{
    if(!ReadValue(in,&o->native_line_id)||!ReadValue(in,&o->temporal_track_id)||!ReadValue(in,&o->mapline_id)||
       !ReadValue(in,&o->frame_id)||!ReadValue(in,&o->keyframe_id)||!ReadValue(in,&o->timestamp)||
       !ReadValue(in,&o->view_id)||!ReadValue(in,&o->line_id)||!ReadValue(in,&o->duplicate_group_id)||
       !ReadValue(in,&o->fisheye_start.x)||!ReadValue(in,&o->fisheye_start.y)||!ReadValue(in,&o->fisheye_end.x)||!ReadValue(in,&o->fisheye_end.y)||
       !ReadValue(in,&o->virtual_start.x)||!ReadValue(in,&o->virtual_start.y)||!ReadValue(in,&o->virtual_end.x)||!ReadValue(in,&o->virtual_end.y)||
       !ReadValue(in,&o->virtual_length_px)||!ReadValue(in,&o->virtual_angle_deg)||!ReadValue(in,&o->response)||!ReadValue(in,&o->width)||
       !ReadValue(in,&o->precision)||!ReadValue(in,&o->sigma_px)||!ReadValue(in,&o->information)||!ReadValue(in,&o->valid_sample_ratio)) return false;
    std::uint8_t flags=0; if(!ReadValue(in,&flags)) return false; o->inlier=flags&1;o->outlier=flags&2;o->descriptor_valid=flags&4;
    if(!ReadPoints(in,&o->fisheye_samples)||!ReadPoints(in,&o->virtual_samples)) return false;
    std::uint64_t n=0;if(!ReadValue(in,&n)||n>10000000)return false;o->sample_valid.resize(n);if(n&&!in.read(reinterpret_cast<char*>(o->sample_valid.data()),n))return false;
    return ReadMat(in,&o->lbd_descriptor);
}

MapLine* Resolve(MapLine* line)
{
    std::set<MapLine*> seen;
    while(line && line->isBad()) { if(!seen.insert(line).second) return NULL; line=line->GetReplaced(); }
    return line && !line->isBad()?line:NULL;
}
} // namespace

ReplaySnapshotWriter& ReplaySnapshotWriter::Instance(){static ReplaySnapshotWriter writer;return writer;}
ReplaySnapshotWriter::ReplaySnapshotWriter()
{
    const char* p=std::getenv("STRUCTURE_SLAM_REPLAY_SNAPSHOT_PATH"); if(p) path_=p;
    const char* f=std::getenv("STRUCTURE_SLAM_CANDIDATE_FINGERPRINT_PATH"); if(f) fingerprint_path_=f;
}
ReplaySnapshotWriter::~ReplaySnapshotWriter(){Flush();}
bool ReplaySnapshotWriter::enabled() const{return !path_.empty();}

void ReplaySnapshotWriter::Capture(const Frame& frame,const std::vector<MapLine*>& lines,const KannalaBrandt8Camera::Parameters& camera)
{
    if(!enabled()) return;
    ReplayFrameState state; state.frame_id=frame.mnId;state.timestamp=frame.mTimeStamp;state.Tcw=frame.mTcw.clone();state.camera=camera;
    state.observations=frame.mvFisheyeLineObservations;
    for(size_t i=0;i<lines.size();++i)
    {
        ReplayMapLineState s; MapLine* line=Resolve(lines[i]); s.valid=line!=NULL;
        if(line){s.id=line->mnId;s.world=line->GetWorldPos();s.min_distance=line->GetMinDistanceInvariance();s.max_distance=line->GetMaxDistanceInvariance();s.descriptor=line->GetDescriptor().clone();
            const std::map<KeyFrame*,size_t> obs=line->GetNativeFisheyeObservations();
            for(auto it=obs.begin();it!=obs.end();++it){const auto stored=line->GetNativeFisheyeObservation(it->first);if(stored.temporal_track_id!=FisheyeLineObservation::kInvalidTrackId)s.temporal_track_ids.push_back(stored.temporal_track_id);}
        }
        state.local_maplines.push_back(s);
    }
    frames_.push_back(state);
}

void ReplaySnapshotWriter::RecordFingerprint(std::uint64_t id,const CandidateFingerprint& fp)
{ if(!fingerprint_path_.empty()) fingerprints_.push_back(std::make_pair(id,fp)); }

void ReplaySnapshotWriter::Flush()
{
    if(flushed_)return;flushed_=true;
    if(!path_.empty())
    {
        std::ofstream out(path_.c_str(),std::ios::binary);out.write(kMagic,8);const std::uint32_t version=1,endian=0x01020304,model=1,desc_type=0,desc_bytes=32;WriteValue(out,version);WriteValue(out,endian);WriteValue(out,model);WriteValue(out,desc_type);WriteValue(out,desc_bytes);const std::uint64_t count=frames_.size();WriteValue(out,count);
        for(const auto& f:frames_){WriteValue(out,f.frame_id);WriteValue(out,f.timestamp);WriteMat(out,f.Tcw);
            WriteValue(out,f.camera.fx);WriteValue(out,f.camera.fy);WriteValue(out,f.camera.cx);WriteValue(out,f.camera.cy);WriteValue(out,f.camera.k1);WriteValue(out,f.camera.k2);WriteValue(out,f.camera.k3);WriteValue(out,f.camera.k4);WriteValue(out,f.camera.width);WriteValue(out,f.camera.height);WriteValue(out,f.camera.newton_tolerance);WriteValue(out,f.camera.newton_max_iterations);WriteValue(out,f.camera.min_positive_z);WriteValue(out,f.camera.min_valid_radius);WriteValue(out,f.camera.max_theta_d);
            std::uint64_t n=f.observations.size();WriteValue(out,n);for(const auto&o:f.observations)WriteObservation(out,o);n=f.local_maplines.size();WriteValue(out,n);for(const auto&m:f.local_maplines){std::uint8_t valid=m.valid;WriteValue(out,valid);WriteValue(out,m.id);for(int i=0;i<6;++i)WriteValue(out,m.world(i));WriteValue(out,m.min_distance);WriteValue(out,m.max_distance);WriteMat(out,m.descriptor);std::uint64_t nt=m.temporal_track_ids.size();WriteValue(out,nt);for(auto id:m.temporal_track_ids)WriteValue(out,id);}}
    }
    if(!fingerprint_path_.empty()){std::ofstream out(fingerprint_path_.c_str());out<<"frame_id,candidate_count,ordered_hash,set_hash\n";for(const auto&r:fingerprints_)out<<r.first<<','<<r.second.count<<','<<r.second.ordered_hash<<','<<r.second.set_hash<<'\n';}
}

bool ReplaySnapshotWriter::Read(const std::string& path,std::vector<ReplayFrameState>* frames,std::string* error)
{
    std::ifstream in(path.c_str(),std::ios::binary);char magic[8];if(!in.read(magic,8)||std::memcmp(magic,kMagic,8)){*error="bad magic";return false;}std::uint32_t version,endian,model,dt,db;if(!ReadValue(in,&version)||!ReadValue(in,&endian)||!ReadValue(in,&model)||!ReadValue(in,&dt)||!ReadValue(in,&db)||version!=1||endian!=0x01020304){*error="unsupported header";return false;}std::uint64_t count;if(!ReadValue(in,&count)||count>1000000){*error="bad frame count";return false;}frames->resize(count);
    for(auto&f:*frames){if(!ReadValue(in,&f.frame_id)||!ReadValue(in,&f.timestamp)||!ReadMat(in,&f.Tcw)){*error="truncated frame";return false;}auto&p=f.camera;if(!ReadValue(in,&p.fx)||!ReadValue(in,&p.fy)||!ReadValue(in,&p.cx)||!ReadValue(in,&p.cy)||!ReadValue(in,&p.k1)||!ReadValue(in,&p.k2)||!ReadValue(in,&p.k3)||!ReadValue(in,&p.k4)||!ReadValue(in,&p.width)||!ReadValue(in,&p.height)||!ReadValue(in,&p.newton_tolerance)||!ReadValue(in,&p.newton_max_iterations)||!ReadValue(in,&p.min_positive_z)||!ReadValue(in,&p.min_valid_radius)||!ReadValue(in,&p.max_theta_d)){*error="truncated camera";return false;}std::uint64_t n;if(!ReadValue(in,&n)||n>1000000){*error="bad observation count";return false;}f.observations.resize(n);for(auto&o:f.observations)if(!ReadObservation(in,&o)){*error="truncated observation";return false;}if(!ReadValue(in,&n)||n>1000000){*error="bad mapline count";return false;}f.local_maplines.resize(n);for(auto&m:f.local_maplines){std::uint8_t valid;if(!ReadValue(in,&valid)||!ReadValue(in,&m.id)){*error="truncated mapline";return false;}m.valid=valid;for(int i=0;i<6;++i)if(!ReadValue(in,&m.world(i))){*error="truncated world";return false;}if(!ReadValue(in,&m.min_distance)||!ReadValue(in,&m.max_distance)||!ReadMat(in,&m.descriptor)){*error="truncated descriptor";return false;}std::uint64_t nt;if(!ReadValue(in,&nt)||nt>1000000){*error="bad temporal count";return false;}m.temporal_track_ids.resize(nt);for(auto&id:m.temporal_track_ids)if(!ReadValue(in,&id)){*error="truncated temporal";return false;}}}
    return true;
}
} // namespace StructureSLAM
