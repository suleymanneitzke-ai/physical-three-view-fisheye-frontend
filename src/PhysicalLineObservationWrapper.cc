#include "PhysicalLineObservationWrapper.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <mutex>
#include <numeric>

namespace StructureSLAM { namespace {
struct Totals { long long frames=0,obs=0,single=0,two=0,three=0,multiple=0,conflict=0,c40=0,c80=0,skip80=0; };
Totals total; std::mutex total_mutex;
void Flush() {
    const char* p=std::getenv("T265_PHYSICAL_WRAPPER_STATS"); if(!p||!*p) return;
    std::lock_guard<std::mutex> lock(total_mutex); std::ofstream o(p);
    o<<"frames,observations,single_view,two_view_confident,three_view_confident,multiple_possible,conflict,checks40,checks80,skipped80\n";
    o<<total.frames<<","<<total.obs<<","<<total.single<<","<<total.two<<","<<total.three<<","<<total.multiple<<","<<total.conflict<<","<<total.c40<<","<<total.c80<<","<<total.skip80<<"\n";
}
struct Register { Register(){std::atexit(Flush);} } reg;
double TraceDistance(const FisheyeLineFrontend::ViewLine& a,const FisheyeLineFrontend::ViewLine& b) {
    if(a.fisheye_samples.empty()||b.fisheye_samples.empty()) return 1e30;
    double sum=0; int n=0;
    for(size_t k=0;k<a.fisheye_samples.size();++k) {
        if(k<a.sample_valid.size()&&!a.sample_valid[k]) continue;
        double best=1e30;
        for(size_t q=0;q<b.fisheye_samples.size();++q) {
            if(q<b.sample_valid.size()&&!b.sample_valid[q]) continue;
            best=std::min(best,(double)cv::norm(a.fisheye_samples[k]-b.fisheye_samples[q]));
        }
        if(best<1e29){sum+=best;++n;}
    }
    return n?sum/n:1e30;
}
}

std::vector<PhysicalLineGroup> PhysicalLineObservationWrapper::Analyze(
    const FisheyeLineFrontend::FrameResult& f,const std::vector<cv::Mat>& masks,
    int frame_id,FrameStats* out) {
    FrameStats s; for(const auto& v:f.lines)s.observations+=(int)v.size();
    std::vector<std::vector<int> > mate(f.lines.size());
    for(size_t v=0;v<f.lines.size();++v) mate[v].assign(f.lines[v].size(),-1);
    auto pair=[&](int a,int b,bool primary){
        if(a>=(int)f.lines.size()||b>=(int)f.lines.size())return;
        if(primary)s.checks40+=(int)(f.lines[a].size()*f.lines[b].size());else s.checks80+=(int)(f.lines[a].size()*f.lines[b].size());
        std::vector<int> ab(f.lines[a].size(),-1),ba(f.lines[b].size(),-1);
        for(size_t i=0;i<f.lines[a].size();++i){double best=1e30;for(size_t j=0;j<f.lines[b].size();++j){double d=TraceDistance(f.lines[a][i],f.lines[b][j]);if(d<best){best=d;ab[i]=(int)j;}}}
        for(size_t j=0;j<f.lines[b].size();++j){double best=1e30;for(size_t i=0;i<f.lines[a].size();++i){double d=TraceDistance(f.lines[b][j],f.lines[a][i]);if(d<best){best=d;ba[j]=(int)i;}}}
        if(!primary){s.skipped80+=(int)f.lines[a].size();return;}
        for(size_t i=0;i<ab.size();++i)if(ab[i]>=0&&ba[ab[i]]==(int)i)mate[a][i]=ab[i];
    };
    pair(0,1,true); pair(1,2,true); pair(0,2,false);
    std::vector<PhysicalLineGroup> groups; std::set<std::pair<int,int> > used; int gid=0;
    for(size_t v=0;v<f.lines.size();++v)for(size_t i=0;i<f.lines[v].size();++i){
        if(used.count({(int)v,(int)i})) continue;
        PhysicalLineGroup g;
        g.group_id=gid++;g.frame_id=frame_id;g.members.push_back({(int)v,(int)i});used.insert({(int)v,(int)i});
        if(v<2&&mate[v][i]>=0){int j=mate[v][i];g.members.push_back({(int)v+1,j});used.insert({(int)v+1,j});if(v==0&&j<(int)mate[1].size()&&mate[1][j]>=0){int k=mate[1][j];g.members.push_back({2,k});used.insert({2,k});}}
        g.type=g.members.size()==3?PhysicalLineGroup::THREE_VIEW_CONFIDENT:(g.members.size()==2?PhysicalLineGroup::TWO_VIEW_CONFIDENT:PhysicalLineGroup::SINGLE_VIEW);
        if(g.type==PhysicalLineGroup::THREE_VIEW_CONFIDENT)++s.three_view;
        else if(g.type==PhysicalLineGroup::TWO_VIEW_CONFIDENT)++s.two_view;
        else ++s.single_view;
        groups.push_back(g);
    }
    if(out)*out=s;
    return groups;
}

SharedMapLineCandidate PhysicalLineObservationWrapper::IntersectCandidateSets(const std::vector<std::set<unsigned long> >& sets){
    SharedMapLineCandidate r;if(sets.empty())return r;r.intersection=sets[0];
    for(const auto& s:sets){r.candidate_union.insert(s.begin(),s.end());std::set<unsigned long> z;std::set_intersection(r.intersection.begin(),r.intersection.end(),s.begin(),s.end(),std::inserter(z,z.begin()));r.intersection.swap(z);}
    r.status=r.intersection.empty()?SharedMapLineCandidate::CROSS_VIEW_CANDIDATE_CONFLICT:(r.intersection.size()==1?SharedMapLineCandidate::UNIQUE_SHARED_CANDIDATE:SharedMapLineCandidate::MULTIPLE_SHARED_CANDIDATES);return r;
}
void PhysicalLineObservationWrapper::AccumulateAndFlushAtExit(const FrameStats& s){std::lock_guard<std::mutex> lock(total_mutex);++total.frames;total.obs+=s.observations;total.single+=s.single_view;total.two+=s.two_view;total.three+=s.three_view;total.multiple+=s.multiple;total.conflict+=s.conflict;total.c40+=s.checks40;total.c80+=s.checks80;total.skip80+=s.skipped80;}
}
