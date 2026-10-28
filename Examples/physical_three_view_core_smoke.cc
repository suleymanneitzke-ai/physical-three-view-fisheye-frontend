#include "PhysicalLineObservationBuilder.h"
#include "optimization/PhysicalLinePlanePoseEdge.h"
#include <iostream>
int main(){StructureSLAM::FisheyeLineFrontend::FrameResult f; f.lines.resize(3); for(int i=0;i<3;++i){StructureSLAM::FisheyeLineFrontend::ViewLine x; x.view_id=i; x.line_id=0; x.virtual_line=cv::Vec4f(210,230,300,230); f.lines[i].push_back(x);} std::vector<cv::Mat> masks(3,cv::Mat(512,512,CV_8U,cv::Scalar(255))); StructureSLAM::PhysicalLineObservationBuilder::Stats s; auto g=StructureSLAM::PhysicalLineObservationBuilder::Build(f,masks,3,4.0,&s); std::cout<<"groups="<<g.size()<<" three="<<s.three<<std::endl; return (g.size()==1 && g[0].members.size()==3 && s.three==1 && g[0].plane_valid)?0:1;}
