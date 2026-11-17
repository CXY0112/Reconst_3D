// Independently authored host implementation of the published G-CVO GN model.
// Compatibility target: ToyotaResearchInstitute/gcvo release/v2 d1268382.
// Upstream is kept unchanged; this file has no CUDA/PCL runtime dependency.
#include "gcvo_api.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

using V3=Eigen::Vector3d;using M3=Eigen::Matrix3d;
using V6=Eigen::Matrix<double,6,1>;using M6=Eigen::Matrix<double,6,6>;
using M4=Eigen::Matrix4d;
using RM3=Eigen::Matrix<double,3,3,Eigen::RowMajor>;
using RM4=Eigen::Matrix<double,4,4,Eigen::RowMajor>;
static thread_local std::string error_message;
static M3 hat(const V3& v){M3 a;a<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return a;}
static M4 exponential(const V6& x){
 V3 w=x.head<3>(),v=x.tail<3>();double t=w.norm();M3 a=hat(w),b=a*a;
 double s,c,d;if(t<1e-5){double q=t*t;s=1-q/6+q*q/120;c=.5-q/24+q*q/720;d=1./6-q/120+q*q/5040;}
 else{s=std::sin(t)/t;c=(1-std::cos(t))/(t*t);d=(t-std::sin(t))/(t*t*t);}
 M4 out=M4::Identity();out.topLeftCorner<3,3>()+=s*a+c*b;out.topRightCorner<3,1>()=(M3::Identity()+c*a+d*b)*v;return out;
}
struct Tree {
 struct Node {int point,left=-1,right=-1,axis;};
 const std::vector<V3>& points;std::vector<int> ids;std::vector<Node> nodes;int root;
 Tree(const std::vector<V3>& p):points(p),ids(p.size()) {std::iota(ids.begin(),ids.end(),0);nodes.reserve(p.size());root=build(0,int(ids.size()),0);}
 int build(int lo,int hi,int depth){if(lo>=hi)return -1;int mid=(lo+hi)/2,axis=depth%3;
  std::nth_element(ids.begin()+lo,ids.begin()+mid,ids.begin()+hi,[&](int a,int b){return points[a][axis]<points[b][axis];});
  int index=int(nodes.size());nodes.push_back({ids[mid],-1,-1,axis});int left=build(lo,mid,depth+1),right=build(mid+1,hi,depth+1);nodes[index].left=left;nodes[index].right=right;return index;}
 void visit(int node,const V3& q,double r2,std::vector<int>& hits)const{if(node<0)return;const auto& n=nodes[node];V3 delta=points[n.point]-q;
  if(delta.squaredNorm()<=r2)hits.push_back(n.point);double split=q[n.axis]-points[n.point][n.axis];
  visit(split<0?n.left:n.right,q,r2,hits);if(split*split<=r2)visit(split<0?n.right:n.left,q,r2,hits);}
 void radius(const V3& q,double r2,std::vector<int>& hits)const{hits.clear();visit(root,q,r2,hits);std::sort(hits.begin(),hits.end());}
};
struct Row {M6 h=M6::Zero();V6 g=V6::Zero();double weight=0;int count=0;};
static std::vector<V3> xyz(const double* p,int n){std::vector<V3> v(n);for(int i=0;i<n;++i){v[i]=Eigen::Map<const V3>(p+3*i);if(!v[i].allFinite())throw std::invalid_argument("nonfinite points");}return v;}
GCVO_API const char* gcvo_backend(){
#ifdef _OPENMP
 return "gcvo_cpu_float64_openmp";
#else
 return "gcvo_cpu_float64_serial";
#endif
}
GCVO_API const char* gcvo_last_error(){return error_message.c_str();}
GCVO_API int gcvo_device_available(){return 1;}

GCVO_API int gcvo_align(const double* fixed,int nf,const double* moving,int nm,
 const double* cf,const double* cm,const double* ff,const double* fm,int fd,
 const double* initial,const double* p,double* output,double* stats){
 error_message.clear();auto start=std::chrono::steady_clock::now();
 try{
  if(nf<3||nm<3||fd<0||fd>64)throw std::invalid_argument("invalid point count/features");
  const int kernel=int(p[13]);if(kernel<0||kernel>2)throw std::invalid_argument("invalid kernel");
  auto x=xyz(fixed,nf),q=xyz(moving,nm);Tree tree(q);
  std::vector<M3> cx(nf),cq(nm),cy(nm);double max_trace=0;
  for(int i=0;i<nf;++i)cx[i]=Eigen::Map<const RM3>(cf+9*i);
  for(int i=0;i<nm;++i){cq[i]=Eigen::Map<const RM3>(cm+9*i);max_trace=std::max(max_trace,cq[i].trace());}
  std::vector<V3> y(nm);std::vector<Row> rows(nf);
  M4 internal=Eigen::Map<const RM4>(initial).inverse();
  double ell=p[0],step_norm=0,objective=0;int k=int(p[8]),weights=0,capped=0,status=1,it=0;
  int warm=int(p[11]),cool=int(p[10]);double ema=0,slow=0,fast=0;bool first=true,windows=false;
  std::queue<double> early,late;double early_sum=0,late_sum=0;
  M6 last_h=M6::Zero();
  for(it=0;it<int(p[6]);++it){
   M3 rt=internal.topLeftCorner<3,3>().transpose();V3 tr=-rt*internal.topRightCorner<3,1>();
   for(int j=0;j<nm;++j){y[j]=rt*q[j]+tr;cy[j]=rt*cq[j]*rt.transpose();}
   const double ell2=ell*ell,geo2=p[2]*p[2],feat2=p[5]*p[5];
   const double normalized_gate=-2*std::log(p[3]/geo2);
   const double gate=kernel==0?normalized_gate:std::min(p[14],normalized_gate);
   const double fgate=-2*p[4]*p[4]*std::log(p[3]/feat2);
   const int threads=std::max(1,int(p[22]));
   #pragma omp parallel for schedule(static) num_threads(threads)
   for(int i=0;i<nf;++i){
    Row row;std::vector<int> candidates;
    double radius2=kernel==0?std::min(p[14],gate*ell2):gate*(cx[i].trace()+max_trace+(p[19]!=0?ell2:0));
    if(kernel!=0)radius2=std::min(radius2,p[15]);
    V3 query=internal.topLeftCorner<3,3>()*x[i]+internal.topRightCorner<3,1>();
    tree.radius(query,std::max(0.,radius2),candidates);
    for(int j:candidates){
     if(row.count>=k)break;
     double fw=1;if(p[16]!=0){double d=0;for(int a=0;a<fd;++a){double z=ff[i*fd+a]-fm[j*fd+a];d+=z*z;}if(d>fgate)continue;fw=feat2*std::exp(-d/(2*p[4]*p[4]));}
     V3 r=x[i]-y[j];M3 s;
     if(kernel==0){if(r.squaredNorm()>p[14])continue;s=M3::Identity()/ell2;}
     else {if(r.squaredNorm()>p[15])continue;M3 cov=cx[i]+cy[j];if(p[19]!=0)cov.diagonal().array()+=ell2;s=cov.inverse();}
     double d=r.dot(s*r);if(!std::isfinite(d)||d>gate)continue;
     double w=geo2*std::exp(-.5*d)*fw;if(w<=p[3])continue;
     Eigen::Matrix<double,3,6> jac;jac.leftCols<3>()=-hat(y[j]);jac.rightCols<3>()=M3::Identity();
     row.h.noalias()+=w*jac.transpose()*s*jac;row.g.noalias()+=w*jac.transpose()*s*r;
     row.weight+=w;++row.count;
    }
    rows[i]=row;
   }
   M6 h=M6::Zero();V6 g=V6::Zero();weights=0;capped=0;objective=0;int max_count=0;
   for(const auto& row:rows){h+=row.h;g+=row.g;weights+=row.count;objective+=row.weight;capped+=row.count==k;max_count=std::max(max_count,row.count);}
   last_h=h;if(weights<100){status=-2;error_message="fewer than 100 kernel entries";break;}
   V6 gradient=-g;M6 gamma=M6::Zero();gamma.topLeftCorner<3,3>()=-.5*hat(gradient.head<3>());gamma.bottomLeftCorner<3,3>()=-hat(gradient.tail<3>());
   if(int(p[17])==1)h-=gamma.transpose();else if(int(p[17])==2)h+=gamma.transpose();
   if(p[18]!=0)h=(.5*(h+h.transpose())).eval();
   Eigen::LDLT<M6> ldlt(h);V6 dx=ldlt.solve(gradient);
   if(ldlt.info()!=Eigen::Success||!dx.allFinite()){status=-3;error_message="GN solve failed";break;}
   step_norm=dx.norm();
   if(step_norm<p[7]&&(kernel!=0||ell<=p[1])){status=0;break;}
   internal=(internal*exponential(dx)).eval();
   bool decay=false;
   if(p[20]!=0){
    if(cool>0)--cool;
    if(first){ema=objective;first=false;}else ema=.8*ema+.2*objective;
    if(!windows){slow=fast=ema;windows=true;}
    else{fast+=(std::min(1.,2/(p[11]+1)))*(ema-fast);slow+=(std::min(1.,2/(2*p[11]+1)))*(ema-slow);
     if(cool<=0){if(warm>0)--warm;else if(std::abs(fast/std::max(std::abs(slow),1e-6)-1)<p[12]){decay=true;windows=false;warm=int(p[11]);}}}
   }else if(it>int(p[10])){
    int win=int(p[11]);if(int(early.size())<win){early.push(objective);early_sum+=objective;}
    else{if(int(late.size())<win){late.push(objective);late_sum+=objective;}
     if(int(late.size())>=win){double ratio=early_sum>1e-12?late_sum/early_sum:1;
      if(std::abs(ratio-1)<p[12]){decay=true;early={};late={};early_sum=late_sum=0;}
      else{late_sum-=late.front();early_sum+=late.front();early.push(late.front());late.pop();early_sum-=early.front();early.pop();late.push(objective);late_sum+=objective;}}}
   }
   if(decay&&ell>p[1])ell=std::max(p[1],ell*p[9]);
   if(p[21]!=0)k=std::min(int(p[8]),std::max(1,int(max_count*1.2)));
  }
  M4 answer=internal.inverse();Eigen::JacobiSVD<M3> svd(answer.topLeftCorner<3,3>(),Eigen::ComputeFullU|Eigen::ComputeFullV);
  M3 correction=M3::Identity();correction(2,2)=(svd.matrixU()*svd.matrixV().transpose()).determinant();answer.topLeftCorner<3,3>()=svd.matrixU()*correction*svd.matrixV().transpose();
  Eigen::Map<RM4> out(output);out=answer;
  Eigen::SelfAdjointEigenSolver<M6> eigen(last_h);
  stats[0]=it;stats[1]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();stats[2]=weights;stats[3]=objective;stats[4]=step_norm;
  stats[5]=eigen.eigenvalues()[0];stats[6]=eigen.eigenvalues()[5];stats[7]=capped;stats[8]=ell;
  if(status==1)error_message="iteration limit reached";return status;
 }catch(const std::exception& e){error_message=e.what();return -10;}
}
