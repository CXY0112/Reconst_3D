// ABI adapter only. The official GPU solver/source is not modified.
#include "gcvo_api.h"
#include <gcvo/GCvoGPU.hpp>
#include <gcvo/utils/PointTypes19.hpp>
#include <chrono>
#include <string>
#include <stdexcept>
#include <cuda_runtime_api.h>
static thread_local std::string message;
using RM4=Eigen::Matrix<double,4,4,Eigen::RowMajor>;
GCVO_API const char* gcvo_backend(){return "official_gcvo_cuda_release_v2";}
GCVO_API const char* gcvo_last_error(){return message.c_str();}
GCVO_API int gcvo_device_available(){int count=0;auto status=cudaGetDeviceCount(&count);if(status!=cudaSuccess||count<1){message=std::string("CUDA unavailable: ")+cudaGetErrorString(status);return 0;}message.clear();return 1;}
GCVO_API int gcvo_align(const double* fixed,int nf,const double* moving,int nm,
 const double* cf,const double* cm,const double* ff,const double* fm,int fd,
 const double* initial,const double* p,double* output,double* stats){
 try{
  int devices=0;auto device_error=cudaGetDeviceCount(&devices);
  if(device_error!=cudaSuccess||devices<1)throw std::runtime_error("No usable CUDA device");
  if(fd!=0&&fd!=3)throw std::runtime_error("CUDA wrapper supports geometry-only or RGB features");
  gcvo::GCvoParams c;c.l_init=p[0];c.l_min=p[1];c.amplitude=p[2];c.sparsity_cutoff=p[3];c.c_l=p[4];c.c_amplitude=p[5];
  c.max_iterations=int(p[6]);c.tol_2=p[7];c.max_neighbors=int(p[8]);c.l_decay_rate=p[9];c.l_decay_start=int(p[10]);c.indicator_window=int(p[11]);c.indicator_threshold=p[12];
  c.kernel_type=static_cast<gcvo::GCvoKernelType>(int(p[13]));c.kernel_eval_max_dist=p[14];c.kernel_euclidean_max_dist=p[15];
  c.use_geometry=1;c.use_features=int(p[16]);c.use_connection_term=int(p[17]);c.use_symmetrization=int(p[18]);c.use_ell2_in_kernel=int(p[19]);c.use_ema_indicator=int(p[20]);c.neighbor_decay=int(p[21]);
  using Cloud=gcvo::GCvoPointCloudT<gcvo::PointS3>;
  auto cloud=[&](const double* xyz,const double* cov,const double* features,int n){Cloud out;out.reserve(n);
   for(int i=0;i<n;++i){gcvo::PointS3 point;point.x=float(xyz[3*i]);point.y=float(xyz[3*i+1]);point.z=float(xyz[3*i+2]);
    for(int j=0;j<9;++j)point.covariance[j]=float(cov[9*i+j]);
    for(int j=0;j<3;++j){point.features[j]=fd?float(features[fd*i+j]):0;point.normal[j]=0;}out.push_back(point);}return out;};
  auto x=cloud(fixed,cf,ff,nf),y=cloud(moving,cm,fm,nm);
  gcvo::GCvoGPU<gcvo::PointS3> solver(c);
  auto r=solver.align(x,y,Eigen::Map<const RM4>(initial).cast<float>());
  Eigen::Map<RM4> out(output);out=r.T_s2t.cast<double>();
  stats[0]=r.num_iters;stats[1]=r.registration_seconds;stats[2]=-1;stats[3]=-1;stats[4]=-1;stats[5]=-1;stats[6]=-1;stats[7]=-1;stats[8]=-1;
  message.clear();return r.return_code!=0?r.return_code:(r.num_iters>=c.max_iterations?1:0);
 }catch(const std::exception& e){message=e.what();return -10;}
}
