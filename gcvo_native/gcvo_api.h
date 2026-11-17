#pragma once
#ifdef _WIN32
#define GCVO_API extern "C" __declspec(dllexport)
#else
#define GCVO_API extern "C" __attribute__((visibility("default")))
#endif
// All numeric buffers use float64, matrices row-major. Public transform maps
// moving(target) -> fixed(source), matching the official public API.
// Parameters: ell,ell_min,amplitude,cutoff,feature_ell,feature_amplitude,
// max_iterations,tol,max_neighbors,ell_decay,decay_start,window,indicator_tol,
// kernel_type,metric_gate,squared_euclidean_gate,features,connection,symmetrize,
// ell2_regularization,ema,neighbor_decay,threads.
// Stats: iterations,seconds,weight_count,objective,step_norm,min_H_eigenvalue,
// max_H_eigenvalue,capped_rows,final_ell.
GCVO_API int gcvo_align(const double* fixed,int nf,const double* moving,int nm,
 const double* fixed_cov,const double* moving_cov,const double* fixed_features,
 const double* moving_features,int feature_dim,const double* initial,
 const double* parameters,double* output,double* stats);
GCVO_API const char* gcvo_last_error();
GCVO_API const char* gcvo_backend();
GCVO_API int gcvo_device_available();
