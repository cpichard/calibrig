#ifndef __SURFNONMAX_H__
#define __SURFNONMAX_H__

#include "SurfHessian.h"

bool computeNonMaxSuppression( CudaImageBuffer<float> &imgSat, HessianData & );

#endif//__SURFNONMAX_H__
