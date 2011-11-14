#ifndef __SURFCOLLECTPOINTS_H__
#define __SURFCOLLECTPOINTS_H__

#include "SurfHessian.h"
#include "SurfDescriptor.h"
#include "VertexBufferObject.h"

bool collectHessianPoints( HessianData &, DescriptorData & );

bool collectPoints( HessianData &, VertexBufferObject &, UInt2 &imgSize );
bool collectPoints( DescriptorData &, VertexBufferObject &, UInt2 &imgSize );

void drawPoints( DescriptorData &, CudaImageBuffer<float> &image );
void drawPoints( HessianData &, CudaImageBuffer<float> &image );
void drawPoints( HessianData &, ImageGL & );

#endif//__SURFCOLLECTPOINTS_H__
