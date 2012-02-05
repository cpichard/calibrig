#include "SurfCollectPoints.h"
#include "SurfHessian.h"

#include <cudpp.h>
#include <iostream>
#include <cstdio>

// Copy Hessian values To Descriptor value if point is valid
__global__
void copyHessianToDescriptor( HessianPoint *hpoints, SurfDescriptorPoint *dpoints, unsigned int *validatedPoints, unsigned int *newIndexes, int numPoints )
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
     if( index >= numPoints )
        return;

    if( validatedPoints[index] == 1 )
    {
        const HessianPoint pt = hpoints[index];
        unsigned int newIndex = newIndexes[index];

        dpoints[newIndex].m_x = pt.m_x;
        dpoints[newIndex].m_y = pt.m_y;
        dpoints[newIndex].m_scale = pt.m_scale;
        dpoints[newIndex].m_laplacian = pt.m_laplacian;
    }
}

__global__
void validatePoint( HessianPoint *points, unsigned int *validatedPoint, int npoints )
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if( index >= npoints )
        return;

    const HessianPoint pt = points[index];
    if( pt.m_x !=0 || pt.m_y !=0 || pt.m_scale !=0 || pt.m_laplacian != 0 )
    {
        validatedPoint[index] = 1;
    }
    else
    {
        validatedPoint[index] = 0;
    }
}

// TODO : in some CudaUtils file ?
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Debugging purpose
__global__
void drawPointf( HessianPoint *points, int npoints, float *img, int width, int height, int step )
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    if( n >= npoints )
        return;

    const HessianPoint pt = points[n];

    if( pt.m_x >=0 && pt.m_x < width && pt.m_y>=0 && pt.m_y< height )
    {
        const unsigned int index = (unsigned int) (pt.m_x + pt.m_y*width);
        img[index] = 0;
    }

}


// Debugging purpose
// Dumb fast code to write red pixels around located point
__global__
void drawPointuc( HessianPoint *points, int npoints, uchar4 *img, int width, int height, int step )
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    if( n >= npoints )
        return;

    const HessianPoint pt = points[n];

    if( pt.m_x >=4 && pt.m_x < width-4 && pt.m_y>=4 && pt.m_y< height-4 )
    {
        const unsigned int index1 = (unsigned int) (pt.m_x + pt.m_y*width);
        const unsigned int index2 = (unsigned int) (pt.m_x+1 + pt.m_y*width);
        const unsigned int index3 = (unsigned int) (pt.m_x-1 + pt.m_y*width);
        const unsigned int index4 = (unsigned int) (pt.m_x+1 + (pt.m_y+1)*width);
        const unsigned int index5 = (unsigned int) (pt.m_x+1 + (pt.m_y-1)*width);
        const unsigned int index6 = (unsigned int) (pt.m_x-1 + (pt.m_y+1)*width);
        const unsigned int index7 = (unsigned int) (pt.m_x-1 + (pt.m_y-1)*width);

        img[index1].x = 255;
        img[index1].y = 0;
        img[index1].z = 0;
        img[index2].x = 255;
        img[index2].y = 0;
        img[index2].z = 0;
        img[index3].x = 255;
        img[index3].y = 0;
        img[index3].z = 0;
        img[index4].x = 255;
        img[index4].y = 0;
        img[index4].z = 0;
        img[index5].x = 255;
        img[index5].y = 0;
        img[index5].z = 0;
        img[index6].x = 255;
        img[index6].y = 0;
        img[index6].z = 0;
        img[index7].x = 255;
        img[index7].y = 0;
        img[index7].z = 0;
    }

}


template<typename Points>
__global__
void collectPointsf( Points *points, float2 *vbo, int npoints, float width, float height )
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    if( n >= npoints )
        return;

    vbo[n].x = points[n].m_x/width;
    vbo[n].y = points[n].m_y/height;
}

// Initialize the descriptor data found hessian points found,
bool collectHessianPoints( HessianData &h_data, DescriptorData &d_data  )
{
    // Count number of validated hessian points
    const size_t numPoints = h_data.capacity();
    unsigned int *validatedPoints;
    unsigned int *newIndexes;

    cudaMalloc((void**)&validatedPoints, numPoints*sizeof(unsigned int));
    cudaMalloc((void**)&newIndexes, numPoints*sizeof(unsigned int));

    // Map validation
    dim3 threads(8);
    dim3 grid(iDivUp(numPoints, 8));
    validatePoint<<<grid, threads>>>(h_data.m_dPoints, validatedPoints, numPoints);
    cudaThreadSynchronize();
    // Scan
	CUDPPConfiguration conf;
	conf.op = CUDPP_ADD;
	conf.datatype = CUDPP_UINT;
	conf.algorithm = CUDPP_SCAN;
	conf.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE; 

    // Allocate plan for rows integration
    CUDPPHandle scanplan = 0;
    CUDPPResult result = cudppPlan(&scanplan, conf, numPoints, 1, 0);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    cudppScan(scanplan, newIndexes, validatedPoints, numPoints);
    
    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }
    
    // Read the number of validated points
    unsigned int numPointFound;
    cudaMemcpy( &numPointFound, newIndexes, 1*sizeof(unsigned int), cudaMemcpyDeviceToHost );

    // Realloc descriptor data with the number of validated hessian points
    d_data.reallocPoints(numPointFound);

    //std::cout << "found" << numPointFound << std::endl;
    // Compact - copy correct hessian to descriptors
    copyHessianToDescriptor<<<grid, threads>>> ( h_data.m_dPoints, d_data.m_descPoints, validatedPoints, newIndexes, numPoints );
    cudaThreadSynchronize();
    
    // Free temporary memory
    cudaFree(validatedPoints);
    cudaFree(newIndexes);

    return true;
}

void drawPoints( HessianData &hdata, ImageGL &image )
{
    // TODO : fonction dans
    if( hdata.capacity() == 0 )
    {
        std::cout << "no points found" << std::endl;
        return;
    }

    dim3 threads(8);
    dim3 grid(iDivUp(hdata.capacity(), 8));
    CudaDevicePtrWrapper<ImageGL,uchar4*> imgPtr(image);
	drawPointuc<<<grid, threads>>>( hdata.m_dPoints, hdata.capacity(),(uchar4*)imgPtr, Width(image), Height(image), Width(image) );
    cudaThreadSynchronize();
}


// Debugging purpose
void drawPoints( DescriptorData &, CudaImageBuffer<float> &image )
{}

// Debugging purpose
void drawPoints( HessianData &hdata, CudaImageBuffer<float> &image )
{
    // TODO : fonction dans
    if( hdata.capacity() == 0 )
    {
        std::cout << "no points found" << std::endl;
        return;
    }

    dim3 threads(8);
    dim3 grid(iDivUp(hdata.capacity(), 8));

	drawPointf<<<grid, threads>>>( hdata.m_dPoints, hdata.capacity(),(float*)image, Width(image), Height(image), Width(image) );
    cudaThreadSynchronize();
}

bool collectPoints( DescriptorData &ddata, VertexBufferObject &vbo, UInt2 &imgSize )
{
    size_t fsize = NbElements(ddata); //max( NbElements(vbo), NbElements(ddata) );
    // REFACTOR setNbElements(vbo, fsize)
    NbElements(vbo) = NbElements(ddata);
    
    //std::cout << vbo.m_bufId << std::endl;

    CudaDevicePtrWrapper<VertexBufferObject,float2*> outDevicePtr(vbo);

    dim3 threads(8);
    dim3 grid(iDivUp(fsize, 8));
    collectPointsf<<<grid, threads >>>(ddata.m_descPoints, (float2*)outDevicePtr, fsize, (float)Width(imgSize), (float)Height(imgSize) );
    cudaThreadSynchronize();
    return true;
}


bool collectPoints( HessianData &hdata, VertexBufferObject &vbo, UInt2 &imgSize )
{
    size_t fsize = max( NbElements(vbo), hdata.capacity() );
    
    // REFACTOR setNbElements(vbo, fsize)
    NbElements(vbo) = fsize;

    // Map VBO
    CudaDevicePtrWrapper<VertexBufferObject,float2*> outDevicePtr(vbo);

    dim3 threads(8);
    dim3 grid(iDivUp(fsize, 8));
    collectPointsf<<<grid, threads >>>(hdata.m_dPoints, (float2*)outDevicePtr, fsize, (float)Width(imgSize), (float)Height(imgSize) );
    cudaThreadSynchronize();
    return true;
}
