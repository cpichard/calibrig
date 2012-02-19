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
bool collectHessianPoints( HessianData &h_data, DescriptorData &d_data )
{
    // Count number of validated hessian points
    const size_t numPoints = h_data.capacity();
    unsigned int *validatedPoints;
    unsigned int *newIndexes;

    cudaMalloc((void**)&validatedPoints, numPoints*sizeof(unsigned int));
    cudaMalloc((void**)&newIndexes, numPoints*sizeof(unsigned int));
    checkLastError();

    // Map validation - write 0 or 1 in validatedPoints depending on the point validated
    dim3 threads(8);
    dim3 grid(iDivUp(numPoints, 8));
    validatePoint<<<grid, threads>>>(h_data.m_dPoints, validatedPoints, numPoints);
    checkLastError();
    cudaDeviceSynchronize();

    // Scan
	CUDPPConfiguration conf;
	conf.op = CUDPP_ADD;
	conf.datatype = CUDPP_UINT;
	conf.algorithm = CUDPP_SCAN;
	conf.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE; 

    // Allocate plan for rows integration
    CUDPPHandle scanplan = 0;
    CUDPPResult result = cudppPlan(&scanplan, conf, numPoints, 1, 0);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(EXIT_FAILURE);
    }

    // Scan to make new positions of points
    cudppScan(scanplan, newIndexes, validatedPoints, numPoints);
    checkLastError();

    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(EXIT_FAILURE);
    }
    
    // Read the number of validated points
    unsigned int numPointFound;
    cudaMemcpy( &numPointFound, newIndexes+(numPoints-1), 1*sizeof(unsigned int), cudaMemcpyDeviceToHost );
    checkLastError();
    
    // Realloc descriptor data with the number of validated hessian points
    d_data.reallocPoints(numPointFound);

    //std::cout << "found" << numPointFound << std::endl;
    // Compact - copy correct hessian to descriptors
    copyHessianToDescriptor<<<grid, threads>>> ( h_data.m_dPoints, d_data.m_descPoints, validatedPoints, newIndexes, numPointFound );
    cudaDeviceSynchronize();
    checkLastError();
    
    // Free temporary memory
    cudaFree(validatedPoints);
    checkLastError();
    cudaFree(newIndexes);
    checkLastError();
    return true;
}

//void drawPoints( HessianData &hdata, ImageGL &image )
//{
//    // TODO : fonction dans
//    if( hdata.capacity() == 0 )
//    {
//        std::cout << "no points found" << std::endl;
//        return;
//    }
//
//    dim3 threads(8);
//    dim3 grid(iDivUp(hdata.capacity(), 8));
//    CudaDevicePtrWrapper<ImageGL,uchar4*> imgPtr(image);
//	drawPointuc<<<grid, threads>>>( hdata.m_dPoints, hdata.capacity(),(uchar4*)imgPtr, Width(image), Height(image), Width(image) );
//    cudaDeviceSynchronize();
//}


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
    cudaDeviceSynchronize();
}

// Copy found points in ddata to a bound vertex buffer object
bool collectPoints( DescriptorData &ddata, VertexBufferObject &vbo, UInt2 &imgSize )
{
    size_t nbPointInDesc = NbElements(ddata);
    
    if(nbPointInDesc<8)
        return false;    

    // REFACTOR setNbElements(vbo, nbPointInDesc)
    NbElements(vbo) = nbPointInDesc; 
   
    // TODO : check that the vertex buffer object has 
    // sufficient memory
 
    CudaDevicePtrWrapper<VertexBufferObject,float2*> outDevicePtr(vbo);

    dim3 threads(8);
    dim3 grid(iDivUp(nbPointInDesc, 8));
    collectPointsf<<<grid, threads >>>(ddata.m_descPoints, (float2*)outDevicePtr, nbPointInDesc, (float)Width(imgSize), (float)Height(imgSize) );
    cudaDeviceSynchronize();
    checkLastError();
    return true;
}


bool collectPoints( HessianData &hdata, VertexBufferObject &vbo, UInt2 &imgSize )
{
    size_t fsize = hdata.capacity();
    if(fsize<8)
        return 0;
    
    // REFACTOR setNbElements(vbo, fsize)
    NbElements(vbo) = fsize;


    // Map VBO
    CudaDevicePtrWrapper<VertexBufferObject,float2*> outDevicePtr(vbo);

    dim3 threads(8);
    dim3 grid(iDivUp(fsize, 8));
    collectPointsf<<<grid, threads >>>(hdata.m_dPoints, (float2*)outDevicePtr, fsize, (float)Width(imgSize), (float)Height(imgSize) );
    checkLastError();
    cudaDeviceSynchronize();
    return true;
}
