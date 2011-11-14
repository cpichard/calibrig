#include "ImageProcessing.h"
#include "CudaImageProcessing.h"

#include <cassert>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cuda.h>
#include <cudaGL.h>
#include <cuda_runtime.h>
#include <cudpp.h>

#include "CudaUtils.h"

// Copy buffer of in in buffer of out converting YCrYcb to RGB
bool convertYCbYCrToRGB( const ImageGL &in, ImageGL &out )
{
    if( HasBuffer(in) == false )
        return false;

    // Copy from video to cuda buffer
    CudaDevicePtrWrapper<ImageGL,uchar4*> inDevicePtr(in);
    CudaDevicePtrWrapper<ImageGL,uchar4*> outDevicePtr(out);

    // Convert YcbYcr to rgb
    cudaYCbYCrToRGBA( (uchar4*)outDevicePtr, (uchar4*)inDevicePtr, Width(out), Height(out));

    return true;
}

// TODO : refactor : Same function as convertYCbYCrToRGB
bool streamsToRGB( ImageGL &srcImg, ImageGL &dstImg )
{
    CudaDevicePtrWrapper<ImageGL,uchar4*> inDevicePtr(srcImg);
    CudaDevicePtrWrapper<ImageGL,uchar4*> outDevicePtr(dstImg);

    // Convert YcbYcr to rgb
    cudaYCbYCrToRGBA( (uchar4*)outDevicePtr, (uchar4*)inDevicePtr, Width(srcImg), Height(srcImg) );

    return true;
}


bool convertYCbYCrToY( const ImageGL &in, ImageGL &out )
{
    if( HasBuffer(in) == false )
        return false;

    CudaDevicePtrWrapper<ImageGL,uchar4*> inDevicePtr(in);
    CudaDevicePtrWrapper<ImageGL,uchar4*> outDevicePtr(out);

    // Convert YcbYcr to rgb
    cudaYCbYCrToY( (uchar4*)outDevicePtr, (uchar4*)inDevicePtr, Width(out), Height(out));
    
    return true;
}

bool convertYToYCbYCr( const ImageGL &in, ImageGL &out )
{
    if( HasBuffer(in) == false )
        return false;

    CudaDevicePtrWrapper<ImageGL,uchar4*> inDevicePtr(in);
    CudaDevicePtrWrapper<ImageGL,uchar4*> outDevicePtr(out);

    // Convert YcbYcr to rgb
    cudaYToYCbYCr( (uchar4*)outDevicePtr, (uchar4*)inDevicePtr, Width(out), Height(out));

    return true;
}

bool convertRGBAToPBOY( const ImageGL &in, PixelBufferObject &out )
{
    if( HasBuffer(in) == false || HasBuffer(out) == false )
        return false;

    CudaDevicePtrWrapper<ImageGL,uchar4*> inDevicePtr(in);
    CudaDevicePtrWrapper<PixelBufferObject,float*> outDevicePtr(out);

    cudaRGBAToFloat((float*)outDevicePtr, (uchar4*)inDevicePtr, Width(out), Height(out));

    return true;
}

bool convertPBOYToRGBA( const PixelBufferObject &in, ImageGL &out )
{
    if( HasBuffer(in) == false || HasBuffer(out) == false )
        return false;

    CudaDevicePtrWrapper<PixelBufferObject,float*> inDevicePtr(in);
    CudaDevicePtrWrapper<ImageGL,uchar4*> outDevicePtr(out);

    cudaFloatToRGBA((uchar4*)outDevicePtr, (float*)inDevicePtr, Width(out), Height(out));

    return true;
}


bool convertRGBAToCudaBufferY( const ImageGL &in, CudaImageBuffer<float> &out )
{
    if( HasBuffer(in) == false || HasBuffer(out) == false )
        return false;

    CudaDevicePtrWrapper<ImageGL,uchar4*> inDevicePtr(in);
    cudaRGBAtoCuda( (float*)out, (uchar4*)inDevicePtr, Width(out), Height(out), out.m_pitchInElements );

    return true;
}

bool convertCudaBufferYToRGBA( const CudaImageBuffer<float> &in, ImageGL &out )
{
    if( HasBuffer(in) == false || HasBuffer(out) == false )
        return false;

    // Map image to CUDA
    CudaDevicePtrWrapper<ImageGL,uchar4*> outDevicePtr(out);

    // Process
    cudaCudatoRGBA( (uchar4*)outDevicePtr, (float*)in, Width(out), Height(out), in.m_pitchInElements );

    return true;
}

// Convert img to an integral image
bool convertToIntegral( CudaImageBuffer<float> &img )
{
    UInt2 TSize = SwapXY( Size(img) );

    // Temporary image buffers
    LocalCudaImageBuffer<float> imgTmp( Size(img) );
    LocalCudaImageBuffer<float> imgTmpTr1( TSize );
    LocalCudaImageBuffer<float> imgTmpTr2( TSize );

    if( ! ( isAllocated(imgTmp) && isAllocated(imgTmpTr1) && isAllocated(imgTmpTr2) ) )
        return false;

    // Use Cudpp to create an integral image
    CUDPPHandle integrateX, integrateY;

	CUDPPConfiguration conf;
	conf.op = CUDPP_ADD;
	conf.datatype = CUDPP_FLOAT;
	conf.algorithm = CUDPP_SCAN;
	conf.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

    // Allocate plan for rows integration
    if( cudppPlan( &integrateX, conf, Width(img), Height(img), img.m_pitchInElements ) != CUDPP_SUCCESS )
    {
        std::cout << "Failed to allocate cudpp plan" << std::endl;
    }

    // Allocate plan for cols integration
    if( cudppPlan( &integrateY, conf, Width(imgTmpTr1), Height(imgTmpTr1), imgTmpTr1.m_pitchInElements ) != CUDPP_SUCCESS )
    {
        std::cout << "Failed to allocate cudpp plan" << std::endl;
    }

    // Compute
    // Integrate Rows
    cudppMultiScan( integrateX, (float*)imgTmp, (float*)img, Width(img), Height(img) );
    cudaTranspose( (float*)imgTmpTr1, imgTmpTr1.m_pitch, (float*)imgTmp, img.m_pitch, Width(img), Height(img) );

    // Integrate columns
    cudppMultiScan( integrateY, (float*)imgTmpTr2, (float*)imgTmpTr1, Width(imgTmpTr1), Height(imgTmpTr1));
    cudaTranspose( (float*)img, img.m_pitch, (float*)imgTmpTr2, imgTmpTr2.m_pitch, Width(imgTmpTr2), Height(imgTmpTr2) );

    cudppDestroyPlan(integrateY);
    cudppDestroyPlan(integrateX);
    
    return true;
}

// TODO in image processing ?
bool warpImage( ImageGL &src, ImageGL &dst, float matrix[9] )
{
    CudaDevicePtrWrapper<ImageGL,uchar4*> srcDevicePtr(src);
    CudaDevicePtrWrapper<ImageGL,uchar4*> dstDevicePtr(dst);
    cudaWarpImage( (uchar4*)dstDevicePtr, (uchar4*)srcDevicePtr, Width(src), Height(src), matrix );
    return true;
}

bool diffImage( ImageGL &src1Img, ImageGL &src2Img, ImageGL &dstImg )
{
    // Map buffer
    CudaDevicePtrWrapper<ImageGL,uchar4*> in1DevicePtr(src1Img);
    CudaDevicePtrWrapper<ImageGL,uchar4*> in2DevicePtr(src2Img);
    CudaDevicePtrWrapper<ImageGL,uchar4*> outDevicePtr(dstImg);

    // Convert YcbYcr to rgb
    cudaDiffRGB( (uchar4*)outDevicePtr, (uchar4*)in1DevicePtr, (uchar4*)in2DevicePtr, Width(src1Img), Height(src1Img) );

    return true;
}

bool copyImageBuffer( ImageGL &src, ImageGL &dst )
{
    // Map buffer
    CudaDevicePtrWrapper<ImageGL,void*> inDevicePtr(src);
    CudaDevicePtrWrapper<ImageGL,void*> outDevicePtr(dst);

    cudaMemcpy( (void*)outDevicePtr, (void*)inDevicePtr, sizeof(unsigned char)*4*Width(src)*Height(src), cudaMemcpyDeviceToDevice );
    return true;
}


bool copyImageBuffer( unsigned char *buffer, unsigned int width, unsigned int heigth, unsigned int depth, ImageGL &dst )
{
    // Copy from video to cuda buffer
    CudaDevicePtrWrapper<ImageGL,uchar4*> outDevicePtr(dst);

    unsigned char *inDevicePtr;
    cudaMalloc( (void**) &inDevicePtr, sizeof(unsigned char)*depth*width*heigth);
    cudaMemcpy( inDevicePtr, buffer, sizeof(unsigned char)*depth*width*heigth, cudaMemcpyHostToDevice );

    // Convert YcbYcr to rgb
    cudaGray1ToRGBA( (uchar4*)outDevicePtr, (unsigned char*)inDevicePtr, Width(dst), Height(dst) );
    
    cudaFree(inDevicePtr);
    return true;
}

bool diffImageBufferYCbYCr( ImageGL &imgA, ImageGL &imgB, ImageGL &result )
{
    CudaDevicePtrWrapper<ImageGL,uchar4*> inDevicePtrA(imgA);
    CudaDevicePtrWrapper<ImageGL,uchar4*> outDevicePtr(result);

    // This is stupid.. but ....
    if( BufId(imgA) == BufId(imgB) )
    {
        cudaDiffFromYCbYCr( (uchar4*)outDevicePtr, (uchar4*)inDevicePtrA,
            (uchar4*)inDevicePtrA, Width(result), Height(result) );
    }
    else
    {
        CudaDevicePtrWrapper<ImageGL,uchar4*> inDevicePtrB(imgB);
        cudaDiffFromYCbYCr( (uchar4*)outDevicePtr, (uchar4*)inDevicePtrA,
                    (uchar4*)inDevicePtrB, Width(result), Height(result) );
    }
    return true;
}