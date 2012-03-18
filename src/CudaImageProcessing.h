#ifndef __CUDAIMAGEPROCESSING_H__
#define __CUDAIMAGEPROCESSING_H__

// TODO REFACTOR

extern "C" void cudaYCbYCrToRGBA( uchar4 *d_dst, uchar4 *d_src, int imageW, int imageH );
extern "C" void cudaGray1ToRGBA( uchar4 *d_dst, unsigned char *d_src, int imageW, int imageH );
extern "C" void cudaYCbYCrToY( uchar4 *d_dst, uchar4 *d_src, int imageW, int imageH );
extern "C" void cudaYToYCbYCr( uchar4 *d_dst, uchar4 *d_src, int imageW, int imageH );
extern "C" void cudaDiffFromYCbYCr( uchar4 *d_dst, uchar4 *d_srcA, uchar4 *d_srcB, int imageW, int imageH );
extern "C" void cudaDiffRGB( uchar4 *d_dst, uchar4 *d_src1, uchar4 *d_src2, int imageW, int imageH );
extern "C" void cudaFloatToRGBA(uchar4*outDevicePtr, float*inDevicePtr, int imageW, int imageH);
extern "C" void cudaRGBAToFloat(float*outDevicePtr, uchar4*inDevicePtr, int imageW, int imageH);
extern "C" void cudaRGBAtoCuda( float *outDevicePtr, uchar4 *inDevicePtr, unsigned int imageW, unsigned int imageH, unsigned int pitch );
extern "C" void cudaCudatoRGBA( uchar4 *outDevicePtr, float *inDevicePtr, unsigned int imageW, unsigned int imageH, unsigned int pitch );
extern "C" void cudaTranspose( float *d_dst, size_t dst_pitch, float *d_src, size_t src_pitch, unsigned int width, unsigned int height );
extern "C" void cudaIntegrate( float*out, float*in, unsigned int width, unsigned int height, unsigned int pitch );
extern "C" void cudaWarpImage( uchar4 *d_dst, uchar4 *d_src, int imageW, int imageH, double matrix[9] );

#endif//__CUDAIMAGEPROCESSING_H__
