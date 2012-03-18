#ifndef __IMAGEPROCESSING_H__
#define __IMAGEPROCESSING_H__

#include "ImageGL.h"

// ImageGL processing
bool convertYCbYCrToRGB( const ImageGL &in, ImageGL &out );
bool convertYCbYCrToY  ( const ImageGL &in, ImageGL &out );
bool convertYToYCbYCr  ( const ImageGL &in, ImageGL &out );
bool convertRGBAToPBOY( const ImageGL &in, PixelBufferObject &out );
bool convertPBOYToRGBA( const PixelBufferObject &in, ImageGL &out );
bool convertToIntegral( CudaImageBuffer<float> &img );
bool convertRGBAToCudaBufferY( const ImageGL &in, CudaImageBuffer<float> &out );
bool convertCudaBufferYToRGBA( const  CudaImageBuffer<float> &in, ImageGL &out );
bool warpImage( ImageGL &src, ImageGL &dst, double matrix[9] );
bool diffImage( ImageGL &src1Img, ImageGL &src2Img, ImageGL &dstImg );
bool diffImageBuffer( ImageGL &imgA, ImageGL &imgB, ImageGL &result );
bool diffImageBufferYCbYCr( ImageGL &imgA, ImageGL &imgB, ImageGL &result );
bool streamsToRGB( ImageGL &srcImg, ImageGL &dstImg );
bool copyImageBuffer( ImageGL &src, ImageGL &dst );
bool copyImageBuffer( unsigned char *buffer, unsigned int width, unsigned int heigth, unsigned int depth, ImageGL &dst );
bool saveGrabbedImage(ImageGL &src, const std::string &filename);

#endif//__IMAGEPROCESSING_H__
