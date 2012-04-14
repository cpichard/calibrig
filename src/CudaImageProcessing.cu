
#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8


#include "cutil_math.h"

texture<uchar4, 2, cudaReadModeElementType> tex;

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// 24-bit multiplication is faster on G80,
// but we must be sure to multiply integers
// only within [-8M, 8M - 1] range
#if defined(CUDA_24BIT_IMUL)
#	define IMUL(a, b) __mul24(a, b)
#else
#	define IMUL(a, b) (a)*(b)
#endif

// TODO deformation
// At the moment this function is only used for testing
__global__
void cuWarpImage( uchar4 *d_dst, int imageW, int imageH, double matrix[9] )
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if( ix < imageW && iy < imageH )
    {
        // Warp is only used for TESTING purposes
        unsigned int pos = ix + iy*imageW;
        const float fx = float(ix);
        const float fy = float(iy);

        const float tu = fx*float(matrix[0]) +fy*float(matrix[1]) + float(matrix[2]);
        const float tv = fx*float(matrix[3]) +fy*float(matrix[4]) + float(matrix[5]);
        const float tw = fx*float(matrix[6]) +fy*float(matrix[7]) + float(matrix[8]); 
        
        const float u = tu/tw;
        const float v = tv/tw;
        
        const uchar4 col = tex2D( tex, u, v ); 
        const float R = (float)col.x;
        const float G = (float)col.y;
        const float B = (float)col.z;
        d_dst[ pos ].x = (unsigned char)( R > 0 ) ? ( ( R <=255 ) ? R : 255 ): 0 ;
        d_dst[ pos ].y = (unsigned char)( G > 0 ) ? ( ( G <=255 ) ? G : 255 ): 0 ;
        d_dst[ pos ].z = (unsigned char)( B > 0 ) ? ( ( B <=255 ) ? B : 255 ): 0 ;
    }
}

__global__
void anaglyphRGB( uchar4 *d_dst, uchar4 *d_src1, uchar4 *d_src2, int imageW, int imageH )
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if(ix < imageW && iy < imageH)
    {
        unsigned int pos = ix + iy*imageW;

        d_dst[ pos ].x = d_src1[pos].x; 
        d_dst[ pos ].y = d_src2[pos].y;
        d_dst[ pos ].z = d_src2[pos].z;
    }
}

__global__
void mixRGB( uchar4 *d_dst, uchar4 *d_src1, uchar4 *d_src2, int imageW, int imageH )
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if(ix < imageW && iy < imageH)
    {
        unsigned int posPix1 = ix + iy*imageW;

        const float3 src1 =  make_float3( d_src1[posPix1].x, d_src1[posPix1].y, d_src1[posPix1].z );
        const float3 src2 =  make_float3( d_src2[posPix1].x, d_src2[posPix1].y, d_src2[posPix1].z );

        const float3 result = src1+src2;

        const float R = (result.x/2.f);
        const float G = (result.y/2.f);
        const float B = (result.z/2.f);

        d_dst[ posPix1 ].x = (unsigned char)( R > 0 ) ? ( ( R <=255 ) ? R : 255 ): 0 ;
        d_dst[ posPix1 ].y = (unsigned char)( G > 0 ) ? ( ( G <=255 ) ? G : 255 ): 0 ;
        d_dst[ posPix1 ].z = (unsigned char)( B > 0 ) ? ( ( B <=255 ) ? B : 255 ): 0 ;
    }
}

__global__
void diffRGB( uchar4 *d_dst, uchar4 *d_src1, uchar4 *d_src2, int imageW, int imageH )
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if(ix < imageW && iy < imageH)
    {
        unsigned int posPix1 = ix + iy*imageW;

        const float3 src1 =  make_float3( d_src1[posPix1].x, d_src1[posPix1].y, d_src1[posPix1].z );
        const float3 src2 =  make_float3( d_src2[posPix1].x, d_src2[posPix1].y, d_src2[posPix1].z );

        const float3 result = fabs(src1-src2);

        const float R = (result.x+0.f);
        const float G = (result.y+0.f);
        const float B = (result.z+0.f);

        d_dst[ posPix1 ].x = (unsigned char)( R > 0 ) ? ( ( R <=255 ) ? R : 255 ): 0 ;
        d_dst[ posPix1 ].y = (unsigned char)( G > 0 ) ? ( ( G <=255 ) ? G : 255 ): 0 ;
        d_dst[ posPix1 ].z = (unsigned char)( B > 0 ) ? ( ( B <=255 ) ? B : 255 ): 0 ;
    }
}

#include <stdio.h>
// Convert 422 ycbycr to rgb
__global__
void
YCbYCrToRGBA(
    uchar4 *dst,
    uchar4 *src,
    int imageW, //960
    int imageH  //1080
)
{
    // Position in src image (960,1080)
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if(ix < imageW && iy < imageH)
    {
        const float Y1 = (float)src[imageW * iy + ix].x;
        const float Cb = (float)src[imageW * iy + ix].y;
        const float Y2 = (float)src[imageW * iy + ix].z;
        const float Cr = (float)src[imageW * iy + ix].w;

        // Conversion in RGB
        const float R1 = Y1 + 1.371*(Cr-128);
        const float G1 = Y1 - 0.698*(Cr-128) - 0.336*(Cb - 128);
        const float B1 = Y1 + 1.732*(Cb-128);

        const float R2 = Y2 + 1.371*(Cr-128);
        const float G2 = Y2 - 0.698*(Cr-128) - 0.336*(Cb - 128);
        const float B2 = Y2 + 1.732*(Cb-128);

        const unsigned int posPix1 = 2*imageW * iy + 2*ix;
        const unsigned int posPix2 = posPix1 + 1;

        dst[ posPix1 ].x = (unsigned char)( R1 > 0 ) ? ( ( R1 <=255 ) ? R1 : 255 ): 0 ;
        dst[ posPix1 ].y = (unsigned char)( G1 > 0 ) ? ( ( G1 <=255 ) ? G1 : 255 ): 0 ;
        dst[ posPix1 ].z = (unsigned char)( B1 > 0 ) ? ( ( B1 <=255 ) ? B1 : 255 ): 0 ;
        dst[ posPix1 ].w = 0;

        dst[ posPix2 ].x = (unsigned char)( R2 > 0 ) ? ( ( R2 <=255 ) ? R2 : 255 ): 0 ;
        dst[ posPix2 ].y = (unsigned char)( G2 > 0 ) ? ( ( G2 <=255 ) ? G2 : 255 ): 0 ;
        dst[ posPix2 ].z = (unsigned char)( B2 > 0 ) ? ( ( B2 <=255 ) ? B2 : 255 ): 0 ;
        dst[ posPix2 ].w = 0;
    }
}

__global__
void
YCbYCrToY(
    uchar4 *dst,
    uchar4 *src,
    int imageW,
    int imageH
)
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if(ix < imageW && iy < imageH)
    {
        const unsigned char Y1 = src[imageW * iy + ix].x;
        const unsigned char Y2 = src[imageW * iy + ix].z;

        const unsigned int posPix1 = 2*imageW * iy + 2*ix;
        const unsigned int posPix2 = posPix1 + 1;

        dst[ posPix1 ].x = Y1;
        dst[ posPix1 ].y = Y1;
        dst[ posPix1 ].z = Y1;
        dst[ posPix1 ].w = 0;

        dst[ posPix2 ].x = Y2;
        dst[ posPix2 ].y = Y2;
        dst[ posPix2 ].z = Y2;
        dst[ posPix2 ].w = 0;
    }
}


__global__
void
YToYCbYCr(
    uchar4 *dst,
    uchar4 *src,
    int imageW,
    int imageH
)
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if(ix < imageW && iy < imageH)
    {
        const unsigned int posPix1 = 2*imageW * iy + 2*ix;
        const unsigned int posPix2 = posPix1 + 1;

        const unsigned char Y1 = src[ posPix1 ].x;
        const unsigned char Y2 = src[ posPix2 ].x;

        dst[imageW * iy + ix].x = Y1;
        dst[imageW * iy + ix].y = 0; // Cb
        dst[imageW * iy + ix].z = Y2;
        dst[imageW * iy + ix].w = 0; // Cr
    }
}

__global__
void
Gray1ToRGBA(
    uchar4 *dst,            // RGB
    unsigned char *src,     // Gray
    int imageW,
    int imageH
)
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if(ix < imageW && iy < imageH)
    {
        const unsigned int position = imageW * iy + ix;
        const unsigned char color = src[position];

        dst[ position ].x = color;
        dst[ position ].y = color;
        dst[ position ].z = color;
        dst[ position ].w = 0;
    }
}

__global__
void
fromDiff(
    uchar4 *dst,
    uchar4 *d_srcA,
    uchar4 *d_srcB,
    int imageW,
    int imageH
)
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if(ix < imageW && iy < imageH)
    {
        const float Y1A = (float)d_srcA[imageW * iy + ix].x;
        const float CbA = (float)d_srcA[imageW * iy + ix].y;
        const float Y2A = (float)d_srcA[imageW * iy + ix].z;
        const float CrA = (float)d_srcA[imageW * iy + ix].w;

        // Conversion in RGB
        const float R1A = Y1A + 1.371*(CrA-128);
        const float G1A = Y1A - 0.698*(CrA-128) - 0.336*(CbA - 128);
        const float B1A = Y1A + 1.732*(CbA-128);

        const float R2A = Y2A + 1.371*(CrA-128);
        const float G2A = Y2A - 0.698*(CrA-128) - 0.336*(CbA - 128);
        const float B2A = Y2A + 1.732*(CbA-128);

        // SRC B
        const float Y1B = (float)d_srcB[imageW * iy + ix].x;
        const float CbB = (float)d_srcB[imageW * iy + ix].y;
        const float Y2B = (float)d_srcB[imageW * iy + ix].z;
        const float CrB = (float)d_srcB[imageW * iy + ix].w;

        // Conversion in RGB
        const float R1B = Y1B + 1.371*(CrB-128);
        const float G1B = Y1B - 0.698*(CrB-128) - 0.336*(CbB - 128);
        const float B1B = Y1B + 1.732*(CbB-128);

        const float R2B = Y2B + 1.371*(CrB-128);
        const float G2B = Y2B - 0.698*(CrB-128) - 0.336*(CbB - 128);
        const float B2B = Y2B + 1.732*(CbB-128);

        const unsigned int posPix1 = 2*imageW * iy + 2*ix;
        const unsigned int posPix2 = posPix1 + 1;

        const float R1 = (R1B-R1A)*0.5 + 127.f;
        const float G1 = (G1B-G1A)*0.5 + 127.f;
        const float B1 = (B1B-B1A)*0.5 + 127.f;
        const float R2 = (R2B-R2A)*0.5 + 127.f;
        const float G2 = (G2B-G2A)*0.5 + 127.f;
        const float B2 = (B2B-B2A)*0.5 + 127.f;

        dst[ posPix1 ].x = (unsigned char)( R1 > 0 ) ? ( ( R1 <=255 ) ? R1 : 255 ): 0 ;
        dst[ posPix1 ].y = (unsigned char)( G1 > 0 ) ? ( ( G1 <=255 ) ? G1 : 255 ): 0 ;
        dst[ posPix1 ].z = (unsigned char)( B1 > 0 ) ? ( ( B1 <=255 ) ? B1 : 255 ): 0 ;

        dst[ posPix2 ].x = (unsigned char)( R2 > 0 ) ? ( ( R2 <=255 ) ? R2 : 255 ): 0 ;
        dst[ posPix2 ].y = (unsigned char)( G2 > 0 ) ? ( ( G2 <=255 ) ? G2 : 255 ): 0 ;
        dst[ posPix2 ].z = (unsigned char)( B2 > 0 ) ? ( ( B2 <=255 ) ? B2 : 255 ): 0 ;
    }
}

__global__
void
RGBAtoFloat(
    float *dst,
    uchar4 *d_src,
    int imageW,
    int imageH,
    int pitch
)
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if(ix < imageW && iy < imageH )
    {
        const unsigned int position = imageW * iy + ix;

        const float col = d_src[position].x;
        dst[position] = col/255.0;
    }
}

__global__
void
FloatToRGBA(
    uchar4 *dst,
    float *d_src,
    int imageW,
    int imageH,
    int pitch
)
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Position in src image
    if(ix < imageW && iy < imageH )
    {
        const unsigned int position = pitch * iy + ix;

        const float pixVal = d_src[position]*255.0;
        const unsigned char p = (unsigned char)( pixVal > 0 ) ? ( ( pixVal <=255 ) ? pixVal : 255 ): 0 ;

        dst[position].x = p;
        dst[position].y = p;
        dst[position].z = p;
        dst[position].w = 0;
    }
}


// Testing purpose
__global__
void
Integrate(
    float *dst,
    float *d_src,
    int imageW,
    int imageH,
    int pitch
)
{
    // Position in dest image
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if(ix < imageW && iy < imageH )
    {
        const unsigned int position = pitch * iy + ix;

        float sum = 0;

        for( int i=0; i < ix; i++ )
            sum += d_src[i];
        
        dst[position] = sum;
    }
}


// Transpose of matrix
__global__ void
transpose(
	float *g_dst, size_t s_dst_pitch,
	const float *g_src, size_t s_src_pitch,
	unsigned int img_width, unsigned int img_height)
{
	extern __shared__ float s_mem[];
	unsigned int x = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int y = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
	const unsigned int src_offset = IMUL(y, s_src_pitch) + x;
	unsigned int smem_offset = IMUL(threadIdx.y, blockDim.x) + threadIdx.x
		+ threadIdx.y;

	// Load data into shared memory
	if (y < img_height)
	{
		s_mem[smem_offset] = g_src[src_offset];
	}

	__syncthreads();

	// Compute smem_offset so that we read the values transposed
	smem_offset = IMUL(threadIdx.x, blockDim.x) + threadIdx.y + threadIdx.x;

	// Compute destination offset
	x = IMUL(blockIdx.y, blockDim.x) + threadIdx.x;
	y = IMUL(blockIdx.x, blockDim.y) + threadIdx.y;
	const unsigned int dst_offset = IMUL(y, s_dst_pitch) + x;

	// Write data back to global memory
	if (y < img_width)
	{
		g_dst[dst_offset] = s_mem[smem_offset];
	}
}

#include <stdio.h>
extern "C" void
cudaYCbYCrToRGBA( uchar4 *d_dst, uchar4 *d_src, int imageW, int imageH)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW/2, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));
    
    YCbYCrToRGBA<<<grid, threads>>>( d_dst, d_src, imageW/2, imageH );
    cudaDeviceSynchronize();
}

extern "C" void
cudaGray1ToRGBA( uchar4 *d_dst, unsigned char *d_src, int imageW, int imageH )
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    Gray1ToRGBA<<<grid, threads>>>( d_dst, d_src, imageW, imageH );
    cudaDeviceSynchronize();
}



extern "C"
void cudaWarpImage( uchar4 *d_dst, uchar4 *d_src, int imageW, int imageH, double matrix[9] )
{
    size_t offset=0;

	tex.filterMode = cudaFilterModePoint; // We don't use interpolation (interpo impossible with uchar)
	tex.normalized = false; // Don't normalize texture coordinates
	/* Clamping saves us some boundary checks */
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.addressMode[2] = cudaAddressModeClamp;

    // Bind texture reference to linear memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaBindTexture2D( &offset, tex, (uchar4*)d_src, channelDesc, imageW, imageH, imageW*4*sizeof(unsigned char) );
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    //matrix[0] = 1;
    //matrix[1] = 0;
    //matrix[2] = 0;
    //matrix[3] = 0;
    //matrix[4] = 1;
    //matrix[5] = 0;
    //matrix[6] = 0;
    //matrix[7] = 0;
    //matrix[8] = 1;

    double *d_matrix; // device matrix
    cudaMalloc((void**)&d_matrix, sizeof(double)*9);
    cudaMemcpy(d_matrix, matrix, sizeof(double)*9, cudaMemcpyHostToDevice);
    cuWarpImage<<<grid, threads>>>( d_dst, imageW, imageH, d_matrix );
    cudaFree(d_matrix);
    cudaDeviceSynchronize();
    cudaUnbindTexture( tex );
}

extern "C"
void cudaDiffRGB( uchar4 *d_dst, uchar4 *d_src1, uchar4 *d_src2, int imageW, int imageH )
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    diffRGB<<<grid, threads>>>( d_dst, d_src1, d_src2, imageW, imageH );
    cudaDeviceSynchronize();
}

extern "C"
void cudaMix( uchar4 *d_dst, uchar4 *d_src1, uchar4 *d_src2, int imageW, int imageH )
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    mixRGB<<<grid, threads>>>( d_dst, d_src1, d_src2, imageW, imageH );
    cudaThreadSynchronize();
}

extern "C"
void cudaAnaglyph( uchar4 *d_dst, uchar4 *d_src1, uchar4 *d_src2, int imageW, int imageH )
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    anaglyphRGB<<<grid, threads>>>( d_dst, d_src1, d_src2, imageW, imageH );
    cudaThreadSynchronize();
}

extern "C"
void cudaDiffFromYCbYCr( uchar4 *d_dst, uchar4 *d_srcA, uchar4 *d_srcB, int imageW, int imageH )
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW/2, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    fromDiff<<<grid, threads>>>( d_dst, d_srcA, d_srcB, imageW/2, imageH );
    cudaDeviceSynchronize();
}

extern "C"
void cudaYCbYCrToY( uchar4 *d_dst, uchar4 *d_src, int imageW, int imageH )
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW/2, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    YCbYCrToY<<<grid, threads>>>(d_dst, d_src, imageW/2, imageH);
    cudaDeviceSynchronize();
}

extern "C"
void cudaYToYCbYCr( uchar4 *d_dst, uchar4 *d_src, int imageW, int imageH )
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW/2, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    YToYCbYCr<<<grid, threads>>>(d_dst, d_src, imageW/2, imageH);
    cudaDeviceSynchronize();
}

extern "C"
void cudaRGBAToFloat( float*outDevicePtr, uchar4*inDevicePtr, int imageW, int imageH)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));
    RGBAtoFloat<<<grid, threads>>>(outDevicePtr,inDevicePtr,imageW,imageH,imageW);
    cudaDeviceSynchronize();
}

extern "C"
void cudaFloatToRGBA( uchar4*outDevicePtr, float*inDevicePtr, int imageW, int imageH)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));
    FloatToRGBA<<<grid, threads>>>(outDevicePtr,inDevicePtr,imageW,imageH,imageW);
    cudaDeviceSynchronize();
}

extern "C"
void
cudaTranspose(
    float *d_dst, size_t dst_pitch,
	float *d_src, size_t src_pitch,
	unsigned int width, unsigned int height )
{
	// execution configuration parameters
	dim3 threads(16, 16);
	dim3 grid(iDivUp(width, 16), iDivUp(height, 16));
	size_t shared_mem_size =
		(threads.x * threads.y + (threads.y - 1)) * sizeof(float);

	transpose<<<grid, threads, shared_mem_size>>>(
		d_dst, dst_pitch / sizeof(float),
		d_src, src_pitch / sizeof(float),
		width, height);
    cudaDeviceSynchronize();

}

extern "C"
void
cudaRGBAtoCuda( float *outDevicePtr, uchar4 *inDevicePtr, unsigned int imageW, unsigned int imageH, unsigned int pitch )
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));
    RGBAtoFloat<<<grid, threads>>>(outDevicePtr,inDevicePtr,imageW,imageH, pitch);
    cudaDeviceSynchronize();
}


extern "C"
void
cudaCudatoRGBA( uchar4 *outDevicePtr, float *inDevicePtr, unsigned int imageW, unsigned int imageH, unsigned int pitch )
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));
    FloatToRGBA<<<grid, threads>>>(outDevicePtr,inDevicePtr,imageW,imageH, pitch);
    cudaDeviceSynchronize();
}


extern "C"
void
cudaIntegrate( float *out, float *in, unsigned int width, unsigned int height, unsigned int pitch )
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(width, BLOCKDIM_X), iDivUp(height, BLOCKDIM_Y));
    Integrate<<<grid, threads>>>( out, in, width, height, pitch );
    cudaDeviceSynchronize();
}
