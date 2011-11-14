#include "SurfDescriptor.h"
#include <cassert>
// 24-bit multiplication is faster on G80,
// but we must be sure to multiply integers
// only within [-8M, 8M - 1] range
#if defined(CUDA_24BIT_IMUL)
#	define IMUL(a, b) __mul24(a, b)
#else
#	define IMUL(a, b) (a)*(b)
#endif

// OpenSURF uses floor((x) + 0.5f), while CUDA recommends rintf(x)
#define fRound(x) (int)rintf(x)


// The following table contains the x coordinate LUT for the orientation detection threads
static const int coord_x[109] = {
	-5, -5, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
	-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,
	 0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
	 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5
};

// The following table contains the y coordinate LUT for the orientation detection threads
static const int coord_y[109] = {
	-3, -2, -1,  0,  1,  2,  3, -4, -3, -2, -1,  0,  1,  2,  3,  4, -5, -4, -3, -2, -1, 0, 1,
	 2,  3,  4,  5, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -5, -4, -3, -2, -1,  0, 1, 2,
	 3,  4,  5, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -5, -4, -3, -2, -1,  0,  1, 2, 3, 4,
	 5, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -5, -4, -3, -2, -1,  0,  1,  2,  3, 4, 5,
	-4, -3, -2, -1,  0,  1,  2,  3,  4, -3, -2, -1,  0,  1,  2,  3
};

// The following table contains the gauss coordinate LUT for the orientation detection threads
// This was originally a 2 dimensional array, but for efficiency reasons and since the orientation detection
// runs in a one-dimensional block, this array has been made one-dimensional and corresponds to the
// x and y lookup tables above
static const float gauss_lin[109] = {
	0.000958195f, 0.00167749f, 0.00250251f, 0.00318132f, 0.00250251f, 0.00167749f, 0.000958195f, 0.000958195f, 0.00196855f, 0.00344628f, 0.00514125f, 0.00653581f, 0.00514125f, 0.00344628f, 0.00196855f, 0.000958195f, 0.000695792f, 0.00167749f,
	0.00344628f, 0.00603331f, 0.00900064f, 0.0114421f, 0.00900064f, 0.00603331f, 0.00344628f, 0.00167749f, 0.000695792f, 0.001038f, 0.00250251f, 0.00514125f, 0.00900064f, 0.0134274f, 0.0170695f, 0.0134274f, 0.00900064f, 0.00514125f, 0.00250251f,
	0.001038f, 0.00131956f, 0.00318132f, 0.00653581f, 0.0114421f, 0.0170695f, 0.0216996f, 0.0170695f, 0.0114421f, 0.00653581f, 0.00318132f, 0.00131956f, 0.00142946f, 0.00344628f, 0.00708015f, 0.012395f, 0.0184912f, 0.0235069f, 0.0184912f, 0.012395f,
	0.00708015f, 0.00344628f, 0.00142946f, 0.00131956f, 0.00318132f, 0.00653581f, 0.0114421f, 0.0170695f, 0.0216996f, 0.0170695f, 0.0114421f, 0.00653581f, 0.00318132f, 0.00131956f, 0.001038f, 0.00250251f, 0.00514125f, 0.00900064f, 0.0134274f,
	0.0170695f, 0.0134274f, 0.00900064f, 0.00514125f, 0.00250251f, 0.001038f, 0.000695792f, 0.00167749f, 0.00344628f, 0.00603331f, 0.00900064f, 0.0114421f, 0.00900064f, 0.00603331f, 0.00344628f, 0.00167749f, 0.000695792f, 0.000958195f, 0.00196855f,
	0.00344628f, 0.00514125f, 0.00653581f, 0.00514125f, 0.00344628f, 0.00196855f, 0.000958195f, 0.000958195f, 0.00167749f, 0.00250251f, 0.00318132f, 0.00250251f, 0.00167749f, 0.000958195f
};
 // Texture reference to the integral image; neede by haarXY()
texture<float, 2, cudaReadModeElementType> integralImage;


__constant__ int dc_coord_x[109];
__constant__ int dc_coord_y[109];
__constant__ float dc_gauss_lin[109];

/*
 * This inline function computes the angle for a X,Y value pair
 * and is a verbatim copy of the CPU version.
 */
__device__ float
getAngle(float X, float Y)
{
	float pi = M_PI;

	if (X >= 0.0f && Y >= 0.0f)
		return atanf(Y/X);

	if (X < 0.0f && Y >= 0.0f)
		return pi - atanf(-Y/X);

	if (X < 0.0f && Y < 0.0f)
		return pi + atanf(Y/X);

	if (X >= 0.0f && Y < 0.0f)
		return 2.0f*pi - atanf(-Y/X);

	return 0.0f;
}


/*
 * This inline function is used by both the orientation and
 * description kernel to take samples from the source image.
 *
 * It computes the Haar X & Y response simultaneously.
 */
__device__ __inline__ void
haarXY(int sampleX, int sampleY, int roundedScale,
	   float *xResponse, float *yResponse, float gauss)
{
	float leftTop, middleTop, rightTop,
		  leftMiddle, rightMiddle,
		  leftBottom, middleBottom, rightBottom;

	int xmiddle = sampleX;
	int ymiddle = sampleY;
	int left = xmiddle - roundedScale;
	int right = xmiddle + roundedScale;
	int top = ymiddle - roundedScale;
	int bottom = ymiddle + roundedScale;

	leftTop = tex2D(integralImage,  left, top);
	leftMiddle = tex2D(integralImage,  left, ymiddle);
	leftBottom = tex2D(integralImage,  left, bottom);
	rightTop = tex2D(integralImage,  right, top);
	rightMiddle = tex2D(integralImage,  right, ymiddle);
	rightBottom = tex2D(integralImage,  right, bottom);
	middleTop = tex2D(integralImage,  xmiddle, top);
	middleBottom = tex2D(integralImage,  xmiddle, bottom);

	float upperHalf = leftTop - rightTop - leftMiddle + rightMiddle;
	float lowerHalf = leftMiddle - rightMiddle - leftBottom + rightBottom;
	*yResponse = gauss * (lowerHalf - upperHalf);

	float rightHalf = middleTop - rightTop - middleBottom + rightBottom;
	float leftHalf = leftTop - middleTop - leftBottom + middleBottom;
	*xResponse = gauss * (rightHalf - leftHalf);
}

/**	\brief Detect orientations of interest points
 *	\param ipoints device pointer to interest points
 *
 *	This kernel runs the entire orientation detection.
 *	Execution configuration:
 *	  Thread block: { 42, 1, 1 }
 *	  Block grid  : { num_ipoints, 1 }
 */
__global__ void
detectIpointOrientiationsCUDA(SurfDescriptorPoint* g_ipoints)
{
	SurfDescriptorPoint *g_ipt = g_ipoints + blockIdx.x; // Get a pointer to the interest point processed by this block

	// 1. Take all samples required to compute the orientation
	int s = fRound(g_ipt->m_scale), x = fRound(g_ipt->m_x), y = fRound(g_ipt->m_y);
	__shared__ float s_resX[109], s_resY[109], s_ang[109];

	// calculate haar responses for points within radius of 6*scale
	for (int index = threadIdx.x; index < 109; index += 42) {
		// Get X&Y offset of our sampling point (unscaled)
		int xOffset = dc_coord_x[index];
		int yOffset = dc_coord_y[index];
		float gauss = dc_gauss_lin[index];

		// Take the sample
		float haarXResult, haarYResult;
		haarXY(x+xOffset*s, y+yOffset*s, 2*s, &haarXResult, &haarYResult, gauss);

		// Store the sample and precomputed angle in shared memory
		s_resX[index] = haarXResult;
		s_resY[index] = haarYResult;
		s_ang[index] = getAngle(haarXResult, haarYResult);
	}

	__syncthreads(); // Wait until all thread finished taking their sample

	// calculate the dominant direction
	float sumX, sumY;
	float ang1, ang2, ang;
	float pi = M_PI;
	float pi_third = pi / 3.0f; // Size of the sliding window

	// Calculate ang1 at which this thread operates, 42 times at most
	ang1 = threadIdx.x * 0.15f;

	// Padded to 48 to allow efficient reduction by 24 threads without branching
	__shared__ float s_metrics[48];
	__shared__ float s_orientations[48];

	// Set the padding to 0, so it doesnt interfere.
	if (threadIdx.x < 6) {
		s_metrics[42 + threadIdx.x] = 0.0f;
	}

	// Each thread now computes one of the windows
	ang2 = ang1+pi_third > 2.0f*pi ? ang1-5.0f*pi_third : ang1+pi_third;
	sumX = sumY = 0.0f;

	// Find all the points that are inside the window
	// The x,y results computed above are now interpreted as points
	for (unsigned int k = 0; k < 109; k++) {
		ang = s_ang[k]; // Angle of vector to point

		// determine whether the point is within the window
		if (ang1 < ang2 && ang1 < ang && ang < ang2) {
			sumX += s_resX[k];
			sumY += s_resY[k];
		} else if (ang2 < ang1 &&
				((ang > 0.0f && ang < ang2) || (ang > ang1 && ang < 2.0f*pi) )) {
			sumX += s_resX[k];
			sumY += s_resY[k];
		}
	}

	// if the vector produced from this window is longer than all
	// previous vectors then this forms the new dominant direction
	s_metrics[threadIdx.x] = sumX*sumX + sumY*sumY;
	s_orientations[threadIdx.x] = getAngle(sumX, sumY);

	__syncthreads();

	/*
	 * The rest of this function finds the longest vector.
	 * The vector length is stored in metrics, while the
	 * corresponding orientation is stored in orientations
	 * with the same index.
	 */
//#pragma unroll 4
	for (int threadCount = 24; threadCount >= 3; threadCount /= 2) {
		if (threadIdx.x < threadCount) {
			if (s_metrics[threadIdx.x] < s_metrics[threadIdx.x + threadCount]) {
				s_metrics[threadIdx.x] = s_metrics[threadIdx.x + threadCount];
				s_orientations[threadIdx.x] = s_orientations[threadIdx.x + threadCount];
			}
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		float max = 0.0f, maxOrientation = 0.0f;
#pragma unroll 3
		for (int i = 0; i < 3; ++i) {
			if (s_metrics[i] > max) {
				max = s_metrics[i];
				maxOrientation = s_orientations[i];
			}
		}

		// assign orientation of the dominant response vector
		g_ipt->m_orientation = maxOrientation;
	}
}



/*
 * The following table is padded to 12x12 (up from 11x11) so it can be loaded
 * efficiently by the 4x4 threads.
 */
static const float gauss33[12][12] = {
	0.014614763f,0.013958917f,0.012162744f,0.00966788f,0.00701053f,0.004637568f,0.002798657f,0.001540738f,0.000773799f,0.000354525f,0.000148179f,0.0f,
	0.013958917f,0.013332502f,0.011616933f,0.009234028f,0.006695928f,0.004429455f,0.002673066f,0.001471597f,0.000739074f,0.000338616f,0.000141529f,0.0f,
	0.012162744f,0.011616933f,0.010122116f,0.008045833f,0.005834325f,0.003859491f,0.002329107f,0.001282238f,0.000643973f,0.000295044f,0.000123318f,0.0f,
	0.00966788f,0.009234028f,0.008045833f,0.006395444f,0.004637568f,0.003067819f,0.001851353f,0.001019221f,0.000511879f,0.000234524f,9.80224E-05f,0.0f,
	0.00701053f,0.006695928f,0.005834325f,0.004637568f,0.003362869f,0.002224587f,0.001342483f,0.000739074f,0.000371182f,0.000170062f,7.10796E-05f,0.0f,
	0.004637568f,0.004429455f,0.003859491f,0.003067819f,0.002224587f,0.001471597f,0.000888072f,0.000488908f,0.000245542f,0.000112498f,4.70202E-05f,0.0f,
	0.002798657f,0.002673066f,0.002329107f,0.001851353f,0.001342483f,0.000888072f,0.000535929f,0.000295044f,0.000148179f,6.78899E-05f,2.83755E-05f,0.0f,
	0.001540738f,0.001471597f,0.001282238f,0.001019221f,0.000739074f,0.000488908f,0.000295044f,0.00016243f,8.15765E-05f,3.73753E-05f,1.56215E-05f,0.0f,
	0.000773799f,0.000739074f,0.000643973f,0.000511879f,0.000371182f,0.000245542f,0.000148179f,8.15765E-05f,4.09698E-05f,1.87708E-05f,7.84553E-06f,0.0f,
	0.000354525f,0.000338616f,0.000295044f,0.000234524f,0.000170062f,0.000112498f,6.78899E-05f,3.73753E-05f,1.87708E-05f,8.60008E-06f,3.59452E-06f,0.0f,
	0.000148179f,0.000141529f,0.000123318f,9.80224E-05f,7.10796E-05f,4.70202E-05f,2.83755E-05f,1.56215E-05f,7.84553E-06f,3.59452E-06f,1.50238E-06f,0.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f
};



__constant__ float dc_gauss33[12][12];

/**	\brief Build SURF descriptor for an interest point
 *	\param g_ipoints device pointer to interest points
 *	\param upright compute upright SURF descriptor or not
 *
 *  This kernel builds the descriptors for an interest point.
 *  Execution configuration:
 *	  Thread block: { 5, 5, 16 }
 *	  Block grid  : { num_ipoints, 1 }
 *
 *  Overview:
 *  Each thread takes a haar X/Y sample and stores it in
 *  shared memory (rx,ry).
 *  For each subsquare (threadIdx.z is the subSquare id),
 *  the four sums (dx,dy,|dx|,|dy|) are built and stored in
 *  global memory (desc, outDesc).
 *  Then, one thread per sub square computes the squared
 *  length of a sub-square and stores it in global memory.
 */
__global__ void
buildSURFDescriptorsCUDA(SurfDescriptorPoint* g_ipoints, int upright)
{
	const int iPointIndex = blockIdx.x; // The index in the one-dimensional ipoint array
	const int samplePointXIndex = threadIdx.x; // The x-position of the sampling point within a sub-square, relative to the upper-left corner of the sub square
	const int samplePointYIndex = threadIdx.y; // The y-position of the sampling point within a sub-square, relative to the upper-left corner of the sub square
	const int subSquareId = threadIdx.z; // The index of the sub-square
	const int subSquareX = (subSquareId % 4); // X-Index of the sub-square
	const int subSquareY = (subSquareId / 4); // Y-Index of the sub-square

	SurfDescriptorPoint *g_ipt = g_ipoints + iPointIndex; // Pointer to the interest point processed by the current block
	int x = fRound(g_ipt->m_x);
	int y = fRound(g_ipt->m_y);
	float scale = g_ipt->m_scale;

	float * const g_desc = g_ipt->m_descriptor; // Pointer to the interest point descriptor
	float co, si; // Precomputed cos&sin values for the rotation of this interest point

	if (!upright) {
		co = cosf(g_ipt->m_orientation);
		si = sinf(g_ipt->m_orientation);
	}

	int roundedScale = fRound(scale);

	// Calculate the relative (to x,y) coordinate of sampling point
	int sampleXOffset = subSquareX * 5 - 10 + samplePointXIndex;
	int sampleYOffset = subSquareY * 5 - 10 + samplePointYIndex;

	// Get Gaussian weighted x and y responses
	float gauss = dc_gauss33[abs(sampleYOffset)][abs(sampleXOffset)];

	// Get absolute coords of sample point on the rotated axis
	int sampleX, sampleY;

	if (!upright) {
		sampleX = fRound(x + (-sampleXOffset*scale*si + sampleYOffset*scale*co));
		sampleY = fRound(y + ( sampleXOffset*scale*co + sampleYOffset*scale*si));
	} else {
		sampleX = fRound(x + sampleXOffset*scale);
		sampleY = fRound(y + sampleYOffset*scale);
	}

	// Take the sample (Haar wavelet response in x&y direction)
	float xResponse, yResponse;
	haarXY(sampleX, sampleY, roundedScale, &xResponse, &yResponse, gauss);

	// Calculate ALL x+y responses for the interest point in parallel
	__shared__ float s_rx[16][5][5];
	__shared__ float s_ry[16][5][5];

	if (!upright) {
		s_rx[subSquareId][samplePointXIndex][samplePointYIndex] = -xResponse*si + yResponse*co;
		s_ry[subSquareId][samplePointXIndex][samplePointYIndex] = xResponse*co + yResponse*si;
	} else {
		s_rx[subSquareId][samplePointXIndex][samplePointYIndex] = xResponse;
		s_ry[subSquareId][samplePointXIndex][samplePointYIndex] = yResponse;
	}

	// TODO: Can this be optimized? It waits for the results of ALL 400 threads, although they are
	// independent in blocks of 25! (Further work)
	__syncthreads(); // Wait until all 400 threads have written their results

	__shared__ float s_sums[16][4][5]; // For each sub-square, for the four values (dx,dy,|dx|,|dy|), this contains the sum over five values.
	__shared__ float s_outDesc[16][4]; // The output descriptor partitioned into 16 bins (one for each subsquare)

	// Only five threads per sub-square sum up five values each
	if (threadIdx.y == 0) {
		// Temporary sums
		float tdx = 0.0f, tdy = 0.0f, tmdx = 0.0f, tmdy = 0.0f;

		for (int sy = 0; sy < 5; ++sy) {
			tdx += s_rx[subSquareId][threadIdx.x][sy];
			tdy += s_ry[subSquareId][threadIdx.x][sy];
			tmdx += fabsf(s_rx[subSquareId][threadIdx.x][sy]);
			tmdy += fabsf(s_ry[subSquareId][threadIdx.x][sy]);
		}

		// Write out the four sums to the shared memory
		s_sums[subSquareId][0][threadIdx.x] = tdx;
		s_sums[subSquareId][1][threadIdx.x] = tdy;
		s_sums[subSquareId][2][threadIdx.x] = tmdx;
		s_sums[subSquareId][3][threadIdx.x] = tmdy;
	}

	__syncthreads(); // Wait until all threads have summed their values

	// Only four threads per sub-square can now write out the descriptor
	if (threadIdx.x < 4 && threadIdx.y == 0) {
		const float* s_src = s_sums[subSquareId][threadIdx.x]; // Pointer to the sum this thread will write out
		float out = s_src[0] + s_src[1] + s_src[2] + s_src[3] + s_src[4]; // Build the last sum for the value this thread writes out
		int subSquareOffset = (subSquareX + subSquareY * 4) * 4; // Calculate the offset in the descriptor for this sub-square
		g_desc[subSquareOffset + threadIdx.x] = out; // Write the value to the descriptor
		s_outDesc[subSquareId][threadIdx.x] = out; // Write the result to shared memory too, this will be used by the last thread to precompute parts of the length
	}

	__syncthreads();

	// One thread per sub-square now computes the length of the description vector for a sub-square and writes it to global memory
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		g_ipt->m_lengths[subSquareX][subSquareY] = s_outDesc[subSquareId][0] * s_outDesc[subSquareId][0]
			+ s_outDesc[subSquareId][1] * s_outDesc[subSquareId][1]
			+ s_outDesc[subSquareId][2] * s_outDesc[subSquareId][2]
			+ s_outDesc[subSquareId][3] * s_outDesc[subSquareId][3];
	}
}


/**	\brief Normalize SURF descriptors
 *	\param g_ipoints device pointer to interest points and their descriptors
 *
 *	Execution configuration:
 *	  Thread block: { 64, 1, 1 }
 *	  Block grid  : { num_ipoints, 1 }
 *
 *	This kernel normalizes the feature vector by using 64 threads per block
 *	and one block per interest point.
 *	First, 4 of the threads sum up the lengths of the 16 sub-vectors (one for
 *	each sub-square). Then, one thread sums up the resulting 4 values and
 *	calculates the inverse square root of the value. The last step of
 *	normalization is performed by all 64 threads.
 */
__global__ void
normalizeSURFDescriptorsCUDA(SurfDescriptorPoint* g_ipoints)
{
	SurfDescriptorPoint* g_ipoint = g_ipoints + blockIdx.x;
	__shared__ float s_sums[4];

	if (threadIdx.x < 4) {
		float* g_lengths = g_ipoint->m_lengths[threadIdx.x];
		s_sums[threadIdx.x] = g_lengths[0] + g_lengths[1] + g_lengths[2] + g_lengths[3];
	}

	__syncthreads();

	float len = rsqrtf(s_sums[0] + s_sums[1] + s_sums[2] + s_sums[3]);

	g_ipoint->m_descriptor[threadIdx.x] *= len;
}

void prepare2_buildSURFDescriptorsGPU(cudaArray *ca_intimg)
{
	integralImage.filterMode = cudaFilterModePoint; // We don't use interpolation
	integralImage.normalized = false; // DointegralImagen't normalize texture coordinates
	/* Clamping saves us some boundary checks */
	integralImage.addressMode[0] = cudaAddressModeClamp;
	integralImage.addressMode[1] = cudaAddressModeClamp;
	integralImage.addressMode[2] = cudaAddressModeClamp;

	cudaBindTextureToArray(integralImage, ca_intimg);
}

void prepare2_detectIpointOrientationsGPU(CudaImageBuffer<float> &img)
{
	// integralImage refers to the texture reference from
	// detectIpointOrientiationsCUDA.cu which is included above.
	integralImage.filterMode = cudaFilterModePoint; // We don't use interpolation
	integralImage.normalized = false; // Don't normalize texture coordinates
	/* Clamping saves us some boundary checks */
	integralImage.addressMode[0] = cudaAddressModeClamp;
	integralImage.addressMode[1] = cudaAddressModeClamp;
	integralImage.addressMode[2] = cudaAddressModeClamp;

    size_t offset;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D( &offset, integralImage, (float*)img, desc, Width(img), Height(img), Width(img)*sizeof(float)) ;
}

void
normalizeSURFDescriptorsGPU(SurfDescriptorPoint *d_ipoints, size_t num_ipoints)
{
	dim3 thread_block(64, 1, 1);
	dim3 block_grid(num_ipoints, 1);

	normalizeSURFDescriptorsCUDA<<<block_grid, thread_block>>>(d_ipoints);
    cudaThreadSynchronize();
//	cutilCheckMsg("normalizeSURFDescriptorsCUDA() execution failed");
}

void detectIpointOrientationsGPU(
	SurfDescriptorPoint *d_ipoints, size_t num_ipoints)
{
	dim3 thread_block(42, 1, 1);
	dim3 block_grid(num_ipoints, 1);

	detectIpointOrientiationsCUDA<<<block_grid, thread_block>>>(d_ipoints);
    cudaThreadSynchronize();
	//cutilCheckMsg("detectIpointOrientationsCUDA() execution failed");
}

bool computeDescriptors( CudaImageBuffer<float> &imgSat, DescriptorData & desc )
{
    bool upright = true; /* rotation invariant mode */
    prepare2_detectIpointOrientationsGPU(imgSat); // TODO : rename

    // Run the CUDA part
 	if (!upright)
    {
		detectIpointOrientationsGPU(desc.m_descPoints, desc.m_nbIPoints);
	}

    // Build surf descriptor GPU
    if(desc.m_descPoints == 0 || NbElements(desc) == 0 )
    {
        return false;
    }

    dim3 thread_block(5, 5, 16);
    dim3 block_grid( NbElements(desc), 1); // Mouarf ca marchera jamais

    buildSURFDescriptorsCUDA<<<block_grid, thread_block>>>(desc.m_descPoints, upright);
    cudaThreadSynchronize();
    //cutilCheckMsg("buildSURFDescriptorsCUDA() execution failed");

    normalizeSURFDescriptorsGPU(desc.m_descPoints, desc.m_nbIPoints);

    return true;
}


DescriptorData::DescriptorData()
:   m_descPoints(NULL),
    m_nbIPoints(0)
{
    cudaMemcpyToSymbol("dc_gauss33", gauss33, sizeof(gauss33));
	cudaMemcpyToSymbol("dc_gauss_lin", gauss_lin, sizeof(gauss_lin));
	cudaMemcpyToSymbol("dc_coord_x", coord_x, sizeof(coord_x));
	cudaMemcpyToSymbol("dc_coord_y", coord_y, sizeof(coord_y));
}

// Realloc surf descriptor points
bool DescriptorData::reallocPoints(unsigned int newSize)
{
    if(m_descPoints)
    {
        cudaFree(m_descPoints);
        m_descPoints = NULL;
        m_nbIPoints = 0;
    }
    
    m_nbIPoints = newSize;
    cudaMalloc((void**)&m_descPoints,m_nbIPoints*sizeof(SurfDescriptorPoint));
    cudaMemset(m_descPoints,0,m_nbIPoints*sizeof(SurfDescriptorPoint));
    return true;
}

DescriptorData::~DescriptorData()
{
    if(m_descPoints)
    {
        cudaFree(m_descPoints);
        m_descPoints = NULL;
        m_nbIPoints = 0;
    }
}