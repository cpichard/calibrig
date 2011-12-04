
#include "SurfHessian.h"

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 8 


// The following code was taken and adapted from SURFGPU :
/* SURFGPU
 * Copyright (C) 2009-2010 Andre Schulz, Florian Jung, Sebastian Hartte,
 *						   Daniel Trick, Christan Wojek, Konrad Schindler,
 *						   Jens Ackermann, Michael Goesele
 * Copyright (C) 2008-2009 Christopher Evans <chris.evans@irisys.co.uk>, MSc University of Bristol
 * */

/////////////////////////////////////////////////////////
// Definitions and constants
/////////////////////////////////////////////////////////

__constant__ int dc_lobe_cache_unique[10];

__device__ float BoxIntegral(float *data, int width, int height, size_t widthStep,
							 int row, int col, int rows, int cols);

/////////////////////////////////////////////////////////
// Kernel Code (Device)
/////////////////////////////////////////////////////////

/** \brief Compute determinants
 *	\param g_img device pointer to integral image
 *	\param g_det device pointer to save resulting determinants to
 *	\param i_width integral image width
 *	\param i_height integral image height
 *	\param i_widthStep number of elements in a row of the integral image
 *	\param intervals number of intervals to compute
 *	\param o octave to compute
 *	\param step step size in X/Y direction in pixels
 *	\param border number of border pixels
 *
 *	Computation is done pixel-wise where one thread computes the determinant of
 *	one pixel.
 *
 *	Recommended execution configuration:
 *	  Thread block: { 16, 8 }
 *	  Block grid  : { ceil(i_width / block.x), ceil(i_height, block.y) }
 */
__global__
void buildDetCUDA(float *g_img, float *g_det,
			 int i_width, int i_height, size_t i_widthStep,
			 int intervals, int o, int step, int border)
{
	// Get current interval
	const int interval_size = gridDim.x / intervals;
	// For octaves > 0, we only compute the higher 2 intervals.
	const int i = blockIdx.x / interval_size + (o > 0) * 2;

	// Get current column and row
	const int c = ((((blockIdx.x % interval_size) * blockDim.x) + threadIdx.x) * step) + border;
	const int r = ((blockIdx.y * blockDim.y + threadIdx.y) * step) + border;

	if (c >= i_width - border || r >= i_height - border)
		return;

	// Construct filter
	const int l = dc_lobe_cache_unique[o * intervals + i];
	const int w = 3 * l;
	const int b = w / 2;

	// Caluclate box integrals
	float Dxx = BoxIntegral(g_img, i_width, i_height, i_widthStep, r - l + 1, c - b, 2 * l - 1, w)
			  - BoxIntegral(g_img, i_width, i_height, i_widthStep, r - l + 1, c - l / 2, 2 * l - 1, l) * 3.0f;
	float Dyy = BoxIntegral(g_img, i_width, i_height, i_widthStep, r - b, c - l + 1, w, 2 * l - 1)
			  - BoxIntegral(g_img, i_width, i_height, i_widthStep, r - l / 2, c - l + 1, l, 2 * l - 1) * 3.0f;
	float Dxy = BoxIntegral(g_img, i_width, i_height, i_widthStep, r - l, c + 1, l, l)
			  + BoxIntegral(g_img, i_width, i_height, i_widthStep, r + 1, c - l, l, l)
			  - BoxIntegral(g_img, i_width, i_height, i_widthStep, r - l, c - l, l, l)
			  - BoxIntegral(g_img, i_width, i_height, i_widthStep, r + 1, c + 1, l, l);

	// Normalise the filter responses with respect to their size
	float inverse_area = 1.0f / (w * w);
	Dxx *= inverse_area;
	Dyy *= inverse_area;
	Dxy *= inverse_area;

	// Get the sign of the laplacian
	const float lap_sign = (Dxx + Dyy >= 0.0f ? 1.0f : -1.0f);

	// Get the determinant of hessian response
	float determinant = (Dxx * Dyy - 0.81f * Dxy * Dxy);
	unsigned int cur_intvl = o * intervals + i;
	unsigned int save_idx = cur_intvl * i_width * i_height + (r * i_width + c);
	g_det[save_idx] = determinant < 0.0f ? 0.0f : lap_sign * determinant;
}

//-------------------------------------------------------

/////////////////////////////////////////////////////////
// Device functions
/////////////////////////////////////////////////////////

//! Computes the sum of pixels within the rectangle specified by the top-left start
//! co-ordinate (row, col) and size (rows, cols).
__device__
float BoxIntegral(float *data, int width, int height, size_t widthStep,
	int row, int col, int rows, int cols)
{
	// The subtraction by one for row/col is because row/col is inclusive.
	const int r1 = min(row, height) - 1;
	const int c1 = min(col, width) - 1;
	const int r2 = min(row + rows, height) - 1;
	const int c2 = min(col + cols, width) - 1;

	const float A = data[r1 * widthStep + c1];
	const float B = data[r1 * widthStep + c2];
	const float C = data[r2 * widthStep + c1];
	const float D = data[r2 * widthStep + c2];

	return max(0.f, A - B - C + D);
}

// TODO : REFACTOR !!
bool computeHessianDet( CudaImageBuffer<float> &img, CudaImageBuffer<float> &det, HessianData &params )
{
	// Calculate step size
	int step = params.m_initSample;

	// Get border size
	int border = params.m_borderCache[0];

    // REFACTOR : it should be in HessianData
    unsigned int total_num_intervals = params.m_intervals + (params.m_octaves - 1) * params.m_intervals / 2;
    size_t det_size = total_num_intervals * params.m_width * params.m_height * sizeof(float);
    cudaMemset(params.m_det, 0, det_size);

	// Calculate grid size
	int steps_x = ( Width(img) - (2 * border)) / step;
	int steps_y = ( Height(img) - (2 * border)) / step;
	dim3 block(16, 4, 6);
	dim3 grid((steps_x + block.x - 1) / block.x,
			  (steps_y + block.y - 1) / block.y);

	// Launch kernel
#ifdef DEBUG
	printf("Call determinant kernel: octave %d, steps %dx%d, border %d, grid dim %dx%d, block dim %dx%dx%d\n",
			0, steps_x, steps_y, border, grid.x, grid.y, block.x, block.y, block.z);
#endif

    cudaThreadSynchronize();
    //	TODO : cutilCheckMsg("buildDetCUDA_smem_bf() execution failed");

	// For octaves > 0, we only compute the higher 2 intervals.
	int intervals = 2; // dans le code original intervals est overridde int intervals = 2
    // o was 1 and the function buildDetCUDA_smem_bf was enabled
    // buildDetCUDA_smem_bf seems to be not adapted on a quaddro 4000
	for (int o = 0; o < params.m_octaves; o++)
    {
		// Calculate step size
		step = params.m_initSample * (1 << o);

		// Get border size
		border = params.m_borderCache[o];

		// Calculate grid size
		steps_x = (Width(img) - (2 * border)) / step;
		steps_y = (Height(img) - (2 * border)) / step;
		dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 grid((steps_x + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (steps_y + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
		grid.x *= intervals;

		// Launch kernel
#ifdef DEBUG
		printf("Call determinant kernel: octave %d, steps %dx%d, border %d, grid dim %dx%d, block dim %dx%d\n",
				o, steps_x, steps_y, border, grid.x, grid.y, block.x, block.y);
#endif
		buildDetCUDA<<<grid, block>>>((float *)img, params.m_det,
			Width(img), Height(img), Width(img),
			intervals, o, step, border);
        cudaThreadSynchronize();
		//cutilCheckMsg("buildDetCUDA() execution failed");
	}

    return false;
}

const int HessianData::m_lobeCache [] = {3,5,7,9,5,9,13,17,9,17,25,33,17,33,49,65};
const int HessianData::m_lobeCacheUnique [] = {3,5,7,9,13,17,25,33,49,65};
const int HessianData::m_lobeMap [] = {0,1,2,3,1,3,4,5,3,5,6,7,5,7,8,9};
const int HessianData::m_borderCache [] = {14,26,50,98};
const int HessianData::IMG_SIZE_DIVISOR = 160;

HessianData::HessianData()
:   m_octaves(4),
    m_intervals(4),
    m_capacity(0),
    m_initSample(2),
    m_det(0),
    m_detPitch(0), // TODO
    m_thres(0.0004f), // init = 0.0004
    m_dPoints(0),
    m_width(0),
    m_height(0)
{
    // 
    cudaMemcpyToSymbol("dc_lobe_cache_unique", m_lobeCacheUnique, sizeof(m_lobeCacheUnique));
}

HessianData::~HessianData()
{
    freeImages();
    // TODO remove constant dc_lobe_cache_unique ?
}


void HessianData::resizeImages( UInt2 imgSize )
{
    freeImages();
    allocImages( imgSize );
}

void HessianData::allocImages( UInt2 imgSize )
{
    m_width = Width(imgSize);
    m_height = Height(imgSize);
    m_capacity = m_width * m_height / IMG_SIZE_DIVISOR;
	size_t points_size = m_capacity * sizeof(HessianPoint);

    // Calculate sizes
    unsigned int total_num_intervals = m_intervals + (m_octaves - 1) * m_intervals / 2;
    size_t det_size = total_num_intervals * m_width * m_height * sizeof(float);

    // TODO safe malloc
    cudaMalloc((void**)&m_det, det_size);
    cudaMalloc((void**)&m_dPoints, points_size);
    cudaMemset(m_det, 0, det_size);
    cudaMemset(m_dPoints, 0, points_size);
}

void HessianData::freeImages()
{
    if(m_dPoints)
    {
        cudaFree(m_dPoints) ;
        m_dPoints = NULL;
    }
    if(m_det)
    {
        cudaFree(m_det);
        m_det=NULL;
    }
}
