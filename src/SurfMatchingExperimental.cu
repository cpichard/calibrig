#include <SurfMatchingExperimental.h>
// Matching 2 VertexBuffer Object
#include <cudpp.h>


inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


__global__
void computeSSDExperimental(    SurfDescriptorPoint *leftDesc,
                                SurfDescriptorPoint *rightDesc,
                                uint nbLeft, uint nbRight,
                                float *result, float *resultTransposed )
{
    // Left and right indexes
    const int leftIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int rightIndex = blockDim.y * blockIdx.y + threadIdx.y;

    if( leftIndex >= nbLeft || rightIndex >= nbRight)
        return;

    float *ldesc = leftDesc[leftIndex].m_descriptor;
    float *rdesc = rightDesc[rightIndex].m_descriptor;

    double sumSquareDiff = 0;
    #pragma unroll 8
    for( unsigned int i=0; i<64; i++ )
    {
        sumSquareDiff += (ldesc[i]-rdesc[i])*(ldesc[i]-rdesc[i]);
    }

    const unsigned int pos = leftIndex + rightIndex*nbLeft;
    const unsigned int posTransposed = rightIndex + leftIndex*nbRight;

    //result[pos] = sqrtf(sumSquareDiff);
    result[pos] = (float)sumSquareDiff;
    resultTransposed[posTransposed] = (float)sumSquareDiff;

};


__global__
void divideBySSD(   float *minBuffer, float *minBufferTransposed,
                    uint nbLeft, uint nbRight,
                    float *result, float *resultTransposed )
{
    // Left and right indexes
    const int leftIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int rightIndex = blockDim.y * blockIdx.y + threadIdx.y;

    // TODO load in shared memory
    if( leftIndex >= nbLeft || rightIndex >= nbRight)
        return;

    // Positions in result buffers
    const unsigned int pos = leftIndex + rightIndex*nbLeft;
    const unsigned int posTransposed = rightIndex + leftIndex*nbRight;

    // Value of the SSD fort this thread
    const float resultValue = result[pos];

    // Value of horizontal scan
    const float minValue = minBuffer[ (nbLeft-1) + rightIndex*nbLeft ];
    if(resultValue == minValue)
    {
        result[pos] = FLT_MAX;
    }
    else
    {
        result[pos] = resultValue/minValue;
    }

    // Result transposed
    const float minValueTr = minBufferTransposed[(nbRight-1) + leftIndex*nbRight];

    if(resultValue == minValue)
    {
        resultTransposed[posTransposed] = FLT_MAX;
    }
    else
    {
        resultTransposed[posTransposed] = resultValue/minValueTr;
    }
}

// TODO
__global__
void selectKernel( float* ssdImage, float*imgMin, float*imgMinTransposed,
        int *selectBuffer, uint nbLeft, uint nbRight )
{
    // Left and right indexes
    const int leftIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int rightIndex = blockDim.y * blockIdx.y + threadIdx.y;

    // TODO load in shared memory
    if( leftIndex >= nbLeft || rightIndex >= nbRight)
        return;
    
    const unsigned int pos = leftIndex + rightIndex*nbLeft;

    // TODO

    selectBuffer[pos] = 0;
}

__global__
void copyLastColumnToLigne( int*scanSelection, int*ligneSelection,
                        uint nbLeft, uint nbRight )
{

}

//
// TEST
//
// Retrieve values of matching points on cpu
bool computeMatchingExperimental( DescriptorData &leftDesc, DescriptorData &rightDesc,
    vector<CvPoint2D32f> &leftPts, vector<CvPoint2D32f> &rightPts,
    UInt2 &imgSize )
{
    // Creates a buffer to store results of all comparisons with ssd
    const unsigned int nbLeftDesc = NbElements(leftDesc);
    const unsigned int nbRightDesc = NbElements(rightDesc);

    if(nbLeftDesc < 8 || nbRightDesc < 8 )
    {
        return false;
    }

    // Allocated 2 buffers that will contain the ssd
    UInt2 ssdImageSize( nbLeftDesc, nbRightDesc );
    UInt2 ssdImageSizeTr = SwapXY(ssdImageSize);
    LocalCudaImageBuffer<float> ssdImage( ssdImageSize );
    LocalCudaImageBuffer<float> ssdImageTransposed( ssdImageSizeTr );

    // Launch comparison kernel
    const unsigned int threadSize = 8;
    dim3 threads(threadSize,threadSize);
    dim3 grid( iDivUp( nbLeftDesc, threadSize ), iDivUp( nbRightDesc, threadSize ) );
    computeSSDExperimental<<<grid, threads>>>
        (   leftDesc.m_descPoints, rightDesc.m_descPoints,
            nbLeftDesc, nbRightDesc,
            (float*)ssdImage, (float*)ssdImageTransposed );
    cudaThreadSynchronize();
    // Buffers for the min prefix scan results
    LocalCudaImageBuffer<float> imgMin( ssdImageSize );
    LocalCudaImageBuffer<float> imgMinTransposed( ssdImageSizeTr );

    // Need cudpp to create min images
    CUDPPConfiguration confForward;
	confForward.op = CUDPP_MIN;
	confForward.datatype = CUDPP_FLOAT;
	confForward.algorithm = CUDPP_SCAN;
	confForward.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

    // Allocate plan for rows integration
    CUDPPHandle integrateXForward;
    if( cudppPlan( &integrateXForward, confForward, 
        Width(imgMin), Height(imgMin),
        imgMin.m_pitchInElements ) != CUDPP_SUCCESS )
    {
        std::cout << "Failed to allocate cudpp plan" << std::endl;
        return false;
    }

    // Allocate plan for cols integration
    CUDPPHandle integrateYForward;
    if( cudppPlan( &integrateYForward, confForward, 
        Width(imgMinTransposed), Height(imgMinTransposed),
        imgMinTransposed.m_pitchInElements ) != CUDPP_SUCCESS )
    {
        std::cout << "Failed to allocate cudpp plan" << std::endl;
        return false;
    }

    // Horizontal integral min image
    cudppMultiScan( integrateXForward, (float*)imgMin,
        (float*)ssdImage,
        Width(imgMin), Height(imgMin) );

    // Vertical integral max image
    cudppMultiScan( integrateYForward, (float*)imgMinTransposed,
        (float*)ssdImageTransposed,
        Width(imgMinTransposed), Height(imgMinTransposed) );

    // Divide SSD with the max value or set FLT_MAX if it's the max value
    // We use ssdImage and ssdImageTransposed as result buffers as we don't need
    // them anymore
    divideBySSD<<< grid, threads >>>(
                (float*)imgMin, (float*)imgMinTransposed,
                nbLeftDesc, nbRightDesc,
                (float*)ssdImage,(float*)ssdImageTransposed);

    cudaThreadSynchronize();

    // Min of the found division
    // Horizontal integral min image
    cudppMultiScan( integrateXForward, 
        (float*)ssdImage, (float*)imgMin,
        Width(imgMin), Height(imgMin) );

    // Vertical integral min image
    cudppMultiScan( integrateYForward, 
        (float*)ssdImageTransposed, (float*)imgMinTransposed,
        Width(imgMinTransposed), Height(imgMinTransposed) );

    // Select kernel
    LocalCudaImageBuffer<int> selectBuffer( ssdImageSize );

    // Launch select kernel
    selectKernel<<<grid, threads>>>( 
        (float*)ssdImage, (float*)imgMin, (float*)imgMinTransposed,
        (int*)selectBuffer, nbLeftDesc, nbRightDesc );
    
    cudaThreadSynchronize();
    
    // Select kernel

    // Scan
    LocalCudaImageBuffer<int> scanSelection( ssdImageSize );
    CUDPPConfiguration confScanPos;
	confScanPos.op = CUDPP_ADD;
	confScanPos.datatype = CUDPP_INT;
	confScanPos.algorithm = CUDPP_SCAN;
	confScanPos.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    CUDPPHandle scanPositionHandle;
    if( cudppPlan( &scanPositionHandle, confScanPos,
        Width(selectBuffer), Height(selectBuffer),
        selectBuffer.m_pitchInElements ) != CUDPP_SUCCESS )
    {
        std::cout << "Failed to allocate cudpp plan" << std::endl;
        return false;
    }
    
    cudppMultiScan( scanPositionHandle,
        (int*)scanSelection, (int*)selectBuffer,
        Width(selectBuffer), Height(selectBuffer) );
    
    
    // Alloc a ligne and copy last column to this ligne
    // Scan ligne
    UInt2 ligneSize(nbRightDesc,1);
    LocalCudaImageBuffer<int> ligneSelection( ligneSize );
    LocalCudaImageBuffer<int> lignePositions( ligneSize );

    dim3 threadsLigne(threadSize);
    dim3 gridLigne( iDivUp( nbRightDesc, threadSize ) );
    copyLastColumnToLigne<<< gridLigne,threadsLigne>>>
            ( (int*)scanSelection, (int*)ligneSelection,
                nbLeftDesc, nbRightDesc
            );

    cudaThreadSynchronize();
    // Scan
    // Allocate a result buffer with the size found in of the ligne
    CUDPPHandle scanLigneHandle;
    if( cudppPlan( &scanLigneHandle, confScanPos,
        nbRightDesc, 1,
        ligneSelection.m_pitchInElements ) != CUDPP_SUCCESS )
    {
        std::cout << "Failed to allocate cudpp plan" << std::endl;
        return false;
    }
    
    cudppScan(
        scanLigneHandle,
        (int*)lignePositions,
        (int*)ligneSelection,
        nbRightDesc);

    // Read number of elements
    //uint nbMatchs = 0; // TODO

    // Copy results from the 



    cudaThreadSynchronize();

    // TODO
    // Scan backward
    // launch kernel to find the best2 match


    // copy results in pt1 and pt2
    //unsigned int matchedPoints = 0;
    
    // Allocate ligne of results


//    // resize to full capacity
//    leftPts.resize(minMatches);
//    rightPts.resize(minMatches);
//
//    for( unsigned int i=0; i < minMatches; i++ )
//    {
//        if( matchedPoints_h[i].ratio < 0.6 && matchedPoints_h[i].ratio != 0 )
//        {
//            leftPts[matchedPoints].x = matchedPoints_h[i].lx;
//            leftPts[matchedPoints].y = matchedPoints_h[i].ly;
//            rightPts[matchedPoints].x = matchedPoints_h[i].rx;
//            rightPts[matchedPoints].y = matchedPoints_h[i].ry;
//
//            matchedPoints++;
//        }
//    }
//
//    // resize to numbers of found values
//    leftPts.resize(matchedPoints);
//    rightPts.resize(matchedPoints);
//
//    // Release result buffer
//    cudaFreeHost(matchedPoints_h);
//    releaseBuffer(ssdImage);

    cudppDestroyPlan(scanLigneHandle);
    cudppDestroyPlan(scanPositionHandle);
    cudppDestroyPlan(integrateYForward);
    cudppDestroyPlan(integrateXForward);
    
    return true;
}