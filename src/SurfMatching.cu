#include <SurfMatching.h>
// Matching 2 VertexBuffer Object
#include <cudpp.h>


inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
// Launch comparison Kernel
// Launch selection kernel

__global__
void computeSSD( SurfDescriptorPoint *leftDesc, SurfDescriptorPoint *rightDesc, uint nbLeft, uint nbRight, float *result )
{
    // Left and right indexes
    const int leftIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int rightIndex = blockDim.y * blockIdx.y + threadIdx.y;

    if( leftIndex >= nbLeft || rightIndex >= nbRight)
        return;
    
    // Due to the way SurfDescriptor is structured, 
    // the memory isn't read efficiently
    float * const ldesc = leftDesc[leftIndex].m_descriptor;
    float * const rdesc = rightDesc[rightIndex].m_descriptor;

    float sumSquareDiff = 0;
    #pragma unroll 64
    for( unsigned int i=0; i<64; i++ )
    {
        const float l = ldesc[i];
        const float r = rdesc[i];
        sumSquareDiff += (l-r)*(l-r); 
    }

    const unsigned int pos = leftIndex + rightIndex*nbLeft;
    result[pos] = (float)sumSquareDiff;
};


__global__
void selectKernel( float* ssdImage, uint nbLeftDesc, uint nbRightDesc, 
    MatchedPoints *matchedPoints, SurfDescriptorPoint *leftDesc, SurfDescriptorPoint *rightDesc,
    float width, float height)
{
    const int rightIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if(rightIndex >= nbRightDesc )
        return;
    
    float best1 = FLT_MAX;
    float best2 = FLT_MAX;
    float ssd;
    uint rBest=0;
    uint pos = rightIndex*nbLeftDesc; // beginning of values
    
    // NOTE that if there is more left desc that right desc
    // the number of total matches could be wrong 
    // with this method
    for(unsigned int i = 0; i < nbLeftDesc; i++ )
    {
        ssd = ssdImage[pos];
        if( ssd < best1 )
        {
            rBest = i;
            best2 = best1;
            best1 = ssd;
        }
        else if( ssd < best2 )
        {
            best2 = ssd;
        }
        pos++;
    }
    
    const float imgRatio = (height/width);
    matchedPoints[rightIndex].m_lx = leftDesc[rBest].m_x/width -0.5;
    matchedPoints[rightIndex].m_ly = (leftDesc[rBest].m_y/height -0.5)*imgRatio;
    matchedPoints[rightIndex].m_rx = rightDesc[rightIndex].m_x/width -0.5;
    matchedPoints[rightIndex].m_ry = (rightDesc[rightIndex].m_y/height -0.5)*imgRatio;

    if( best2 != 0.f)
        matchedPoints[rightIndex].m_ratio = best1/best2;
    else
        matchedPoints[rightIndex].m_ratio = FLT_MAX;
}


// Retrieve values of matching points on cpu
bool computeMatching( DescriptorData &leftDesc, DescriptorData &rightDesc, 
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

    UInt2 ssdImageSize( nbLeftDesc, nbRightDesc );
    CudaImageBuffer<float> ssdImage;
    if( ! allocBuffer( ssdImage, ssdImageSize ) )
    {
        std::cout << "unable to create result buffer" << std::endl;
        return false;
    }

    // Launch comparison kernel
    // Compute ssd between all descriptors
    const unsigned int threadSize = 16;
    dim3 threads(threadSize,threadSize);
    dim3 grid( iDivUp( nbLeftDesc, threadSize ), iDivUp( nbRightDesc, threadSize ) );
    computeSSD<<<grid, threads>>>( leftDesc.m_descPoints, rightDesc.m_descPoints, nbLeftDesc, nbRightDesc, (float*)ssdImage );
    checkLastError();
    //cudaDeviceSynchronize();
    checkLastError();

    // Select best matches
    // They will be transfered on the host
    MatchedPoints *matchedPoints_h=NULL; // Host
    MatchedPoints *matchedPoints_d=NULL; // Device
    //cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc( (void**)&matchedPoints_h, nbRightDesc*sizeof(MatchedPoints), cudaHostAllocMapped );
    assert(matchedPoints_h!=NULL);
    checkLastError(); 
    cudaHostGetDevicePointer((void **)&matchedPoints_d, (void *)matchedPoints_h, 0);
    assert(matchedPoints_d!=NULL); 
    checkLastError();

    dim3 threads2( threadSize );
    dim3 grid2( iDivUp( nbRightDesc, threadSize ));
    selectKernel<<< grid2, threads2 >>>( (float*)ssdImage,
        nbLeftDesc, nbRightDesc, matchedPoints_d,
        leftDesc.m_descPoints, rightDesc.m_descPoints,
         (float)Width(imgSize), (float)Height(imgSize) );
    checkLastError();
    // at this point we get some pair of matching points
    // retrieve buffer on cpu memory
    cudaDeviceSynchronize();

    // resize points buffers to full capacity
    leftPts.resize(leftPts.capacity());
    rightPts.resize(rightPts.capacity());

    // Disambiguation of matches
    unsigned int nbMatchedPoints = 0;
    copyPoints( leftPts, rightPts, nbRightDesc, matchedPoints_h, nbMatchedPoints  );
    checkLastError();
    // Resize to numbers of found values
    leftPts.resize(nbMatchedPoints);
    rightPts.resize(nbMatchedPoints);

    // Release result buffer
    cudaFreeHost(matchedPoints_h);
    releaseBuffer(ssdImage);
    checkLastError();
    return true;
}


