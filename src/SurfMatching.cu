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
    
    float *ldesc = leftDesc[leftIndex].m_descriptor;
    float *rdesc = rightDesc[rightIndex].m_descriptor;

    double sumSquareDiff = 0;
    #pragma unroll 8
    for( unsigned int i=0; i<64; i++ )
    {
        sumSquareDiff += (ldesc[i]-rdesc[i])*(ldesc[i]-rdesc[i]);
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
    const unsigned int threadSize = 8;
    dim3 threads(threadSize,threadSize);
    dim3 grid( iDivUp( nbLeftDesc, threadSize ), iDivUp( nbRightDesc, threadSize ) );
    computeSSD<<<grid, threads>>>( leftDesc.m_descPoints, rightDesc.m_descPoints, nbLeftDesc, nbRightDesc, (float*)ssdImage );
    cudaThreadSynchronize();

    // Select best matches
    // They will be transfered on the host
    MatchedPoints *matchedPoints_h; // Host
    MatchedPoints *matchedPoints_d; // Device
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc( (void**)&matchedPoints_h, nbRightDesc, cudaHostAllocMapped );
    cudaHostGetDevicePointer((void **)&matchedPoints_d, (void *)matchedPoints_h, 0);

    dim3 threads2( threadSize );
    dim3 grid2( iDivUp( nbRightDesc, threadSize ));
    selectKernel<<< grid2, threads2 >>>( (float*)ssdImage,
        nbLeftDesc, nbRightDesc, matchedPoints_d,
        leftDesc.m_descPoints, rightDesc.m_descPoints,
         (float)Width(imgSize), (float)Height(imgSize) );

    // at this point we get some pair of matching points
    // retrieve buffer on cpu memory
    cudaThreadSynchronize();

    // resize points buffers to full capacity
    leftPts.resize(leftPts.capacity());
    rightPts.resize(rightPts.capacity());

    // Disambiguation of matches
    unsigned int nbMatchedPoints = 0;
    copyPoints( leftPts, rightPts, nbRightDesc, matchedPoints_h, nbMatchedPoints  );
//    MatchedPointSet matchedPointSet;
//    std::pair<MatchedPointSet::iterator,bool> insertOk;
//    for( unsigned int i=0; i < nbRightDesc; i++ )
//    {
//        if( matchedPoints_h[i].m_ratio < 0.6 )
//        {
//            boost::hash<MatchedPoints> hasher;
//            std::size_t key = hasher(matchedPoints_h[i]);
//
//            insertOk = matchedPointSet.insert( std::make_pair(key,matchedPoints_h[i] ) );
//            if( insertOk.second == false )
//            {
//                if( (*insertOk.first).second.m_ratio > matchedPoints_h[i].m_ratio )
//                {
//                    matchedPointSet.erase(insertOk.first);
//                    matchedPointSet.insert( std::make_pair(key,matchedPoints_h[i] )  );
//                }
//            }
//        }
//    }
//
//    // copy results in pt1 and pt2
//    unsigned int nbMatchedPoints = matchedPointSet.size();
//    MatchedPointSet::iterator it = matchedPointSet.begin();
//    //std::cout << "Matched points = " << matchedPoints << std::endl;
//    for( unsigned int i=0; i < nbMatchedPoints; i++, ++it )
//    {
//        leftPts[i].x = it->second.m_lx;
//        leftPts[i].y = it->second.m_ly;
//        rightPts[i].x = it->second.m_rx;
//        rightPts[i].y = it->second.m_ry;
//    }

    // Resize to numbers of found values
    leftPts.resize(nbMatchedPoints);
    rightPts.resize(nbMatchedPoints);

    // Release result buffer
    cudaFreeHost(matchedPoints_h);
    releaseBuffer(ssdImage);

    return true;
}


