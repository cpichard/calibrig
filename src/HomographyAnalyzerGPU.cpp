#include "HomographyAnalyzerGPU.h"
#include "ComputationDataGPU.h"
#include "CudaUtils.h"

#include <cuda_runtime.h>

#include "SurfHessian.h"
#include "SurfNonMax.h"
#include "SurfDescriptor.h"
#include "SurfMatching.h"

#include "ImageProcessing.h"

HomographyAnalyzerGPU::HomographyAnalyzerGPU()
: StereoAnalyzer()
, m_hessianData()
, m_rightDescriptors()
, m_leftDescriptors()
, m_result(NULL)
, m_histoRange(20)
, m_sentThreshold(500)
, m_toDispose(NULL)
{}

HomographyAnalyzerGPU::~HomographyAnalyzerGPU()
{
    if(m_result)
        delete m_result;
    freeImages();
}

void HomographyAnalyzerGPU::updateRightImageWithSDIVideo( ImageGL &videoPBO )
{
    boost::mutex::scoped_lock sl(m_imgMutex);
    convertYCbYCrToY(videoPBO, m_imgRight ); 
    checkLastError();

    // Copy RGBA image [0,255] in float buffer [0,1] allocated by cuda
    convertRGBAToCudaBufferY( m_imgRight, m_satRightImage );
    checkLastError();
    m_rightImageIsNew = true;
    cudaDeviceSynchronize();
}

void HomographyAnalyzerGPU::updateLeftImageWithSDIVideo ( ImageGL &videoPBO )
{
    boost::mutex::scoped_lock sl(m_imgMutex);
    convertYCbYCrToY(videoPBO, m_imgLeft ); 
    checkLastError();
    
    // Copy RGBA image [0,255] in float buffer [0,1] allocated by cuda
    convertRGBAToCudaBufferY( m_imgLeft, m_satLeftImage );
    checkLastError();
    m_leftImageIsNew = true;
    cudaDeviceSynchronize();
}

void HomographyAnalyzerGPU::acceptCommand( const Command &command )
{
    if( command.m_action == "OCVTHRESHOLD" )
    {
        // Still TODO normalise value 0 to 100
        m_sentThreshold = (float)command.m_value;
        m_hessianData.m_thres = ((float)command.m_value)/100000.f;
    }
    if( command.m_action == "HISTOGRAMRANGE")
    {
        m_histoRange = command.m_value;
    }
}

void HomographyAnalyzerGPU::analyse()
{
    if( m_imgWidth == 0 || m_imgHeight == 0 )
        return;

    if(m_toDispose)
    {
        delete m_toDispose;
        m_toDispose=NULL;
    }

    if( m_result != NULL )
    {
        return;
    }
    computeSurfDescriptors( m_satRightImage, m_rightDescriptors );
    checkLastError();
    computeSurfDescriptors( m_satLeftImage, m_leftDescriptors );
    checkLastError();

    collectPoints( m_leftDescriptors, m_leftPoints, Size(m_imgLeft) );
    checkLastError();
    collectPoints( m_rightDescriptors, m_rightPoints, Size(m_imgRight) );
    checkLastError();

    m_result = new ComputationDataGPU( m_imgLeft, m_imgRight, m_rightPoints, m_leftPoints );

    computeMatching( m_leftDescriptors, m_rightDescriptors,
       m_result->m_leftMatchedPts, m_result->m_rightMatchedPts,
       Size(m_imgRight));
    checkLastError();
    cudaDeviceSynchronize();
    m_leftImageIsNew = m_rightImageIsNew = false;

    Deformation &d = m_result->m_d;
    d.m_nbMatches = m_result->m_leftMatchedPts.size();
    d.m_nbPtsRight = NbElements(m_rightPoints);
    d.m_nbPtsLeft = NbElements(m_leftPoints);
    d.m_mode = "GPU";

    // Compute homography descriptors for left and right images
    if( d.m_nbMatches > 8 )
    {
        CvMat _pt1, _pt2;
        _pt1 = cvMat(1, d.m_nbMatches, CV_32FC2, &m_result->m_leftMatchedPts[0] );
        _pt2 = cvMat(1, d.m_nbMatches, CV_32FC2, &m_result->m_rightMatchedPts[0] );
        CvMat _h = cvMat(3, 3, CV_64F, d.m_h1);
        if( cvFindHomography( &_pt1, &_pt2, &_h, CV_LMEDS, 1.0 ) )
        {
            // Simple decomposition
            double Tx = cvmGet(&_h,0,2);
            double Ty = cvmGet(&_h,1,2);
            double rot =  asin(cvmGet(&_h,1,0));
            double scale =  cvmGet(&_h,0,0)/cos(rot);

            //std::cout << "Tx = " << Tx << " , Ty = " << Ty << std::endl;

            d.m_tx = Tx*(float)m_imgWidth;
            d.m_ty = -Ty*(float)m_imgWidth;
            d.m_scale = scale;
            d.m_rot = rot*180.f/3.14; // approx
            d.m_succeed = true;
            d.m_nbPtsRight = NbElements(m_rightPoints);
            d.m_nbPtsLeft = NbElements(m_leftPoints);
            d.m_nbMatches = m_result->m_leftMatchedPts.size();
            computeDisparity();
            
            m_result->m_thresholdUsed = m_sentThreshold; 
        }
    }
}

void HomographyAnalyzerGPU::resizeImages( UInt2 imgSize )
{
    freeImages();
    allocImages( imgSize );

    m_hessianData.resizeImages(imgSize);
}

void HomographyAnalyzerGPU::allocImages( UInt2 imgSize )
{
    m_imgWidth = Width(imgSize);
    m_imgHeight = Height(imgSize);

    if( m_imgWidth == 0 || m_imgHeight == 0 )
        return;

    if( ! allocBufferAndTexture( m_imgRight, imgSize ) )
        return;
    if( ! allocBufferAndTexture( m_imgLeft, imgSize ) )
        return;

    if( ! allocBufferAndTexture( m_warpedTmp, imgSize ) )
        return;

    if( ! allocBuffer( m_satLeftImage, imgSize ) )
        return;
    if( ! allocBuffer( m_satRightImage, imgSize ) )
        return;

    allocBuffer( m_hesImage, imgSize );

    //
    m_hessianData.allocImages(imgSize);
    checkLastError();

    allocBuffer( m_rightPoints, m_hessianData.capacity() );
    checkLastError();
    allocBuffer( m_leftPoints,  m_hessianData.capacity() );
    checkLastError();

}

void HomographyAnalyzerGPU::freeImages()
{
    m_hessianData.freeImages();
    releaseBuffer( m_hesImage );
    releaseBuffer( m_satLeftImage );
    releaseBuffer( m_satRightImage );
    releaseBufferAndTexture( m_warpedTmp );
    releaseBuffer(m_leftPoints);
    releaseBuffer(m_rightPoints);
    releaseBufferAndTexture( m_imgLeft );
    releaseBufferAndTexture( m_imgRight );
    checkLastError();
}

void HomographyAnalyzerGPU::computeSurfDescriptors( CudaImageBuffer<float> &satImage, DescriptorData &descriptorsData )
{
    // Create integral image
    convertToIntegral( satImage );
    cudaDeviceSynchronize();
    checkLastError();

    // Compute hessian and determinants
    computeHessianDet( satImage, m_hesImage, m_hessianData );
    cudaDeviceSynchronize();
    checkLastError();

    // Find position of maximum values in determinant image
    computeNonMaxSuppression( m_hesImage, m_hessianData );
    cudaDeviceSynchronize();
    checkLastError();

    // Copy hessian points in descriptors
    collectHessianPoints( m_hessianData, descriptorsData );
    cudaDeviceSynchronize();
    checkLastError();

    // Compute Surf descriptors
    computeDescriptors( satImage, descriptorsData );
    cudaDeviceSynchronize();
    checkLastError();
}

ComputationData * HomographyAnalyzerGPU::acquireLastResult()
{
    ComputationData *ret = NULL;
    ret = m_result;
    m_result = NULL;
    return ret;
}

void HomographyAnalyzerGPU::disposeResult(ComputationData *toDispose)
{
    m_toDispose = (ComputationDataGPU*)toDispose;
}


void HomographyAnalyzerGPU::computeDisparity()
{
    if( m_result && Width(m_imgLeft) > 0 )
    {
        Deformation &d = m_result->m_d;

        for( int i = 0; i < d.s_histogramBinSize; i++ )
        {
            d.m_hdisp[i] = 0;
            d.m_vdisp[i] = 0;
        }

        for( int i = 0; i < d.m_nbMatches; i++ )
        {
            const float distH = m_result->m_rightMatchedPts[i].x - m_result->m_leftMatchedPts[i].x;
            const float distV = m_result->m_rightMatchedPts[i].y - m_result->m_leftMatchedPts[i].y;

            // TODO : range of the histogram
            const float range = float(m_histoRange)/Width(m_imgLeft); // 20 pixels wide
            int indexH = (int)( 0.5 + ((distH + 0.5*range)/range)*(float)d.s_histogramBinSize);
            int indexV = (int)( 0.5 + ((distV + 0.5*range)/range)*(float)d.s_histogramBinSize);

            if( indexH <= 0)
                indexH = 0;

            if( indexH >= d.s_histogramBinSize )
                indexH = d.s_histogramBinSize-1;

            d.m_hdisp[indexH] = d.m_hdisp[indexH] + 1;

            if( indexV <= 0)
                indexV = 0;

            if( indexV >= d.s_histogramBinSize )
                indexV = d.s_histogramBinSize-1;

            d.m_vdisp[indexV] = d.m_vdisp[indexV] + 1;

        }
    }
}
