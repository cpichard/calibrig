#include "StereoAnalyzerGPU.h"
#include "ComputationDataGPU.h"
#include "CudaUtils.h"

#include <cuda_runtime.h>

#include "SurfHessian.h"
#include "SurfNonMax.h"
#include "SurfDescriptor.h"
#include "SurfMatching.h"

#include "ImageProcessing.h"

StereoAnalyzerGPU::StereoAnalyzerGPU()
: StereoAnalyzer()
, m_hessianData()
, m_rightDescriptors()
, m_leftDescriptors()
, m_result(NULL)
, m_histoRange(20)
, m_sentThreshold(500)
, m_maxNumberOfPoints(8000)
{}

StereoAnalyzerGPU::~StereoAnalyzerGPU()
{
    if(m_result)
        delete m_result;
    m_leftMatchedPts.clear();
    m_rightMatchedPts.clear();
    freeImages();
}

void StereoAnalyzerGPU::updateRightImageWithSDIVideo( ImageGL &videoPBO )
{
    m_imgMutex.lock();
    // TODO : save image right and left here
    convertYCbYCrToY( videoPBO, m_imgRight );
    checkLastError();
    m_rightImageIsNew = true;
    m_imgMutex.unlock();
    computeSurfDescriptors( m_imgRight, m_rightDescriptors );
    checkLastError();
    collectPoints( m_rightDescriptors, m_rightPoints, Size(m_imgRight) );
    checkLastError();
}

void StereoAnalyzerGPU::updateLeftImageWithSDIVideo ( ImageGL &videoPBO )
{
    m_imgMutex.lock();
    convertYCbYCrToY( videoPBO, m_imgLeft );
    checkLastError();
    m_leftImageIsNew = true;
    m_imgMutex.unlock();
    computeSurfDescriptors( m_imgLeft, m_leftDescriptors );
    checkLastError();
    collectPoints( m_leftDescriptors, m_leftPoints, Size(m_imgLeft) );
    checkLastError();
}

void StereoAnalyzerGPU::acceptCommand( const Command &command )
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
    if( command.m_action == "MAXPOINTS")
    {
        m_maxNumberOfPoints = command.m_value;
    }
}

void StereoAnalyzerGPU::processImages()
{
    // Store descriptors values for testing and viewing
    // if m_leftImageIsNew or m_rightImageIsNew, the analysis thread is working
    m_imgMutex.lock();
    if( m_leftImageIsNew == true && m_rightImageIsNew == true )
    {
        m_imgMutex.unlock();
        // Compute matching
        if( m_matchMutex.try_lock() && m_result == NULL )
        {
            // Compute matching only if the number of points is under a threshold
            if( NbElements(m_leftDescriptors) < m_maxNumberOfPoints
            &&  NbElements(m_rightDescriptors) < m_maxNumberOfPoints )
            {
                checkLastError();
                computeMatching( m_leftDescriptors, m_rightDescriptors,
                    m_leftMatchedPts, m_rightMatchedPts,
                    Size(m_imgRight));
            }

            checkLastError();
            // Prepare data to be computed
            m_result = new ComputationDataGPU( m_imgLeft, m_imgRight, m_rightPoints, m_leftPoints );

            m_matchMutex.unlock();
        }
    }
    else
    {
        m_imgMutex.unlock();
    }
}

void StereoAnalyzerGPU::analyse()
{
    if( m_imgWidth == 0 || m_imgHeight == 0 )
        return;

    m_imgMutex.lock();
    m_leftImageIsNew = m_rightImageIsNew = false;
    m_imgMutex.unlock();

    // If no result has been prepared, go away

    m_matchMutex.lock();
    if( m_result == NULL )
    {
        m_matchMutex.unlock();
        return;
    }
    Deformation &d = m_result->m_d;
    assert( m_leftMatchedPts.size() == m_rightMatchedPts.size() );
    d.m_nbMatches = m_leftMatchedPts.size();
    d.m_nbPtsRight = NbElements(m_rightPoints);
    d.m_nbPtsLeft = NbElements(m_leftPoints);
    d.m_mode = "GPU";

    if( NbElements(m_leftDescriptors) >= m_maxNumberOfPoints
    ||  NbElements(m_rightDescriptors) >= m_maxNumberOfPoints )
    {
        d.m_nbMatches = 0;
    }
    // Compute homography descriptors for left and right images
    if( d.m_nbMatches > 8 )
    {
        CvMat _pt1, _pt2;
        _pt1 = cvMat(1, d.m_nbMatches, CV_32FC2, &m_leftMatchedPts[0] );
        _pt2 = cvMat(1, d.m_nbMatches, CV_32FC2, &m_rightMatchedPts[0] );
        CvMat _h = cvMat(3, 3, CV_64F, d.m_h);
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
            d.m_nbMatches = m_leftMatchedPts.size();
            computeDisparity();
            
            m_result->m_leftMatchedPts = m_leftMatchedPts;
            m_result->m_rightMatchedPts = m_rightMatchedPts;
            m_result->m_thresholdUsed = m_sentThreshold; 
        }
    }
    
    m_matchMutex.unlock();
}

void StereoAnalyzerGPU::resizeImages( UInt2 imgSize )
{
    freeImages();
    allocImages( imgSize );

    m_hessianData.resizeImages(imgSize);
}

void StereoAnalyzerGPU::allocImages( UInt2 imgSize )
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

    if( ! allocBuffer( m_satImage, imgSize ) )
        return;

    allocBuffer( m_hesImage, imgSize );

    //
    m_hessianData.allocImages(imgSize);

    allocBuffer( m_rightPoints, m_hessianData.capacity() );
    allocBuffer( m_leftPoints,  m_hessianData.capacity() );

    m_leftMatchedPts.reserve( m_hessianData.capacity());
    m_rightMatchedPts.reserve( m_hessianData.capacity());
}

void StereoAnalyzerGPU::freeImages()
{
    m_rightMatchedPts.clear();
    m_leftMatchedPts.clear();
    m_hessianData.freeImages();
    releaseBuffer( m_hesImage );
    releaseBuffer( m_satImage );
    releaseBufferAndTexture( m_warpedTmp );
    releaseBuffer(m_leftPoints);
    releaseBuffer(m_rightPoints);
    releaseBufferAndTexture( m_imgLeft );
    releaseBufferAndTexture( m_imgRight );
}

void StereoAnalyzerGPU::computeSurfDescriptors( ImageGL & img, DescriptorData &descriptorsData )
{
    // Copy RGBA image [0,255] in float buffer [0,1] allocated by cuda
    convertRGBAToCudaBufferY( img, m_satImage );
    checkLastError();
    // Create integral image
    convertToIntegral( m_satImage );
    checkLastError();
    // Compute hessian and determinants
    computeHessianDet( m_satImage, m_hesImage, m_hessianData );
    checkLastError();
    // Find position of maximum values in determinant image
    computeNonMaxSuppression( m_hesImage, m_hessianData );
    checkLastError();
    // Copy hessian points in descriptors
    collectHessianPoints( m_hessianData, descriptorsData );
    checkLastError();
    // Compute Surf descriptors
    computeDescriptors( m_satImage, descriptorsData );
    checkLastError();
}

ComputationData * StereoAnalyzerGPU::acquireLastResult()
{
    ComputationData *ret = NULL;
    if( m_matchMutex.try_lock() )
    {
        ret = m_result;
        m_result = NULL;
        m_matchMutex.unlock();
    }
    
    return ret;
}

void StereoAnalyzerGPU::computeDisparity()
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
            const float distH = m_rightMatchedPts[i].x - m_leftMatchedPts[i].x;
            const float distV = m_rightMatchedPts[i].y - m_leftMatchedPts[i].y;

            // range of the histogram
            const float range = float(m_histoRange)/float(Width(m_imgLeft)); 
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
