#include "StereoAnalyzerCPU.h"

#include "NvSDIin.h"


#include <vector>
#include <iomanip>
#include <iostream>

using std::vector;

// Utility
void printMat( std::string T, CvMat *H, int r, int c )
{
    //return; // CYWILL
    std::cout << T << std::endl;
    std::cout << std::setprecision( 3 ) << std::right << std::fixed;
    for ( int row = 0; row < r; ++ row )
    {
        for ( int col = 0; col < c; ++ col )
        {
            std::cout << std::setw( 5 ) << (double)cvmGet( H, row, col ) << " ";
        }
        std::cout << std::endl;
    }
}

// Constructor
StereoAnalyzerCPU::StereoAnalyzerCPU( unsigned int nbChannelsInSDIVideo )
:StereoAnalyzer( nbChannelsInSDIVideo )
, m_imgWidth(0)
, m_imgHeight(0)
, m_surfThreshold(500)
, m_imgRight(NULL)
, m_imgLeft(NULL)
, m_imgTmp(NULL)
, m_tmpBuf(0)
, m_warpMatrix(NULL)
, m_result(NULL)
, m_maxNumberOfPoints(8000)
{
    m_warpMatrix = cvCreateMat(2,3,CV_64FC1);
    m_toNormMatrix = cvCreateMat(3,3,CV_64FC1);
    m_fromNormMatrix = cvCreateMat(3,3,CV_64FC1);

    cvSetIdentity(m_warpMatrix);
}

StereoAnalyzerCPU::~StereoAnalyzerCPU()
{
    // Release OpenCV images
    freeImages( );

    cvFree(&m_warpMatrix);
    m_warpMatrix = NULL;
    
    cvFree(&m_toNormMatrix);
    m_toNormMatrix = NULL;

    cvFree(&m_fromNormMatrix);
    m_fromNormMatrix = NULL;

    if(m_result)
    {
        delete m_result;
        m_result=NULL;
    }
}

void StereoAnalyzerCPU::updateRightImageWithSDIVideo( ImageGL &videoPBO )
{
    boost::mutex::scoped_lock sl(m_imgMutex);updateImageWithSDIVideo(videoPBO, m_imgRight); m_rightImageIsNew = true;
}

void StereoAnalyzerCPU::updateLeftImageWithSDIVideo ( ImageGL &videoPBO )
{
    boost::mutex::scoped_lock sl(m_imgMutex);updateImageWithSDIVideo(videoPBO, m_imgLeft ); m_leftImageIsNew = true;
}

void StereoAnalyzerCPU::resizeImages( UInt2 imgSize )
{
    freeImages();
    allocImages( imgSize );
}

void StereoAnalyzerCPU::allocImages( UInt2 imgSize )
{
    m_imgWidth = Width(imgSize);
    m_imgHeight = Height(imgSize);

    if( m_imgWidth == 0 || m_imgHeight == 0 )
        return;
    
    // Create 2 luminance images
    assert(m_imgRight==NULL);
    m_imgRight = cvCreateImage( cvSize(m_imgWidth,m_imgHeight), IPL_DEPTH_8U, 1);
    assert(m_imgLeft==NULL);
    m_imgLeft  = cvCreateImage( cvSize(m_imgWidth,m_imgHeight), IPL_DEPTH_8U, 1);
    assert(m_imgTmp==NULL);
    m_imgTmp   = cvCreateImage( cvSize(m_imgWidth,m_imgHeight), IPL_DEPTH_8U, 1);

    // TODO : test with cuda malloc
    // TODO : check encoding
    assert(m_tmpBuf==NULL);
    m_tmpBuf = (char *)malloc( m_nbChannelsInSDIVideo*m_imgWidth*m_imgHeight );

    const float L = (float)(m_imgWidth-1);
    const float H = (float)(m_imgHeight-1);

    cvSetIdentity(m_toNormMatrix);
    cvmSet( m_toNormMatrix, 0, 2, -0.5*L);
    cvmSet( m_toNormMatrix, 1, 2, -0.5*H);
    cvmSet( m_toNormMatrix, 2, 2, L);

    cvSetIdentity(m_fromNormMatrix);
    cvmSet( m_fromNormMatrix, 0, 2, 0.5);
    cvmSet( m_fromNormMatrix, 1, 2, 0.5*H/L);
    cvmSet( m_fromNormMatrix, 2, 2, 1.0/L);
}

void StereoAnalyzerCPU::freeImages( )
{
    if( m_imgRight )
    {
        cvReleaseImage( &m_imgRight);
        m_imgRight = NULL;
    }

    if( m_imgLeft )
    {
        cvReleaseImage( &m_imgLeft);
        m_imgLeft = NULL;
    }

    if( m_imgTmp )
    {
        cvReleaseImage( &m_imgTmp);
        m_imgTmp = NULL;
    }

    if(m_tmpBuf)
    {
        free(m_tmpBuf);
        m_tmpBuf = NULL;
    }    
    m_imgWidth = 0;
    m_imgHeight = 0;
}


void StereoAnalyzerCPU::acceptCommand( const Command &command )
{
    if( command.m_action == "OCVTHRESHOLD" )
    {
        // value 0 to 100
        m_surfThreshold = command.m_value;// < 0 ? 10 : ( (command.m_value > 100 ) ? 10000 : command.m_value * 100);
    }
    if( command.m_action == "MAXPOINTS" )
    {
        m_maxNumberOfPoints = command.m_value;    
    }
}

//void StereoAnalyserCPU::saveImage( IplImage *ocvImg, const std::string &filename )
//{
//    // Write location in image
//    CvSeqReader it;
//    cvStartReadSeq(objectKeypoints,&it);
//    char *ocvBuf = (char*)imgBuf[0]->imageData;
//    for(unsigned int i=0; i <objectKeypoints->total; i++ )
//    {
//        const CvSURFPoint *kp = (const CvSURFPoint *)it.ptr;
//        ocvBuf[ ((unsigned int)kp->pt.x +((unsigned int)kp->pt.y)* card.getWidth()) ] = 255;
//        CV_NEXT_SEQ_ELEM( it.seq->elem_size, it );
//    }
//}

void StereoAnalyzerCPU::updateImageWithSDIVideo( ImageGL &videoPBO, IplImage *ocvImg )
{
    if( m_imgWidth == 0 || m_imgHeight == 0 )
        return;

    if(BufId(videoPBO)==0)
        return;

    // Copy from device memory to host memory
    glBindBufferARB( GL_VIDEO_BUFFER_NV, BufId(videoPBO) );
    assert(glGetError() == GL_NO_ERROR);
    glGetBufferSubDataARB( GL_VIDEO_BUFFER_NV, 0, m_nbChannelsInSDIVideo*m_imgWidth*m_imgHeight, m_tmpBuf );
    assert(glGetError() == GL_NO_ERROR);
    glBindBufferARB( GL_VIDEO_BUFFER_NV, 0 );
    assert(glGetError() == GL_NO_ERROR);
    // Convert to gray scale, we keep only luminance
    // It works only for 422 format
    char *sdiBuf = (char*)m_tmpBuf;
    char *tmpBuf = (char*)m_imgTmp->imageData; // TESTING PURPOSE
    //char *tmpBuf = (char*)ocvImg->imageData;
    double tt = (double)cvGetTickCount();
    for( unsigned int j=0; j < m_imgHeight; j++)
    {
        for( unsigned int k=0; k < m_imgWidth; k++ )
        {
            *tmpBuf++ = *sdiBuf; // first component = Y
            sdiBuf+=m_nbChannelsInSDIVideo;
        }
    }

   //printMat( "m_warpMatrix", m_warpMatrix, 2, 3 );

    // TESTING PURPOSE
    cvWarpAffine( m_imgTmp, ocvImg, m_warpMatrix );

    tt = (double)cvGetTickCount() - tt;
    //printf( "Convert to gray time = %gms\n", tt/(cvGetTickFrequency()*1000.));
}

// TESTING PURPOSE
void StereoAnalyzerCPU::setTransform(float tx, float ty, float scale, float rot )
{
    // Create transform mat
    CvMat *transform = cvCreateMat(3,3,CV_64FC1);
    cvSetIdentity(transform);
    cvmSet(transform, 0, 0, scale*cos(rot));
    cvmSet(transform, 0, 1, -sin(rot));
    cvmSet(transform, 0, 2, tx);
    cvmSet(transform, 1, 0, sin(rot));
    cvmSet(transform, 1, 1, scale*cos(rot));
    cvmSet(transform, 1, 2, ty);
    cvmSet(transform, 2, 2, 1.0);

    // Tmp matrices
    CvMat *result = cvCreateMat(3,3,CV_64FC1);
    cvSetIdentity(result);
    CvMat *tmpMat = cvCreateMat(3,3,CV_64FC1);
    cvSetIdentity(tmpMat);
    
    // Multiplication
    cvMatMul( transform, m_toNormMatrix, tmpMat);
    cvMatMul( m_fromNormMatrix, tmpMat,  result);

    // Set Warp matrix
    cvmSet(m_warpMatrix, 0, 0, cvmGet(result,0,0));
    cvmSet(m_warpMatrix, 0, 1, cvmGet(result,0,1));
    cvmSet(m_warpMatrix, 0, 2, cvmGet(result,0,2));
    cvmSet(m_warpMatrix, 1, 0, cvmGet(result,1,0));
    cvmSet(m_warpMatrix, 1, 1, cvmGet(result,1,1));
    cvmSet(m_warpMatrix, 1, 2, cvmGet(result,1,2));
}

double compareSURFDescriptors( const float* d1, const float* d2, double best, int length )
{
    double total_cost = 0;
    assert( length % 4 == 0 );
    for( int i = 0; i < length; i += 4 )
    {
        double t0 = d1[i] - d2[i];
        double t1 = d1[i+1] - d2[i+1];
        double t2 = d1[i+2] - d2[i+2];
        double t3 = d1[i+3] - d2[i+3];
        total_cost += t0*t0 + t1*t1 + t2*t2 + t3*t3;
        if( total_cost > best )
            break;
    }
    return total_cost;
}

int naiveNearestNeighbor( const float* vec, int laplacian,
                      const CvSeq* model_keypoints,
                      const CvSeq* model_descriptors )
{
    int length = (int)(model_descriptors->elem_size/sizeof(float));
    int i, neighbor = -1;
    double d, dist1 = 1e6, dist2 = 1e6;
    CvSeqReader reader, kreader;
    cvStartReadSeq( model_keypoints, &kreader, 0 );
    cvStartReadSeq( model_descriptors, &reader, 0 );

    for( i = 0; i < model_descriptors->total; i++ )
    {
        const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
        const float* mvec = (const float*)reader.ptr;
    	CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        if( laplacian != kp->laplacian )
            continue;
        d = compareSURFDescriptors( vec, mvec, dist2, length );
        if( d < dist1 )
        {
            dist2 = dist1;
            dist1 = d;
            neighbor = i;
        }
        else if ( d < dist2 )
            dist2 = d;
    }
    if ( dist1 < 0.6*dist2 )
        return neighbor;
    return -1;
}

void
findPairs(const CvSeq* rightKeypoints, const CvSeq* rightDescriptors,
          const CvSeq* leftKeypoints, const CvSeq* leftDescriptors, std::vector<int>& ptpairs )
{
    int i;
    CvSeqReader reader, kreader;
    cvStartReadSeq( rightKeypoints, &kreader );
    cvStartReadSeq( rightDescriptors, &reader );

    // TODO : OPTIMIZE HERE
    // TODO : disambiguate match
    ptpairs.clear();

    for( i = 0; i < rightDescriptors->total; i++ )
    {
        const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
        const float* descriptor = (const float*)reader.ptr;
        CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        int nearest_neighbor = naiveNearestNeighbor( descriptor, kp->laplacian, leftKeypoints, leftDescriptors );
        if( nearest_neighbor >= 0 )
        {
            ptpairs.push_back(i);
            ptpairs.push_back(nearest_neighbor);
        }
    }
}

void normalise( CvSeq* keyPoints, unsigned int imgWidth, unsigned int imgHeight )
{
    double w = static_cast<double>(imgWidth);
    double h = static_cast<double>(imgHeight);
    CvSeqReader reader;
    cvStartReadSeq( keyPoints, &reader );
    unsigned int nbElements = keyPoints->total;
    for( unsigned int i = 0; i < nbElements; i++ )
    {
        CvSURFPoint* kp = (CvSURFPoint*)reader.ptr;
        kp->pt.x = kp->pt.x/w - 0.5;
        kp->pt.y = ((kp->pt.y / h ) -0.5)*(h/w);
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
    }
}

int
findHomography( const CvSeq* rightKeypoints, const CvSeq* rightDescriptors,
                const CvSeq* leftKeypoints, const CvSeq* leftDescriptors, 
                double *h, vector<int> &ptpairs )
{
    CvMat _h = cvMat(3, 3, CV_64F, h);

    vector<CvPoint2D32f> pt1, pt2;
    CvMat _pt1, _pt2;
    int i, n;

    findPairs( rightKeypoints, rightDescriptors, leftKeypoints, leftDescriptors, ptpairs );

    n = ptpairs.size()/2;
    if( n < 4 )
        return 0;

    pt1.resize(n);
    pt2.resize(n);
    for( i = 0; i < n; i++ )
    {
        pt1[i] = ((CvSURFPoint*)cvGetSeqElem(rightKeypoints,ptpairs[i*2]))->pt;
        pt2[i] = ((CvSURFPoint*)cvGetSeqElem(leftKeypoints,ptpairs[i*2+1]))->pt;
    }

    _pt1 = cvMat(1, n, CV_32FC2, &pt1[0] );
    _pt2 = cvMat(1, n, CV_32FC2, &pt2[0] );
    if( !cvFindHomography( &_pt1, &_pt2, &_h, CV_LMEDS, 1 ))
        return 0;

    return 1;
}

// Launch analysis and return
void StereoAnalyzerCPU::analyse()
{
    m_leftImageIsNew = m_rightImageIsNew = false;

    if( m_imgWidth == 0 || m_imgHeight == 0 )
        return;
    
    // If no one has taken the result,
    if( m_result != NULL )
    {
        return;
    }

    // Allocate new result
    // BAD !! optimize here
    // DO not allocate a result each time
    ComputationDataCPU *resultTmp = new ComputationDataCPU();
    m_result = resultTmp;

    // Check if we have the buffers
    if( m_imgRight == NULL || m_imgLeft == NULL )
    {
        return;
    }
    
    // Params
    CvSURFParams params = cvSURFParams(m_surfThreshold, 1);
    resultTmp->m_storage = cvCreateMemStorage(0);

    // Extract surf descriptors
    double tt = (double)cvGetTickCount();
    cvExtractSURF( m_imgRight, NULL, 
        &resultTmp->m_rightKeypoints,
        &resultTmp->m_rightDescriptors,
        resultTmp->m_storage, params );

    cvExtractSURF( m_imgLeft, NULL,
        &resultTmp->m_leftKeypoints,
        &resultTmp->m_leftDescriptors,
        resultTmp->m_storage, params );

    tt = (double)cvGetTickCount() - tt;
    // TODO : is it needed in result ?
    //printf( "Extraction time = %gms\n", tt/(cvGetTickFrequency()*1000.));

    // No or too many points found
    if(  ! resultTmp->m_rightKeypoints || ! resultTmp->m_leftKeypoints)
    {
        delete resultTmp;
        resultTmp = NULL;
        m_result = NULL;
        return;
    }

    // Keypoints are in the image basis => normalize keypoints between -0.5 and 0.5
    normalise( resultTmp->m_rightKeypoints, m_imgWidth, m_imgHeight );
    normalise( resultTmp->m_leftKeypoints,  m_imgWidth, m_imgHeight );

    AnalysisResult &d = resultTmp->m_d;
    d.m_nbPtsRight = resultTmp->m_rightKeypoints->total;
    d.m_nbPtsLeft = resultTmp->m_leftKeypoints->total;
    d.m_nbMatches = 0;
    d.m_mode = "CPU";

    // Compute homography only if number of points is under a certain level
    if( resultTmp->m_rightKeypoints->total < m_maxNumberOfPoints
    &&  resultTmp->m_leftKeypoints->total < m_maxNumberOfPoints
    &&  findHomography( resultTmp->m_leftKeypoints, resultTmp->m_leftDescriptors,
                        resultTmp->m_rightKeypoints, resultTmp->m_rightDescriptors,
                        d.m_h, resultTmp->m_ptpairs ) )
    {
        //std::cout << "Found Homography" << std::endl;
        CvMat H = cvMat(3, 3, CV_64F, d.m_h);

        // Simple decomposition
        double Tx = cvmGet(&H,0,2);
        double Ty = cvmGet(&H,1,2);
        double rot =  asin(cvmGet(&H,1,0));
        double scale =  cvmGet(&H,0,0)/cos(rot);

        //std::cout << "Tx = " << Tx << " , Ty = " << Ty << std::endl;

        d.m_tx = Tx*(float)m_imgWidth;
        d.m_ty = -Ty*(float)m_imgWidth;
        d.m_scale = scale;
        d.m_rot = rot*180.f/3.14; // approx
        d.m_succeed = true;
        d.m_nbPtsRight = resultTmp->m_rightKeypoints->total;
        d.m_nbPtsLeft = resultTmp->m_leftKeypoints->total;
        d.m_nbMatches = resultTmp->m_ptpairs.size()/2;
        resultTmp->m_thresholdUsed = m_surfThreshold;
        resultTmp->setRightImage( m_imgRight);
        resultTmp->setLeftImage ( m_imgLeft );

        computeDisparity();
    }
}

void StereoAnalyzerCPU::computeDisparity()
{
    if( m_result )
    {
        AnalysisResult &d = m_result->m_d;

        for( int i = 0; i < d.s_histogramBinSize; i++ )
        {
            d.m_hdisp[i] = 0;
            d.m_vdisp[i] = 0;
        }
        
        ComputationDataCPU *result = static_cast<ComputationDataCPU*>(m_result);
        for( int i = 0; i < d.m_nbMatches; i++ )
        {
            CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem(result->m_rightKeypoints, result->m_ptpairs[i*2+1]);
            CvSURFPoint* l = (CvSURFPoint*)cvGetSeqElem(result->m_leftKeypoints, result->m_ptpairs[i*2]);

            const float distH = r->pt.x - l->pt.x;
            const float distV = r->pt.y - l->pt.y;

            // TODO : range of the histogram
            const float range = 20.f/1920.f; // 20 pixels wide
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
