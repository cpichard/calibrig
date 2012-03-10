#include "FMatrixAnalyzerCPU.h"

#include "NvSDIin.h"


#include <vector>
#include <iomanip>
#include <iostream>

using std::vector;

extern void normalise( CvSeq* keyPoints, unsigned int imgWidth, unsigned int imgHeight );

extern void
findPairs(const CvSeq* rightKeypoints, const CvSeq* rightDescriptors,
          const CvSeq* leftKeypoints, const CvSeq* leftDescriptors, std::vector<int>& ptpairs );
// Constructor
FMatrixAnalyzerCPU::FMatrixAnalyzerCPU( unsigned int nbChannelsInSDIVideo )
: HomographyAnalyzerCPU( nbChannelsInSDIVideo )
{}

FMatrixAnalyzerCPU::~FMatrixAnalyzerCPU()
{}

int
findFMatrix( const CvSeq* rightKeypoints, const CvSeq* rightDescriptors,
                const CvSeq* leftKeypoints, const CvSeq* leftDescriptors, 
                double *f, double *h1, double *h2, vector<int> &ptpairs,
                CvSize &imgSize)
{
    CvMat _f = cvMat(3, 3, CV_64F, f);
    CvMat _h1 = cvMat(3, 3, CV_64F, h1);
    CvMat _h2 = cvMat(3, 3, CV_64F, h2);

    vector<CvPoint2D32f> pt1, pt2;
    CvMat _pt1, _pt2;
    int i, n;

    findPairs( rightKeypoints, rightDescriptors, leftKeypoints, leftDescriptors, ptpairs );

    n = ptpairs.size()/2;
    if( n < 7 )
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

    // TODO : recuperer les points matches et faire la rectification avec 
    if( cvFindFundamentalMat( &_pt1, &_pt2, &_f, CV_LMEDS, 1.0, 0.99 /*, points matches*/ ))
    {
        cvStereoRectifyUncalibrated( &_pt1, &_pt2, &_f, imgSize, &_h1, &_h2);
        return 1;
    }

    return 0;
}

// Launch analysis and return
void FMatrixAnalyzerCPU::analyse()
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

    // No points found
    if( ! resultTmp->m_rightKeypoints || ! resultTmp->m_leftKeypoints )
    {
        delete resultTmp;
        resultTmp = NULL;
        m_result = NULL;
        return;
    }

    // Keypoints are in the image basis => normalize keypoints between -0.5 and 0.5
    normalise( resultTmp->m_rightKeypoints, m_imgWidth, m_imgHeight );
    normalise( resultTmp->m_leftKeypoints,  m_imgWidth, m_imgHeight );

    Deformation &d = resultTmp->m_d;
    d.m_nbPtsRight = resultTmp->m_rightKeypoints->total;
    d.m_nbPtsLeft = resultTmp->m_leftKeypoints->total;
    d.m_nbMatches = 0;
    d.m_mode = "CPU";
    
    CvSize imgSize;
    imgSize.width=m_imgWidth;
    imgSize.height=m_imgHeight;
    if( findFMatrix( resultTmp->m_leftKeypoints, resultTmp->m_leftDescriptors,
                        resultTmp->m_rightKeypoints, resultTmp->m_rightDescriptors,
                        d.m_f, d.m_h1, d.m_h2, resultTmp->m_ptpairs, imgSize ) )
    {
        // Copy result
        d.m_succeed = true;
        d.m_nbPtsRight = resultTmp->m_rightKeypoints->total;
        d.m_nbPtsLeft = resultTmp->m_leftKeypoints->total;
        d.m_nbMatches = resultTmp->m_ptpairs.size()/2;
        resultTmp->m_thresholdUsed = m_surfThreshold;
        resultTmp->setRightImage( m_imgRight);
        resultTmp->setLeftImage ( m_imgLeft );
    }

    // TODO :
    // computeDisparity();
}

