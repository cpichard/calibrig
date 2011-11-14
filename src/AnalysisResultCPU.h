#ifndef __ANALYSISRESULT_H__
#define __ANALYSISRESULT_H__


#include "cv.h"
#include "highgui.h"

#include "Deformation.h"

//! Store data used during computation
// REFACTOR rename to ComputationDataCPU ?
typedef struct AnalysisResult
{
    AnalysisResult()
    :m_imgRight(0),
    m_imgLeft(0),
    m_rightKeypoints(NULL),
    m_leftKeypoints(NULL),
    m_rightDescriptors(NULL),
    m_leftDescriptors(NULL),
    m_storage(NULL)
    {}

    ~AnalysisResult()
    {
        if(m_storage)
        {
            cvReleaseMemStorage(&m_storage);
            m_storage = NULL;
            m_rightKeypoints = NULL;
            m_leftKeypoints = NULL;
        }
    }

    inline IplImage * rightImage(){return m_imgRight;}
    inline IplImage * leftImage() {return m_imgLeft; }
    inline void setRightImage(IplImage *r){m_imgRight=r;}
    inline void setLeftImage(IplImage *l) {m_imgLeft=l; }

    IplImage *m_imgRight;
    IplImage *m_imgLeft;

    // Deformation
    Deformation m_d;

    // Buffer used
    std::vector<int> m_ptpairs;
    CvSeq *m_rightKeypoints;
    CvSeq *m_leftKeypoints;
    CvSeq *m_rightDescriptors;
    CvSeq *m_leftDescriptors;
    CvMemStorage* m_storage;
}
AnalysisResult;


#endif //
