#ifndef __COMPUTATIONDATACPU_H__
#define __COMPUTATIONDATACPU_H__


#include "cv.h"
#include "highgui.h"

#include "ComputationData.h"


// Store data used for a computation
// Provide an interface for displaying data
class ComputationDataCPU : public ComputationData
{
public:
    ComputationDataCPU()
    :ComputationData(),
    m_imgRight(0),
    m_imgLeft(0),
    m_rightKeypoints(NULL),
    m_leftKeypoints(NULL),
    m_rightDescriptors(NULL),
    m_leftDescriptors(NULL),
    m_storage(NULL)
    {}

    virtual ~ComputationDataCPU()
    {
        if(m_storage)
        {
            cvReleaseMemStorage(&m_storage);
            m_storage = NULL;
            m_rightKeypoints = NULL;
            m_leftKeypoints = NULL;
        }
    }

    // Computation data knows how to update a monitor
    // Rename drawLeftImage
    virtual void updateRightMonitor( Monitor &mon );
    virtual void updateLeftMonitor ( Monitor &mon );

    virtual void drawLeftKeyPoints(float ratio);
    virtual void drawRightKeyPoints(float ratio);


    virtual void drawVerticalDisparity();
    virtual void drawHorizontalDisparity();

    // REFACTOR REMOVE right
    inline IplImage * rightImage(){return m_imgRight;}
    inline IplImage * leftImage() {return m_imgLeft; }
    inline void setRightImage(IplImage *r){m_imgRight=r;}
    inline void setLeftImage(IplImage *l) {m_imgLeft=l; }

    // PRIVATE
    IplImage *m_imgRight;
    IplImage *m_imgLeft;

    // Buffer used
    std::vector<int> m_ptpairs;
    CvSeq *m_rightKeypoints;
    CvSeq *m_leftKeypoints;
    CvSeq *m_rightDescriptors;
    CvSeq *m_leftDescriptors;
    CvMemStorage* m_storage;
};


#endif//__COMPUTATIONDATACPU_H__
