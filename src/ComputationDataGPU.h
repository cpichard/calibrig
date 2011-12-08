#ifndef __COMPUTATIONDATAGPU_H__
#define __COMPUTATIONDATAGPU_H__

#include "ComputationData.h"
#include "VertexBufferObject.h"

class ComputationDataGPU : public ComputationData
{
public:
    ComputationDataGPU( ImageGL &l, ImageGL &r, VertexBufferObject &pointsR,  VertexBufferObject &pointsL );
    ~ComputationDataGPU();

    // Computation data knows how to update a monitor
    virtual void updateRightMonitor( Monitor &mon );
    virtual void updateLeftMonitor ( Monitor &mon );

    virtual void drawLeftKeyPoints(float ratio);
    virtual void drawRightKeyPoints(float ratio);
    virtual void drawVerticalDisparity();
    virtual void drawHorizontalDisparity();

    virtual std::string infos();

    // TODO GPU buffer used for surf
    ImageGL m_imgRight;
    ImageGL m_imgLeft;

    // VBO !
    VertexBufferObject m_pointsR;
    VertexBufferObject m_pointsL;

    // TODO : use full GPU for matched points
    vector<CvPoint2D32f> m_leftMatchedPts, m_rightMatchedPts;

private:
    void drawHistogram(float *);

};

#endif // __COMPUTATIONDATAGPU_H__
