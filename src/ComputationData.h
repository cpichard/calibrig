#ifndef __COMPUTATIONDATA_H__
#define __COMPUTATIONDATA_H__


#include "cv.h"
#include "highgui.h"
#include "AnalysisResult.h"
#include "Monitor.h"

// Store data used for a computation
class ComputationData
{
public:
    ComputationData()
    :m_thresholdUsed(0)
    {}

    virtual ~ComputationData(){}

    // Computation data knows how to update a monitor
    virtual void updateRightMonitor( Monitor &mon )=0;
    virtual void updateLeftMonitor ( Monitor &mon )=0;

    virtual void drawLeftKeyPoints(float ratio)=0;
    virtual void drawRightKeyPoints(float ratio)=0;

    virtual void drawVerticalDisparity(){};
    virtual void drawHorizontalDisparity(){};

    virtual std::string infos()=0; // TODO draw infos CPU/GPU

    // AnalysisResult found
    AnalysisResult m_d;

    // Analysis infos;
    float m_thresholdUsed;

    // Graphic value for histogram size
    // it shouldn't be here, shouldn't it ?
    static const float m_binWidth = 0.03;
};


#endif//__COMPUTATIONDATA_H__
