#ifndef __HOMOGRAPHYANALYZERCPU_H__
#define __HOMOGRAPHYANALYZERCPU_H__

#include "cv.h"
#include "highgui.h"
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <vector>
#include "Utils.h"

#include <boost/thread.hpp>
#include "ComputationDataCPU.h"
#include "StereoAnalyzer.h"

// TODO : rename to StereoAnalyZerCPU
class HomographyAnalyzerCPU : public StereoAnalyzer
{
public:
    // Constructor
    HomographyAnalyzerCPU( unsigned int nbChannelsInSDIVideo=2 );
    ~HomographyAnalyzerCPU();

    // Update left and right images
    virtual void updateRightImageWithSDIVideo( ImageGL &videoPBO );
    virtual void updateLeftImageWithSDIVideo ( ImageGL &videoPBO );

    // Launch analysis and return results
    virtual void analyse();
    virtual void computeDisparity();
    virtual ComputationData * acquireLastResult() { ComputationData *result = m_result; m_result = NULL; return result; };
    virtual void disposeResult(ComputationData *result){ delete result;}
    virtual void resizeImages( UInt2 imgSize );
    virtual void allocImages( UInt2 imgSize );
    virtual void freeImages( );

    virtual void acceptCommand(const Command &);

    // TESTING PURPOSE
    void setTransform(float tx, float ty, float scale, float rot );

protected:
    void updateImageWithSDIVideo( ImageGL &videoPBO, IplImage * );
    //void saveImage ( IplImage *img, const std::string &filename );
    
    // Image size
    unsigned int m_imgWidth;
    unsigned int m_imgHeight;

    // Threshold for surf extraction
    unsigned int m_surfThreshold;
    
    // Maximum number of points
    unsigned int m_maxNumberOfPoints;
    
    // below variables In computation data instead ???
    //
    // Right image
    IplImage *m_imgRight;
    
    // Left image
    IplImage *m_imgLeft;

    // Tmp image
    IplImage *m_imgTmp;

    // Tmp buffer used to copy raw image (YcrYcb) to memory
    char *m_tmpBuf;

    // TESTING PURPOSE
    CvMat *m_warpMatrix;
    CvMat *m_toNormMatrix;
    CvMat *m_fromNormMatrix;
    
    ComputationData *m_result;
};

#endif

