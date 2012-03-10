#ifndef __FMATRIXANALYZERCPU_H__
#define __FMATRIXANALYZERCPU_H__

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
#include "HomographyAnalyzerCPU.h"

// TODO : rename to StereoAnalyZerCPU
class FMatrixAnalyzerCPU : public HomographyAnalyzerCPU
{
public:
    // Constructor
    FMatrixAnalyzerCPU( unsigned int nbChannelsInSDIVideo=2 );
    ~FMatrixAnalyzerCPU();

    // Launch analysis and return results
    virtual void analyse();
};

#endif

