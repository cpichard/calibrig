#ifndef __STEREOANALYZER_H__
#define __STEREOANALYZER_H__

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
#include "CommandStack.h"

//! Base class for a stereo analyzer.
//! All derived classes need to implement update left and right image functions
class StereoAnalyzer
{
public:
    // Constructor
    StereoAnalyzer( unsigned int nbChannelsInSDIVideo=2 );
    virtual ~StereoAnalyzer();

    bool try_lock(){ return m_mutex.try_lock(); }
    void lock(){ m_mutex.lock(); }
    void unlock(){ m_mutex.unlock();}

    // Update left and right images
    virtual void updateRightImageWithSDIVideo( ImageGL &videoPBO )=0;
    virtual void updateLeftImageWithSDIVideo ( ImageGL &videoPBO )=0;
    virtual void processImages(){}
    virtual void computeDisparity(){}
    
    // Launch analysis and return results
    virtual void analyse()=0;


    inline bool imagesAreNew() { boost::mutex::scoped_lock sl(m_imgMutex); return m_leftImageIsNew && m_rightImageIsNew; }

    virtual ComputationData * acquireLastResult() = 0; 

    virtual void resizeImages( UInt2 imgSize )=0;
    virtual void allocImages( UInt2 imgSize )=0;
    virtual void freeImages( )=0;

    virtual void acceptCommand(const Command &){};
    
protected:
    // Image size
    unsigned int m_imgWidth;
    unsigned int m_imgHeight;
     
    unsigned int m_nbChannelsInSDIVideo;

    bool m_leftImageIsNew;
    bool m_rightImageIsNew;
    
    //
    boost::mutex m_mutex;
    boost::mutex m_imgMutex;
};

#endif//__STEREOANALYZER_H__

