#ifndef __GRABBER_H__
#define __GRABBER_H__

#include "ImageGL.h"
#include "GraphicSystemX11.h"

// Interface for the grabber class
class Grabber
{
public:
    Grabber( Display *dpy, HGPUNV *gpu, GLXContext ctx );
    virtual ~Grabber();    
    
    virtual bool init()=0;
    virtual bool captureVideo()=0;
    virtual void shutdown()=0;

    // Returns streams
    inline ImageGL & stream1() {return m_stream1CaptureHandle;}
    inline ImageGL & stream2() {return m_stream2CaptureHandle;}
    inline UInt2 & videoSize() {return m_videoSize;}

    void saveImages();

protected:
    // External data
    Display         *m_dpy;
    HGPUNV          *m_gpu;
    GLXContext      m_ctx;
    
    // Output streams
    UInt2           m_videoSize;
    ImageGL         m_stream1CaptureHandle;
    ImageGL         m_stream2CaptureHandle;
};


#endif//__GRABBER_H__
