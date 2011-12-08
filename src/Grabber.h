#ifndef __GRABBER_H__
#define __GRABBER_H__

#include "NvSDIin.h"
#include "ImageGL.h"

// FrameGrabber
// Grabber == SDI card + OpenGL 
// DRAFT class in progress
class Grabber
{
public:
    Grabber( Display *dpy, HGPUNV *gpu, GLXContext ctx );

    // The functions to call from outside
    bool init();
    bool captureVideo();
    void shutdown();

    inline ImageGL & stream1() {return m_stream1CaptureHandle;}
    inline ImageGL & stream2() {return m_stream2CaptureHandle;}

    inline UInt2 & videoSize() {return m_videoSize;}

    void saveImages();

private:
    GLenum cardCaptureVideo();
    bool setupCardGL();
    void selectStreams();
    void registerCudaBuffers();
    void unregisterCudaBuffers();

    // External data
    Display         *m_dpy;
    HGPUNV          *m_gpu;
    GLXContext      m_ctx;

    typedef enum {
        OFFLINE,
        CARD_INITIALIZED,
        INPUT_READY,
        CAPTURE_STARTED,
        CAPTURE_FAILED
    }
    State;

    // Grabber data
    CNvSDIin        m_card;
    CaptureOptions  m_captureOptions;
    UInt2           m_videoSize;
    ImageGL         m_stream1CaptureHandle;
    ImageGL         m_stream2CaptureHandle;
    GLuint          m_sequenceNum;
    GLuint          m_prevSequenceNum;
    int             m_numFails;

    State           m_state;
};

#endif//__GRABBER_H__
