#ifndef __GRABBERSDI_H__
#define __GRABBERSDI_H__

#include "NvSDIin.h"

#include "Grabber.h"

// FrameGrabberSDI
// GrabberSDI == SDI card + OpenGL 
// DRAFT class in progress
class GrabberSDI : public Grabber
{
public:
    GrabberSDI( Display *dpy, HGPUNV *gpu, GLXContext ctx );

    // The functions to call from outside
    bool init();
    bool captureVideo();
    void shutdown();

private:
    GLenum cardCaptureVideo();
    bool setupCardGL();
    void selectStreams();

    typedef enum {
        OFFLINE,
        CARD_INITIALIZED,
        INPUT_READY,
        CAPTURE_STARTED,
        CAPTURE_FAILED
    }
    State;

    // GrabberSDI data
    CNvSDIin        m_card;
    CaptureOptions  m_captureOptions;
    GLuint          m_sequenceNum;
    GLuint          m_prevSequenceNum;
    int             m_numFails;

    State           m_state;
};

#endif//__GRABBERSDI_H__
