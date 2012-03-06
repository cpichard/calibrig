#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <cuda.h>
#include <cudaGL.h>

#include "CudaUtils.h"

#include "GrabberSDI.h"
#include <iostream>
#include <ctime>

GrabberSDI::GrabberSDI( Display *dpy, HGPUNV *gpu, GLXContext ctx )
: Grabber(dpy, gpu, ctx)
, m_captureOptions(gpu->deviceXScreen)
, m_sequenceNum(0)
, m_prevSequenceNum(0)
, m_numFails(0)
, m_state(OFFLINE)
{
    // Set capture options
    m_card.setCaptureOptions( m_dpy, m_captureOptions );
}

void GrabberSDI::shutdown()
{
    m_card.endCapture();
    m_card.destroyCaptureDeviceNVCtrl();
    m_card.destroyCaptureDeviceGL();

    // Delete OpenGL rendering context.
    glXMakeCurrent(m_dpy, NULL, NULL);
    if(m_ctx)
    {
        glXDestroyContext(m_dpy, m_ctx);
        m_ctx = NULL;
    }
}


// State from OFFLINE to CAPTURE_STARTED
bool GrabberSDI::init()
{
    // if the card has already be initialized
    if( m_state != OFFLINE )
        return true;

    // Check if a physical card is present and count number of inputs
    if( m_card.initCaptureDeviceNVCtrl() != TRUE )
    {
        printf("Error setting up video capture.\n");
        return false;
    }
    m_state = CARD_INITIALIZED;
    std::cout << "Card initialized" << std::endl;

    if( m_card.initInputDeviceGL() )
    {
        setupCardGL();

        //
        selectStreams();
        m_state = INPUT_READY;

        //
        bool captureStarted = m_card.startCapture();

        std::cout << "Starting capture" << std::endl;
        if( captureStarted == false )
        {
            m_numFails = 1000;
        }
        else
        {
            m_state = CAPTURE_STARTED;
        }
        return true;
    }

    return false;
}

GLenum GrabberSDI::cardCaptureVideo( )
{
	GLenum ret;
    static GLuint64EXT captureTime;

    // Capture the video to a buffer object
    ret = m_card.capture(&m_sequenceNum, &captureTime);
    // TODO in a log module
    //if(m_sequenceNum - m_prevSequenceNum > 1)
        //printf("glVideoCaptureNV: Dropped %d frames\n",m_sequenceNum - m_prevSequenceNum);
    m_prevSequenceNum = m_sequenceNum;
    switch(ret)
    {
        case GL_SUCCESS_NV:
            //printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, card.m_gpuTime,card.m_gviTime);
            m_numFails = 0;
            break;
        case GL_PARTIAL_SUCCESS_NV:
            printf("glVideoCaptureNV: GL_PARTIAL_SUCCESS_NV\n");
            m_numFails = 0;
            break;
        case GL_FAILURE_NV:
            printf("glVideoCaptureNV: GL_FAILURE_NV - Video capture failed.\n");
            m_numFails++;
            break;
        default:
            printf("glVideoCaptureNV: Unknown return value.\n");
            break;
    } // switch

    return ret;
}

bool GrabberSDI::setupCardGL()
{
    // Claer buffers ?
    glClearColor( 0.0, 0.0, 0.0, 0.0);
    glClearDepth( 1.0 );

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);

    // TODO tests
    return m_card.initCaptureDeviceGL();
}

void GrabberSDI::selectStreams()
{
    switch( m_card.getNumStreams() )
    {
        case 0:
            m_videoSize = ZeroUInt2;
        break;
        default:
            SetWidth(m_videoSize, m_card.getWidth());
            SetHeight(m_videoSize, m_card.getHeight());
        break;
    }

    // Select stream input
    SetBufId(m_stream1CaptureHandle, m_card.getBufferObjectHandle(0));
    SetSize(m_stream1CaptureHandle, m_videoSize);
    m_stream2CaptureHandle = m_stream1CaptureHandle;
    if( m_card.getNumStreams() >= 2 )
    {
        SetBufId(m_stream2CaptureHandle, m_card.getBufferObjectHandle(1));
        SetSize(m_stream2CaptureHandle, m_videoSize);
    }

}


bool GrabberSDI::captureVideo()
{
    if( m_state == OFFLINE )
    {
        if( !init() )
        {
            return false;
        }
    }

    if( m_state == CARD_INITIALIZED )
    {
        // Try to check input
        if( m_card.initInputDeviceGL() )
        {
            //
            setupCardGL();

            //
            selectStreams();
            
            m_state = INPUT_READY;
        }
    }

    if( m_state == INPUT_READY )
    {
        if( m_card.startCapture() )
        {
            m_state = CAPTURE_STARTED;
        }
    }

    if( m_state == CAPTURE_STARTED )
    {
        if( m_numFails < 100 )
        {
            cardCaptureVideo();
            return true;
        }
        else
        {
            m_state = CAPTURE_FAILED;
        }
    }

    if( m_state == CAPTURE_FAILED )
    {
        // TODO Release card buffer;
        // Change values
        // Change configuration
        m_videoSize = ZeroUInt2;

        // Stop capture
        m_card.endCapture();
        //m_card.destroyCaptureDeviceNVCtrl();
        //m_card.destroyCaptureDeviceGL();
        
        // Change state to CARD-INITIALIZED
        m_state = CARD_INITIALIZED;
    }

    return false;
}

