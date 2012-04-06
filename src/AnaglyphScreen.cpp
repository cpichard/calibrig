
#include "AnaglyphScreen.h"

#include "Monitor.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cuda.h>
#include <cudaGL.h>

#include "CudaUtils.h"
#include "ImageProcessing.h"

// BAD
#define PI 3.14


AnaglyphScreen::AnaglyphScreen( Display *dpy,  UInt2 winSize  )
:ScreenLayout( dpy, winSize ),
    m_mon()
{}

AnaglyphScreen::~AnaglyphScreen()
{
    freeImage();
}


void AnaglyphScreen::draw()
{
    // Set view parameters.
    glViewport(0, 0, winWidth(), winHeight());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);

    // BAckground color
    glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Set draw color.
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glLoadIdentity();
    m_mon.drawGL();
}

void AnaglyphScreen::allocImage( UInt2 &imgSize )
{
    allocBufferAndTexture( m_anaglyphImg, imgSize);
    allocBufferAndTexture( m_leftImg, imgSize );
    allocBufferAndTexture( m_rightImg, imgSize );
}

void AnaglyphScreen::freeImage()
{
    releaseBufferAndTexture( m_rightImg );
    releaseBufferAndTexture( m_leftImg );
    releaseBufferAndTexture( m_anaglyphImg );
}

void AnaglyphScreen::resizeImage( UInt2 &imgSize )
{
    freeImage();
    allocImage( imgSize );
    m_mon.resizeImage(imgSize);
}

void AnaglyphScreen::updateResult()
{
    
}

void AnaglyphScreen::nextFrame()
{
    if(BufId(m_stream1)==BufId(m_stream2))
    {
        streamsToRGB(m_stream1, m_leftImg);
        m_mon.updateWithImageBuffer(m_leftImg,4 );
        return;
    }

    // Convert to RGB
    streamsToRGB( m_stream1, m_leftImg  );
    streamsToRGB( m_stream2, m_rightImg );

    anaglyph( m_leftImg, m_rightImg, m_anaglyphImg );
    m_mon.updateWithImageBuffer( m_anaglyphImg, 4 );
}
