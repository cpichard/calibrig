#include "VisualComfortScreen.h"

#include "ImageProcessing.h"

VisualComfortScreen::VisualComfortScreen( Display *dpy, UInt2 winSize )
: ScreenLayout( dpy, winSize )
, m_mon()
{}

VisualComfortScreen::~VisualComfortScreen()
{
    freeImage();
}

void VisualComfortScreen::draw()
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

void VisualComfortScreen::allocImage( UInt2 &imgSize )
{
    allocBufferAndTexture( m_leftImg, imgSize );
    allocBufferAndTexture( m_rightImg, imgSize );
    allocBufferAndTexture( m_warpedRightImg, imgSize );
    allocBufferAndTexture( m_warpedLeftImg, imgSize );
    allocBufferAndTexture( m_warpedImg, imgSize );
}

void VisualComfortScreen::freeImage()
{
    releaseBufferAndTexture( m_rightImg );
    releaseBufferAndTexture( m_leftImg );
    releaseBufferAndTexture( m_warpedRightImg );
    releaseBufferAndTexture( m_warpedLeftImg );
    releaseBufferAndTexture( m_warpedImg );
}

void VisualComfortScreen::resizeImage( UInt2 &imgSize )
{
    freeImage();
    allocImage( imgSize );
    m_mon.resizeImage(imgSize);
}

void VisualComfortScreen::updateResult()
{
}

void VisualComfortScreen::nextFrame()
{
    streamsToRGB( m_stream1, m_leftImg  );
    streamsToRGB( m_stream2, m_rightImg );
    
    Deformation &d = m_analysisResult->m_d;
    warpImage(m_leftImg, m_warpedLeftImg, d.m_h2); 
    warpImage(m_rightImg, m_warpedRightImg, d.m_h1); 
    
    diffImage( m_warpedLeftImg, m_warpedRightImg, m_warpedImg );
    
    m_mon.updateWithImageBuffer( m_warpedImg, 4 );

}

