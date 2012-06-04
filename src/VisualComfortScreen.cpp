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
    //std::cout << "allocating" << Width(imgSize) << " " << Height(imgSize) << std::endl;
    allocBufferAndTexture( m_leftImg, imgSize );
    allocBufferAndTexture( m_rightImg, imgSize );
    allocBufferAndTexture( m_warpedRightImg, imgSize );
    allocBufferAndTexture( m_warpedLeftImg, imgSize );
    allocBufferAndTexture( m_warpedImg, imgSize );

    // Down rez size
    UInt2 downRezSize(Width(imgSize)/4, Height(imgSize)/1);

    allocBufferAndTexture(m_downRezImgRight, downRezSize);
    allocBufferAndTexture(m_downRezImgLeft, downRezSize);
    allocBufferAndTexture(m_resultImgRight, downRezSize);
    allocBufferAndTexture(m_resultImgLeft, downRezSize);
}

void VisualComfortScreen::freeImage()
{
    releaseBufferAndTexture( m_rightImg );
    releaseBufferAndTexture( m_leftImg );
    releaseBufferAndTexture( m_warpedRightImg );
    releaseBufferAndTexture( m_warpedLeftImg );
    releaseBufferAndTexture( m_warpedImg );
    releaseBufferAndTexture( m_downRezImgLeft);
    releaseBufferAndTexture( m_downRezImgLeft);
    releaseBufferAndTexture( m_resultImgLeft);
    releaseBufferAndTexture( m_resultImgRight);
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
    //warpImage(m_leftImg, m_warpedLeftImg, d.m_h2);
    //warpImage(m_rightImg, m_warpedRightImg, d.m_h1);
    //resizeImageGL(m_warpedLeftImg, m_downRezImgRight);
    //resizeImageGL(m_warpedRightImg, m_downRezImgLeft);
    //copyImageBuffer(m_leftImg, m_warpedImg);   
    resizeImageGL(m_leftImg, m_downRezImgRight);
    resizeImageGL(m_rightImg, m_downRezImgLeft);
    visualComfort( m_downRezImgLeft, m_downRezImgRight, m_resultImgLeft, m_resultImgRight );
    resizeImageGL( m_resultImgLeft, m_warpedImg );
    //resizeImageGL(m_downRezImgLeft, m_warpedImg);
    m_mon.updateWithImageBuffer( m_warpedImg, 4 );

    //m_mon.updateWithImageBuffer( m_leftImg, 4 );
}

