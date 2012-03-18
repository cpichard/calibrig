
#include "DiffScreen.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cuda.h>
#include <cudaGL.h>

#include "CudaUtils.h"
#include "ImageProcessing.h"

// BAD
#define PI 3.14


DiffScreen::DiffScreen( Display *dpy,  UInt2 winSize  )
: ScreenLayout( dpy, winSize )
, m_mon()
{}

DiffScreen::~DiffScreen()
{
    freeImage();
}

void drawTarget()
{
    glBegin(GL_LINES);
        glVertex2f(0, 0.05);
        glVertex2f(0, -0.05);
    glEnd();
    glBegin(GL_LINES);
        glVertex2f(0.05, 0);
        glVertex2f(-0.05, 0);
    glEnd();

    glBegin(GL_LINES);
        glVertex2f(0.3+.05, 0);
        glVertex2f(0.3-0.05, 0);
    glEnd();

    glBegin(GL_LINES);
        glVertex2f(0.0,0.3+.05);
        glVertex2f(0.0, 0.3-0.05);
    glEnd();

    glBegin(GL_LINES);
        glVertex2f(-0.3+.05, 0);
        glVertex2f(-0.3-0.05, 0);
    glEnd();

    glBegin(GL_LINES);
        glVertex2f(0.0,-0.3+.05);
        glVertex2f(0.0,-0.3-0.05);
    glEnd();

    glBegin(GL_LINE_STRIP);
    for(int i=0; i <= 360; i++)
        glVertex3f(sin(i*PI/180.f)*0.3, cos(i*PI/180.f)*0.3, 1);
    glEnd();
}

void DiffScreen::draw()
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

    if( m_analysisResult )
    {
        Deformation &d = m_analysisResult->m_d;
        // Result values
        std::stringstream strstr;
        strstr << std::setprecision(4) << std::right << std::fixed
                 << " Tx = "    << std::setw( 5 ) << d.m_tx
                 << " Ty = "    << std::setw( 5 ) << d.m_ty
                 << " Rot = "   << std::setw( 5 ) << d.m_rot
                 << " Scale = " << std::setw( 5 ) << d.m_scale;
        std::string resultText = strstr.str();

        glLoadIdentity();
        glScalef( 0.5, 0.5, 0);
        glTranslatef( -1, 0.2, 0 );
        glRasterPos3f(0.0, 0.0, 0.0f);
        glColor4f(1.0f, 0.0, 0.0f, 0.2f);
        glPushAttrib(GL_LIST_BIT);
            glListBase(m_fontDisplayList - ' ');
            glRasterPos3f(-0.95, 1.1, 0.0f);
            glCallLists(resultText.size(), GL_BYTE, resultText.c_str());
        glPopAttrib();

        glLoadIdentity();
        drawTarget();

        const float texWidth = (float)Width( m_mon.imageSize() );
        const float texHeight = (float)Height( m_mon.imageSize() );
        glLoadIdentity();
        glScalef( d.m_scale, d.m_scale, 0 );
        glRotatef( d.m_rot, 0.0, 0.0, 1.0 );
        glTranslatef( 2.f*d.m_tx/texWidth, -2.f*d.m_ty/texWidth, 0);

        glColor4f(0.0f, 1.0, 0.0f, 0.2f);
        drawTarget();
    }
    // DRAW TARGET
}

void DiffScreen::allocImage( UInt2 &imgSize )
{
    allocBufferAndTexture( m_warpedImg, imgSize);
    allocBufferAndTexture( m_leftImg, imgSize );
    allocBufferAndTexture( m_rightImg, imgSize );
}

void DiffScreen::freeImage()
{
    releaseBufferAndTexture( m_rightImg );
    releaseBufferAndTexture( m_leftImg );
    releaseBufferAndTexture( m_warpedImg );
}

void DiffScreen::resizeImage( UInt2 &imgSize )
{
    freeImage();
    allocImage( imgSize );
    m_mon.resizeImage(imgSize);
}

void DiffScreen::updateResult()
{
    
}

void DiffScreen::nextFrame()
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
    diffImage( m_leftImg, m_rightImg, m_warpedImg );

    // TEST
    //float matrix[9];
    //warpImage( m_leftImg, m_warpedImg, matrix );

    m_mon.updateWithImageBuffer( m_warpedImg, 4 );
}
