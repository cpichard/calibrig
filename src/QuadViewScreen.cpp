#include "QuadViewScreen.h"

#include <iomanip>
#include <iostream>
#include <cstdlib>

QuadViewScreen::QuadViewScreen( Display *dpy, UInt2 winSize  )
:ScreenLayout( dpy, winSize ),
    m_mon1(),
    m_mon2(),
    m_mon3(),
    m_mon4()
{}

void QuadViewScreen::draw()
{
    // Set view parameters.
    glViewport(0, 0, winWidth(), winHeight());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);

    // Background color
    glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Set draw color.
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    // Draw textured quad 1 in graphics window.
    glLoadIdentity();
    glScalef(0.5,0.5,0);
    glTranslatef( -1, 1,0 );
    m_mon1.drawGL();
    drawText(-0.95,0.80,"Monitor 1" );

    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
    if( BufId(m_stream1) == 0 )
    {
        drawText(0,0, "No Signal" );
    }

    // Draw textured quad 2 in graphics window.
    glLoadIdentity();
    glScalef( 0.5, 0.5, 0);
    glTranslatef( 1, 1, 0 );
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    m_mon2.drawGL();
    drawText(-0.95, 0.80, "Monitor 2" );

    if( BufId(m_stream2) == 0 )
    {
        drawText(0,0, "No Signal" );
    }

    // Draw textured quad 3 in graphics window.
    glLoadIdentity();
    glScalef( 0.5, 0.5, 0);
    glTranslatef( -1, -1, 0 );
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    m_mon3.drawGL();
    drawText(-0.95, 0.80, "Monitor 3 - Left" );

    // Draw textured quad 4 in graphics window.
    glLoadIdentity();
    glScalef( 0.5, 0.5, 0);
    glTranslatef( 1, -1, 0 );
    m_mon4.drawGL();
    drawText(-0.95, 0.80, "Monitor 4 - Right" );

    if( m_analysisResult )
    {
        // TODO : use functions instead of d
        Deformation &d = m_analysisResult->m_d;
        // Result values
        std::stringstream strstr;
        strstr << std::setprecision(4) << std::right << std::fixed
                 << " Tx = "    << std::setw( 5 ) << d.m_tx
                 << " Ty = "    << std::setw( 5 ) << d.m_ty
                 << " Rot = "   << std::setw( 5 ) << d.m_rot
                 << " Scale = "   << std::setw( 5 ) << d.m_scale;

        glLoadIdentity();
        glScalef( 0.5, 0.5, 0);
        glTranslatef( -1, -1, 0 );
        glColor4f(1.0f, 0.0, 0.0f, 0.2f);
        drawText(-0.95, 1.1, strstr.str());

        // Draw second result informations
        std::stringstream strstr2;
        strstr2 << std::setprecision(4) << std::right << std::fixed
                 << " R = " << std::setw( 5 ) << d.m_nbPtsLeft << "pts"
                 << " L = " << std::setw( 5 ) << d.m_nbPtsRight << "pts"
                 << " Matches = " << std::setw( 5 ) << d.m_nbMatches;

        glLoadIdentity();
        glScalef( 0.5, 0.5, 0);
        glTranslatef( -1, -1, 0 );
        glColor4f(1.0f, 0.0, 0.0f, 0.2f);
        drawText(-0.95, 0.95, strstr2.str());

        // Draw points on left monitor
        const float imgWidth = static_cast<float> ( Width( m_mon1.imageSize() ) );
        const float imgHeight = static_cast<float>( Height(m_mon1.imageSize() ) );
        const float ratio = imgWidth/imgHeight;
        glPointSize(4);
        glLoadIdentity();
        glTranslatef( -1, -1, 0 );
        m_analysisResult->drawLeftKeyPoints(ratio);

        // Draw points on right monitor
        glPointSize(4);
        glLoadIdentity();
        glTranslatef( 0, -1, 0 );
        glColor4f(1.0f, 0.0, 0.0f, 0.2f);
        m_analysisResult->drawRightKeyPoints(ratio);
        
    }
}

void QuadViewScreen::resizeImage( UInt2 &imgSize )
{
    m_mon1.resizeImage(imgSize);
    m_mon2.resizeImage(imgSize);
    m_mon3.resizeImage(imgSize);
    m_mon4.resizeImage(imgSize);
}

void QuadViewScreen::updateResult()
{
    if( m_analysisResult )
    {
        m_analysisResult->updateLeftMonitor(m_mon3);
        m_analysisResult->updateRightMonitor(m_mon4);
    }
}

void QuadViewScreen::nextFrame()
{
    m_mon1.updateWithSDIVideo( m_stream1 );
    m_mon2.updateWithSDIVideo( m_stream2 );
}

