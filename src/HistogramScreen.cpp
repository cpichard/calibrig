#include <HistogramScreen.h>

HistogramScreen::HistogramScreen( Display *dpy, UInt2 winSize )
:ScreenLayout( dpy,winSize )
{}

HistogramScreen::~HistogramScreen()
{}

void HistogramScreen::draw()
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

    // GPU Memory
    #define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
    #define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

    GLint total_mem_kb = 0;
    glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX,
                  &total_mem_kb);

    GLint cur_avail_mem_kb = 0;
    glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX,
                  &cur_avail_mem_kb);

    std::stringstream gpumemstr;
    gpumemstr << "Available GPU memory " << cur_avail_mem_kb << "/" << total_mem_kb ;

    glLoadIdentity();
    glScalef( 0.5, 0.5, 0);
    glTranslatef( -1, -2.3, 0 );
    glColor4f(0.7, 0.7, 0.7, 0.2f);
    drawText(-0.95,0.50, gpumemstr.str());

    if(m_analysisResult)
    {
        Deformation &d = m_analysisResult->m_d;
        const float mid = ((float)d.s_histogramBinSize+1.0)*m_analysisResult->m_binWidth/2.f;

        // Vertical disparity
        glLoadIdentity();
        glScalef( 0.25, 0.5, 0);
        glTranslatef( -3.6, 0.4, 0 );
        glColor4f(1.0f, 1.0, 0.5f, 0.2f);
        m_analysisResult->drawVerticalDisparity();
        drawText(0.0,1.0,"Vertical disparity" );

        // Draw horizontal ligne
        glColor4f(1.0f, 0.0, 0.f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(mid,0.0,0);
        glVertex3f(mid,1.0,0);
        glEnd();

        // Horizontal disparity
        glLoadIdentity();
        glScalef( 0.25, 0.5, 0);
        glTranslatef( -3.6, -1.6, 0 );
        glColor4f(1.0f, 1.0, 0.5f, 0.2f);
        m_analysisResult->drawHorizontalDisparity();
        drawText(0.0,1.0,"Horizontal disparity");

        glColor4f(1.0f, 0.0, 0.f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(mid,0.0,0);
        glVertex3f(mid,1.0,0);
        glEnd();
    }
    
    // Current threshold value
    glLoadIdentity();
    glTranslatef( 0.2, 0.60, 0 );
    std::stringstream threshold;
    threshold << "Threshold : " << m_analysisResult->m_thresholdUsed; 
    glColor4f(0.95, 0.97, 0.95, 0.0f);
    drawText(0.0, 0.0, threshold.str());    

    // Current mode
    glLoadIdentity();
    glTranslatef( 0.2, 0.70, 0 );
    glColor4f(0.95, 0.97, 0.95, 0.0f);
    drawText(0.0, 0.0, m_analysisResult->infos());    
}

void HistogramScreen::resizeImage( UInt2 &imgSize )
{

}

void HistogramScreen::updateResult()
{

}

void HistogramScreen::nextFrame()
{

}
