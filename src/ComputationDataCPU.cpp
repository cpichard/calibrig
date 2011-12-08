#include "ComputationDataCPU.h"

void ComputationDataCPU::updateRightMonitor( Monitor &mon )
{
    if( leftImage() )
    {
        mon.updateWithImageBuffer( (unsigned char*)rightImage()->imageData,rightImage()->width, rightImage()->height, 1 );
    }
}

void ComputationDataCPU::updateLeftMonitor( Monitor &mon )
{
    if( rightImage() )
    {
        mon.updateWithImageBuffer( (unsigned char*)leftImage()->imageData, leftImage()->width, leftImage()->height, 1 );
    }
}


void ComputationDataCPU::drawLeftKeyPoints(float ratio)
{
    glBegin(GL_POINTS);
    for( int i = 0; i < m_leftKeypoints->total; i++ )
    {
        CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( m_leftKeypoints, i );
        glVertex2f((r->pt.x+0.5), (r->pt.y*ratio+0.5));
    }
    glColor4f(0.0f, 1.0, 0.0f, 0.2f);
    for( int i = 0; i < m_d.m_nbMatches; i++ )
    {
        CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( m_leftKeypoints,m_ptpairs[i*2]);
        glVertex2f((r->pt.x+0.5), (r->pt.y*ratio+0.5));
    }
    glEnd();
}


void ComputationDataCPU::drawRightKeyPoints(float ratio)
{
    glBegin(GL_POINTS);
    for( int i = 0; i < m_rightKeypoints->total; i++ )
    {
        CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( m_rightKeypoints, i );
        glVertex2f((r->pt.x+0.5), (r->pt.y*ratio+0.5));
    }
    glColor4f(0.0f, 1.0, 0.0f, 0.2f);
    for( int i = 0; i < m_d.m_nbMatches; i++ )
    {
        CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem(m_rightKeypoints,m_ptpairs[i*2+1]);
        glVertex2f((r->pt.x+0.5), (r->pt.y*ratio+0.5));
    }
    glEnd();
}

void ComputationDataCPU::drawVerticalDisparity()
{
    glColor4f(0.3, 0.3, 0.35f, 0.2f);
    glBegin(GL_QUADS);
        glVertex3f(0.0, 0.0, 0.1 );
        glVertex3f(0.0, 1.0, 0.1);
        glVertex3f((m_d.s_histogramBinSize)*m_binWidth, 1.0, 0.1 );
        glVertex3f((m_d.s_histogramBinSize)*m_binWidth, 0.0, 0.1 );
    glEnd();

    glColor4f(0.3, 0.3, 0.5f, 0.2f);
    for(unsigned int i=0; i < m_d.s_histogramBinSize; i++ )
    {
        float ratio = 2.f*fabs( (float)i/m_d.s_histogramBinSize -0.5);
        glColor4f(ratio/2.f+0.5, (1.f-ratio), 0, 0.2f);
        glBegin(GL_QUADS);
            glVertex3f(i*m_binWidth, 0,0 );
            glVertex3f(i*m_binWidth, (float)m_d.m_vdisp[i]/(float)m_d.m_nbMatches,0);
            glVertex3f((i+1)*m_binWidth, (float)m_d.m_vdisp[i]/(float)m_d.m_nbMatches,0 );
            glVertex3f((i+1)*m_binWidth,0,0);
        glEnd();
    }
};

void ComputationDataCPU::drawHorizontalDisparity()
{
    glColor4f(0.3, 0.3, 0.35f, 0.2f);
    glBegin(GL_QUADS);
        glVertex3f(0.0, 0.0, 0.1 );
        glVertex3f(0.0, 1.0, 0.1);
        glVertex3f((m_d.s_histogramBinSize)*m_binWidth, 1.0, 0.1 );
        glVertex3f((m_d.s_histogramBinSize)*m_binWidth, 0.0, 0.1 );
    glEnd();

    glColor4f(0.3, 0.3, 0.5f, 0.2f);
    for(unsigned int i=0; i < m_d.s_histogramBinSize; i++ )
    {
        float ratio = 2.f*fabs( (float)i/m_d.s_histogramBinSize -0.5);
        glColor4f(ratio/2.f+0.5, (1.f-ratio), 0, 0.2f);
        glBegin(GL_QUADS);
            glVertex3f(i*m_binWidth, 0,0 );
            glVertex3f(i*m_binWidth, (float)m_d.m_hdisp[i]/(float)m_d.m_nbMatches, 0);
            glVertex3f((i+1)*m_binWidth, (float)m_d.m_hdisp[i]/(float)m_d.m_nbMatches, 0);
            glVertex3f((i+1)*m_binWidth,0,0);
        glEnd();
    }
}

std::string ComputationDataCPU::infos()
{
    return std::string("Mode CPU");
}
