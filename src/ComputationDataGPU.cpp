#include "ComputationDataGPU.h"
#include "ImageProcessing.h"
#include "VertexBufferObject.h"
#include "SurfDescriptor.h"

ComputationDataGPU::ComputationDataGPU( ImageGL &l, ImageGL &r, VertexBufferObject &pointsR, VertexBufferObject &pointsL )
:ComputationData()
{
    m_leftMatchedPts.reserve(3000);
    m_rightMatchedPts.reserve(3000);
    
    allocBufferAndTexture(m_imgLeft,Size(l));
    allocBufferAndTexture(m_imgRight,Size(r));

    allocBuffer(m_pointsL, Capacity(pointsL));
    allocBuffer(m_pointsR, Capacity(pointsR));

    copyImageBuffer( l, m_imgLeft);
    copyImageBuffer( r, m_imgRight);

    copyVertexBuffer(pointsR, m_pointsR);
    copyVertexBuffer(pointsL, m_pointsL);

}

ComputationDataGPU::~ComputationDataGPU()
{
    releaseBuffer(m_pointsL);
    releaseBuffer(m_pointsR);
    releaseBufferAndTexture(m_imgLeft);
    releaseBufferAndTexture(m_imgRight);
    m_rightMatchedPts.clear();
    m_leftMatchedPts.clear();
}

// Computation data knows how to update a monitor
void ComputationDataGPU::updateRightMonitor( Monitor &mon )
{
    if( BufId(m_imgRight) != 0 )
    {
        mon.updateWithImageBuffer( m_imgRight, 4 );
    }
}

void ComputationDataGPU::updateLeftMonitor ( Monitor &mon )
{
    if( BufId(m_imgLeft) != 0 )
    {
        mon.updateWithImageBuffer( m_imgLeft, 4 );
    }
}

void ComputationDataGPU::drawLeftKeyPoints(float ratio)
{
    drawObject( m_pointsL );
    
    // TODO replace with full GPU
    glBegin(GL_POINTS);
    glColor4f(0.0f, 1.0, 0.0f, 0.2f);
    for( int i = 0; i < m_d.m_nbMatches; i++ )
    {
        glVertex2f((m_leftMatchedPts[i].x+0.5), (m_leftMatchedPts[i].y*ratio+0.5));
    }
    glEnd();
}

void ComputationDataGPU::drawRightKeyPoints(float ratio)
{
    drawObject( m_pointsR );

    // TODO replace with full GPU
    glBegin(GL_POINTS);
    glColor4f(0.0f, 1.0, 0.0f, 0.2f);
    for( int i = 0; i < m_d.m_nbMatches; i++ )
    {
        glVertex2f((m_rightMatchedPts[i].x+0.5), (m_rightMatchedPts[i].y*ratio+0.5));
    }
    glEnd();
}

void ComputationDataGPU::drawVerticalDisparity()
{
    drawHistogram( m_d.m_vdisp );
}

void ComputationDataGPU::drawHorizontalDisparity()
{
    drawHistogram( m_d.m_hdisp );
}

void ComputationDataGPU::drawHistogram( float *hist )
{
    // TODO : full gpu use vertex buffer object
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
            glVertex3f(i*m_binWidth, (float)hist[i]/(float)m_d.m_nbMatches, 0);
            glVertex3f((i+1)*m_binWidth, (float)hist[i]/(float)m_d.m_nbMatches, 0);
            glVertex3f((i+1)*m_binWidth,0,0);
        glEnd();
    }
}

std::string ComputationDataGPU::infos()
{
    return std::string("Mode GPU");    
}

