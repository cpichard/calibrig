#ifndef __HOMOGRAPHYANALYZERGPU_H___
#define __HOMOGRAPHYANALYZERGPU_H___ 

#include "StereoAnalyzer.h"
#include "SurfGPUData.h"
#include "SurfHessian.h"
#include "SurfDescriptor.h"
#include "SurfCollectPoints.h"
#include "ComputationDataGPU.h"

class HomographyAnalyzerGPU : public StereoAnalyzer
{
public:
    HomographyAnalyzerGPU();
    virtual ~HomographyAnalyzerGPU();
    
    void updateRightImageWithSDIVideo( ImageGL &videoPBO );
    void updateLeftImageWithSDIVideo ( ImageGL &videoPBO );

    void analyse();

    void resizeImages( UInt2 imgSize );
    void allocImages( UInt2 imgSize );
    void freeImages();

    virtual ComputationData * acquireLastResult();
    virtual void disposeResult(ComputationData *);
    void acceptCommand( const Command &command );
    
    // Displaying
    ImageGL m_imgRight;
    ImageGL m_imgLeft;

    VertexBufferObject m_rightPoints;
    VertexBufferObject m_leftPoints;

private:

    void computeSurfDescriptors( CudaImageBuffer<float> &satImage, DescriptorData &descripors );
    void computeDisparity();
    
    ImageGL m_warpedTmp;

    // Temporary buffers needed
    CudaImageBuffer<float> m_satLeftImage;
    CudaImageBuffer<float> m_satRightImage;
    CudaImageBuffer<float> m_hesImage;
    HessianData     m_hessianData;

    // TODO : refactor ?? replace by SurfGPUComputationData ?
    DescriptorData  m_rightDescriptors;
    DescriptorData  m_leftDescriptors;

    // Points used to compute homography
    ComputationDataGPU *        m_result;
    ComputationDataGPU *        m_toDispose;
    int                         m_histoRange;
    float                       m_sentThreshold;
    unsigned int                m_maxNumberOfPoints;
};

#endif // __HOMOGRAPHYANALYZERGPU_H__ 

