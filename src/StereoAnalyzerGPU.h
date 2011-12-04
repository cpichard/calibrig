#ifndef __STEREOANALYZERGPU_H__
#define __STEREOANALYZERGPU_H__

#include "StereoAnalyzer.h"
#include "SurfGPUData.h"
#include "SurfHessian.h"
#include "SurfDescriptor.h"
#include "SurfCollectPoints.h"
#include "ComputationDataGPU.h"
class StereoAnalyzerGPU : public StereoAnalyzer
{
public:
    StereoAnalyzerGPU();
    virtual ~StereoAnalyzerGPU();
    
    void updateRightImageWithSDIVideo( ImageGL &videoPBO );
    void updateLeftImageWithSDIVideo ( ImageGL &videoPBO );
    void processImages();

    void analyse();

    void resizeImages( UInt2 imgSize );
    void allocImages( UInt2 imgSize );
    void freeImages();

    virtual ComputationData * acquireLastResult();
    void acceptCommand( const Command &command );
    
    // Displaying
    ImageGL m_imgRight;
    ImageGL m_imgLeft;

    VertexBufferObject m_rightPoints;
    VertexBufferObject m_leftPoints;

private:

    void computeSurfDescriptors( ImageGL &img, DescriptorData &descripors );
    void computeDisparity();
    
    ImageGL m_warpedTmp;

    // Temporary buffers needed
    CudaImageBuffer<float> m_satImage;
    CudaImageBuffer<float> m_hesImage;
    HessianData     m_hessianData;

    // TODO : refactor ?? replace by SurfGPUComputationData ?
    DescriptorData  m_rightDescriptors;
    DescriptorData  m_leftDescriptors;

    // Points used to compute homography
    vector<CvPoint2D32f> m_leftMatchedPts, m_rightMatchedPts;
    boost::mutex                m_matchMutex;
    ComputationDataGPU *        m_result;
    int                         m_histoRange;
    float                       m_sentThreshold;
};

#endif // __STEREOANALYSERGPU_H__

