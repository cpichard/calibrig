#ifndef __SURF_HESSIAN_H__
#define __SURF_HESSIAN_H__

#include "ImageGL.h"

// Data stored in gpu, basic structure.
// TODO test with array of float instead of array of HessianPoint
// ie typedef struct{
//  float *x,
//
//}
typedef struct
{
    float m_x;
    float m_y;
    float m_scale;
    int m_laplacian;
}
HessianPoint;

// Data needed for an Hessian computation
struct HessianData
{
    HessianData();
    ~HessianData();

    // Alloc undelying data depending on image size
    void resizeImages( UInt2 imgSize );
    void allocImages( UInt2 imgSize );
    void freeImages();

    inline size_t & capacity() { return m_capacity; }

    // Static data
    static const int m_lobeCache [];
    static const int m_lobeCacheUnique [];
    static const int m_lobeMap [];
    static const int m_borderCache [];
    
    /*! The maximum number of interest points in an image is computed as follows:
     *   (image_width * image_height) / IMG_SIZE_DIVISOR
     */
    static const int IMG_SIZE_DIVISOR;

    // Octaves and intervals 
    unsigned int    m_octaves;
    unsigned int    m_intervals;
    size_t          m_capacity;

    //! Initial sampling step for interest point detection
    unsigned int m_initSample;

    // Image size refactor
    int m_width, m_height;

    //! Array stack of determinant of hessian values
    float *m_det;
    size_t m_detPitch;

    //! Threshold value for blob responses
    float m_thres;

    // Points found
    HessianPoint *m_dPoints; // device points
};


//
bool computeHessianDet( CudaImageBuffer<float> &img, CudaImageBuffer<float> &det, HessianData &params );
inline size_t & NbElements( HessianData &hd ){ return hd.capacity(); }

#endif // __SURF_HESSIAN_H__