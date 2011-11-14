#ifndef __SURFDESCRIPTOR_H__
#define __SURFDESCRIPTOR_H__

#include "SurfHessian.h"

// Ipoint struct for CUDA implementation of SURF descriptor computation.
typedef struct
{
    //! Coordinates of the detected interest point
    float m_x;
    float m_y;

    //! Detected scale
    float m_scale;

    //! Orientation measured anti-clockwise from +ve x-axis
    float m_orientation;

    //! Sign of laplacian for fast matching purposes
    int m_laplacian;

    //! Vector of descriptor components
    float m_descriptor[64];

    //! Length of the partial descriptor vector for each 4x4 subsquare
    float m_lengths[4][4];

    //! Placeholds for point motion (can be used for frame to frame motion analysis)
    float m_dx, m_dy;

    //! Used to store cluster index
    int m_clusterIndex;
} SurfDescriptorPoint; 

struct DescriptorData
{
    DescriptorData();
    ~DescriptorData();

    // Realloc m_descPoints
    bool reallocPoints(unsigned int newSize);

    //! Device memory containing interest points
    SurfDescriptorPoint *m_descPoints;
    size_t              m_nbIPoints;


};

bool computeDescriptors( CudaImageBuffer<float> &imgSat, DescriptorData & );
inline unsigned int & NbElements( DescriptorData &dd ){ return dd.m_nbIPoints; }

#endif//__SURFDESCRIPTOR_H__
