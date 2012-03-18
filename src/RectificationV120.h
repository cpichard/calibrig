#ifndef __RECTIFICATIONV120_H__
#define __RECTIFICATIONV120_H__


//
// Rectification V120
//
// Implementation of a method initially used to rectify multiple view.
// This implementation use only a stereo pair.
//
// Details of the method can be found in 
// Multiple View Image Rectification,
// Vincent Nozick, in proc. of IEEE-ISAS 2011, 
// International Symposium on Access Spaces, pages 277–282, Yokohama, Japan, 17-19, june 2011.
//

#include "OCVUtils.h"

// Take a list of points, image size and returns 2 camera rectification homographies for a stereo pair
int v120Rectify( const CvMat* _points1, const CvMat* _points2, CvSize imgSize, CvMat* _H1, CvMat* _H2 );

// Calculation functor
struct RectifyV120Functor
{
    // Constant values 
    const CvSize &m_imgSize;
    const CvMat*m_points1;
    const CvMat*m_points2;
    CvMat *m_K1ori; // Cam1 original matrix
    CvMat *m_K2ori; // Cam2 original matrix

    // Constructor
    RectifyV120Functor(const CvSize &imgSize, const CvMat* _points1, const CvMat* _points2);
    ~RectifyV120Functor();

    // Allocate a parameters vector 
    inline CvVector * allocParamVector(const CvSize &imgSize)
    {
        CvVector *retVect = cvCreateVector(8, CV_64F);
        // R1x R1y R1z F1 R2x R2y R2z F2
        cvSetZero(retVect); 
        return retVect;
    }

    // Allocate a vector with the size of the result vector
    inline CvVector * allocResultVector(){return cvCreateVector(1, CV_64F);}

    // Compute cam1 correction homography from params 
    inline void setH1(CvMat *H1, CvVector *params)
    {
        rectificationMatrix( H1, m_K1ori, cvGet(params,0), cvGet(params,1), cvGet(params,2), cvGet(params,3) ); 
    }

    // Compute cam2 correction homography from params 
    inline void setH2(CvMat *H2, CvVector *params)
    {
        rectificationMatrix( H2, m_K2ori, cvGet(params,4), cvGet(params,5), cvGet(params,6), cvGet(params,7) ); 
    }

    // Function :  results = f(params)
    void operator()(CvVector *params, CvVector *results); 

    // Utils
    void eulerAngles(CvMat *M, double rx, double ry, double rz);
    void rectificationMatrix(CvMat *M, CvMat *Kori, double rx, double ry, double rz, double f);
};


#endif//__RECTIFICATIONV120_H__

