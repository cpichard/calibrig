#ifndef __OCVUTILS_H__
#define __OCVUTILS_H__

#include "cv.h"
#include "highgui.h"

//
// OpenCV vectors Utilities.
//

// Define a new type : vectors.
// Used just to simplify the writing
typedef CvMat CvVector;

// Alloc/Free vector
inline CvVector *cvCreateVector( int size, int type ) { return static_cast<CvVector*>(cvCreateMat(size, 1, type));}
inline void cvReleaseVector(CvVector **v){cvReleaseMat(v);}

// Get/Set
// TODO : change to cvvGet and cvvSet
inline double cvGet(CvVector *v, int index){ return cvmGet(v, index, 0); }
inline void cvSet(CvVector *v, int index, double value){cvmSet(v, index, 0, value); }

// Clone vectors
inline CvVector *cvCloneVector(CvVector *src){ return static_cast<CvVector*>(cvCloneMat(static_cast<CvMat*>(src))); }

// Returns the size of a vector
inline size_t cvSize(CvVector*v){return static_cast<size_t>(v->rows);}

// 
// Utils
//

inline void cvSetCol(CvMat *dst, CvVector *src, int col)
{
    for(int i=0; i<cvSize(src); i++)
    {
        cvmSet(dst, i, col, cvGet(src,i));
    }
}

inline CvMat * cvCreateTranspose(CvMat *mat)
{
    CvMat *ret = cvCreateMat(mat->cols, mat->rows, mat->type);
    cvTranspose(mat, ret);
    return ret;
}


#endif//__OCVUTILS_H__

