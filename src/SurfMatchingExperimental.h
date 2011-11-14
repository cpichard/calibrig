#ifndef __SURFMATCHINGEXPERIMENTAL_H__
#define __SURFMATCHINGEXPERIMENTAL_H__

#include <SurfDescriptor.h>
#include <SurfMatching.h>
#include "cv.h"
#include <vector>


bool computeMatchingExperimental( DescriptorData &leftDesc, DescriptorData &rightDesc,
        vector<CvPoint2D32f> &pt1, vector<CvPoint2D32f> &pt2,
        UInt2 &imgSize);

#endif//__SURFMATCHINGEXPERIMENTAL_H__
