#ifndef __SURFMATCHING_H__
#define __SURFMATCHING_H__

#include <SurfDescriptor.h>
#include "cv.h"
#include <vector>

struct MatchedPoints
{
    float m_lx;   // left X
    float m_ly;
    float m_rx;
    float m_ry;
    float m_ratio; // Matching ratio between the best and second best point
};

inline
bool operator == (const MatchedPoints &r, const MatchedPoints &l )
{
    return r.m_lx == l.m_lx
        && r.m_ly == l.m_ly
        && r.m_rx == l.m_rx
        && r.m_ry == l.m_ry
        && r.m_ratio == l.m_ratio;
}

#include<map>
#include <boost/functional/hash.hpp>
inline
std::size_t hash_value( MatchedPoints const &m )
{
    std::size_t seed = 0;
    boost::hash_combine(seed,m.m_lx);
    boost::hash_combine(seed,m.m_ly);

    return seed;
}

typedef
std::map< std::size_t, MatchedPoints > MatchedPointSet;

    
bool computeMatching( DescriptorData &leftDesc, DescriptorData &rightDesc,
        vector<CvPoint2D32f> &pt1, vector<CvPoint2D32f> &pt2,
        UInt2 &imgSize);

bool computeMatchingExperimental( DescriptorData &leftDesc, DescriptorData &rightDesc,
        vector<CvPoint2D32f> &pt1, vector<CvPoint2D32f> &pt2,
        UInt2 &imgSize);
void copyPoints( vector<CvPoint2D32f> &leftPts, vector<CvPoint2D32f> &rightPts, unsigned int nbRighDesc, MatchedPoints *matchedPoints_h, unsigned int &nbMatchedPoints  );
#endif//__SURFMATCHING_H__
