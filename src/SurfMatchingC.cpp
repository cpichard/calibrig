
#include <SurfMatching.h>


void copyPoints( vector<CvPoint2D32f> &leftPts, vector<CvPoint2D32f> &rightPts, unsigned int nbRightDesc, MatchedPoints *matchedPoints_h, unsigned int &nbMatchedPoints  )
{
	// Disambiguation of matches
	MatchedPointSet matchedPointSet;
	std::pair<MatchedPointSet::iterator,bool> insertOk;
	for( unsigned int i=0; i < nbRightDesc; i++ )
	{
	    if( matchedPoints_h[i].m_ratio < 0.6 )
	    {
            // Use of a map and a hash
            boost::hash<MatchedPoints> hasher;
            std::size_t key = hasher(matchedPoints_h[i]);

            insertOk = matchedPointSet.insert( std::make_pair(key,matchedPoints_h[i] ) );
            if( insertOk.second == false )
            {
                if( (*insertOk.first).second.m_ratio > matchedPoints_h[i].m_ratio )
                {
                    matchedPointSet.erase(insertOk.first);
                    matchedPointSet.insert( std::make_pair(key,matchedPoints_h[i] )  );
                }
            }
	    }
	}

	// Resize to numbers of found values
	nbMatchedPoints = matchedPointSet.size();
	leftPts.resize(nbMatchedPoints);
	rightPts.resize(nbMatchedPoints);

    // Copy points
	MatchedPointSet::iterator it = matchedPointSet.begin();
	for( unsigned int i=0; i < nbMatchedPoints; i++, ++it )
	{
	    leftPts[i].x = it->second.m_lx;
	    leftPts[i].y = it->second.m_ly;
	    rightPts[i].x = it->second.m_rx;
	    rightPts[i].y = it->second.m_ry;
	}
}

