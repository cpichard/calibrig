
#include <SurfMatching.h>


void copyPoints( vector<CvPoint2D32f> &leftPts, vector<CvPoint2D32f> &rightPts, unsigned int nbRightDesc, MatchedPoints *matchedPoints_h, unsigned int &nbMatchedPoints  )
{
	// resize points buffers to full capacity
	leftPts.resize(leftPts.capacity());
	rightPts.resize(rightPts.capacity());

	// Disambiguation of matches
	MatchedPointSet matchedPointSet;
	std::pair<MatchedPointSet::iterator,bool> insertOk;
	for( unsigned int i=0; i < nbRightDesc; i++ )
	{
	    if( matchedPoints_h[i].m_ratio < 0.6 )
	    {
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

	// copy results in pt1 and pt2
	nbMatchedPoints = matchedPointSet.size();
	MatchedPointSet::iterator it = matchedPointSet.begin();
	//std::cout << "Matched points = " << matchedPoints << std::endl;
	for( unsigned int i=0; i < nbMatchedPoints; i++, ++it )
	{
	    leftPts[i].x = it->second.m_lx;
	    leftPts[i].y = it->second.m_ly;
	    rightPts[i].x = it->second.m_rx;
	    rightPts[i].y = it->second.m_ry;
	}

	// Resize to numbers of found values
	leftPts.resize(nbMatchedPoints);
	rightPts.resize(nbMatchedPoints);
}

