#include "StereoAnalyzer.h"

#include "NvSDIin.h"


#include <vector>
#include <iomanip>
#include <iostream>

using std::vector;


// Constructor
StereoAnalyzer::StereoAnalyzer( unsigned int nbChannelsInSDIVideo )
    :   m_imgWidth(0)
    ,   m_imgHeight(0)
    ,   m_nbChannelsInSDIVideo(nbChannelsInSDIVideo)
    ,   m_leftImageIsNew(false)
    ,   m_rightImageIsNew(false)
    ,   m_mutex()
    ,   m_imgMutex()
{
}

StereoAnalyzer::~StereoAnalyzer()
{
}
