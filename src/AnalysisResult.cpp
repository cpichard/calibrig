#include <sstream>
#include "AnalysisResult.h"

// Application version
extern const char *version;

// Create a reply
void makeResultReply( const AnalysisResult &d, std::string &reply )
{
    std::stringstream str;
    str << "{ \"tx\":" << d.m_tx;
    str << ", \"ty\":" << d.m_ty;
    str << ", \"rot\":" << d.m_rot;
    str << ", \"scale\":" << d.m_scale;
    str << ", \"pts_r\":" << d.m_nbPtsRight;
    str << ", \"pts_l\":" << d.m_nbPtsLeft;
    str << ", \"matches\":" << d.m_nbMatches;
    str << ", \"success\":" << ( d.m_succeed ? "true" : "false") ;
    str << " }\n";
    reply = str.str();
}


// Histogram horizontal
void makeHistogramHorizontalReply( const AnalysisResult &d, std::string &reply )
{
    std::stringstream str;
    str << "{ \"hdisp\":[" << d.m_hdisp[0];
    for( unsigned int i=1; i< d.s_histogramBinSize; i++)
    {
        str << ", " << d.m_hdisp[i];
    }
    str << "] }\n";

    reply = str.str();
}

// Histogram vertical
void makeHistogramVerticalReply(  const AnalysisResult &d, std::string &reply )
{
    std::stringstream str;
    str << "{ \"vdisp\":[" << d.m_vdisp[0];
    for( unsigned int i=1; i< d.s_histogramBinSize; i++)
    {
        str << ", " << d.m_vdisp[i];
    }
    str << "] }\n";

    reply = str.str();
}

// Create a reply
void makeHistogramReply(  const AnalysisResult &d, std::string &reply )
{
    std::stringstream str;
    str << "{ \"hdisp\":[" << d.m_hdisp[0];
    for( unsigned int i=1; i< d.s_histogramBinSize; i++)
    {
        str << ", " << d.m_hdisp[i];
    }
    str << "], \"vdisp\":[" << d.m_vdisp[0];
    for( unsigned int i=1; i< d.s_histogramBinSize; i++)
    {
        str << ", " << d.m_vdisp[i];
    }

    str << "] }\n";

    reply = str.str();
}

void makeVersionReply( const AnalysisResult &d, std::string &reply )
{
    std::stringstream str;
    str << "{ \"version\":\"" << std::string(version) << "\", \"mode\":\"" << d.m_mode << "\" }\n";
    reply = str.str();
}

// Rig information 
//void makeCameraReply( const AnalysisResult &d, std::string &reply )
//{
//    std::string cam;
//    m_cameraJson.get(cam);
//    std::stringstream str;
//    str << cam << std::endl;
//    reply = str.str();
//}
