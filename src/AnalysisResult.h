#ifndef __DEFORMATION_H__
#define __DEFORMATION_H__

#include <string>
/**
 * The result of an analysis contains only the data we want to send to the rig
 */
struct AnalysisResult
{
    AnalysisResult()
    : m_succeed(false)
    , m_rot(0)
    , m_tx(0)
    , m_ty(0)
    , m_scale(1)
    , m_nbPtsRight(0)
    , m_nbPtsLeft(0)
    , m_nbMatches(0)
    , m_mode("unknown")
    {
        // TODO memset ?
        for(unsigned int i=0; i < s_histogramBinSize; i++ )
        {
            m_hdisp[i] = m_vdisp[i] = 0;
        }
    }

    static const unsigned int s_histogramBinSize = 128;

    double m_h[9];
    bool m_succeed;
    double m_rot;
    double m_tx;
    double m_ty;
    double m_scale;
    unsigned int m_nbPtsRight;
    unsigned int m_nbPtsLeft;
    unsigned int m_nbMatches;
    float m_hdisp[s_histogramBinSize]; // Horizontal disparity histogram
    float m_vdisp[s_histogramBinSize]; // Vertical disparity histogram
    std::string m_mode;
};

// Function to convert values of analysis result to a formatted json string
void makeResultReply                ( const AnalysisResult &, std::string & );
void makeHistogramHorizontalReply   ( const AnalysisResult &, std::string & );
void makeHistogramVerticalReply     ( const AnalysisResult &, std::string & );
void makeHistogramReply             ( const AnalysisResult &, std::string & );
void makeVersionReply               ( const AnalysisResult &, std::string & );
// rig information void makeCameraReply                ( const AnalysisResult &, std::string &reply );

#endif // __DEFORMATION_H__
