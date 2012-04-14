#ifndef __DEFORMATION_H__
#define __DEFORMATION_H__

// TODO find a better name 
//
struct Deformation
{
    Deformation()
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

    double m_h1[9]; // Homography first image
    double m_h2[9]; // Homography second image
    double m_f[9];  // Fundamental matrix
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

#endif // __DEFORMATION_H__
