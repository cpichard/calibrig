#ifndef __PROGRAMOPTIONS_H__
#define __PROGRAMOPTIONS_H__

#include "Utils.h"

struct ProgramOptions
{
    ProgramOptions(int argc, char *argv[], const char *version);
    ~ProgramOptions(){}    
    
    // Port the server is running on    
    unsigned int m_serverPort;
    // CPU or GPU mode
    bool m_useGPU;
    // Skip threads, used for debugging
    bool m_noThread;
    // Default window size 
    static const UInt2 m_winSize;
};

#endif//__PROGRAMOPTIONS_H__

