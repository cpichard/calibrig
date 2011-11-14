#include "StereoAnalyzer.h"

// TODO rename analyzer
struct AnalyzerFunctor
{
    AnalyzerFunctor( StereoAnalyzer &analyzer )
    :m_analyzer(analyzer),m_running(true)
    {}

    void stop(){m_running=false;}

    void operator()()
    {
        while( m_running == true )
        {
            if( m_analyzer.imagesAreNew() == false )
            {
                sleep(0.1);
            }
            else
            if( m_analyzer.try_lock() )
            {
                m_analyzer.analyse();
                m_analyzer.unlock();
            }
        }
    }

    StereoAnalyzer &m_analyzer;
    bool            m_running;
};

