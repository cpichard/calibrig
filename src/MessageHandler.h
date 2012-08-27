#ifndef __MESSAGEHANDLER_H__
#define __MESSAGEHANDLER_H__

#include <boost/thread.hpp>

#include "StereoAnalyzer.h"
#include "CommandStack.h"
//
template <typename StoredType>
class LockDecorator
{
public:
    // TODO : copy constructors+operator+
    LockDecorator(){}
    ~LockDecorator(){}

    // Accessors
    inline void set( const StoredType &d )
    {
        boost::lock_guard<boost::mutex> l(m_mutex);
        m_d = d;
    }

    inline void get( StoredType &d )
    {
        boost::lock_guard<boost::mutex> l(m_mutex);
        d = m_d;
    }

private:
    StoredType m_d;
    boost::mutex m_mutex;
};

/**
 * This is used to process the incoming messages from the network
 */
class MessageHandler
{
public:
    MessageHandler( LockDecorator<AnalysisResult> &result
                  , CommandStack &commandStack );
    ~MessageHandler();

    // Take the incoming message, process it a
    bool process( const std::string &messageIn, std::string &messageOut );

private:
    LockDecorator<AnalysisResult>   &m_result;
    CommandStack                    &m_commandStack;
    LockDecorator<std::string>      m_cameraJson;
};

#endif
