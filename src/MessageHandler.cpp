#include <boost/algorithm/string.hpp> // trim
#include "MessageHandler.h"

using namespace boost::algorithm; // trim

MessageHandler::MessageHandler( LockDecorator<AnalysisResult> &result
                              , CommandStack &commandStack )
: m_result(result)
, m_commandStack(commandStack)
{
    m_cameraJson.set("{}");
}

MessageHandler::~MessageHandler()
{}

bool MessageHandler::process( const std::string &msgReceived, std::string &messageOut )
{
    // Get a local copy of the current stored result
    AnalysisResult d;
    m_result.get(d);

    //std::cout << msgReceived << std::endl;
    if( msgReceived.find("GETR") != string::npos )
    {
        //std::cout << "Find GETR" << std::endl;
        makeResultReply(d, messageOut);
    }
    else if( msgReceived.find("THRES")!= string::npos )
    {
        std::string com;
        int value = 0;
        std::stringstream ss(msgReceived);
        ss >> com; // THRES
        ss >> value;

        if(!ss.fail())
        {
            m_commandStack.pushCommand("ANALYSER","OCVTHRESHOLD", value );
        }
    }
    else if( msgReceived.find("MAXPOINTS")!= string::npos )
    {
        std::string com;
        int value = 0;
        std::stringstream ss(msgReceived);
        ss >> com; // MAXNBPOINTS
        ss >> value;

        if(!ss.fail())
        {
            m_commandStack.pushCommand("ANALYSER","MAXPOINTS", value );
        }
    }
    else if( msgReceived.find("SCREEN")!= string::npos )
    {
        std::string com;
        int value = 0;
        std::stringstream ss(msgReceived);
        ss >> com; // SCREEN NUMBER
        ss >> value;

        if(!ss.fail())
        {
            m_commandStack.pushCommand("MAIN","SCREEN", value );
        }
    }
    // Histogram range
    else if( msgReceived.find("HRANGE")!= string::npos )
    {
        std::string com;
        int value = 0;
        std::stringstream ss(msgReceived);
        ss >> com;
        ss >> value;

        if(!ss.fail())
        {
            m_commandStack.pushCommand("ANALYSER","HISTOGRAMRANGE", value );
        }
    }
    else if( msgReceived.find("EXIT") != string::npos)
    {
        m_commandStack.pushCommand("MAIN", "EXIT" );
    }
    else if( msgReceived.find("CLOSE") != string::npos)
    {
        // Close connection
        return false;
    }
    else if( msgReceived.find("GETHH") != string::npos)
    {
        makeHistogramHorizontalReply(d, messageOut);
    }
    else if( msgReceived.find("GETHV") != string::npos)
    {
        makeHistogramVerticalReply(d, messageOut);
    }
    else if( msgReceived.find("GETH") != string::npos)
    {
        makeHistogramReply(d, messageOut);
    }
    else if( msgReceived.find("SNAPSHOT") != string::npos)
    {
        m_commandStack.pushCommand("MAIN", "SNAPSHOT");
    }
    else if( msgReceived.find("VERSION") != string::npos)
    {
        makeVersionReply(d, messageOut);
    }
    else if( msgReceived.find("SETCAM ") != string::npos)
    {
        std::string com;
        std::string cam;
        std::stringstream ss(msgReceived);
        ss >> com;
        // Read the rest of the line
        std::getline(ss, cam);
        trim(cam);
        m_cameraJson.set(cam);
    }
    else if( msgReceived.find("GETCAM") != string::npos )
    {
        std::string cam;
        m_cameraJson.get(cam);
        std::stringstream str;
        str << cam << std::endl;
        messageOut = str.str();
    }
    else
    {
        messageOut = "{\"help\" : \"unknown command, use GETR, GETH, GETHV, GETHH, MAXPOINTS <value>,THRES <value>, HRANGE <value>, SNAPSHOT, VERSION, CLOSE or EXIT\"}\n";
    }
    return true;
}


