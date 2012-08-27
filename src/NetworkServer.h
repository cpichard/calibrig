#ifndef __NETWORKSERVER_H__
#define __NETWORKSERVER_H__


#include <boost/asio.hpp>
#include "TCPServer.h"

using namespace std;
using namespace boost::algorithm;

using boost::asio::ip::tcp;


#include "MessageHandler.h"

/** Network TCP and UDP server
*/
class NetworkServer
{
public:
    NetworkServer( MessageHandler &msgHandler
                 , unsigned int tcpServerPort
                 , unsigned int udpServerPort );

    void operator()();
    inline void stop(){ m_io.stop(); }

private:
    MessageHandler              &m_msgHandler;
    boost::asio::io_service     m_io;
    unsigned int                m_tcpServerPort;
    unsigned int                m_udpServerPort;
};


#endif//__NETWORKSERVER_H__
