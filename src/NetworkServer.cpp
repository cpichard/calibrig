#include "NetworkServer.h"
#include "TCPServer.h"
#include "UDPServer.h"

// TODO : split in multiple files


NetworkServer::NetworkServer( MessageHandler &msgHandler
                            , unsigned int tcpServerPort
                            , unsigned int udpServerPort )
: m_msgHandler(msgHandler)
, m_tcpServerPort(tcpServerPort)
, m_udpServerPort(udpServerPort)
{}

void NetworkServer::operator()()
{
    // Asio service
    try
    {
        TcpServer tcpServer( m_io, m_msgHandler, m_tcpServerPort );
        UdpServer udpServer( m_io, m_msgHandler, m_udpServerPort );
        m_io.run();
    }
    catch(std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}



