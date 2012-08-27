#ifndef __UDPSERVER_H__
#define __UDPSERVER_H__

#include <iostream>
#include <string>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/asio.hpp>

#include "MessageHandler.h"

using boost::asio::ip::udp;

class UdpServer
{
public:
    UdpServer( boost::asio::io_service &io
             , MessageHandler &msgHandler
             , unsigned int serverPort );

private:
    void startReceive();
    void handleReceive(const boost::system::error_code& error, size_t msgSize);
    void handleSend(boost::shared_ptr<std::string> msgToSend);

    static const int s_msgMaxSize=512;

    udp::socket                         m_socket;
    udp::endpoint                       m_remoteEndpoint;
    boost::array<char, s_msgMaxSize>    m_receiveBuffer;
    MessageHandler                      &m_msgHandler;
};

#endif//__UDPSERVER_H__                                             

