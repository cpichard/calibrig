#include "UDPServer.h"


UdpServer::UdpServer( boost::asio::io_service &io
                    , MessageHandler &msgHandler
                    , unsigned int serverPort )
: m_socket( io, udp::endpoint(udp::v4(), serverPort) )
, m_msgHandler(msgHandler)
{
    startReceive();
}

void UdpServer::startReceive() {
    m_socket.async_receive_from
        ( boost::asio::buffer(m_receiveBuffer)
        , m_remoteEndpoint
        , boost::bind( &UdpServer::handleReceive
                     , this
                     , boost::asio::placeholders::error
                     , boost::asio::placeholders::bytes_transferred));
}


void UdpServer::handleReceive(const boost::system::error_code& error, size_t msgSize) {

    if (!error || error == boost::asio::error::message_size)
    {
        if (msgSize >= s_msgMaxSize) {
            std::cout << "UDPServer: message received is too large, not processing" << std::endl;
            startReceive();
        }

        std::string msgReceived(m_receiveBuffer.data());
        std::string msgProcessed;
        m_msgHandler.process( msgReceived, msgProcessed);
        boost::shared_ptr<std::string> msgToSend(new std::string(msgProcessed));
        m_socket.async_send_to( boost::asio::buffer(*msgToSend)
                              , m_remoteEndpoint
                              , boost::bind(&UdpServer::handleSend
                                           , this
                                           , msgToSend));
        // Next connection
        startReceive();
    }
}

void UdpServer::handleSend(boost::shared_ptr<std::string> msgToSend)
{}

