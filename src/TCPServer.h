#ifndef __TCPSERVER_H__
#define __TCPSERVER_H__

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/algorithm/string.hpp>

#include "MessageHandler.h"

using boost::asio::ip::tcp;

class TcpConnection : public boost::enable_shared_from_this<TcpConnection>
{
public:
    typedef boost::shared_ptr<TcpConnection> SharedPtr;

    static SharedPtr create( boost::asio::io_service &io_service
                           , MessageHandler &msgHandler );

    ~TcpConnection();

    tcp::socket & socket();

    void start();
    void close();

    void handleReadCommand(const boost::system::error_code& error);
    void sendMessage();

private:
    TcpConnection( boost::asio::io_service &io_service
                 , MessageHandler &msgHandler );

    void handleWrite( const boost::system::error_code &, size_t );

    tcp::socket             m_socket;
    MessageHandler          &m_msgHandler;
    std::string             m_msgToSend;
    boost::asio::streambuf  m_msgReceived;
};


class TcpServer
{
public:
    TcpServer( boost::asio::io_service &io
             , MessageHandler &msgHandler
             , unsigned int serverPort );

private:
    void startAccept();
    void handleAccept( TcpConnection::SharedPtr newConnection, const boost::system::error_code &error );

    tcp::acceptor               m_acceptor;
    MessageHandler              &m_msgHandler;
};

#endif//__TCPSERVER_H__
