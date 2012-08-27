#include "TCPServer.h"


TcpConnection::SharedPtr
TcpConnection::create( boost::asio::io_service &io_service
                     , MessageHandler &msgHandler )
{
    return SharedPtr( new TcpConnection(io_service, msgHandler ));
}

TcpConnection::~TcpConnection()
{}

tcp::socket & TcpConnection::socket(){ return m_socket; }

void TcpConnection::start()
{
    //std::cout << "connection started" << std::endl;

    // Read the command
    boost::asio::async_read_until(socket(),
        m_msgReceived,
        '\n',
        boost::bind(
          &TcpConnection::handleReadCommand, shared_from_this(),
          boost::asio::placeholders::error));

}

void TcpConnection::handleReadCommand(const boost::system::error_code& error)
{
    //std::cout << "handleReadCommand" << std::endl;
    if( !error )
    {
        // Get message from streambuffer
        std::istream is(&m_msgReceived);
        std::string msgReceived;
        std::getline(is, msgReceived);

        // processMessage
        m_msgToSend="";
        if( m_msgHandler.process( msgReceived, m_msgToSend ) )
        {
            if( m_msgToSend != "" )
            {
                sendMessage();
            }

            // Read next message
            boost::asio::async_read_until(socket(),
                m_msgReceived,
                '\n',
                boost::bind(
                  &TcpConnection::handleReadCommand, shared_from_this(),
                  boost::asio::placeholders::error));
        }
    }
    else
    {
        // todo close connection properly
        std::cout << "error in NetworkServer.h" << std::endl;
    }
}


void TcpConnection::sendMessage()
{
    boost::asio::async_write( m_socket, boost::asio::buffer(m_msgToSend),
        boost::bind( &TcpConnection::handleWrite, shared_from_this(),
            boost::asio::placeholders::error,
            boost::asio::placeholders::bytes_transferred));
}

void TcpConnection::close(){ m_socket.close();}

TcpConnection::TcpConnection( boost::asio::io_service &io_service, MessageHandler &msgHandler)
: m_socket(io_service)
, m_msgHandler(msgHandler)
{}

void TcpConnection::handleWrite( const boost::system::error_code &, size_t){}




TcpServer::TcpServer( boost::asio::io_service &io
                    , MessageHandler &msgHandler
                    , unsigned int serverPort )
: m_acceptor(io, tcp::endpoint(tcp::v4(),serverPort))
, m_msgHandler(msgHandler)
{
    startAccept();
}

void TcpServer::startAccept()
{
    TcpConnection::SharedPtr newConnection = TcpConnection::create(m_acceptor.io_service(), m_msgHandler);
    m_acceptor.async_accept(
        newConnection->socket(),
        boost::bind(    &TcpServer::handleAccept ,
                        this,
                        newConnection,
                        boost::asio::placeholders::error
        )
    );
}

void TcpServer::handleAccept( TcpConnection::SharedPtr newConnection, const boost::system::error_code &error )
{
    if(!error)
    {
        newConnection->start();
        startAccept();
    }
}
