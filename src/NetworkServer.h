#ifndef __NETWORKSERVER_H__
#define __NETWORKSERVER_H__


#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/algorithm/string.hpp>
#include "StereoAnalyzer.h"
#include "CommandStack.h"

using namespace std;
using namespace boost::algorithm;

using boost::asio::ip::tcp;


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
