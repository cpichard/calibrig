#ifndef __RESULTSERVER_H__
#define __RESULTSERVER_H__


#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include "StereoAnalyzer.h"
#include "CommandStack.h"


using boost::asio::ip::tcp;

namespace po = boost::program_options;

class SharedResult
{
public:
    SharedResult(){}
    ~SharedResult(){}

    // Data
    void setResult( const Deformation &d )
    {
        boost::lock_guard<boost::mutex> l(m_mutex);
        m_d = d;
    }

    void getResult( Deformation &d )
    {
        boost::lock_guard<boost::mutex> l(m_mutex);
        d = m_d;
    }

    Deformation m_d;
    boost::mutex m_mutex;
};


class TcpConnection : public boost::enable_shared_from_this<TcpConnection>
{
public:
    typedef boost::shared_ptr<TcpConnection> pointer;

    static pointer create( boost::asio::io_service &io_service, SharedResult &result, CommandStack &commandStack  )
    {
        return pointer( new TcpConnection(io_service, result, commandStack ));
    }

    ~TcpConnection()
    {}

    tcp::socket & socket(){ return m_socket; }

    void start()
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

    void handleReadCommand(const boost::system::error_code& error)
    {
        //std::cout << "handleReadCommand" << std::endl;
        if( !error )
        {
            // Get message from streambuffer
            std::istream is(&m_msgReceived);
            std::string msgReceived;
            std::getline( is, msgReceived);

            //std::cout << msgReceived << std::endl;
            if( msgReceived.find("GETR") != string::npos )
            {
                //std::cout << "Find GETR" << std::endl;
                makeResultReply();
                sendMessage();
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
            // Histogram range - TODO 
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
                return;
            }
            else if( msgReceived.find("GETH") != string::npos)
            {
                makeHistogramReply();
                sendMessage();
            }
            else
            {
                m_msgToSend = "{\"help\" : \"unknown command, use GETR, GETH, THRES <value>, HRANGE <value>, CLOSE or EXIT\"}";
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
        else
        {
            // todo close connection properly
            std::cout << "error in NetworkServer.h" << std::endl;
        }
    }

    // Create a reply
    void makeResultReply( )
    {
        Deformation d;
        m_result.getResult( d );
        std::stringstream str;
        str << "{ \"tx\":" << d.m_tx;
        str << ", \"ty\":" << d.m_ty;
        str << ", \"rot\":" << d.m_rot;
        str << ", \"scale\":" << d.m_scale;
        str << ", \"pts_r\":" << d.m_nbPtsRight;
        str << ", \"pts_l\":" << d.m_nbPtsLeft;
        str << ", \"matches\":" << d.m_nbMatches;
        str << ", \"success\":" << ( d.m_succeed ? "true" : "false") ;
        str << " }";
        m_msgToSend = str.str();
    }

    // Create a reply
    void makeHistogramReply( )
    {
        Deformation d;
        m_result.getResult( d );
        std::stringstream str;
        str << "{ \"hdisp\":[" << d.m_hdisp[0];
        for( unsigned int i=1; i< d.s_histogramBinSize; i++)
        {
            str << ", " << d.m_hdisp[i];
        }
        str << "], \"vdisp\":[" << d.m_vdisp[0];
        for( unsigned int i=1; i< d.s_histogramBinSize; i++)
        {
            str << ", " << d.m_vdisp[i];
        }

        str << "] }";

        m_msgToSend = str.str();
    }


    void sendMessage()
    {
        boost::asio::async_write( m_socket, boost::asio::buffer(m_msgToSend),
            boost::bind( &TcpConnection::handleWrite, shared_from_this(),
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
    }

    void close(){ m_socket.close();}

private:
    TcpConnection( boost::asio::io_service &io_service, SharedResult &result, CommandStack &commandStack  )
    : m_socket(io_service), m_result(result), m_commandStack(commandStack)
    {}

    void handleWrite( const boost::system::error_code &, size_t){}

    tcp::socket m_socket;
    std::string m_msgToSend;
    boost::asio::streambuf m_msgReceived;

    SharedResult &m_result;
    CommandStack &m_commandStack;
};

class TcpServer
{
public:
    TcpServer( boost::asio::io_service &io, SharedResult &result, CommandStack &commandStack, unsigned int serverPort )
    : m_acceptor(io, tcp::endpoint(tcp::v4(),serverPort)), m_result(result), m_commandStack(commandStack)
    {
        startAccept();
    }
    void setResult( std::string &message )
    {}

private:
    void startAccept()
    {
        TcpConnection::pointer newConnection = TcpConnection::create(m_acceptor.io_service(), m_result, m_commandStack );
        m_acceptor.async_accept(
            newConnection->socket(),
            boost::bind(    &TcpServer::handleAccept ,
                            this,
                            newConnection,
                            boost::asio::placeholders::error
            )
        );
    }

    void handleAccept( TcpConnection::pointer newConnection, const boost::system::error_code &error )
    {
        if(!error)
        {
            newConnection->start();
            startAccept();
        }
    }

    tcp::acceptor m_acceptor;
    SharedResult &m_result;
    CommandStack &m_commandStack;
};

class NetworkServer
{
public:
    NetworkServer(SharedResult &result, CommandStack &commandStack, unsigned int serverPort)
    :m_result(result), m_commandStack(commandStack), m_serverPort(serverPort)
    {}

    void operator()()
    {
        // Asio service
        try
        {
            TcpServer server( m_io, m_result, m_commandStack, m_serverPort );
            m_io.run();
        }
        catch(std::exception &e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    inline void stop(){m_io.stop();}

    TcpServer   *m_server;
    SharedResult &m_result;
    CommandStack &m_commandStack;
    std::string m_message;
    boost::asio::io_service m_io;
    unsigned int m_serverPort;
};


#endif//__RESULTSERVER_H__
