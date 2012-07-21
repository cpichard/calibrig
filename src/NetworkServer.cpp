#include "NetworkServer.h"


// TODO : split in multiple files

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
        TcpServer server( m_io, m_msgHandler, m_tcpServerPort );
        // TODO UdpServer server( m_io, m_result, m_cameraJson, m_commandStack, m_tcpServerPort );
        m_io.run();
    }
    catch(std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}


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


