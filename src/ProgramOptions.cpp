#include "ProgramOptions.h"

#include <iostream>

#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
namespace po = boost::program_options;


// Default window size static for now
const UInt2 ProgramOptions::m_winSize(1024,768);

// Debugging tests 
#define TEST 1

ProgramOptions::ProgramOptions(int argc, char *argv[], const char *version)
: m_serverPort(8090)
, m_useGPU(false)
, m_noThread(false)
{
    std::cout << "calibrig v" << version << " - cpu + gpu beta" << std::endl;
    std::cout << "Copyright (C) 2010-2012  C. Pichard"<< std::endl;
    std::cout << "This program comes with ABSOLUTELY NO WARRANTY;" << std::endl;
    std::cout << "This is free software, and you are welcome to redistribute it" << std::endl;
    std::cout << "under certain conditions; " << std::endl;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("port", po::value<unsigned int>(), "server port")
        ("gpu", "enable gpu computing")
        ("nothread", "remove gpu multi threading ")
    ;

	// Parse command line
    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if( vm.count("help") )
        {
            std::cout << desc << "\n";
            exit(EXIT_SUCCESS);
        }

        if( vm.count("port") )
        {
            m_serverPort = vm["port"].as<unsigned int>();
        }
        else
        {
            std::cout << "Using port " << m_serverPort << std::endl;
        }

        if( vm.count("gpu") )
        {
            m_useGPU = true;
        }
        if( vm.count("nothread") )
        {
            m_noThread = true;
        }
    }

    catch(std::exception &e)
    {
        std::cout << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
#if TEST
    // Development, force use of GPU mode
    //useGPU = true;
#endif
}
