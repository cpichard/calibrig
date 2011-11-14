#ifndef __COMMANDSTACK_H__
#define __COMMANDSTACK_H__

#include <stack>
#include <boost/thread.hpp>

//! Dummy command system at this point
//! TODO improve it to make it work with each part of the system
struct Command
{
    Command( const std::string &dest, const std::string &action, int value=0)
            :m_dest(dest), m_action(action), m_value(value){}

    Command()
    :m_dest(""), m_action(""), m_value(0)
    {}

    std::string m_dest;    //! Destination
    std::string m_action;  //! What to do
    int m_value;           //! Optionnal value
};

// Mutexed stack for command
class CommandStack : public std::stack<Command>
{
public:
    CommandStack()
    :std::stack<Command>(){};
    
    inline 
    void pushCommand( const std::string &dest, const std::string &action, int value=0 )
    {
        boost::lock_guard<boost::mutex> l(m_mutex);
        Command c(dest, action, value );
        push(c);
    }

    inline
    bool popCommand( Command &lastCommand )
    {
        boost::lock_guard<boost::mutex> l(m_mutex);
        if(empty())
        {
            return false;
        }
        else
        {
            lastCommand = top();
            pop();
            return true;
        }
    }

    boost::mutex m_mutex;
};

#endif//__COMMANDSTACK_H__