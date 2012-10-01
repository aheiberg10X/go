#ifndef DOMAIN_H
#define DOMAIN_H

#include "state.h"
#include <string>
#include <set>
/*#include "action.h"*/

//TODO
//have an action type, instead of strings or ints?

class Domain {
    public :
        virtual int getPlayerIx( State state ) = 0;

        virtual State copyState( State state ) = 0;

        virtual void applyAction( State state, 
                                  std::string action,
                                  bool side_effects ) = 0;

        //TODO
        //to_fill_len should be defined by the number of players
        //should this be a memember of Domain?
        virtual void getRewards( int* to_fill,
                                 State state ) = 0;

        virtual std::string randomAction( 
                State state,
                std::set<std::string> to_exclude ) = 0;

        virtual bool fullyExpanded( std::string action ) = 0;

        virtual bool isTerminal( State state ) = 0;
};

#endif
