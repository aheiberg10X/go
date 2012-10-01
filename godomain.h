#ifndef GODOMAIN_H
#define GODOMAIN_H

#include "domain.h"
#include <string>
#include "gostate.h"
#include <set>
/*#include "goaction.h"*/

class GoDomain : public Domain {
    public :
        int getPlayerIx( State state );
        State copyState( State state );
        void applyAction( State state, 
                string action,
                bool side_effects );
        
        void getRewards( int* to_fill,
                                 State state );

        std::string randomAction( State state,
                                  std::set<std::string> to_exclude );

        bool fullyExpanded( std::string action );

        bool isTerminal( State state );

};

#endif
