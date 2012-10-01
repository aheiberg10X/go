#include "godomain.h"

using namespace std;

int GoDomain::getPlayerIx( State state ){
    return 42;
}

State GoDomain::copyState( State state ){
    return GoState( "asdf", 1, false );
};

void GoDomain::applyAction( State state, 
        string action,
        bool side_effects ){
    return;
}

void GoDomain::getRewards( int* to_fill,
                           State state ){
    return;
}

string GoDomain::randomAction( State state,
                               set<string> to_exclude ){
    return "adsfasdf";
}
 
bool GoDomain::fullyExpanded( std::string action ){
    return false;
}

bool GoDomain::isTerminal( State state ){
    return true;
}

