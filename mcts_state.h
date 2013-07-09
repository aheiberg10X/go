#ifndef MCTState
#define MCTState

#include "bitmask.h"
#include <string>

using namespace std;

//a virtual class defining the methods mcts.h will require to work
class MCTS_State {
public :
    virtual const int getNumPlayers() = 0;

    virtual int getNumActions() = 0;

    virtual int getPlayerIx() = 0;

    virtual int movesMade() = 0;

    virtual void copyInto( MCTS_State* target ) = 0;

    virtual MCTS_State* copy( ) = 0;

    virtual void deleteState( ) = 0;

    virtual bool applyAction( int action,
                              bool side_effects ) = 0;

    virtual void getRewards( int* to_fill ) = 0;
    
    virtual int randomAction( BitMask* to_exclude,
                              bool side_effects ) = 0;
    
    virtual bool isTerminal() = 0;

    virtual bool fullyExpanded( int action ) = 0;

    virtual bool isChanceAction() = 0;

    virtual string toString() = 0;

};



#endif
