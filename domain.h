#ifndef DOMAIN_H
#define DOMAIN_H

#include "bitmask.h"
#include <string>

class Domain {
    public :
        virtual const int getNumPlayers( void* state ) = 0;
        virtual int getNumActions( void* state ) = 0;
        virtual int getPlayerIx( void* state ) = 0;
        virtual int movesMade( void* state ) = 0;

        virtual void copyStateInto( void* source, void* target ) = 0;
        virtual void* copyState( void* source ) = 0;
        virtual void deleteState( void* state ) = 0;

        virtual bool applyAction( void* state, 
                                  int action,
                                  bool side_effects ) = 0;

        virtual void getRewards( int* to_fill,
                                 void* state ) = 0;
        
        virtual int randomAction( void* state,
                                  BitMask* to_exclude,
                                  bool side_effects ) = 0;
        
        virtual bool isTerminal( void* state ) = 0;

        virtual bool fullyExpanded( int action ) = 0;

        virtual bool isChanceAction( void* state ) = 0;

};

#endif
