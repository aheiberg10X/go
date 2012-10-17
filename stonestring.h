#ifndef STONESTRING_H
#define STONESTRING_H

#include <vector>
#include <string>
#include "gostate.h"

class StoneString {
public :
    int string_id;
    int* members;
    int members_len;
    COLOR color;
    std::vector<int> territory;

    StoneString( int astring_id, 
                 int* amembers, 
                 int amembers_len, 
                 COLOR acolor );

    /*~StoneString();*/

    std::string toString();
};

#endif
