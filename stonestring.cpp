#include "stonestring.h"
#include <sstream>

using namespace std;

StoneString::StoneString( int astring_id,
                          int* amembers,
                          int amembers_len,
                          COLOR acolor ){
    string_id = astring_id;
    members = amembers;
    members_len = amembers_len;
    color = acolor;
}

//StoneString::~StoneString(){
//}

string StoneString::toString(){
    stringstream out;
    out << "id :" << string_id << " \nmembers: ";

    for( int i=0; i < members_len; i++ ){
       out << members[i] << ", ";
    }
    out << "\n color: " << (int) color << " \nterritories: ";

    vector<int>::iterator it;
    for( it=territory.begin(); it != territory.end(); it++ ){
        out << *it;
    }
    return out.str();
}
