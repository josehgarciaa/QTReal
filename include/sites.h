#ifndef SITES_H
#define SITES_H
#include "typedef.h"

class Site{
    size_t tag;
    Index binIdx{-1,-1,-1};
    Position position;
    public:
        Site(Position X, size_t tag);
        Site();
        ~Site();
        Site(const Site& A);
        Site(Site && A);
        Site& operator = (const Site& A);
        Site& operator = (Site&& A);
        Position displacement(Site& A);
        Float distance(Site &A);
        inline Position& getPosition(){return position;};
        inline void setPosition(Position X){ this -> position  = X;};
        inline size_t getTag(){return tag;};
        inline void setTag(size_t Id){ this -> tag = Id;};
        inline void setBinIdx(Index bin){this -> binIdx =bin;};
        inline Index getBinIdx(){return this -> binIdx;};
};



#endif