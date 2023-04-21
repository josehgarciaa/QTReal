#ifndef SPATIAL
#define SPATIAL

#include "typedef.h"
#include <vector>
#include <unordered_map>
#include "sites.h"
#include <boost/algorithm/string.hpp>
#include <iostream>

class SpatialGrid{
    private:
        size_t BinHint=0;
        Float CutOff = -1.;
        Index NumCells{-1,-1,-1};
        Position CellSize = Position({-1,-1,-1});
        Position BoxLow = Position({-1,-1,-1});
        Position BoxHi = Position({-1,-1,-1});
        Position Width = Position({-1,-1,-1});
        std::unordered_map<size_t,std::vector<Site> > Grid;
    public:
        SpatialGrid(Position CellSize, Position Lows, Position Hi);
        SpatialGrid();
        ~SpatialGrid();
        size_t  Key(Site& A);
        void BuildFromSiteList(std::vector<Site>& SiteList);
        std::vector<Site>& Query (Site& A);
        template <size_t dim> std::vector<Site*> GetNeighbors(Site & A);
        template <size_t dim> std::unordered_map<size_t,std::vector<Site> >  GetAllNeighbors();
        template <size_t dim> std::unordered_map<size_t,std::vector<Site> >  GetAllFiniteNeighbors();
        void SetBins();
        void Insert(Site& site);
        void Delete(Site& site);
        void Update(Site& site);       
        void UpdateFromSiteList(std::vector<Site>& SiteLIst);
        inline Float & GetCutoff(){return this -> CutOff;};
        inline Position getWidth(){return this -> Width;};
        inline Position getBoxLow() {return this -> BoxLow;};
        inline Position getBoxHi(){return this -> BoxHi;};
        inline Index getNumCells(){return this -> NumCells;};
        inline void SetBinHint(size_t size){this -> BinHint = size;};
        inline size_t GetBinHint(){return this -> BinHint;};
        inline size_t getMapIdx(Index  idx){ return idx[0]+(size_t)NumCells[0]*idx[1]+(size_t)NumCells[0]*(size_t)NumCells[1]*idx[2];};
        inline std::unordered_map<size_t, std::vector<Site> >& getGrid(){return this -> Grid;};



};





#endif