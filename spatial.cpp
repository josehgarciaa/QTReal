#include "spatial.h"

SpatialGrid::SpatialGrid(Position CellSize, Position Lows, Position Highs)
: CellSize(CellSize), BoxLow(Lows), BoxHi(Highs), Width(Highs - Lows){};


SpatialGrid::SpatialGrid(){};


SpatialGrid::~SpatialGrid(){
    this -> CutOff = -1.;
    this -> NumCells=Index({0,0,0});
    this -> CellSize = Position({-1,-1,1});
    this -> BoxLow = Position({-1,-1,-1});
    this -> BoxHi = Position({-1,-1,-1});
    this -> CellSize = Position({-1,-1,-1});
    this -> Width = Position({-1,-1,-1});
    Grid.clear();
}


size_t SpatialGrid::Key(Site& Site){
    Position out = Site.getPosition();
    out -= BoxLow;
    for(auto i=0; i<2; i++)
        out[i] = std::floor(out[i]/CellSize[i]);
    if(CellSize[2]==0)
            out[2] = std::floor(0.0);
    else
        out[2] = std::floor(out[2]/CellSize[2]);
    return (size_t)out[0] + this ->NumCells[0]*(size_t)out[1] + this ->NumCells[0]* this ->NumCells[1]*(size_t)out[2];
}



void SpatialGrid::SetBins(){
    if(CellSize.sum()<=0.0 || Width.sum()<=0.0){
        std::cerr<<"The box and the cells aren't defined... Exiting\n"<<std::endl;
        exit(EXIT_FAILURE);
    }
    Position MaxIdx = Width;
    for(auto i = 0; i<3; i++)
        MaxIdx[i] = std::ceil(MaxIdx[i]/CellSize[i]);
    NumCells = Index({(Integer)MaxIdx[0], (Integer)MaxIdx[1], (Integer)MaxIdx[2]});
    //Now we create the keys in the dictionary
    if(NumCells[2]!=0){
        for(auto i=0; i<NumCells[0]; i++)
            for(auto j = 0; j<NumCells[1]; j++)
                for(auto k = 0; k<=NumCells[2]; k++){
                    const size_t key =getMapIdx(Index(i,j,k));
                    Grid[key] = std::vector<Site>();
                    Grid[key].reserve(BinHint);
                }
    }
    else{
        for(Integer i=0; i<NumCells[0]; i++)
            for(Integer j = 0; j<NumCells[1]; j++){
                const size_t key = i+NumCells[0]*j;
                Grid[key] = std::vector<Site>();
                Grid[key].reserve(BinHint);
            }
    }
    return;
}

void SpatialGrid::Insert(Site & site){
    Position out = site.getPosition();
    out -= BoxLow;
    for(auto i=0; i<2; i++)
        out[i] = std::floor(out[i]/CellSize[i]);
    if(CellSize[2]==0)
            out[2] = std::floor(0.0);
    else
        out[2] = std::floor(out[2]/CellSize[2]);
    site.setBinIdx(Index({(Integer)out[0],(Integer)out[1],(Integer)out[2]}));
    Grid.at(Key(site)).emplace_back(site);
    return;
}


void SpatialGrid::Update(Site& site){
    Position out = site.getPosition();
    out -= BoxLow;
    //Calculation of the new indices
    for(auto i=0; i<2; i++)
        out[i] = std::floor(out[i]/CellSize[i]);
    if(CellSize[2]==0)
            out[2] = std::floor(0.0);
    else
        out[2] = std::floor(out[2]/CellSize[2]);
    //Here we make the comparison
    const auto old = site.getBinIdx();
    site.getBinIdx() = Index((Integer)out[0],(Integer)out[1],(Integer)out[2]);
    if(site.getBinIdx()[0]==old[0] && site.getBinIdx()[1]==old[1] && site.getBinIdx()[2]==old[2]  ){
        //Here we iterate at the grid and find the vector
        const auto idx = getMapIdx(site.getBinIdx());
        for(auto & v : Grid.at(idx)){
            if(v.getTag()==site.getTag()){
                v.getPosition() = site.getPosition();
                break;
            }
            else{ }
        }
    }
    //if different
    else{
        const auto idx0 = getMapIdx(old);
        for(auto& v: Grid.at(idx0)){
            if(v.getTag()==site.getTag()){
                std::swap(v,Grid.at(idx0)[Grid.at(idx0).size()-1]);
                Grid.at(idx0).pop_back();
                break;
            }
            else{ }
        }
        //setting the new indexes
        const auto idx = getMapIdx(site.getBinIdx());
        Grid.at(idx).emplace_back(site);
    }
    return;
}


std::vector<Site> &  SpatialGrid::Query(Site& site){
    return Grid.at(Key(site));
}

void SpatialGrid::UpdateFromSiteList(std::vector<Site>& SiteList){
    if(NumCells[0]+NumCells[1]+NumCells[2]<=0 && Grid.size()==0){
        std::cerr<<"Error: Lattice not initialized..."<<std::endl;
        exit(EXIT_FAILURE);
    }
    for(auto & v : SiteList)
        Update(v);
    return;

}

void SpatialGrid::BuildFromSiteList(std::vector<Site> & Sitelist){
    if(NumCells[0]+NumCells[1]+NumCells[2]<=0 || Grid.size()==0)
        SetBins();
    for(auto & v: Sitelist)
        Insert(v);
    return;
}

template <> std::unordered_map<size_t,std::vector<Site> >  SpatialGrid::GetAllNeighbors<1>(){
    std::cerr<<"not implemented yet bruh"<<std::endl;
    std::unordered_map<size_t,std::vector<Site> > neigh;
    return neigh;
}

template <> std::vector<Site *> SpatialGrid::GetNeighbors<1>(Site& site){
    std::cerr<<"not implemented yet bruh"<<std::endl;
    std::vector<Site *> neigh;
    return neigh;
}


template <> std::unordered_map<size_t,std::vector<Site> >  SpatialGrid::GetAllNeighbors<3>(){
    std::cerr<<"not implemented yet bruh"<<std::endl;
    std::unordered_map<size_t,std::vector<Site> > neigh;
    return neigh;
}

template <> std::vector<Site *> SpatialGrid::GetNeighbors<3>(Site& site){
    std::cerr<<"not implemented yet bruh"<<std::endl;
    std::vector<Site *> neigh;
    return neigh;
}



template <> std::vector<Site *> SpatialGrid::GetNeighbors<2>(Site& site){
    size_t idx1,idx2,idx3;
    std::vector<Site *> neigh;
    return neigh;
    
}

template <> std::unordered_map<size_t,std::vector<Site> >  SpatialGrid::GetAllNeighbors<2>(){
    std::unordered_map<size_t,std::vector<Site> > neigh;
    Integer idx1, idx2, idx3;
    for(auto& v : Grid){
        for(auto& k: v.second){
            for(auto di=-2; di<3; di++){
                for(auto dj = -2; dj<3; dj++){
                    //For cell periodic conditions
                    idx1 = k.getBinIdx()[0]+di;
                    idx2 = k.getBinIdx()[1]+dj;
                    idx3 = k.getBinIdx()[2];
                    //Evaluation of the pbc
                    if(idx1 >=NumCells[0]) idx1-=NumCells[0];
                    if(idx2>=NumCells[1]) idx2 -= NumCells[1];
                    if(idx1<0) idx1+=NumCells[0];
                    if(idx2<0) idx2 += NumCells[1];
                    if(true){
                        for(auto & item :Grid[getMapIdx(Index(idx1,idx2,idx3))]){
                            Position displacement = k.displacement(item);
                            if(displacement[0]>Width[0]*0.5) displacement[0]-=Width[0];
                            if(displacement[0]<-Width[0]*0.5) displacement[0]+=Width[0];
                            if(displacement[1]>Width[1]*0.5) displacement[1]-=Width[1];
                            if(displacement[1]<-Width[1]*0.5) displacement[1]+=Width[1];
                            if(displacement.squaredNorm() < CutOff*CutOff && k.getTag()!=item.getTag()){
                                neigh[k.getTag()].emplace_back(item);
                            }    
                        }
                    }
                    else{ }
                }
            }
        }
    }
    return neigh;
}


template <> std::unordered_map<size_t,std::vector<Site> >  SpatialGrid::GetAllFiniteNeighbors<2>(){
    std::unordered_map<size_t,std::vector<Site> > neigh;
    Integer idx1, idx2, idx3;
    for(auto& v : Grid){
        for(auto& k: v.second){
            for(auto di=-2; di<3; di++){
                for(auto dj = -2; dj<3; dj++){
                    //For cell periodic conditions
                    idx1 = k.getBinIdx()[0]+di;
                    idx2 = k.getBinIdx()[1]+dj;
                    idx3 = k.getBinIdx()[2];
                    //Evaluation of the pbc in the bin space
                    if(idx1 >=NumCells[0]) idx1-=NumCells[0];
                    if(idx2>=NumCells[1]) idx2 -= NumCells[1];
                    if(idx1<0) idx1+=NumCells[0];
                    if(idx2<0) idx2 += NumCells[1];
                    if(true){
                        for(auto & item :Grid[getMapIdx(Index(idx1,idx2,idx3))]){
                            Position displacement = k.displacement(item);
                            if(displacement.squaredNorm() < CutOff*CutOff && k.getTag()!=item.getTag()){
                                neigh[k.getTag()].emplace_back(item);
                            }    
                        }
                    }
                    else{ }
                }
            }
        }
    }
    return neigh;
}








