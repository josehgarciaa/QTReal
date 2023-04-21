#ifndef GRAPHGEN
#define GRAPHGEN
#include "sites.h"
#include "spatial.h"


//const Float a0 = 0.245;
const Float a0 = 2.552; //This is for the use of the potential SiC of lammps examples
const Float acc = a0/std::sqrt(3.);


void GenerateGraphene4At(Integer ncellsv1, Integer ncellsv2,std::vector<Site> & sitelist){
    std::vector<Position> Latvecs(2);
    Latvecs[0]=(Position(a0,0.,0.));
    Latvecs[1]=(Position(0.,3.0*acc,0.));
    std::vector<Position> LatPos(4);
    LatPos[0] = Position(0.,0.,0.);
    LatPos[1] = Position(0.,acc,0.);
    LatPos[2] = Position(0.5*a0,3.*0.5*acc,0.);
    LatPos[3] = Position(0.5*a0,5.0*0.5*acc,0.);
    Integer sitecount = 0;
    for(auto idx =0; idx<ncellsv1*ncellsv2; idx++){
        const Integer n1 = (Integer)(idx/ncellsv2);
        const Integer n2 = (Integer)(idx%ncellsv2);
        for(auto i=0; i<LatPos.size(); i++){
            Position p = LatPos[i]+(Float)n1*Latvecs[0] + n2*Latvecs[1];
            sitelist.push_back(Site(p,sitecount));
            sitecount+=1;
        }
    }
    return;

}


void GenerateGraphene4AtTD(Float displacement,Float Freq,Float Tstep, std::vector<Site> & sitelist,std::vector<Site>& SiteOutput){
    #pragma omp parallel for schedule(static) firstprivate(displacement,Freq,Tstep)
    for(auto v =0; v<sitelist.size(); v++){
        SiteOutput[v].getPosition() = sitelist[v].getPosition();
        SiteOutput[v].getPosition() += Position(0,std::pow(-1.0,v%2)*displacement,0)*std::sin(Freq*Tstep);
    }
    return;

}


void GenerateBox(Integer ncellsv1, Integer ncellsv2,std::vector<Position>&Limits){
    std::vector<Position> Latvecs(2);
    Latvecs[0]=(Position(a0,0.,0.));
    Latvecs[1]=(Position(0.,3.0*acc,0.));
    std::vector<Position> LatPos(4);
    LatPos[0] = Position(0.,0.,0.);
    LatPos[1] = Position(0,acc,0.);
    LatPos[2] = Position(0.5*a0,3.*0.5*acc,0.);
    LatPos[3] = Position(0.5*a0,5.0*0.5*acc,0.);
    //Position Lower = Position(0,0,0);
    Position Lower = Position(-0.025,-0.025,0);
    //Position Higher=(Float)(ncellsv1)*Latvecs[0] + (Float)(ncellsv2)*Latvecs[1];
    Position Higher=(Float)(ncellsv1)*Latvecs[0] + (Float)(ncellsv2)*Latvecs[1]+Lower;
    Limits.push_back(Lower);
    Limits.push_back(Higher);
    return;
}


void GenerateHamiltonian(std::vector<Integer>& rows, std::vector<Integer> & cols, std::vector<scalar>& vals, SpatialGrid& grid){
    const Float hop = -2.71;
    const Float beta = -3.37;
    auto nn = grid.GetAllNeighbors<2>();//FillNeighbour list
    //Now we iterate over all the neighbours
    for (auto & v: grid.getGrid()){ //over the grid
        for(auto & k: v.second){ //over the sites on each bucket of the grid
            for(auto & l : nn.at(k.getTag())){ //over the neighbors of the site k
                if(l.getTag()<k.getTag()){
                    Position displacement = k.displacement(l);
                    const auto Width = grid.getWidth();
                    if(displacement[0]>Width[0]*0.5) displacement[0]-=Width[0];            
                    if(displacement[0]<-Width[0]*0.5) displacement[0]+=Width[0];
                    if(displacement[1]>Width[1]*0.5) displacement[1]-=Width[1];
                    if(displacement[1]<-Width[1]*0.5) displacement[1]+=Width[1];
                    cols.emplace_back(k.getTag());
                    rows.emplace_back(l.getTag());
                    vals.emplace_back(hop*std::exp(beta*(displacement.norm()/acc - 1.0)));
                }
                else{ }
                
                
            }
            
        }
    }
    return;
}


void GenerateFiniteHamiltonian(std::vector<Integer>& rows, std::vector<Integer> & cols, std::vector<scalar>& vals, SpatialGrid& grid){
    const Float hop = -2.71;
    const Float beta = -3.37;
    auto nn = grid.GetAllFiniteNeighbors<2>();//FillNeighbour list
    //Now we iterate over all the neighbours
    for (auto & v: grid.getGrid()){ //over the grid
        for(auto & k: v.second){ //over the sites on each bucket of the grid
            for(auto & l : nn.at(k.getTag())){ //over the neighbors of the site k
                if(l.getTag()<k.getTag()){
                    Position displacement = k.displacement(l);
                    const auto Width = grid.getWidth();
                    if(displacement[0]>Width[0]*0.5) displacement[0]-=Width[0];            
                    if(displacement[0]<-Width[0]*0.5) displacement[0]+=Width[0];
                    if(displacement[1]>Width[1]*0.5) displacement[1]-=Width[1];
                    if(displacement[1]<-Width[1]*0.5) displacement[1]+=Width[1];
                    cols.emplace_back(k.getTag());
                    rows.emplace_back(l.getTag());
                    vals.emplace_back(hop*std::exp(beta*(displacement.norm()/acc - 1.0)));
                }
                else{ }
                
                
            }
            
        }
    }
    return;
}



void GenerateFiniteHamiltonianAB(std::vector<Integer>& rows, std::vector<Integer> & cols, std::vector<scalar>& vals, SpatialGrid& grid, Float AB=0.0){
    const Float hop = -2.71;
    const Float beta = -3.37;
    auto nn = grid.GetAllFiniteNeighbors<2>();//FillNeighbour list
    //Now we iterate over all the neighbours
    for (auto & v: grid.getGrid()){ //over the grid
        for(auto & k: v.second){ //over the sites on each bucket of the grid
            rows.emplace_back(k.getTag());
            cols.emplace_back(k.getTag());
            vals.emplace_back(AB*std::pow(-1.0,k.getTag()));
            for(auto & l : nn.at(k.getTag())){ //over the neighbors of the site k
                if(l.getTag()<k.getTag()){
                    Position displacement = k.displacement(l);
                    const auto Width = grid.getWidth();
                    if(displacement[0]>Width[0]*0.5) displacement[0]-=Width[0];            
                    if(displacement[0]<-Width[0]*0.5) displacement[0]+=Width[0];
                    if(displacement[1]>Width[1]*0.5) displacement[1]-=Width[1];
                    if(displacement[1]<-Width[1]*0.5) displacement[1]+=Width[1];
                    cols.emplace_back(k.getTag());
                    rows.emplace_back(l.getTag());
                    vals.emplace_back(hop*std::exp(beta*(displacement.norm()/acc - 1.0)));
                }
                else{ }
                
                
            }
            
        }
    }
    return;
}


void GenerateFiniteHamiltonianABTDep(std::vector<Integer>& rows, std::vector<Integer> & cols, std::vector<scalar>& vals, SpatialGrid& grid, Float AB, Float freq, Float time,Float pol=0.0){
    const Float hop = -2.71;
    const Float beta = -3.37;
    const scalar I(0.,1.);
    auto nn = grid.GetAllFiniteNeighbors<2>();//FillNeighbour list
    //Now we iterate over all the neighbours
    for (auto & v: grid.getGrid()){ //over the grid
        for(auto & k: v.second){ //over the sites on each bucket of the grid
            rows.emplace_back(k.getTag());
            cols.emplace_back(k.getTag());
            vals.emplace_back(AB*std::pow(-1.0,k.getTag()));
            for(auto & l : nn.at(k.getTag())){ //over the neighbors of the site k
                if(l.getTag()<k.getTag()){
                    Position displacement = k.displacement(l);
                    const auto Width = grid.getWidth();
                    if(displacement[0]>Width[0]*0.5) displacement[0]-=Width[0];            
                    if(displacement[0]<-Width[0]*0.5) displacement[0]+=Width[0];
                    if(displacement[1]>Width[1]*0.5) displacement[1]-=Width[1];
                    if(displacement[1]<-Width[1]*0.5) displacement[1]+=Width[1];
                    cols.emplace_back(k.getTag());
                    rows.emplace_back(l.getTag());
                    const scalar hopping = hop*std::exp(beta*(displacement.norm()/acc - 1.0))*std::exp((I*M_PI*4.e-2/a0)*(displacement[0]*std::cos(freq*time)+ pol*displacement[1]*std::sin(freq*time)));
                    
                    vals.emplace_back(hopping);
                }
                else{ }
                
                
            }
            
        }
    }
    return;
}



void GenerateFinitePseudoSpin(std::vector<Integer>& rows, std::vector<Integer> & cols, std::vector<scalar>& vals, SpatialGrid& grid){
    const Float hop = -2.71;
    const Float beta = -3.37;
    const scalar I(0.,1.);
    auto nn = grid.GetAllFiniteNeighbors<2>();//FillNeighbour list
    //Now we iterate over all the neighbours
    for (auto & v: grid.getGrid()){ //over the grid
        for(auto & k: v.second){ //over the sites on each bucket of the grid
            rows.emplace_back(k.getTag());
            cols.emplace_back(k.getTag());
            vals.emplace_back((std::pow(-1.0,k.getTag())));
            
            
        }
    }
    return;
}



void GenerateForceMat(std::vector<Integer>& rows, std::vector<Integer> & cols, std::vector<scalar>& Xvals,std::vector<scalar>& Yvals,std::vector<scalar>& Zvals, SpatialGrid& grid){
    const Float hop = -2.71;
    const Float beta = -3.37;
    auto nn = grid.GetAllNeighbors<2>();//FillNeighbour list
    //Now we iterate over all the neighbours
    for (auto & v: grid.getGrid()){ //over the grid
        for(auto & k: v.second){ //over the sites on each bucket of the grid
            for(auto & l : nn.at(k.getTag())){ //over the neighbors of the site k
                if(l.getTag()<k.getTag()){
                    Position displacement = k.displacement(l);
                    const auto Width = grid.getWidth();
                    if(displacement[0]>Width[0]*0.5) displacement[0]-=Width[0];            
                    if(displacement[0]<-Width[0]*0.5) displacement[0]+=Width[0];
                    if(displacement[1]>Width[1]*0.5) displacement[1]-=Width[1];
                    if(displacement[1]<-Width[1]*0.5) displacement[1]+=Width[1];
                    cols.emplace_back(k.getTag());
                    rows.emplace_back(l.getTag());
                    Xvals.emplace_back(hop*std::exp(beta*(displacement.norm()/acc - 1.0))*(beta/acc)*displacement[0]/displacement.norm());
                    Yvals.emplace_back(hop*std::exp(beta*(displacement.norm()/acc - 1.0))*(beta/acc)*displacement[1]/displacement.norm());
                    Zvals.emplace_back(hop*std::exp(beta*(displacement.norm()/acc - 1.0))*(beta/acc)*displacement[2]/displacement.norm());
                    cols.emplace_back(l.getTag());
                    rows.emplace_back(k.getTag());
                    Xvals.emplace_back(-std::conj(hop)*std::exp(beta*(displacement.norm()/acc - 1.0))*(beta/acc)*displacement[0]/displacement.norm());
                    Yvals.emplace_back(-std::conj(hop)*std::exp(beta*(displacement.norm()/acc - 1.0))*(beta/acc)*displacement[1]/displacement.norm());
                    Zvals.emplace_back(-std::conj(hop)*std::exp(beta*(displacement.norm()/acc - 1.0))*(beta/acc)*displacement[2]/displacement.norm());
                }
                else{ }
                
                
            }
            
        }
    }
    return;
}

void ProjectionSite(Integer site, Integer bsize, const std::vector<scalar> & initialv, std::vector<scalar>& temp){
    if(temp.size()!=initialv.size())
        temp = std::vector<scalar>(initialv.size(),0);
    else{
        #pragma omp parallel for schedule (dynamic)
        for(auto i =0; i<initialv.size(); i++ )
            temp[i] = 0.0;
    }
    for(auto b = 0; b<bsize; b++)
        temp[site + (initialv.size()/bsize)*b] = initialv[site+(initialv.size()/bsize)*b];
    return;
}
 
#endif