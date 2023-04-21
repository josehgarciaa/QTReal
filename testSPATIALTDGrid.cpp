#include <iostream>
#include "unitary.h"
#include "periodichamiltonian.h"
#include "sparsematrix.h"
#include "moments.h"
#include <chrono>
#include "chebyshevsolver.h"
#include "spatial.h"
#include "grapheneGenerator.h"

int main(Integer argc, char **argv){

    if(argc != 7){
        printf("Incorrect Number of parameters. The parameters are nv1, nv2, M, R, Bsize, threads \n");
        exit(EXIT_FAILURE);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    Integer nv1 = std::atoi(argv[1]);
    Integer nv2 = std::atoi(argv[2]);
    Integer M = std::atoi(argv[3]);
    Integer R = std::atoi(argv[4]);
    Integer bsize = std::atoi(argv[5]);
    Integer threads = std::atoi(argv[6]);
    Float eta = 1.0;
    Float cutoff = 1.2*acc;
    std::string name = "./DOS.dat";
    Float displacement = 0.0014748445992209544;
    omp_set_num_threads(threads);
    //PeriodicHamiltonian filter
    std::vector<Integer> rows, cols, Orows, Ocols;
    std::vector<scalar> vals, Ovals;
    std::vector<Site> SiteList;
    std::vector<Position> Limits;
    //here We generate the hamiltonian
    GenerateGraphene4At(nv1,nv2,SiteList);
    GenerateBox(nv1,nv2,Limits);
    Position CellSize = Position(a0,a0,1e-3);
    SpatialGrid grid(CellSize,Limits[0],Limits[1]);
    grid.BuildFromSiteList(SiteList);
    auto Stdep = SiteList;
    GenerateGraphene4AtTD(displacement,1.,M_PI*0.5, SiteList,Stdep);
    grid.GetCutoff() = cutoff;
    GenerateHamiltonian(rows, cols, vals, grid);
    SparseMat Hoppings;
    Hoppings.SetLabel("SpatialGraphene");
    Hoppings.SetDimensions(SiteList.size(), SiteList.size());
    Hoppings.SetNelem(rows.size());
    const scalar emax = 9.1;
    Hoppings.SetNelem(rows.size());
    Hoppings.ConvertFromCOO(rows, cols, vals);
    SparseMat Onsites = SparseMat();
    Onsites.SetLabel(Hoppings.GetLabel());
    Onsites.SetDimensions(SiteList.size(), SiteList.size());
    Onsites.SetNelem(Orows.size());
    Onsites.ConvertFromCOO(Orows, Ocols, Ovals);
    chebyshev::Moments1D chebMom(M);
    chebMom.SystemLabel(Hoppings.GetLabel());
    chebMom.BandWidth(2*emax.real());
    chebMom.BandCenter(0.0);
    chebMom.BlockSize(bsize);
	chebMom.SetAndRescaleHamiltonian(Hoppings,Onsites);
    chebMom.Print();
    SparseMat ID;
    ID.MakeIdentity();
    factory::generator gen;
    chebyshev::SpectralMoments(ID, chebMom, gen, R/bsize);
    //chebyshev::SpectralMoments(Filter, chebMom, gen, R/bsize);
    chebMom.ApplyJacksonKernel(0.0000001);
    std::ofstream output("./DOSSpatial.dat");
    Float dx = 2.0*(chebyshev::CUTOFF)/(Float)65535;
    for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
        scalar approx=0;
        for(Float m =0 ; m<chebMom.HighestMomentNumber(); m++){
            approx += delta_chebF(x,m)*chebMom.MomentVector(m);
        }
        output<<x<<"\t"<<approx.real()<<std::endl;
    }
    printf("size of first grid\n");   
    rows = std::vector<Integer>();
    cols = std::vector<Integer>();
    vals = std::vector<scalar>();
    Hoppings.ReleaseMatrix();
    Onsites.ReleaseMatrix();
    Onsites.ConvertFromCOO(Orows,Ocols,Ovals);
    grid.UpdateFromSiteList(Stdep);
    //for(auto v = 0; v<rows.size(); v++)
    //    std::cout<<rows[v]<<" "<<cols[v]<<" "<<vals[v]<<std::endl;
    GenerateHamiltonian(rows,cols, vals,grid);
    auto nn = grid.GetAllNeighbors<2>();
    for(auto &v:nn){
        if(v.second.size()!=3)
            std::cout<<"Error at Site: "<<v.first<<" Neighbors"<<v.second.size()<<std::endl;
        else { }
    }
    Hoppings.ConvertFromCOO(rows, cols, vals);
    chebMom.SetAndRescaleHamiltonian(Hoppings,Onsites);//El problema es uqe no esta reescalando
    chebMom.MomentVector() = std::vector<scalar>(M,0);
    gen.ResetCount();
    chebyshev::SpectralMoments(ID,chebMom,gen,R/bsize);
    chebMom.ApplyJacksonKernel(0.0000001);
    std::ofstream output2("./DOSSpatial2.dat");
     dx = 2.0*(chebyshev::CUTOFF)/(Float)65535;
    for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
        scalar approx=0;
        for(Float m =0 ; m<chebMom.HighestMomentNumber(); m++){
            approx += delta_chebF(x,m)*chebMom.MomentVector(m);
        }
        output2<<x<<"\t"<<approx.real()<<std::endl;
    }

    
    auto t2 = std::chrono::high_resolution_clock::now();

    auto s = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);

    std::cout<<" The elapsed time is "<<s.count()<<std::endl;
    
    


    return 0;
}   
