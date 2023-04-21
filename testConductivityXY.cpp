#include <iostream>
#include "unitary.h"
#include "periodichamiltonian.h"
#include "sparsematrix.h"
#include "moments.h"
#include <chrono>
#include "chebyshevsolver.h"

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

    std::string name = "./DOS.dat";

    omp_set_num_threads(threads);
    //Unitary testlat("./pybuild","squaremine");
    //Unitary testlat("./pybuild","GrapheneSpinful");
    Unitary testlat("./pybuild","HaldaneModel");
    testlat.RegisterOrbitals();
    testlat.RegisterHoppings();
    testlat.SetDeltas();
    const scalar emax = 3.6;
    PeriodicHamiltonian HamBuilder(testlat, nv1, nv2);
    std::vector<Integer> rows, cols, Orows, Ocols, Vrows, Vcols;
    std::vector<scalar> vals, Ovals, VXvals,VYvals;
    HamBuilder.SetPeriodicHoppings<2>(rows, cols ,vals);
    HamBuilder.SetOnSitePotentials<2>(Orows,Ocols,Ovals);
    HamBuilder.SetVelocities(Vrows, Vcols, VXvals,VYvals);
    SparseMat Hoppings, Onsites, VX, VY;
    Hoppings.SetLabel(testlat.getName());
    Integer dimension = 0;
    for(auto v: HamBuilder.getUnitCell().getOnSiteEnergies())
        dimension += v.cols();
    Hoppings.SetDimensions(dimension*nv1*nv2, dimension*nv1*nv2);
    Hoppings.SetNelem(rows.size());
    Onsites.SetLabel(testlat.getName());
    Onsites.SetDimensions(dimension*nv1*nv2, dimension*nv1*nv2);
    Onsites.SetNelem(Orows.size());
    VX.SetLabel(testlat.getName());
    VX.SetDimensions(dimension*nv1*nv2, dimension*nv1*nv2);
    VX.SetNelem(Vrows.size());
    VY.SetLabel(testlat.getName());
    VY.SetDimensions(dimension*nv1*nv2, dimension*nv1*nv2);
    VY.SetNelem(Vrows.size());
    Hoppings.ConvertFromCOO(rows, cols, vals);
    Onsites.ConvertFromCOO(Orows, Ocols, Ovals);
    VX.ConvertFromCOO(Vrows, Vcols, VXvals);
    VY.ConvertFromCOO(Vrows, Vcols, VYvals);
    chebyshev::Moments2D chebmom(M,M);
    chebmom.SystemLabel(Hoppings.GetLabel());
    chebmom.BandWidth(2.*emax.real());
    chebmom.BandCenter(0.0);
    chebmom.BlockSize(bsize);
    chebmom.SetAndRescaleHamiltonian(Hoppings, Onsites);
    factory::generator gen;
    const size_t Dim = chebmom.SystemSize();
    const auto NumMoms= chebmom.MomentNumber();
    chebyshev::CorrelationExpansionMoments(VY,VX,chebmom,gen,R/bsize);
    chebmom.ApplyJacksonKernel(0.0000001,0.0000001);
    Integer numdiv = 60*chebmom.HighestMomentNumber();
    const Float xbound = chebyshev::CUTOFF;
   

    std::ofstream cond("./ConductivityXY.dat");
    
    scalar I(0,1);
    scalar approxf;
    scalar factor;
    scalar sigma;
    Float dE = (2*chebyshev::CUTOFF)/(Float)(65535);
    

    for(auto x = -chebyshev::CUTOFF;x<=chebyshev::CUTOFF; x+=dE){
        sigma = 0;
        for(Integer m0 = 0; m0< chebmom.HighestMomentNumber(0); m0++)
        for(Integer m1 = 0; m1< chebmom.HighestMomentNumber(1); m1++){
            sigma += 2.*delta_chebF(x,m0)*(DGreenR_chebFAlt(x,m1)*chebmom(m0,m1)).imag();
        }
        sigma = -4.*sigma*chebyshev::CUTOFF*chebyshev::CUTOFF*(Float)chebmom.SystemSize()/(chebmom.BandWidth()*chebmom.BandWidth());
        cond<<x<<" "<<sigma.real()<<std::endl;
    }

    cond.close();

    auto t2 =  std::chrono::high_resolution_clock::now();
    auto s = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
    std::cout<<"The elapsed time is "<<s.count()<<" miliseconds"<<std::endl;

   


    return 0;
}   
