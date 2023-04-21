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

    //auto t1 = std::chrono::high_resolution_clock::now();
    Integer nv1 = std::atoi(argv[1]);
    Integer nv2 = std::atoi(argv[2]);
    Integer M = std::atoi(argv[3]);
    Integer R = std::atoi(argv[4]);
    Integer bsize = std::atoi(argv[5]);
    Integer threads = std::atoi(argv[6]);

    std::string name = "./DOS.dat";

    omp_set_num_threads(threads);
    //Unitary testlat("./pybuild","squaremine");
    Unitary testlat("./pybuild","GrapheneSpinful");
    //Unitary testlat("./pybuild","HaldaneModel");
    //Unitary testlat("./pybuild","MoS2");
    //Unitary testlat("./pybuild","MoS2LzBasis");
    testlat.RegisterOrbitals();
    testlat.RegisterHoppings();    
    PeriodicHamiltonian HamBuilder(testlat,nv1,nv2);
    //PeriodicHamiltonian filter
    std::vector<Integer> Oprows, Opcols;
    std::vector<scalar> Opvals;
    Oprows.reserve(testlat.getFromHoppings().size()*nv1*nv2);
    Opcols.reserve(testlat.getFromHoppings().size()*nv1*nv2);
    Opvals.reserve(testlat.getFromHoppings().size()*nv1*nv2);
    //
    std::vector<Integer> rows, cols, Orows, Ocols;

    std::vector<scalar> vals, Ovals;
    rows.reserve(testlat.getFromHoppings().size()*nv1*nv2);
    cols.reserve(testlat.getFromHoppings().size()*nv1*nv2);
    vals.reserve(testlat.getFromHoppings().size()*nv1*nv2);
    Orows.reserve(testlat.getFromHoppings().size()*nv1*nv2);
    Ocols.reserve(testlat.getFromHoppings().size()*nv1*nv2);
    Ovals.reserve(testlat.getFromHoppings().size()*nv1*nv2);

    HamBuilder.SetPeriodicHoppings<2>(rows, cols, vals);
    HamBuilder.SetOnSitePotentials<2>(Orows, Ocols, Ovals);
    Integer dimension = 0;
    for(auto v: HamBuilder.getUnitCell().getOnSiteEnergies())
        dimension += v.cols();
    std::cout<<dimension<<std::endl;
    const scalar emax = 12.1 ;
    
    auto t1 = std::chrono::high_resolution_clock::now();

    SparseMat Hoppings,Filter;

    Hoppings.SetLabel(testlat.getName());
    Hoppings.SetDimensions(nv1*nv2*dimension, nv1*nv2*dimension);
    Hoppings.SetNelem(rows.size());
    Hoppings.ConvertFromCOO(rows, cols, vals);
    SparseMat Onsites = SparseMat();
    Onsites.SetLabel(testlat.getName());
    Onsites.SetDimensions(nv1*nv2*dimension, nv1*nv2*dimension);
    Onsites.SetNelem(Orows.size());
    std::cout<<Onsites.NumNNZ()<<" "<<Orows.size()<<std::endl;
    Onsites.ConvertFromCOO(Orows, Ocols, Ovals);
	
    //
    //HamBuilder.SetOpeartorName("ProjLzZero");
    //HamBuilder.SetOperator<2>(Oprows,Opcols,Opvals);
    //Filter.SetLabel("ProjLzZero");
    //Filter.SetDimensions(dimension*nv1*nv2, dimension*nv1*nv2);
    //Filter.SetNelem(Oprows.size());
    //Filter.ConvertFromCOO(Oprows,Opcols, Opvals);
    


    
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
    printf("Ahora entro en la funcion del spectral moments\n");
    chebyshev::SpectralMoments(ID, chebMom, gen, R/bsize);
    //chebyshev::SpectralMoments(Filter, chebMom, gen, R/bsize);
    chebMom.ApplyJacksonKernel(0.0000001);
    std::ofstream output("./DOS.dat");
    Float dx = 2.0*(chebyshev::CUTOFF)/(Float)65535;
    for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
        scalar approx=0;
        for(Float m =0 ; m<chebMom.HighestMomentNumber(); m++){
            approx += delta_chebF(x,m)*chebMom.MomentVector(m);
        }
        output<<x<<"\t"<<approx.real()<<std::endl;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    auto s = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);

    std::cout<<" The elapsed time is "<<s.count()<<std::endl;
    
    


    return 0;
}   
