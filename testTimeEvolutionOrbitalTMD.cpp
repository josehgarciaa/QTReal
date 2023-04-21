#include <iostream>
#include "unitary.h"
#include "periodichamiltonian.h"
#include "sparsematrix.h"
#include "moments.h"
#include <chrono>
#include "chebyshevsolver.h"


int main(int argc, char **argv){
    srand(time(NULL));
    if(argc!=9){
        printf("Incorrect number of parameters. These are nv1, nv2, M, R bsize, threads numtimes maxtime\n");
        exit(EXIT_FAILURE);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    Integer nv1 = std::atoi(argv[1]);
    Integer nv2 = std::atoi(argv[2]);
    Integer M = std::atoi(argv[3]);
    Integer R = std::atoi(argv[4]);
    Integer bsize = std::atoi(argv[5]);
    Integer threads = std::atoi(argv[6]);
    Integer numtimes = std::atoi(argv[7]);
    Float MaxTime = std::stod(argv[8]);
    omp_set_num_threads(threads);
    Unitary testlat("./pybuild","MoS2LzBasis");
    testlat.RegisterOrbitals();
    testlat.RegisterHoppings();
    const scalar emax = 4.1;
    PeriodicHamiltonian HamBuilder(testlat, nv1, nv2);
    std::vector<Integer> rows, cols, Orows, Ocols, Oprows, Opcols;
    std::vector<scalar> vals, Ovals, Opvals;
    HamBuilder.SetPeriodicHoppings<2>(rows, cols ,vals);
    HamBuilder.SetOnSitePotentials<2>(Orows,Ocols,Ovals);
    HamBuilder.SetOpeartorName("Lz");
    HamBuilder.SetOperator<2>(Oprows,Opcols,Opvals);
    SparseMat Hoppings, Onsites, LzUpFilter, Lz;
    Hoppings.SetLabel(HamBuilder.getUnitCell().getName());
    Integer dimension = 0;
    for(auto v: HamBuilder.getUnitCell().getOnSiteEnergies())
        dimension += v.cols();
    Hoppings.SetDimensions(dimension*nv1*nv2, dimension*nv1*nv2);
    Hoppings.SetNelem(rows.size());
    Onsites.SetLabel(HamBuilder.getUnitCell().getName());
    Onsites.SetDimensions(dimension*nv1*nv2, dimension*nv1*nv2);
    Onsites.SetNelem(Orows.size());
    Lz.SetLabel("Lz");
    Lz.SetDimensions(dimension*nv1*nv2, dimension*nv1*nv2);
    Lz.SetNelem(Oprows.size());
    Hoppings.ConvertFromCOO(rows, cols, vals);
    Onsites.ConvertFromCOO(Orows, Ocols, Ovals);
    Lz.ConvertFromCOO(Oprows,Opcols, Opvals);
    Oprows = std::move(std::vector<Integer>());
    Opcols = std::move(std::vector<Integer>());
    Opvals = std::move(std::vector<scalar>());
    HamBuilder.SetOpeartorName("ProjLzUp");
    HamBuilder.SetOperator<2>(Oprows,Opcols,Opvals);
    LzUpFilter.SetLabel("ProjLzUp");
    LzUpFilter.SetDimensions(dimension*nv1*nv2, dimension*nv1*nv2);
    LzUpFilter.SetNelem(Oprows.size());
    LzUpFilter.ConvertFromCOO(Oprows,Opcols, Opvals);
    chebyshev::MomentsTD chebMom(numtimes,M);
    chebyshev::Moments1D Reference(M,bsize);
    chebMom.SystemLabel(Hoppings.GetLabel());
    chebMom.BandWidth(2.*emax.real());
    chebMom.BandCenter(0.0);
    chebMom.BlockSize(bsize);
    chebMom.MaxTimeStep(numtimes);
    chebMom.TimeDiff(MaxTime/(Float)(numtimes-1));
    chebMom.SetAndRescaleHamiltonian(Hoppings,Onsites);
    const size_t Dim = chebMom.SystemSize();
    const size_t maxmom = chebMom.HighestMomentNumber();
    const size_t NumTimes = chebMom.MaxTimeStep();
    Reference.getMomentsParams(chebMom);
    factory::generator gen;
    gen.NumberOfStates(R/bsize);
    gen.BlockSize(chebMom.BlockSize());
    gen.SystemSize(chebMom.SystemSize());
    chebMom.Print();
    scalar dot;
    while(gen.getQuantumState()){
        auto OPL = std::vector<scalar>(gen.Out().size());
        auto SR = std::vector<scalar>(gen.Out().size(),0);
        LzUpFilter.BlockMultiply(gen.Out(), OPL,chebMom.BlockSize());
        auto OPR = OPL;
        while(chebMom.CurrentTimeStep() != chebMom.MaxTimeStep()){
            const auto n = chebMom.CurrentTimeStep();
            Lz.BlockMultiply(OPL,SR,chebMom.BlockSize());
            chebMom.SetInitBlock(OPR);
            for(size_t m = 0; m <M; m++){
                double scal = 2.0/(gen.NumberOfStates()*gen.BlockSize());
                if(m==0)
                    scal *=0.5;
                linalg::dot(gen.Out().size(),SR,chebMom.ChebV0(),&dot);
                chebMom(n,m) += scal*dot*0.5;
                chebMom.Iterate();

            }

            chebMom.SetInitBlock(SR);
            for(size_t m = 0; m <M; m++){
                double scal = 2.0/(gen.NumberOfStates()*gen.BlockSize());
                if(m==0)
                    scal *=0.5;
                linalg::dot(gen.Out().size(),OPR,chebMom.ChebV0(),&dot);
                chebMom(n,m) += scal*dot*0.5;
                chebMom.Iterate();

            }

            if(chebMom.CurrentTimeStep()==0){
                Reference.SetInitBlock(OPR);
                for(size_t m = 0; m <M; m++){
                    double scal = 2.0/(gen.NumberOfStates()*gen.BlockSize());
                    if(m==0)
                        scal *=0.5;
                    linalg::dot(gen.Out().size(),OPR,Reference.ChebV0(),&dot);
                    Reference(m) += scal*dot;
                    Reference.Iterate();

                }   
            }
            chebMom.IncreaseTimeStep();
            chebMom.Evolve(OPR);
            chebMom.Evolve(OPL);

        }
    }
    chebMom.ApplyJacksonKernel(0.0000000001);
    std::string filename= "./"+Hoppings.GetLabel()+"NVec1"+std::to_string(nv1)+"Nvec2"+std::to_string(nv2)+"M"+std::to_string(M);
    Float dx = 2.0*(chebyshev::CUTOFF)/(Float)65535;
    for(auto t =0; t<chebMom.MaxTimeStep(); t++){
        std::ofstream output(filename+"Step"+std::to_string(t)+"x"+std::to_string(chebMom.TimeDiff())+".dat");
        for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
            scalar approx=0;
            for(Float m = 0; m<chebMom.HighestMomentNumber(); m++){
                approx+=delta_chebF(x,m)*chebMom(t,(size_t)m);
            }
            output<<x<<"\t"<<approx.real()<<std::endl;
        }
    }

    Reference.ApplyJacksonKernel(0.000000001);
    std::ofstream outputref("./REF"+Hoppings.GetLabel()+"NVec1"+std::to_string(nv1)+"NVec2"+std::to_string(nv2)+"M"+std::to_string(M)+".dat");

    for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
            scalar approx=0;
            for(Float m = 0; m<Reference.HighestMomentNumber(); m++){
                approx+=delta_chebF(x,m)*Reference((size_t)m);
            }
            outputref<<x<<"\t"<<approx.real()<<std::endl;
        }


    auto t2 = std::chrono::high_resolution_clock::now();

    auto s = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    std::cout << "Runtime: " << s.count() << "\t"
              << "Miliseconds" << std::endl;


    



    return 0;
}