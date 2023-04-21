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

    if(argc != 10){
        printf("Incorrect Number of parameters. The parameters are nv1, nv2, M, R, Bsize, threads numtimes evolutionsxmeasurement maxtime \n");
        exit(EXIT_FAILURE);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    //Parameters for teh simulation
    Integer nv1 = std::atoi(argv[1]);
    Integer nv2 = std::atoi(argv[2]);
    Integer M = std::atoi(argv[3]);
    Integer R = std::atoi(argv[4]);
    Integer bsize = std::atoi(argv[5]);
    Integer threads = std::atoi(argv[6]);
    Integer numtimes = std::atoi(argv[7]);
    Integer evoxmeasure = std::atoi(argv[8]);
    Float MaxTime = std::stod(argv[9]);
    Float ephonon = 0.8;
    Float phononFreq = ephonon/chebyshev::Hbar;
    Float displacement = 0.0014748445992209544; // in nm
    Float Mu =0;
    Float Temp = 1e-10*chebyshev::Boltzmann;
    Float eta = 1.0;
    Float cutoff = 1.2*acc;
        
    const scalar emax = 12.1;

    //Declaration of teh environment variables
    omp_set_num_threads(threads);
    //Declaration of the containers in the system, Oxx are for the onsite and Dxx for disorder (TBD)
    std::vector<Integer> rows, cols, Orows, Ocols;
    std::vector<scalar> vals, Ovals;
    std::vector<Site> SiteList; //List of postions to be updated
    std::vector<Position> Limits; //Limits of the simulation box
    //here We generate the Points. This will be handled by LAMMPS
    GenerateGraphene4At(nv1,nv2,SiteList);
    GenerateBox(nv1,nv2,Limits); //Setting the box
    Position CellSize = Position(a0,a0,1e-3); //Setting the binsize
    SpatialGrid grid(CellSize,Limits[0],Limits[1]);
    grid.BuildFromSiteList(SiteList); 
    grid.GetCutoff() = cutoff;
    GenerateHamiltonian(rows, cols, vals, grid);
    //Set the Hamiltonian
    //Here new site list with indexes from the grid
    auto Stdep = SiteList;
    //This sitelist here will be updated as time goes by
    SparseMat Hoppings;
    SparseMat HopRef;
    Hoppings.SetLabel("SpatialGraphene");
    Hoppings.SetDimensions(SiteList.size(), SiteList.size());
    Hoppings.SetNelem(rows.size());
    Hoppings.ConvertFromCOO(rows, cols, vals);

    HopRef.SetLabel("SpatialGraphene");
    HopRef.SetDimensions(SiteList.size(), SiteList.size());
    HopRef.SetNelem(rows.size());
    HopRef.ConvertFromCOO(rows, cols, vals);
    SparseMat Onsites = SparseMat();
    Onsites.SetLabel(Hoppings.GetLabel());
    Onsites.SetDimensions(SiteList.size(), SiteList.size());
    Onsites.SetNelem(Orows.size());
    Onsites.ConvertFromCOO(Orows, Ocols, Ovals);
    //Declaration ofthe moments that will be used in the calculation
    chebyshev::MomentsNEq chebMom(numtimes, M, evoxmeasure); //TimeDependent
    chebyshev::Moments1D Reference(M,bsize); //Reference
    chebMom.SystemLabel(Hoppings.GetLabel());
    chebMom.BandWidth(2*emax.real());
    chebMom.BandCenter(0.0);
    chebMom.BlockSize(bsize);
    chebMom.MaxTimeStep(numtimes);
    chebMom.TimeDiff(MaxTime/(Float)(numtimes-1));
	chebMom.SetAndRescaleHamiltonian(Hoppings,Onsites);
    Reference.getMomentsParams(chebMom);
    Reference.SetAndRescaleHamiltonian(HopRef,Onsites);
    const size_t Dim = chebMom.SystemSize();
    const size_t maxmom = chebMom.HighestMomentNumber();
    const size_t NumTimes = chebMom.MaxTimeStep();
    factory::generator gen;
    gen.NumberOfStates(R/bsize);
    gen.BlockSize(chebMom.BlockSize());
    gen.SystemSize(chebMom.SystemSize());
    chebyshev::VectorFilter Prepr(chebyshev::FilterType::fermi,std::array<Float,2>({Mu,Temp}),M);
    Prepr.getMomentsParams(chebMom);
    Prepr.FillCoef();
    Prepr.ApplyJacksonKernel(1e-10);
    chebMom.Print();
    scalar dot;
    //Now we start the time iteration
    while(gen.getQuantumState()){
        //Filter step
        if(gen.BlockSize()==1){
            Prepr.SetInitVectors(gen.Out());
            Prepr.PrepareOutVec();
            Reference.SetInitVectors(gen.Out());
        }
        else{
            Prepr.SetInitBlock(gen.Out());
            Prepr.PrepareOutVec();
            Reference.SetInitBlock(gen.Out());
        }

        Integer counter =0; //this will track the number of evolutions prior the measurements
        Integer midx =0; //This tracks the measurement
        while(chebMom.CurrentTimeStep() != chebMom.MaxTimeStep()){
            if(midx!=0){
                GenerateGraphene4AtTD(displacement,phononFreq,chebMom.CurrentTimeStep()*chebMom.TimeDiff(),SiteList,Stdep);
                grid.UpdateFromSiteList(Stdep);
                Hoppings.ReleaseMatrix();
                Onsites.ReleaseMatrix();
                rows = std::move(std::vector<Integer>());
                cols = std::move(std::vector<Integer>());
                vals = std::move(std::vector<scalar>());
                GenerateHamiltonian(rows, cols, vals, grid);
                Hoppings.ConvertFromCOO(rows, cols, vals);
                Onsites.ConvertFromCOO(Orows,Ocols,Ovals);
                chebMom.SetAndRescaleHamiltonian(Hoppings,Onsites);
            }
            else{ }
            if(counter == 0 || counter == evoxmeasure){
                chebMom.SetHamiltonianHoppings(HopRef);
                if(chebMom.BlockSize()==1)
                    chebMom.SetInitVectors(Prepr.OPV());
                else
                    chebMom.SetInitBlock(Prepr.OPV());

                for(size_t m = 0; m<M; m++){
                    double scal = 2.0/(gen.NumberOfStates()*gen.BlockSize());
                    if(m==0)
                        scal*=0.5;
                    linalg::dot(gen.Out().size(), gen.Out(),chebMom.ChebV0(),&dot);
                    chebMom(midx,m) += scal*dot;
                    chebMom.Iterate();
                }
                counter = 0;
                midx++;
                assert(midx<=(numtimes/evoxmeasure)+1);
                chebMom.SetHamiltonianHoppings(Hoppings);
            }
            if(chebMom.CurrentTimeStep()==0){
                for(size_t m=0; m<M; m++){
                    double scal = 2.0/(gen.NumberOfStates()*gen.BlockSize());
                    if(m==0)
                        scal*=0.5;
                    linalg::dot(gen.Out().size(), gen.Out(),Reference.ChebV0(),&dot);
                    Reference(m) += scal*dot;
                    Reference.Iterate();
                }
            }
            counter++;
            chebMom.IncreaseTimeStep();
            chebMom.Evolve(Prepr.OPV());
            chebMom.Evolve(gen.Out());
        }
    }

    //Clearing all the moments
    chebMom.ApplyJacksonKernel(1e-10);
    std::string filename = "./NEQ"+Hoppings.GetLabel()+"Nv1"+std::to_string(nv1)+"xNv2"+std::to_string(nv2)+"M"+std::to_string(M);
    Float dx = 2.0*(chebyshev::CUTOFF)/(Float)65535;
    for(auto t = 0; t<chebMom.NumMeasurements(); t++){
        std::ofstream output(filename+"EvoxMeasurement"+std::to_string(chebMom.EvolutionsxMeasurements())+"Measurement"+std::to_string(t)+"x"+std::to_string(chebMom.TimeDiff())+".dat");
        for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
            scalar approx = 0;
            for(Float m=0; m<chebMom.HighestMomentNumber(); m++){
                approx+=delta_chebF(x,m)*chebMom(t,(size_t)m);
            }
            output<<x<<"\t"<<approx.real()<<std::endl;
        }
    }
    

    Reference.ApplyJacksonKernel(1e-10);
    std::ofstream outputref("./NEQREF"+Hoppings.GetLabel()+"Nv1"+std::to_string(nv1)+"xNv2"+std::to_string(nv2)+"M"+std::to_string(M)+".dat");
    for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
            scalar approx=0;
            for(Float m = 0; m<Reference.HighestMomentNumber(); m++){
                approx+=delta_chebF(x,m)*Reference((size_t)m);
            }
            outputref<<x<<"\t"<<approx.real()<<std::endl;
        }
    auto t2 = std::chrono::high_resolution_clock::now();

    auto s = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);

    std::cout<<" The elapsed time is "<<s.count()<<std::endl;
    
    


    return 0;
}   
