#include "typedef.h"
#include "sparsematrix.h"
#include "moments.h"
#include "spatial.h"
#include "grapheneGenerator.h"
#include "chebyshevsolver.h"
//KPM structure to be passed through callback function in lammps
struct KPM{
    //Hamiltonian elements, the codifictation goes as H = Hhop + Honsites
    std::unique_ptr<SparseMat> ptrHoppings;
    std::unique_ptr<SparseMat> ptrRefHoppings;
    std::unique_ptr<SparseMat> ptrOnsites;
    std::unique_ptr<SparseMat> ptrRefOnsites;
    //Force matrices, the forces are skew hermitian to encode the direction of the hoppings
    std::unique_ptr<SparseMat> ptrForceMatX;
    std::unique_ptr<SparseMat> ptrForceMatY;
    std::unique_ptr<SparseMat> ptrForceMatZ;
    //Pointer to the MomentumFile
    std::unique_ptr<chebyshev::MomentsNEq> ptrMom;
    //Pointer to the Reference moments To compute the DOS at every "measurement"
    std::unique_ptr<chebyshev::Moments1D> ptrRefMom;
    //Ptr to the factory generator of random vectors
    std::unique_ptr<factory::generator> ptrFactory;
    //Filter for the dynamic conditions
    std::unique_ptr<chebyshev::VectorFilter> ptrFilter;
    //Filter for the final configuration of the lattice
    std::unique_ptr<chebyshev::VectorFilter> ptrFilterEnd;
    //pointer to the realspace grid
    std::unique_ptr<SpatialGrid> gridPtr;
    //Uniqueptr to the optimization class with NLOPT TBD
    
    //
    //KPM parameters
    Integer Moments;
    Integer Random;
    Integer bsize;
    Integer threads;
    Integer Niter;
    Integer EvolutionsXMeasurement;
    //Box parameter
    Float cellsize;
    //Thermodynamic variables
    Float Tlattice;
    Float Ttarget;
    Float PreviousPE;
    Float PreviousEel;
    //Parameters for the optimizator
    Float MuOpt, TempOpt, EnergyChange;
    //Prefix of the files
    std::string prefix;
    //Containers of the relevant data of the simulation
    //  List of sites
    std::vector<Site> SiteList;
    // Rows of Hoppings, Onsites, and forces
    std::vector<Integer> rows;
    std::vector<Integer> Orows;
    std::vector<Integer> frows;
    // Cols of Hoppings, Onsites and forces
    std::vector<Integer> cols;
    std::vector<Integer> Ocols;
    std::vector<Integer> fcols;
    //  Values of hopping, onsites, forces xyz
    std::vector<scalar> vals;
    std::vector<scalar> Ovals;
    std::vector<scalar> fXvals;
    std::vector<scalar> fYvals;
    std::vector<scalar> fZvals;
    // Forces acting on the filtered random phase
    std::vector<scalar> FXFilter;
    std::vector<scalar> FYFilter;
    std::vector<scalar> FZFilter;
    //  Forces acting on the filtered random phase equilibrium config
    std::vector<scalar> FXFilterEnd;
    std::vector<scalar> FYFilterEnd;
    std::vector<scalar> FZFilterEnd;
    // Forces acting on the full random vector
    std::vector<scalar> FXRV;
    std::vector<scalar> FYRV;
    std::vector<scalar> FZRV;
    

};


struct Info{
    char *input;
    AuxMemory *memory;
    LAMMPS_NS::LAMMPS *lmp;
    KPM *kpm;
    int me;
    std::ofstream thermoOut;

};





void InitializeHamiltonianAndForces(void * info){
    Info * kpmstruct = (Info *) info;
    kpmstruct ->kpm->gridPtr->BuildFromSiteList(kpmstruct -> kpm -> SiteList);
    kpmstruct -> kpm ->gridPtr->GetCutoff() = 1.2*kpmstruct->kpm ->cellsize/std::sqrt(3);
    const Integer natoms =kpmstruct-> kpm ->SiteList.size();
    GenerateHamiltonian(kpmstruct -> kpm ->rows, kpmstruct -> kpm -> cols, kpmstruct -> kpm -> vals, *(kpmstruct -> kpm ->gridPtr));
    GenerateForceMat(kpmstruct -> kpm ->frows, kpmstruct -> kpm -> fcols, kpmstruct -> kpm -> fXvals, kpmstruct -> kpm -> fYvals, kpmstruct -> kpm -> fZvals, *(kpmstruct -> kpm ->gridPtr));
    //Initialization of the Hamiltonian Matrix
    kpmstruct -> kpm -> ptrHoppings = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
    kpmstruct -> kpm -> ptrHoppings -> SetLabel("GrapheneLammps");
    kpmstruct -> kpm -> ptrHoppings -> SetDimensions(natoms, natoms);
    kpmstruct -> kpm -> ptrHoppings -> SetNelem(kpmstruct -> kpm ->rows.size());
    //Initialization of the reference matrix
    kpmstruct -> kpm -> ptrRefHoppings = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
    kpmstruct -> kpm -> ptrRefHoppings -> SetLabel("GrapheneLammpsRef");
    kpmstruct -> kpm -> ptrRefHoppings -> SetDimensions(natoms, natoms);
    kpmstruct -> kpm -> ptrRefHoppings -> SetNelem(kpmstruct -> kpm ->rows.size());
    //Force Matrices
    //// X force
    kpmstruct -> kpm -> ptrForceMatX = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
    kpmstruct -> kpm -> ptrForceMatX -> SetLabel("GrapheneFX");
    kpmstruct -> kpm -> ptrForceMatX -> SetDimensions(natoms, natoms);
    kpmstruct -> kpm -> ptrForceMatX -> SetNelem(kpmstruct -> kpm ->frows.size());
    kpmstruct -> kpm -> ptrForceMatX ->SetSkewHermitian();
    kpmstruct -> kpm -> ptrForceMatX -> ConvertFromCOO(kpmstruct -> kpm ->frows, kpmstruct -> kpm -> fcols, kpmstruct -> kpm -> fXvals);
    //// Y Force
    kpmstruct -> kpm -> ptrForceMatY = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
    kpmstruct -> kpm -> ptrForceMatY -> SetLabel("GrapheneFY");
    kpmstruct -> kpm -> ptrForceMatY -> SetDimensions(natoms, natoms);
    kpmstruct -> kpm -> ptrForceMatY -> SetNelem(kpmstruct -> kpm ->frows.size());
    kpmstruct -> kpm -> ptrForceMatY ->SetSkewHermitian();
    kpmstruct -> kpm -> ptrForceMatY -> ConvertFromCOO(kpmstruct -> kpm ->frows, kpmstruct -> kpm -> fcols, kpmstruct -> kpm -> fYvals);
    //// Z Force
    kpmstruct -> kpm -> ptrForceMatZ = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
    kpmstruct -> kpm -> ptrForceMatZ -> SetLabel("GrapheneFZ");
    kpmstruct -> kpm -> ptrForceMatZ -> SetDimensions(natoms, natoms);
    kpmstruct -> kpm -> ptrForceMatZ -> SetNelem(kpmstruct -> kpm ->frows.size());
    kpmstruct -> kpm -> ptrForceMatZ ->SetSkewHermitian();
    kpmstruct -> kpm -> ptrForceMatZ -> ConvertFromCOO(kpmstruct -> kpm ->frows, kpmstruct -> kpm -> fcols, kpmstruct -> kpm -> fZvals);
    //Generation of MKL matrices
    kpmstruct -> kpm -> ptrHoppings -> ConvertFromCOO(kpmstruct -> kpm ->rows, kpmstruct -> kpm -> cols, kpmstruct -> kpm -> vals);
    kpmstruct -> kpm -> ptrRefHoppings -> ConvertFromCOO(kpmstruct -> kpm ->rows, kpmstruct -> kpm -> cols, kpmstruct -> kpm -> vals);
    //Preparing onsite matrices
    kpmstruct -> kpm -> ptrOnsites = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
    kpmstruct -> kpm -> ptrOnsites -> SetLabel("GrapheneLammps");
    kpmstruct -> kpm -> ptrOnsites -> SetDimensions(natoms, natoms);
    kpmstruct -> kpm -> ptrOnsites -> SetNelem(kpmstruct -> kpm ->Orows.size());
    kpmstruct -> kpm -> ptrOnsites -> ConvertFromCOO(kpmstruct -> kpm ->Orows, kpmstruct -> kpm -> Ocols, kpmstruct -> kpm -> Ovals);
    //Preparing reference OnsiteMatrices
    kpmstruct -> kpm -> ptrRefOnsites = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
    kpmstruct -> kpm -> ptrRefOnsites -> SetLabel("GrapheneLammps");
    kpmstruct -> kpm -> ptrRefOnsites -> SetDimensions(natoms, natoms);
    kpmstruct -> kpm -> ptrRefOnsites -> SetNelem(kpmstruct -> kpm ->Orows.size());
    kpmstruct -> kpm -> ptrRefOnsites -> ConvertFromCOO(kpmstruct -> kpm ->Orows, kpmstruct -> kpm -> Ocols, kpmstruct -> kpm -> Ovals);
}

void InitializeMomentsAndVectors(void * info,const std::vector<Float>& Par, const std::vector<Float>& ParEnd, Float unittime, Float emax){
    if(Par.size() !=2 || ParEnd.size() !=2){
        std::cout<<"The parameters for the initial and final configurations are not properly set\n"<<std::endl;
        exit(EXIT_FAILURE);
    }


    Info * kpmstruct = (Info *) info;


    //Constant quantities to generate the moments
    const Integer niter = kpmstruct -> kpm -> Niter;
    const Integer moments = kpmstruct -> kpm -> Moments;
    const Integer evoxmeasurement = kpmstruct -> kpm -> EvolutionsXMeasurement;
    const Integer blocksize = kpmstruct -> kpm -> bsize;


    const Float tempini = Par[1];
    const Float Mu = Par[0];
    const Float Muend = ParEnd[0];//To guarantee particle conservation
    kpmstruct -> kpm -> Tlattice = ParEnd[1]; //Fixed for ps relaxation

    kpmstruct -> kpm -> ptrMom = std::move(std::unique_ptr<chebyshev::MomentsNEq>(new chebyshev::MomentsNEq(niter,moments, evoxmeasurement, blocksize)));
    kpmstruct -> kpm -> ptrRefMom = std::move(std::unique_ptr<chebyshev::Moments1D>(new chebyshev::Moments1D(moments)));
    kpmstruct -> kpm -> ptrFactory = std::move(std::unique_ptr<factory::generator>(new factory::generator()));
    kpmstruct -> kpm -> ptrFilter = std::move(std::unique_ptr<chebyshev::VectorFilter>(new chebyshev::VectorFilter()));
    kpmstruct -> kpm -> ptrFilterEnd = std::move(std::unique_ptr<chebyshev::VectorFilter>(new chebyshev::VectorFilter()));
    //Initialization of the NeqMoments
    kpmstruct -> kpm -> ptrMom -> SystemLabel(kpmstruct -> kpm -> ptrOnsites -> GetLabel());
    kpmstruct -> kpm -> ptrMom -> BandWidth(2.*emax);
    kpmstruct -> kpm -> ptrMom -> BandCenter(0); //Here I'm fixing particle hole symmetry
    kpmstruct -> kpm -> ptrMom -> TimeDiff(unittime);
    kpmstruct -> kpm -> ptrMom -> SetAndRescaleHamiltonian(*(kpmstruct -> kpm -> ptrHoppings), *(kpmstruct -> kpm -> ptrOnsites));
    kpmstruct -> kpm -> ptrMom -> Print();
    //Definition of the simulation paramters
     //Eventually we can use the optimization class from NLOPT to determine the time-dependent chemical potential
    //Filling the reference moments
    kpmstruct -> kpm -> ptrRefMom -> getMomentsParams(*(kpmstruct -> kpm -> ptrMom));
    kpmstruct -> kpm -> ptrRefMom -> SetAndRescaleHamiltonian(*(kpmstruct -> kpm -> ptrRefHoppings), *(kpmstruct -> kpm -> ptrRefOnsites));
    //Configuring the filter of the random phase state
    kpmstruct -> kpm -> ptrFactory -> NumberOfStates(kpmstruct -> kpm -> Random/ kpmstruct -> kpm -> bsize);
    kpmstruct -> kpm -> ptrFactory -> BlockSize(kpmstruct -> kpm -> bsize);
    kpmstruct -> kpm -> ptrFactory -> SystemSize( kpmstruct -> kpm -> ptrMom->SystemSize());
    //Start the vectors
    kpmstruct -> kpm -> ptrFactory -> getQuantumState();
    
    
    //Load the Parameters in the filter
    kpmstruct -> kpm -> ptrFilter->getMomentsParams(*(kpmstruct -> kpm -> ptrMom));
    kpmstruct -> kpm -> ptrFilter->setFilterType(chebyshev::FilterType::fermi);
    kpmstruct -> kpm -> ptrFilter -> SetInitBlock(kpmstruct -> kpm -> ptrFactory ->Out());
    kpmstruct -> kpm -> ptrFilter -> setFilterParams(std::array<Float,2>({Mu,tempini*chebyshev::Boltzmann}));
    kpmstruct -> kpm -> ptrFilter -> MomentVector(std::vector<scalar>(kpmstruct -> kpm -> Moments,0));
    kpmstruct -> kpm -> ptrFilter -> setMomentNumber(kpmstruct -> kpm -> Moments);
    kpmstruct -> kpm -> ptrFilter -> FillCoef();
    kpmstruct -> kpm -> ptrFilter -> ApplyJacksonKernel(1E-10);
    kpmstruct -> kpm -> ptrFilter -> PrepareOutVec();
    //Parameters for the end vector
    kpmstruct -> kpm -> ptrFilterEnd->getMomentsParams(*(kpmstruct -> kpm -> ptrMom));
    kpmstruct -> kpm -> ptrFilterEnd->setFilterType(chebyshev::FilterType::fermi);
    kpmstruct -> kpm -> ptrFilterEnd -> SetInitBlock(kpmstruct -> kpm -> ptrFactory ->Out());
    kpmstruct -> kpm -> ptrFilterEnd -> setFilterParams(std::array<Float,2>({Muend,kpmstruct -> kpm -> Tlattice*chebyshev::Boltzmann}));
    kpmstruct -> kpm -> ptrFilterEnd -> MomentVector(std::vector<scalar>(kpmstruct -> kpm -> Moments,0));
    kpmstruct -> kpm -> ptrFilterEnd -> setMomentNumber(kpmstruct -> kpm -> Moments);
    kpmstruct -> kpm -> ptrFilterEnd -> FillCoef();
    kpmstruct -> kpm -> ptrFilterEnd -> ApplyJacksonKernel(1E-10);
    kpmstruct -> kpm -> ptrFilterEnd -> PrepareOutVec();



}


void ComputeQuantumForces(void *info,double **fexternal ){
    Info * kpmstruct = (Info *) info;
    const Integer blocksize = kpmstruct -> kpm -> bsize;
    //Compute FX_i|PsiFD(t)>
    kpmstruct -> kpm -> ptrForceMatX -> BlockMultiply(kpmstruct -> kpm -> ptrFilter -> OPV(),kpmstruct -> kpm -> FXFilter, kpmstruct -> kpm -> bsize);
    kpmstruct -> kpm -> ptrForceMatY -> BlockMultiply(kpmstruct -> kpm -> ptrFilter -> OPV(),kpmstruct -> kpm -> FYFilter, kpmstruct -> kpm -> bsize);
    kpmstruct -> kpm -> ptrForceMatZ -> BlockMultiply(kpmstruct -> kpm -> ptrFilter -> OPV(),kpmstruct -> kpm -> FZFilter, kpmstruct -> kpm -> bsize);
    //Compute FX_i|PsiEq(t)>
    kpmstruct -> kpm -> ptrForceMatX -> BlockMultiply(kpmstruct -> kpm -> ptrFilterEnd -> OPV(),kpmstruct -> kpm -> FXFilterEnd, kpmstruct -> kpm -> bsize);
    kpmstruct -> kpm -> ptrForceMatY -> BlockMultiply(kpmstruct -> kpm -> ptrFilterEnd -> OPV(),kpmstruct -> kpm -> FYFilterEnd, kpmstruct -> kpm -> bsize);
    kpmstruct -> kpm -> ptrForceMatZ -> BlockMultiply(kpmstruct -> kpm -> ptrFilterEnd -> OPV(),kpmstruct -> kpm -> FZFilterEnd, kpmstruct -> kpm -> bsize);
    //Compute FX:i|PRV(t)>
    kpmstruct -> kpm -> ptrForceMatX -> BlockMultiply(kpmstruct -> kpm -> ptrFilter -> OPV(),kpmstruct -> kpm -> FXRV, kpmstruct -> kpm -> bsize);
    kpmstruct -> kpm -> ptrForceMatY -> BlockMultiply(kpmstruct -> kpm -> ptrFilter -> OPV(),kpmstruct -> kpm -> FYRV, kpmstruct -> kpm -> bsize);
    kpmstruct -> kpm -> ptrForceMatZ -> BlockMultiply(kpmstruct -> kpm -> ptrFilter -> OPV(),kpmstruct -> kpm -> FZRV, kpmstruct -> kpm -> bsize);
    //Now we iterate over all the sites while each thread write over each force
    const Integer dimm = kpmstruct -> kpm -> SiteList.size();
    #pragma omp parallel for schedule(dynamic) firstprivate(blocksize, dimm)
    for(auto n = 0; n<kpmstruct -> kpm -> SiteList.size(); n++){
        for(auto b = 0; b<blocksize; b++){
            
            //NonEquilibrium Part
            //First term 
            fexternal[n][0] += (Float)dimm*(std::conj(kpmstruct -> kpm -> ptrFactory -> Out()[n+b*(dimm)])*kpmstruct -> kpm -> FXFilter[n+b*(dimm)]/(scalar)blocksize).real();
            fexternal[n][1] += (Float)dimm*(std::conj(kpmstruct -> kpm -> ptrFactory -> Out()[n+b*(dimm)])*kpmstruct -> kpm -> FYFilter[n+b*(dimm)]/(scalar)blocksize).real();
            fexternal[n][2] += (Float)dimm*(std::conj(kpmstruct -> kpm -> ptrFactory -> Out()[n+b*(dimm)])*kpmstruct -> kpm -> FZFilter[n+b*(dimm)]/(scalar)blocksize).real();
            //Second term
            fexternal[n][0] += (Float)dimm*(std::conj(kpmstruct -> kpm -> FXRV[n+b*(dimm)])*kpmstruct -> kpm -> ptrFilter->OPV()[n+b*(dimm)]/(scalar)blocksize).real();
            fexternal[n][1] += (Float)dimm*(std::conj(kpmstruct -> kpm -> FYRV[n+b*(dimm)])*kpmstruct -> kpm -> ptrFilter->OPV()[n+b*(dimm)]/(scalar)blocksize).real();
            fexternal[n][2] += (Float)dimm*(std::conj(kpmstruct -> kpm -> FZRV[n+b*(dimm)])*kpmstruct -> kpm -> ptrFilter->OPV()[n+b*(dimm)]/(scalar)blocksize).real();
            //Equilibrium Part
            //First term 
            fexternal[n][0] -= (Float)dimm*(std::conj(kpmstruct -> kpm -> ptrFactory -> Out()[n+b*(dimm)])*kpmstruct -> kpm -> FXFilterEnd[n+b*(dimm)]/(scalar)blocksize).real();
            fexternal[n][1] -= (Float)dimm*(std::conj(kpmstruct -> kpm -> ptrFactory -> Out()[n+b*(dimm)])*kpmstruct -> kpm -> FYFilterEnd[n+b*(dimm)]/(scalar)blocksize).real();
            fexternal[n][2] -= (Float)dimm*(std::conj(kpmstruct -> kpm -> ptrFactory -> Out()[n+b*(dimm)])*kpmstruct -> kpm -> FZFilterEnd[n+b*(dimm)]/(scalar)blocksize).real();
            //Second ter-
            fexternal[n][0] -= (Float)dimm*(std::conj(kpmstruct -> kpm -> FXRV[n+b*(dimm)])*kpmstruct -> kpm -> ptrFilterEnd->OPV()[n+b*(dimm)]/(scalar)blocksize).real();
            fexternal[n][1] -= (Float)dimm*(std::conj(kpmstruct -> kpm -> FYRV[n+b*(dimm)])*kpmstruct -> kpm -> ptrFilterEnd->OPV()[n+b*(dimm)]/(scalar)blocksize).real();
            fexternal[n][2] -= (Float)dimm*(std::conj(kpmstruct -> kpm -> FZRV[n+b*(dimm)])*kpmstruct -> kpm -> ptrFilterEnd->OPV()[n+b*(dimm)]/(scalar)blocksize).real();
        }
    }
    return;
}

