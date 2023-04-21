#include "typedef.h"
#include <chrono>
#include <mpi.h>
#include "many2one.h"
#include "one2many.h"
#include "files.h"
#include "memory.h"
#include "error.h"
#include "lmpintegration.h"
#include "sparsematrix.h"
#include "moments.h"
#include "spatial.h"
#include "grapheneGenerator.h"
#include <iostream>
#include "chebyshevsolver.h"

struct KPM{
    std::unique_ptr<SparseMat> ptrHoppings;
    std::unique_ptr<SparseMat> ptrRefHoppings;
    std::unique_ptr<SparseMat> ptrOnsites;
    std::unique_ptr<SparseMat> ptrRefOnsites;
    std::unique_ptr<chebyshev::MomentsNEq> ptrMom;
    std::unique_ptr<chebyshev::Moments1D> ptrRefMom;
    std::unique_ptr<factory::generator> ptrFactory;
    std::unique_ptr<chebyshev::VectorFilter> ptrFilter;
    std::unique_ptr<SpatialGrid> gridPtr;
    Integer Moments;
    Integer Random;
    Integer bsize;
    Integer threads;
    Integer Niter;
    Integer EvolutionsXMeasurement;
    Float cellsize;
    std::string prefix;
    std::vector<Site> SiteList;
    std::vector<Integer> rows;
    std::vector<Integer> Orows;
    std::vector<Integer> cols;
    std::vector<Integer> Ocols;
    std::vector<scalar> vals;
    std::vector<scalar> Ovals;
    std::vector<scalar> ProjectedRV;
    std::vector<scalar> RVHolder;

};


struct Info{
    char *input;
    AuxMemory *memory;
    LAMMPS_NS::LAMMPS *lmp;
    KPM *kpm;
    int me;
    std::ofstream thermoOut;

};

void trialcallback(void* , LAMMPS_NS::bigint, int, int*, double **, double**);


int main(int argc, char **argv)
{
    //This is a copy of the input files of the files of the couple folder of lammps
    Integer n;
    char str[128];
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    Integer me, nprocs;
    MPI_Comm_rank(comm,&me);
    MPI_Comm_size(comm,&nprocs);
    Integer rank = 0;
    while (rank < nprocs){
        if(me == rank)
            omp_set_num_threads(std::atoi(argv[4]));
        rank ++;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    


    AuxMemory *memory = new AuxMemory(comm);
    AuxError * error = new AuxError(comm);

    if(argc!=8) error->all("Incorrect Number of Parameters. The parameters are numomm, R, bsize, threads, evoxMeasurement, MDiter input\n");
    
    Integer niter = std::atoi(argv[6]);
    n = strlen(argv[7])+1;
    char *lammps_input = new char[n];
    strcpy(lammps_input,argv[7]);

    //Instatiate lammps

    LAMMPS_NS::LAMMPS *lmp = new LAMMPS_NS::LAMMPS(0,NULL,MPI_COMM_WORLD);
    lmp ->input->file(lammps_input);

    Info info;
    info.me = me;
    info.memory = memory;
    info.lmp = lmp;
    MPI_Barrier(MPI_COMM_WORLD);
    if(info.me == 0){
        //Initialization of the KPM variables
        printf("Debug: This is the lead Node that will carry the KPM calculation");
        info.kpm = new KPM();
        info.kpm->Moments = std::atoi(argv[1]);
        info.kpm ->Random = std::atoi(argv[2]);
        info.kpm ->bsize = std::atoi(argv[3]);
        info.kpm ->threads = std::atoi(argv[4]);
        info.kpm->EvolutionsXMeasurement = std::atoi(argv[5]);
        info.kpm->Niter =std::atoi(argv[6]);
        info.kpm ->cellsize = 2.522; //in Angstrom
        info.kpm ->prefix ="./DOSGrapheneWLammpsM"+std::to_string(info.kpm->Moments)+"R"+std::to_string(info.kpm->Random);
        omp_set_num_threads(info.kpm->threads);
        printf("Debug: Now we initialize the containers within the KPM structure\n");
        info.kpm -> ptrHoppings = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
        info.kpm -> ptrOnsites = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
        info.thermoOut.open("./ThermoDynamicVars.dat");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(info.me==0)
        printf("Paso de la inicializacion\n");
    int ifix = lmp->modify->find_fix("3");
    LAMMPS_NS::FixExternal *fix = (LAMMPS_NS::FixExternal *) lmp->modify->fix[ifix];
    fix->set_callback(trialcallback,&info);
    sprintf(str,"run %d",niter);
    if(info.me == 0){
	    printf("leyo todo el input\n");
        printf("ahora esta a punto de llamar el input de lammps\n");
    }
    lmp ->input->one(str);
    delete lmp;
    delete memory;
    delete error;
    delete [] lammps_input;
    MPI_Finalize();

    return 0;
}

void trialcallback(void *ptr, LAMMPS_NS::bigint ntimestep, int nlocal, int *id, double **x, double **f){
    Info *info = (Info *) ptr;

    double unittime =  *((double *) lammps_extract_global(info->lmp,"dt"));
    double boxxlo = *((double *) lammps_extract_global(info->lmp,"boxxlo"));
    double boxxhi = *((double *) lammps_extract_global(info->lmp,"boxxhi"));
    double boxylo = *((double *) lammps_extract_global(info->lmp,"boxylo"));
    double boxyhi = *((double *) lammps_extract_global(info->lmp,"boxyhi"));
    double boxzlo = *((double *) lammps_extract_global(info->lmp,"boxzlo"));
    double boxzhi = *((double *) lammps_extract_global(info->lmp,"boxzhi"));
    double boxxy = *((double *) lammps_extract_global(info->lmp,"xy"));
    double boxxz = *((double *) lammps_extract_global(info->lmp,"xz"));
    double boxyz = *((double *) lammps_extract_global(info->lmp,"yz"));
    char units = *((char *) lammps_extract_global(info->lmp,"units"));
    if(units == 'm')
        unittime = unittime*1e3;
    int natoms;
    MPI_Allreduce(&nlocal,&natoms,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    Many2One *lmp2external = new Many2One(MPI_COMM_WORLD);
    lmp2external -> setup(nlocal,id,natoms);
    double **xexternal = NULL;
    double **fexternal = NULL;
    //This creates the memory of all the states 
    if(info -> me ==0 ){
        xexternal = info ->memory->create_2d_double_array(natoms,3,"lammps:linqt");
        fexternal = info ->memory->create_2d_double_array(natoms,3,"lammps:linqt");
    }
    
    //now we get the position of all the processors
    if(info -> me == 0){
        lmp2external ->gather(&x[0][0],3,&xexternal[0][0]);
        for(auto n=0; n<natoms; n++){
            fexternal[n][0] = 0.0;
            fexternal[n][1] = 0.0;
            fexternal[n][2] = 0.0;
        }
    }
    else lmp2external ->gather(&x[0][0],3,NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    if(info ->me == 0 && ntimestep==0){
        printf("Initalizing the grid for the Hamiltonian\n");
        Position CellBounds {info->kpm->cellsize,info->kpm->cellsize,info->kpm->cellsize};
        Position BoxLow {boxxlo,boxylo,boxzlo};
        Position BoxHi {boxxhi,boxyhi,boxzhi};
        info->kpm -> gridPtr = std::move(std::unique_ptr<SpatialGrid>(new SpatialGrid(CellBounds,BoxLow,BoxHi)));
        //Creation of the Site Vector
        info -> kpm ->SiteList = std::vector<Site>(natoms);
        for(auto n = 0; n<natoms; n++)
            info -> kpm -> SiteList[n] = Site(Position(xexternal[n][0],xexternal[n][1],xexternal[n][2]),n);
        info ->kpm->gridPtr->BuildFromSiteList(info -> kpm -> SiteList);
        info -> kpm ->gridPtr->GetCutoff() = 1.2*info->kpm ->cellsize/std::sqrt(3);
        GenerateHamiltonian(info -> kpm ->rows, info -> kpm -> cols, info -> kpm -> vals, *(info -> kpm ->gridPtr));
        //Initialization of the Hamiltonian Matrix
        info -> kpm -> ptrHoppings = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
        info -> kpm -> ptrHoppings -> SetLabel("GrapheneLammps");
        info -> kpm -> ptrHoppings -> SetDimensions(natoms, natoms);
        info -> kpm -> ptrHoppings -> SetNelem(info -> kpm ->rows.size());

        info -> kpm -> ptrRefHoppings = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
        info -> kpm -> ptrRefHoppings -> SetLabel("GrapheneLammps");
        info -> kpm -> ptrRefHoppings -> SetDimensions(natoms, natoms);
        info -> kpm -> ptrRefHoppings -> SetNelem(info -> kpm ->rows.size());


        const scalar emax = 12.5;
        info -> kpm -> ptrHoppings -> ConvertFromCOO(info -> kpm ->rows, info -> kpm -> cols, info -> kpm -> vals);
        info -> kpm -> ptrRefHoppings -> ConvertFromCOO(info -> kpm ->rows, info -> kpm -> cols, info -> kpm -> vals);
        info -> kpm -> ptrOnsites = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
        info -> kpm -> ptrOnsites -> SetLabel("GrapheneLammps");
        info -> kpm -> ptrOnsites -> SetDimensions(natoms, natoms);
        info -> kpm -> ptrOnsites -> SetNelem(info -> kpm ->Orows.size());
        info -> kpm -> ptrOnsites -> ConvertFromCOO(info -> kpm ->Orows, info -> kpm -> Ocols, info -> kpm -> Ovals);
        
        info -> kpm -> ptrRefOnsites = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
        info -> kpm -> ptrRefOnsites -> SetLabel("GrapheneLammps");
        info -> kpm -> ptrRefOnsites -> SetDimensions(natoms, natoms);
        info -> kpm -> ptrRefOnsites -> SetNelem(info -> kpm ->Orows.size());
        info -> kpm -> ptrRefOnsites -> ConvertFromCOO(info -> kpm ->Orows, info -> kpm -> Ocols, info -> kpm -> Ovals);


        info -> kpm -> ptrMom = std::move(std::unique_ptr<chebyshev::MomentsNEq>(new chebyshev::MomentsNEq(info ->kpm->Niter,info -> kpm ->Moments, info -> kpm ->EvolutionsXMeasurement, info -> kpm ->bsize)));
        info -> kpm -> ptrRefMom = std::move(std::unique_ptr<chebyshev::Moments1D>(new chebyshev::Moments1D(info ->kpm->Moments)));
        info -> kpm -> ptrFactory = std::move(std::unique_ptr<factory::generator>(new factory::generator()));
        info -> kpm -> ptrFilter = std::move(std::unique_ptr<chebyshev::VectorFilter>(new chebyshev::VectorFilter()));
        //Initialization of the NeqMoments
        info -> kpm -> ptrMom -> SystemLabel(info -> kpm -> ptrOnsites -> GetLabel());
        info -> kpm -> ptrMom -> BandWidth(2*emax.real());
        info -> kpm -> ptrMom -> BandCenter(0);
        info -> kpm -> ptrMom -> TimeDiff(unittime);
        info -> kpm -> ptrMom -> SetAndRescaleHamiltonian(*(info -> kpm -> ptrHoppings), *(info -> kpm -> ptrOnsites));
        info -> kpm -> ptrMom -> Print();
        //Definition of the simulation paramters
        //const Float temp = 1E-10*chebyshev::Boltzmann;
        const Float temp = 300.*chebyshev::Boltzmann;
        const Float Mu = 0.;
        //Filling the reference moments
        info -> kpm -> ptrRefMom -> getMomentsParams(*(info -> kpm -> ptrMom));
        info -> kpm -> ptrRefMom -> SetAndRescaleHamiltonian(*(info -> kpm -> ptrRefHoppings), *(info -> kpm -> ptrRefOnsites));
        //Configuring the filter of the random phase state
        info -> kpm -> ptrFactory -> NumberOfStates(info -> kpm -> Random/ info -> kpm -> bsize);
        info -> kpm -> ptrFactory -> BlockSize(info -> kpm -> bsize);
        info -> kpm -> ptrFactory -> SystemSize( info -> kpm -> ptrMom->SystemSize());
        //Start the vectors
        info -> kpm -> ptrFactory -> getQuantumState();
        
        info -> kpm -> ProjectedRV = std::move(std::vector<scalar>(info->kpm->ptrFactory->Out().size(),0));
        info -> kpm -> RVHolder = std::move(std::vector<scalar>(info->kpm->ptrFactory->Out().size(),0));

        //Load the Parameters in the filter
        info -> kpm -> ptrFilter->getMomentsParams(*(info -> kpm -> ptrMom));
        info -> kpm -> ptrFilter->setFilterType(chebyshev::FilterType::fermi);
        info -> kpm -> ptrFilter -> SetInitBlock(info -> kpm -> ptrFactory ->Out());
        info -> kpm -> ptrFilter -> setFilterParams(std::array<Float,2>({Mu,temp}));
        info -> kpm -> ptrFilter -> MomentVector(std::vector<scalar>(info -> kpm -> Moments,0));
        info -> kpm -> ptrFilter -> setMomentNumber(info -> kpm -> Moments);
        info -> kpm -> ptrFilter -> FillCoef();
        info -> kpm -> ptrFilter -> ApplyJacksonKernel(1E-10);
        info -> kpm -> ptrFilter -> PrepareOutVec();
        //Initialization of the vectors within the moments
        info -> kpm ->ptrRefMom -> SetInitBlock(info -> kpm -> ptrFactory ->Out());
        info -> kpm ->ptrMom -> SetInitBlock(info -> kpm -> ptrFilter ->OPV());
        scalar dot;
        //Here the forces will be added in the future

        //Here the forces will be added in the future
        const Integer measurement = ntimestep/info -> kpm -> EvolutionsXMeasurement;
        for(size_t m = 0; m<info -> kpm -> Moments; m++){
            double scal = 2.0/(info -> kpm ->ptrFactory->NumberOfStates()*info -> kpm -> ptrFactory -> BlockSize());
            if(m==0)
                scal *= 0.5;
            //Coefficients for the TD moments
            linalg::dot(info -> kpm -> ptrFactory -> Out().size(), info -> kpm -> ptrFactory ->Out(), info -> kpm -> ptrMom -> ChebV0(), & dot);
            (*(info -> kpm -> ptrMom)) (measurement,m) += scal*dot;
            linalg::dot( info -> kpm -> ptrFactory -> Out().size(), info -> kpm -> ptrFactory -> Out(), info -> kpm -> ptrRefMom -> ChebV0(), &dot);
            (*(info -> kpm -> ptrRefMom))(m) = scal*dot;
            info ->kpm -> ptrRefMom -> Iterate();
            info ->kpm -> ptrMom -> Iterate();
        }
        
        printf("DEBUG: Calling time evolution of teh states at step %lu  \n",ntimestep);
        info -> kpm -> ptrMom -> Evolve( info -> kpm -> ptrFactory -> Out());
        info -> kpm -> ptrMom -> Evolve(info -> kpm -> ptrFilter -> OPV());

        printf("DEBUG: Calculation of the Density of states\n");
        info -> kpm -> ptrMom -> ApplyJacksonKernel(1E-10);
        info -> kpm -> ptrRefMom -> ApplyJacksonKernel(1E-10);
        std::string filenameRef = info -> kpm -> prefix+"Reference"+std::to_string(ntimestep)+".dat";
        Float dx = 2.0*chebyshev::CUTOFF/(Float)65535;
        std::ofstream oref(filenameRef);
        std::ofstream odos(info -> kpm ->prefix+"Step"+std::to_string(ntimestep)+".dat");
        scalar approx;
        for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
            approx =0;
            for(Float m =0; m<info -> kpm -> ptrMom->HighestMomentNumber(); m++){
                approx += delta_chebF(x,m)*((*(info ->kpm -> ptrMom))(measurement,(size_t)m));
            }
            odos<<x<<"\t"<<approx.real()<<std::endl;
        }

        for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
            approx =0;
            for(Float m =0; m<info -> kpm -> ptrRefMom->HighestMomentNumber(); m++){
                approx += delta_chebF(x,m)*((*(info ->kpm -> ptrRefMom))((size_t)m));
            }
            oref<<x<<"\t"<<approx.real()<<std::endl;
        }
        printf("DEBUG: Finishing the calculation of the density of states\n");

        





        
    }
    
    else if(info -> me == 0 && ntimestep !=0 &&(ntimestep%info->kpm->EvolutionsXMeasurement) == 0){
        printf("DEBUG: The second routine for various measurements\n");
        std::cout<<"DEBUG: The current timestep is "<<ntimestep<<" and the Mod is "<<(ntimestep%info->kpm->EvolutionsXMeasurement)<<std::endl;
        //Pare aca toca revisar que esta pasando con la matriz aqui. Algo pasa con la memoria
        info -> kpm -> ptrHoppings -> ReleaseMatrix();
        info -> kpm -> ptrOnsites -> ReleaseMatrix();
        for(auto n = 0; n<natoms; n++)
            info -> kpm -> SiteList[n].getPosition() = Position(xexternal[n][0],xexternal[n][1],xexternal[n][2]);
        info ->kpm->gridPtr->UpdateFromSiteList(info -> kpm -> SiteList);
        while(!(info -> kpm -> rows.empty() && info -> kpm -> cols.empty() && info -> kpm -> vals.empty())){
        //info -> kpm ->rows = std::move(std::vector<Integer>());
        //info -> kpm ->cols = std::move(std::vector<Integer>());
        //info -> kpm ->vals = std::move(std::vector<scalar>());
            info -> kpm -> rows.pop_back();
            info -> kpm -> cols.pop_back();
            info -> kpm -> vals.pop_back();
        }
        GenerateHamiltonian(info -> kpm ->rows, info -> kpm -> cols, info -> kpm -> vals, *(info -> kpm ->gridPtr));
        info -> kpm -> ptrHoppings -> ConvertFromCOO(info -> kpm ->rows, info -> kpm -> cols, info -> kpm -> vals);
        info -> kpm -> ptrOnsites -> ConvertFromCOO(info -> kpm ->Orows, info -> kpm -> Ocols, info -> kpm -> Ovals); //Pare aca entonces
        info -> kpm -> ptrMom -> SetAndRescaleHamiltonian(*(info -> kpm -> ptrHoppings),*(info -> kpm -> ptrOnsites));
        info -> kpm -> ptrRefMom ->SetHamiltonianHoppings(*(info -> kpm -> ptrHoppings));
        info -> kpm -> ptrRefMom ->SetHamiltonianOnSite(*(info -> kpm -> ptrOnsites));
        //info -> kpm -> ptrMom -> SetHamiltonianHoppings(*(info -> kpm -> ptrRefHoppings));
        //info -> kpm -> ptrMom -> SetHamiltonianOnSite(*(info -> kpm -> ptrRefOnsites));
        //Aca tiene que ir el calculo de las fuerzas

        //Aca termina el calculo de las fuerzas
        info -> kpm ->ptrRefMom -> SetInitBlock(info -> kpm -> ptrFactory ->Out());
        info -> kpm -> ptrMom -> SetInitBlock(info -> kpm -> ptrFilter -> OPV());
        scalar dot;
        const Integer measurement = ntimestep/info -> kpm -> EvolutionsXMeasurement;
        for(size_t m = 0; m<info -> kpm -> Moments; m++){
            double scal = 2.0/(info -> kpm ->ptrFactory->NumberOfStates()*info -> kpm -> ptrFactory -> BlockSize());
            if(m==0)
                scal *= 0.5;
            //Coefficients for the TD moments
            linalg::dot(info -> kpm -> ptrFactory -> Out().size(), info -> kpm -> ptrFactory ->Out(), info -> kpm -> ptrMom -> ChebV0(), & dot);
            (*(info -> kpm -> ptrMom)) (measurement,m) += scal*dot;
            
            linalg::dot( info -> kpm -> ptrFactory -> Out().size(), info -> kpm -> ptrFactory -> Out(), info -> kpm -> ptrRefMom -> ChebV0(), &dot);
            (*(info -> kpm -> ptrRefMom))(m) = scal*dot;
            info ->kpm -> ptrRefMom -> Iterate();
            info ->kpm -> ptrMom -> Iterate();

        }
        //info -> kpm -> ptrMom -> SetAndRescaleHamiltonian(*(info -> kpm -> ptrHoppings),*(info -> kpm -> ptrOnsites));
        printf("DEBUG: Calling time evolution of teh states at step %lu  \n",ntimestep);
        info -> kpm -> ptrMom -> Evolve( info -> kpm -> ptrFactory -> Out());
        info -> kpm -> ptrMom -> Evolve(info -> kpm -> ptrFilter -> OPV());

        printf("DEBUG: Calculation of the Density of states\n");
        info -> kpm -> ptrMom -> ApplyJacksonKernel(1E-10);
        info -> kpm -> ptrRefMom -> ApplyJacksonKernel(1E-10);
        Float dx = 2.0*chebyshev::CUTOFF/(Float)65535;
        std::string filenameRef = info -> kpm -> prefix+"Reference"+std::to_string(ntimestep)+".dat";
        std::ofstream oref(filenameRef);
        std::ofstream odos(info -> kpm ->prefix+"Step"+std::to_string(ntimestep)+".dat");
        scalar approx;
        for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
            approx =0;
            for(Float m =0; m<info -> kpm -> ptrMom->HighestMomentNumber(); m++){
                approx += delta_chebF(x,m)*((*(info ->kpm -> ptrMom))(measurement,(size_t)m));
            }
            odos<<x<<"\t"<<approx.real()<<std::endl;
        }


        for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
            approx =0;
            for(Float m =0; m<info -> kpm -> ptrRefMom->HighestMomentNumber(); m++){
                approx += delta_chebF(x,m)*((*(info ->kpm -> ptrRefMom))((size_t)m));
            }
            oref<<x<<"\t"<<approx.real()<<std::endl;
        }


        printf("DEBUG: Finishing the calculation of the density of states\n");
           
    }

    else if(info -> me == 0 && ntimestep !=0 &&(ntimestep%info->kpm->EvolutionsXMeasurement) != 0){
        printf("DEBUG: The routine for various measurements outside the interval\n");
        info -> kpm -> ptrHoppings -> ReleaseMatrix();
        info -> kpm -> ptrOnsites -> ReleaseMatrix();
        for(auto n = 0; n<natoms; n++)
            info -> kpm -> SiteList[n].getPosition() = Position(xexternal[n][0],xexternal[n][1],xexternal[n][2]);
        info ->kpm->gridPtr->UpdateFromSiteList(info -> kpm -> SiteList);
        //info -> kpm ->rows = std::move(std::vector<Integer>());
        //info -> kpm ->cols = std::move(std::vector<Integer>());
        //info -> kpm ->vals = std::move(std::vector<scalar>());
        while(!(info -> kpm -> rows.empty() && info -> kpm -> cols.empty() && info -> kpm -> vals.empty())){
        //info -> kpm ->rows = std::move(std::vector<Integer>());
        //info -> kpm ->cols = std::move(std::vector<Integer>());
        //info -> kpm ->vals = std::move(std::vector<scalar>());
            info -> kpm -> rows.pop_back();
            info -> kpm -> cols.pop_back();
            info -> kpm -> vals.pop_back();
        }
        GenerateHamiltonian(info -> kpm ->rows, info -> kpm -> cols, info -> kpm -> vals, *(info -> kpm ->gridPtr));
        info -> kpm -> ptrHoppings -> ConvertFromCOO(info -> kpm ->rows, info -> kpm -> cols, info -> kpm -> vals);
        info -> kpm -> ptrOnsites -> ConvertFromCOO(info -> kpm ->Orows, info -> kpm -> Ocols, info -> kpm -> Ovals);
        info -> kpm -> ptrMom -> SetAndRescaleHamiltonian(*(info -> kpm -> ptrHoppings),*(info -> kpm -> ptrOnsites));
        //Aca tiene que ir el calculo de las fuerzas

        //Aca termina el calculo de las fuerzas
        info -> kpm -> ptrMom -> SetInitBlock(info -> kpm -> ptrFilter -> OPV());
        
        printf("DEBUG: Calling time evolution of the states at step %lu  \n",ntimestep);
        info -> kpm -> ptrMom -> Evolve(info -> kpm -> ptrFilter -> OPV());
        info -> kpm -> ptrMom -> Evolve( info -> kpm -> ptrFactory -> Out());
    }

    else{ }



    MPI_Barrier(MPI_COMM_WORLD);
    One2Many *external2lmp = new One2Many(MPI_COMM_WORLD);
    external2lmp-> setup(natoms,nlocal,id);
    double *fvector = NULL;
    if(f)
        fvector = &f[0][0];
    if(info -> me ==0)
        external2lmp -> scatter(&fexternal[0][0],3,fvector);
    else
        external2lmp -> scatter(NULL,3,fvector);
    if(info -> me == 0){
        info ->memory->destroy_2d_double_array(xexternal);
        info ->memory ->destroy_2d_double_array(fexternal);
        
    }
    delete external2lmp;
    delete lmp2external;
    MPI_Barrier(MPI_COMM_WORLD);

    return;
    
    
}
