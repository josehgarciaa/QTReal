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
    std::unique_ptr<SparseMat> ptrOnsites;
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

    
    double boxxlo = *((double *) lammps_extract_global(info->lmp,"boxxlo"));
    double boxxhi = *((double *) lammps_extract_global(info->lmp,"boxxhi"));
    double boxylo = *((double *) lammps_extract_global(info->lmp,"boxylo"));
    double boxyhi = *((double *) lammps_extract_global(info->lmp,"boxyhi"));
    double boxzlo = *((double *) lammps_extract_global(info->lmp,"boxzlo"));
    double boxzhi = *((double *) lammps_extract_global(info->lmp,"boxzhi"));
    double boxxy = *((double *) lammps_extract_global(info->lmp,"xy"));
    double boxxz = *((double *) lammps_extract_global(info->lmp,"xz"));
    double boxyz = *((double *) lammps_extract_global(info->lmp,"yz"));
    double temperature = lammps_get_thermo(info -> lmp, "temp");
    double kinetic = lammps_get_thermo(info ->lmp,"ke");
    double potential = lammps_get_thermo(info ->lmp,"pe");
    double etotal = lammps_get_thermo(info -> lmp, "etotal");
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
        (info -> thermoOut)<<ntimestep<<"\t"<<temperature<<"\t"<<potential<<"\t"<<kinetic<<"\t"<<etotal<<std::endl;
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
        const scalar emax = 12.5;
        info -> kpm -> ptrHoppings -> ConvertFromCOO(info -> kpm ->rows, info -> kpm -> cols, info -> kpm -> vals);
        info -> kpm -> ptrOnsites = std::move(std::unique_ptr<SparseMat>(new SparseMat()));
        info -> kpm -> ptrOnsites -> SetLabel("GrapheneLammps");
        info -> kpm -> ptrOnsites -> SetDimensions(natoms, natoms);
        info -> kpm -> ptrOnsites -> SetNelem(info -> kpm ->Orows.size());
        info -> kpm -> ptrOnsites -> ConvertFromCOO(info -> kpm ->Orows, info -> kpm -> Ocols, info -> kpm -> Ovals);
        info -> kpm -> ptrRefMom = std::move(std::unique_ptr<chebyshev::Moments1D>(new chebyshev::Moments1D(info ->kpm->Moments)));
        info -> kpm -> ptrFactory = std::move(std::unique_ptr<factory::generator>(new factory::generator()));
        //Initialization ofthe moment
        info -> kpm -> ptrRefMom -> SystemLabel(info -> kpm -> ptrOnsites->GetLabel());
        info -> kpm -> ptrRefMom -> BandWidth(2*emax.real());
        info -> kpm -> ptrRefMom -> BandCenter(0.0);
        info -> kpm -> ptrRefMom -> BlockSize(info -> kpm ->bsize);
        info -> kpm -> ptrRefMom -> SetAndRescaleHamiltonian(*(info -> kpm -> ptrHoppings), *(info -> kpm -> ptrOnsites));
        info -> kpm  -> ptrRefMom -> Print();
        SparseMat ID;
        ID.MakeIdentity();
        chebyshev::SpectralMoments(ID,*(info -> kpm -> ptrRefMom),*(info -> kpm -> ptrFactory),info ->kpm->Random/info->kpm->bsize);
        info -> kpm -> ptrRefMom->ApplyJacksonKernel(1e-8);
        Float dx = 2.0*(chebyshev::CUTOFF)/(Float)65535;
        std::ofstream output(info -> kpm ->prefix+"DOS.dat");
        for(Float x = -chebyshev::CUTOFF; x<=chebyshev::CUTOFF; x+=dx){
            scalar approx=0;
            for(Float m =0 ; m<info -> kpm -> ptrRefMom->HighestMomentNumber(); m++){
                approx += delta_chebF(x,m)*info -> kpm -> ptrRefMom->MomentVector(m);
            }
            output<<x<<"\t"<<approx.real()<<std::endl;
        }    
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(info -> me ==0)
        (info -> thermoOut)<<ntimestep<<"\t"<<temperature<<"\t"<<potential<<"\t"<<kinetic<<"\t"<<etotal<<std::endl;
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
