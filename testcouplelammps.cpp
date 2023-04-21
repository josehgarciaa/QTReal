#include "typedef.h"
#include <chrono>
#include <mpi.h>
#include "many2one.h"
#include "one2many.h"
#include "files.h"
#include "memory.h"
#include "error.h"
#include "lmpintegration.h"

struct Info{
    int me;
    AuxMemory *memory;
    LAMMPS_NS::LAMMPS *lmp;
    char *input;

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

    AuxMemory *memory = new AuxMemory(comm);
    AuxError * error = new AuxError(comm);

    if(argc!=4) error->all(".out niter parameter2 in.lammps\n");

    Integer niter = std::atoi(argv[1]);
    Integer par2 = std::atoi(argv[2]);
    n = strlen(argv[3])+1;
    char *lammps_input = new char[n];
    strcpy(lammps_input,argv[3]);

    //Instatiate lammps

    LAMMPS_NS::LAMMPS *lmp = new LAMMPS_NS::LAMMPS(0,NULL,MPI_COMM_WORLD);
    lmp ->input->file(lammps_input);

    Info info;
    info.me = me;
    info.memory = memory;
    info.lmp = lmp;
    if(info.me == 0)
	    printf("creo todo funciona\n");

    //int ifix = lmp->modify->find_fix("4");
    //LAMMPS_NS::FixExternal *fix = (LAMMPS_NS::FixExternal *) lmp->modify->fix[ifix];
    //fix ->set_callback(trialcallback,&info);
    sprintf(str,"run %d",niter);
    if(info.me == 0)
	    printf("leyo todo el input\n");
    printf("ahora esta a punto de llamar el input de lammps\n");
    lmp ->input->one(str);
    delete lmp;
    delete memory;
    delete error;
    delete [] lammps_input;
    MPI_Finalize();

    return 0;
}

void trialcallback(void *ptr, LAMMPS_NS::bigint ntimestep, int nlocal, int *id, double **x, double **f){
    int i,j;
    char str[128];
    Info *info = (Info *) ptr;
    char **boxlines = NULL;
    if(info->me == 0){
        boxlines = new char * [3];
        for(i=0; i<3; i++)
        boxlines[i] = new char[128];
    }
    
    
    double boxxlo = *((double *) lammps_extract_global(info->lmp,"boxxlo"));
    double boxxhi = *((double *) lammps_extract_global(info->lmp,"boxxhi"));
    double boxylo = *((double *) lammps_extract_global(info->lmp,"boxylo"));
    double boxyhi = *((double *) lammps_extract_global(info->lmp,"boxyhi"));
    double boxzlo = *((double *) lammps_extract_global(info->lmp,"boxzlo"));
    double boxzhi = *((double *) lammps_extract_global(info->lmp,"boxzhi"));
    double boxxy = *((double *) lammps_extract_global(info->lmp,"xy"));
    double boxxz = *((double *) lammps_extract_global(info->lmp,"xz"));
    double boxyz = *((double *) lammps_extract_global(info->lmp,"yz"));
    int natoms;
    MPI_Allreduce(&nlocal,&natoms,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    Many2One *lmp2external = new Many2One(MPI_COMM_WORLD);
    lmp2external -> setup(nlocal,id,natoms);
    char **xlines = NULL;
    double **xexternal = NULL;
    //This creates the memory of all the states 
    if(info -> me ==0 ){
        xexternal = info ->memory->create_2d_double_array(natoms,3,"lammps:linqt");
        xlines = new char * [natoms];
        for(i =0; i<natoms; i++)
            xlines[i] = new char[128];
    }
    
    //now we get the position of all the processors
    if(info -> me == 0)
        lmp2external ->gather(&x[0][0],3,&xexternal[0][0]);
    else lmp2external ->gather(&x[0][0],3,NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    if(info ->me == 0){
        printf("Aca en el thread %d tenemos que hay %d atomos y que\n", info->me,nlocal);
        std::cout<<"las dimensiones de la caja son boxxlo: "<<boxxlo<<" boxxhi: "<<boxxhi<<std::endl;
        std::cout<<"las dimensiones de la caja son boxylo: "<<boxylo<<" boxyhi: "<<boxyhi<<std::endl;
        std::cout<<"las dimensiones de la caja son boxzlo: "<<boxzlo<<" boxzhi: "<<boxzhi<<std::endl;
        printf("And the diagonals are\n");
        std::cout<<"boxxy: "<<boxxy<<" boxxz: "<<boxxz<<" boxyz: "<<boxyz<<std::endl;
        std::cout<<"Locally there are "<<nlocal<<" atoms, and in the whole simulation there are  "<<natoms<<std::endl;
	std::cout<<"the size of the box is "<<boxxhi-boxxlo<<std::endl;
        printf("The position of the atoms are \n");
        for(i=0; i<natoms; i++)
            std::cout<<"index:"<<i<<" x: "<<xexternal[i][0]<<" y: "<<xexternal[i][1]<<" z: "<<xexternal[i][2]<<std::endl;
        for(i=0; i<natoms-1;i++)
        std::cout<<"index:"<<i<<","<<i+1<<" x: "<<xexternal[i+1][0]-xexternal[i][0]<<" y: "<<xexternal[i+1][1]-xexternal[i][1]<<" z: "<<xexternal[i+1][2]-xexternal[i][2]<<std::endl;   
        MPI_Barrier(MPI_COMM_WORLD);    
    }
    
    if(info ->me == 1){
        printf("Aca en el thread %d tenemos que hay %d atomos y que\n", info->me,nlocal);
        std::cout<<"las dimensiones de la caja son boxxlo: "<<boxxlo<<" boxxhi: "<<boxxhi<<std::endl;
        std::cout<<"las dimensiones de la caja son boxylo: "<<boxylo<<" boxyhi: "<<boxyhi<<std::endl;
        std::cout<<"las dimensiones de la caja son boxzlo: "<<boxzlo<<" boxzhi: "<<boxzhi<<std::endl;
        printf("And the diagonals are\n");
        std::cout<<"boxxy: "<<boxxy<<" boxxz: "<<boxxz<<" boxyz: "<<boxyz<<std::endl;
        std::cout<<"Locally there are "<<nlocal<<" atoms, and in the whole simulation there are  "<<natoms<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if(info ->me == 2){
        printf("Aca en el thread %d tenemos que hay %d atomos y que\n", info->me,nlocal);
        std::cout<<"las dimensiones de la caja son boxxlo: "<<boxxlo<<" boxxhi: "<<boxxhi<<std::endl;
        std::cout<<"las dimensiones de la caja son boxylo: "<<boxylo<<" boxyhi: "<<boxyhi<<std::endl;
        std::cout<<"las dimensiones de la caja son boxzlo: "<<boxzlo<<" boxzhi: "<<boxzhi<<std::endl;
        printf("And the diagonals are\n");
        std::cout<<"boxxy: "<<boxxy<<" boxxz: "<<boxxz<<" boxyz: "<<boxyz<<std::endl;
        std::cout<<"Locally there are "<<nlocal<<" atoms, and in the whole simulation there are  "<<natoms<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
    }


    if(info ->me == 3){
        printf("Aca en el thread %d tenemos que hay %d atomos y que\n", info->me,nlocal);
        std::cout<<"las dimensiones de la caja son boxxlo: "<<boxxlo<<" boxxhi: "<<boxxhi<<std::endl;
        std::cout<<"las dimensiones de la caja son boxylo: "<<boxylo<<" boxyhi: "<<boxyhi<<std::endl;
        std::cout<<"las dimensiones de la caja son boxzlo: "<<boxzlo<<" boxzhi: "<<boxzhi<<std::endl;
        printf("And the diagonals are\n");
        std::cout<<"boxxy: "<<boxxy<<" boxxz: "<<boxxz<<" boxyz: "<<boxyz<<std::endl;
        std::cout<<"Locally there are "<<nlocal<<" atoms, and in the whole simulation there are  "<<natoms<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return;
    
    
}
