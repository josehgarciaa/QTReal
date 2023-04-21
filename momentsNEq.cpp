#include "moments.h"

size_t chebyshev::MomentsNEq::Evolve(std::vector<scalar>& phi){
    const auto dim = this -> SystemSize();
    const auto size =  phi.size();
    const auto bsize = size/dim;
    const scalar zero = (scalar)0;
    const scalar I(0,1);
    if(bsize > 1){
        if(this -> ChebV0().size()/bsize != dim){
            this -> ChebV0() = std::vector<scalar>(size,0);
            this -> BlockSize(bsize);
        }

        if(this -> ChebV1().size()/bsize != dim){
            this -> ChebV1() = std::vector<scalar> (size,0);
        }

        const auto x = this ->TimeDiff()*this->ChebyshevFreq();
        const Float tol = 1E-15;
        Float momentcutoff = 2.0*tol;
        //aca si se neceseita podriamos usar un pointer normal, pero esto sera para luego
        #pragma omp parallel for schedule(static)
        for(size_t v = 0; v<size; v++)
            this ->ChebV0()[v] = phi[v];
        
        this -> HamiltonianHoppings().BlockMultiply(this -> ChebV0(), this -> ChebV1(), bsize);
        if( this -> HamiltonianOnSite().NumNNZ() !=0)
            this -> HamiltonianOnSite().BlockMultiply(1.0,this -> ChebV0(), bsize, 1.0, this -> ChebV1());
        size_t n = 0;
        scalar nIp = scalar(1,0);
        Float Jn = (Float)0.5*Bessel(n,x);
        linalg::scal(size,&zero,phi);
        scalar pref;
        while(momentcutoff>tol){
            pref = nIp*(Float)2*Jn;
            linalg::axpy(size, &pref, this -> ChebV0(), phi);
            this -> HamiltonianHoppings().BlockMultiply(2.0, this ->ChebV1(), bsize, -1.0, this -> ChebV0());
            if( this -> HamiltonianOnSite().NumNNZ() !=0)
                this ->HamiltonianOnSite().BlockMultiply(2.0, this -> ChebV1(), bsize, 1.0, this -> ChebV0());
            this -> ChebV0().swap(this -> ChebV1());
            nIp*=-I;
            n++;
            Jn = Bessel(n,x);
            momentcutoff = std::fabs((Float)2*Jn);
        }

        const auto exp0 = std::exp(-I*this->ChebyshevFreq0()*this->TimeDiff());
        linalg::scal(size, &exp0, phi);
        return 0;

        
    }

    if(bsize == 1){
        if(this -> ChebV0().size()/bsize != dim){
            this -> ChebV0() = std::vector<scalar>(size,0);
            this -> BlockSize(bsize);
        }

        if(this -> ChebV1().size()/bsize != dim){
            this -> ChebV1() = std::vector<scalar>(size,0);
        }
        
        const auto x = this ->TimeDiff()*this->ChebyshevFreq();
        const Float tol = 1E-15;
        Float momentcutoff = 2.0*tol;
        //aca si se neceseita podriamos usar un pointer normal, pero esto sera para luego
        #pragma omp parallel for schedule(static)
        for(size_t v = 0; v<size; v++)
            this ->ChebV0()[v] = phi[v];
        
        this -> HamiltonianHoppings().Multiply(this -> ChebV0(), this -> ChebV1());
        if(this -> HamiltonianOnSite().NumNNZ()!=0)
            this -> HamiltonianOnSite().Multiply(this -> ChebV0(), this -> ChebV1());
        size_t n = 0;
        scalar nIp = scalar(1,0);
        Float Jn = (Float)0.5*Bessel(n,x);
        linalg::scal(size,&zero,phi);
        scalar pref;
        while(momentcutoff>tol){
            pref = nIp*(Float)2*Jn;
            linalg::axpy(size, &pref, this -> ChebV0(), phi);
            this -> HamiltonianHoppings().Multiply(2.0, this ->ChebV1(), -1.0, this -> ChebV0());
            if(this -> HamiltonianOnSite().NumNNZ() != 0)
                this -> HamiltonianOnSite().Multiply(2.0,this -> ChebV1(), 1.0,this -> ChebV0());
            this -> ChebV0().swap(this -> ChebV1());
            nIp*=-I;
            n++;
            Jn = Bessel(n,x);
            momentcutoff = std::fabs((Float)2*Jn);
        }

        const auto exp0 = std::exp(-I*this->ChebyshevFreq0()*this->TimeDiff());
        linalg::scal(size, &exp0, phi);
        return 0;

        
    }

    else{
        std::cerr<<"Error: The vector to iterate is empty..."<<std::endl;
        assert(false);
        return 0;
    }
};

void chebyshev::MomentsNEq::Print(){
  std::cout<<"\n\nCHEBYSHEV NEQ MOMENTS INFO"<<std::endl;
  std::cout<<"\tSYSTEM:\t\t\t"<<this->SystemLabel()<<std::endl;
  if( this-> SystemSize() > 0 )
    std::cout<<"\tSIZE:\t\t\t"<<this-> SystemSize()<<std::endl;

  std::cout<<"\tMOMENTS SIZE:\t\t"<<"("
	   <<this->HighestMomentNumber()<< " x " <<(this->MaxTimeStep() / this -> EvolutionsxMeasurements())<<"+1"<<")"<<std::endl;
  printf("DEBUG: THE NUMBER OF MEASUREMENTS IS %ld",this ->NumMeasurements());
  std::cout<<"\tSCALE FACTOR:\t\t"<<this->ScaleFactor()<<std::endl;
  std::cout<<"\tSHIFT FACTOR:\t\t"<<this->ShiftFactor()<<std::endl;
  std::cout<<"\tENERGY SPECTRUM:\t("
	   <<-this->HalfWidth()+this->BandCenter()<<" , "
	   << this->HalfWidth()+this->BandCenter()<<")"<<std::endl<<std::endl;
  std::cout<<"\tTIME STEP:\t\t"<<this->MaxTimeStep()<<std::endl;
  std::cout<<"\tTIME DIFF:\t"<<this->TimeDiff()<<std::endl;
  std::cout<<"\tNUM MEASUREMENTS:\t"<<this ->NumMeasurements()<<std::endl;
  std::cout<<"\tBLOCK SIZE:\t"<<this->BlockSize()<<std::endl;
  std::cout<<"\tVECTOR SIZE:\t"<<this ->ChebV0().size()<<std::endl;
};

void chebyshev::MomentsNEq::saveIn(std::string filename){
    typedef std::numeric_limits<Float> dbl;
    std::ofstream outputfile(filename);
    outputfile.precision(dbl::max_digits10);
    outputfile << this -> SystemSize()<<" "<< this -> BandWidth()<<" "<< this -> BandCenter()<<" "
               << this -> MaxTimeStep()<<" "<<this -> TimeDiff()<< " "<< this -> NumMeasurements()<< std::endl;
    
    outputfile << this -> HighestMomentNumber()<< " "<< this -> MaxTimeStep()<<" "<<std::endl;
    outputfile <<this->BlockSize();
    for(size_t v = 0; v<this->HighestMomentNumber(); v++){
        const scalar var = this -> MomentVector(v);
        outputfile <<var.real()<<" "<<var.imag()<<std::endl;
    }
    outputfile.close();

}

chebyshev::MomentsNEq::MomentsNEq(std::string momfilename){
    size_t ext_pos = momfilename.find(".chebmomNEq");
    if(ext_pos == std::string::npos){
        std::cerr<<"The first argument does not seem to be a valid .chebmomNEq file"<<std::endl;
        assert(false);
    }

    this -> SystemLabel(momfilename.substr(0,ext_pos));

    std::ifstream momfile(momfilename);
    assert(momfile.is_open());
    size_t ibuff; Float dbuff;
    momfile>>ibuff; this -> SystemSize(ibuff);
    momfile>>dbuff; this -> BandWidth(dbuff);
    momfile>>dbuff; this -> BandCenter(dbuff);
    momfile>>ibuff; this -> MaxTimeStep(ibuff);
    momfile>>dbuff; this -> TimeDiff(dbuff);
    momfile>>ibuff; this -> NumMeasurements(ibuff); // this is a new omdification for the case in which the number of measurements is required
    momfile>>this ->_numMoms >> this -> _maxTimeStep;
    momfile >>ibuff; BlockSize(ibuff);
    this -> MomentVector(std::vector<scalar>(this->HighestMomentNumber()* this -> NumMeasurements(),0));
    Float rmu, imu;
    for(size_t measurement = 0; measurement< this-> NumMeasurements(); measurement++){
        for(size_t moment = 0; moment< this -> HighestMomentNumber(); moment++){
            momfile>>rmu>>imu;
            this -> operator()(measurement, moment) = scalar(rmu, imu);
        }
    }
    momfile.close();
}

void chebyshev::MomentsNEq::MomentNumber(const size_t numMoms){
    assert(numMoms<=this -> HighestMomentNumber());
    const size_t measurements = this -> NumMeasurements();
    const size_t maxtimestep = this ->MaxTimeStep();
    const size_t evolutionsxmeasurements = this ->EvolutionsxMeasurements();
    chebyshev::MomentsNEq new_mom(maxtimestep,numMoms,evolutionsxmeasurements);
    assert(new_mom.NumMeasurements() == this -> NumMeasurements() && new_mom.EvolutionsxMeasurements() == this ->EvolutionsxMeasurements());
    printf("DEBUG: PASO DEL CHECK\n");
    printf("###############################\n");
    for(size_t measure = 0; measure<=measurements; measure++){
        for(size_t moments = 0; moments<numMoms; moments++){
            new_mom(measure, moments) = this -> operator()(measure, moments);
        }
    }
    this -> _numMoms = new_mom._numMoms;
    this -> MomentVector(new_mom.MomentVector());
    
};

void chebyshev::MomentsNEq::ApplyJacksonKernel(const Float broad){
    assert(broad > 0);
    const Float eta = (Float)2.0*broad/(Float)1000/this -> BandWidth();
    size_t maxMom = std::ceil((Float)M_PI/eta);
    if(maxMom> this -> HighestMomentNumber())
        maxMom = this -> HighestMomentNumber();
    std::cout<<"Kernel reduced the number of moments to "<<maxMom<<"  for a broadening of "<<(Float)M_PI/(Float)maxMom<<std::endl;
    this -> MomentNumber(maxMom);
    const Float phi_J = (Float)M_PI/(Float)(maxMom+1);
    Float g_D_m;
    for(size_t measurement = 0; measurement<= this -> NumMeasurements(); measurement++){
        for(size_t moment = 0; moment<maxMom; moment++){
          g_D_m = ( (Float)(maxMom - moment + 1) * cos(phi_J * (Float)moment) + sin(phi_J * (Float)moment) * cos(phi_J) / sin(phi_J) ) * phi_J/(Float)M_PI;
          this -> operator()(measurement,moment) *= g_D_m;
        }
   }

}


