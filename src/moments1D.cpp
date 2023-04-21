#include "moments.h"


void chebyshev::Moments1D::Print(){
    std::cout<<"\n\nCHEBYSHEV 1D MOMENTS INFO"<<std::endl;
	std::cout<<"\tSYSTEM:\t\t\t"<<this->SystemLabel()<<std::endl;
	if( this-> SystemSize() > 0 )
		std::cout<<"\tSIZE:\t\t\t"<<this-> SystemSize()<<std::endl;

	std::cout<<"\tMOMENTS SIZE:\t\t"<<this->HighestMomentNumber()<<std::endl;
	std::cout<<"\tSCALE FACTOR:\t\t"<<this->ScaleFactor()<<std::endl;
	std::cout<<"\tSHIFT FACTOR:\t\t"<<this->ShiftFactor()<<std::endl;
	std::cout<<"\tENERGY SPECTRUM:\t("
			 <<-this->HalfWidth()+this->BandCenter()<<" , "
			 << this->HalfWidth()+this->BandCenter()<<")"<<std::endl<<std::endl;
    std::cout<<"\tBLOCKSIZE:\t"<<this->BlockSize()<<std::endl;
}

void chebyshev::Moments1D::saveIn(std::string filename){
    typedef std::numeric_limits<Float> dbl;
    std::ofstream outputfile(filename);
    outputfile.precision(dbl::max_digits10);
    outputfile << this -> SystemSize()<<" "<< this ->BandWidth()<<" "<<this ->BandCenter()<<std::endl;
    outputfile << this -> HighestMomentNumber()<<std::endl;
    outputfile << this -> BlockSize()<<std::endl;
    for(size_t v = 0; v<HighestMomentNumber(); v++){
        const scalar var = this ->MomentVector(v);
        outputfile << var.real()<<" "<<var.imag()<<std::endl;
    }
    outputfile.close();       
}




chebyshev::Moments1D::Moments1D( const std::string momfilename){
    size_t ext_pos =  momfilename.find(".chebmom1D");
    if(ext_pos == std::string::npos){
        std::cerr<<"The first argument does not seem to be a valid .chebmom1D file"<<std::endl;
        assert(false);
    }

    this ->SystemLabel(momfilename.substr(0,ext_pos));
    std::ifstream momfile(momfilename);
    assert(momfile.is_open());

    size_t ibuff; Float dbuff;
    momfile>>ibuff; this ->SystemSize(ibuff);
    momfile>>dbuff; this ->BandWidth(dbuff);
    momfile>>dbuff; this ->BandCenter(dbuff);
    momfile>>this -> _numMoms;
    momfile>>ibuff; BlockSize(ibuff);
    this -> MomentVector(std::vector<scalar>(_numMoms,0));
    Float rmu, imu;
    for(size_t m0 = 0; m0<_numMoms; m0++){
        momfile>>rmu>>imu;
        this ->operator()(m0) = scalar(rmu,imu);
    }
    momfile.close();
};

void chebyshev::Moments1D::ApplyJacksonKernel(const Float broad){
    assert(broad>0);
    const Float eta = (Float)2*broad/(Float)1000/this->BandWidth();
    size_t maxMom = std::ceil((Float)M_PI/eta);
    if(maxMom> this -> HighestMomentNumber())
        maxMom = this -> HighestMomentNumber();
    std::cout<<"Kernel reduced the number of moments to "<<maxMom<<" for a broadening of "<<(Float)M_PI/maxMom<<std::endl;
    const Float phi_J = (Float)M_PI/(Float)(maxMom+(Float)1.0);
    Float g_D_m;
    for( size_t m = 0 ; m < maxMom ; m++)
    {
	  g_D_m = ( (Float)(maxMom - m + 1) * cos(phi_J * (Float)m) + sin(phi_J * (Float)m) * cos(phi_J) / sin(phi_J) ) * phi_J/(Float)M_PI;
	  this->operator()(m) *= g_D_m;
	}
}