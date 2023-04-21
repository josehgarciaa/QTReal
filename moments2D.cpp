#include "moments.h"

void chebyshev::Moments2D::MomentNumber(const size_t mom0, const size_t mom1){

    assert(mom0<=_numMoms[0] && mom1<=_numMoms[1] && this->BlockSize()!= 0);
    chebyshev::Moments2D new_mom(mom0,mom1);
    for(size_t m0 = 0; m0<mom0; m0++)
    for(size_t m1 = 0; m1<mom1; m1++)
        new_mom(m0,m1) = this -> operator()(m0, m1);
    this -> _numMoms = new_mom.MomentNumber();
	this->MomentVector( new_mom.MomentVector() );
};




void chebyshev::Moments2D::Print(){
    std::cout<<"\n\nCHEBYSHEV 2D MOMENTS INFO"<<std::endl;
	std::cout<<"\tSYSTEM:\t\t\t"<<this->SystemLabel()<<std::endl;
	if( this-> SystemSize() > 0 )
		std::cout<<"\tSIZE:\t\t\t"<<this-> SystemSize()<<std::endl;

	std::cout<<"\tMOMENTS SIZE:\t\t"<<"("<<this->HighestMomentNumber(0)<<" x " <<this->HighestMomentNumber(1)<<")"<<std::endl;
	std::cout<<"\tSCALE FACTOR:\t\t"<<this->ScaleFactor()<<std::endl;
	std::cout<<"\tSHIFT FACTOR:\t\t"<<this->ShiftFactor()<<std::endl;
	std::cout<<"\tENERGY SPECTRUM:\t("
			 <<-this->HalfWidth()+this->BandCenter()<<" , "
			 << this->HalfWidth()+this->BandCenter()<<")"<<std::endl<<std::endl;
    std::cout<<"\tVECTOR SIZE\t"<<this->ChebV0().size()<<std::endl;
    std::cout<<"\tBLOCK SIZE\t"<< this ->BlockSize()<<std::endl;
};

void chebyshev::Moments2D::saveIn(std::string filename){
    typedef std::numeric_limits<Float> dbl;
    std::ofstream outputfile(filename);
    outputfile << this -> SystemSize() <<" "<< this -> BandWidth()<<" "<< this -> BandCenter()<<std::endl;
    for(const auto& x : _numMoms)
        outputfile << x<< " ";
    outputfile<<std::endl;
    outputfile << this->BlockSize()<<std::endl;
    for(size_t v = 0; v<_numMoms[0]*_numMoms[1]; v++){
        const scalar var = this->MomentVector(v);
        outputfile<<var.real()<<" "<<var.imag()<<std::endl;
    }
    outputfile.close();
}


chebyshev::Moments2D::Moments2D(std::string momfilename){
    size_t ext_pos = momfilename.find(".chebmom2D");
    if(ext_pos == std::string::npos){
        std::cerr<<"The first argument does not seem to be a valid .chebmom2D file... Exiting..."<<std::endl;
        assert(false);
    }

    this -> SystemLabel(momfilename.substr(0,ext_pos));
    
    std::ifstream momfile(momfilename);
    assert(momfile.is_open());
    size_t ibuff; Float dbuff;
    momfile >>ibuff; this -> SystemSize();
    momfile >>dbuff; this -> BandWidth();
    momfile >>dbuff; this -> BandCenter();
    momfile >> this ->_numMoms[0]>>this->_numMoms[1];
    momfile >>ibuff; this -> BlockSize(ibuff);
    this ->MomentVector(std::vector<scalar>(_numMoms[0]*_numMoms[1],0));
    Float rmu, imu;
    for(size_t m0 = 0; m0<_numMoms[0]; m0++)
    for(size_t m1 = 0; m1<_numMoms[1]; m1++){
        momfile>>rmu>>imu;
        this ->operator()(m0,m1) = scalar(rmu,imu);
    }
    momfile.close();

}

void chebyshev::Moments2D::ApplyJacksonKernel( const Float b0, const Float b1 )
{
	assert( b0 >0 && b1>0);
	const Float eta0   =  2.0*b0/1000/this->BandWidth();
	const Float eta1   =  2.0*b1/1000/this->BandWidth();
		
	Integer maxMom0=  ceil(M_PI/eta0);
	Integer maxMom1=  ceil(M_PI/eta1);

	if(  maxMom0 > _numMoms[0] ) maxMom0 = _numMoms[0];
	if(  maxMom1 > _numMoms[1] ) maxMom1 = _numMoms[1];
	std::cout<<"Kernel reduced the number of moments to "<<maxMom0<<" "<<maxMom1<<std::endl;
	this->MomentNumber( maxMom0,maxMom1 ) ;



	const double
	phi_J0 = M_PI/(double)(_numMoms[0]+1.0),
	phi_J1 = M_PI/(double)(_numMoms[1]+1.0);
		
	double g_D_m0,g_D_m1;
	for( int m0 = 0 ; m0 < _numMoms[0] ; m0++)
	{
		g_D_m0=( (_numMoms[0]-m0+1)*cos( phi_J0*m0 )+ sin(phi_J0*m0)*cos(phi_J0)/sin(phi_J0) )*phi_J0/M_PI;
		for( int m1 = 0 ; m1 < _numMoms[1] ; m1++)
		{
			g_D_m1=( (_numMoms[1]-m1+1)*cos( phi_J1*m1 )+ sin(phi_J1*m1)*cos(phi_J1)/sin(phi_J1) )*phi_J1/M_PI;
			this->operator()(m0,m1) *= g_D_m0*g_D_m1;
		}
	}
}