#include "moments.h"


void chebyshev::VectorFilter::FillCoef(){
    assert(this -> _numMoms !=0);
    if(this ->MomentVector().size() ==0)
        this -> MomentVector() = std::move(std::vector<scalar>(_numMoms));
    switch(this ->Filter()){
        case fermi:{
            std::cout<<"Selected Fermi Dirac filter with chemical potential "<<this->_filterpar[0]<<" and temperature "<<this -> _filterpar[1]<<std::endl;
            this -> _Sample = 8*this -> MomentVector().size();
            Float dx = 2.0*chebyshev::CUTOFF/(Float)_Sample;
            const Float emax = 0.5*(this -> BandWidth() +  this -> BandCenter());
            const Float emin = 0.5*(-this -> BandWidth() +  this -> BandCenter());
            for(size_t m = 0; m<this -> MomentVector().size(); m++){
                #pragma omp parallel
                {
                    scalar localcoef = 0.;
                    #pragma omp for schedule(static), firstprivate(emax, emin, dx)
                    for(size_t n =0; n<=_Sample;n++){
                        const scalar pos = -chebyshev::CUTOFF+(Float)n*dx;
                        localcoef+=2.0*dx*FermiFilter(pos,_filterpar[0],_filterpar[1],emax,emin,chebyshev::CUTOFF)*pol1(pos,m)/((Float)M_PI*std::sqrt(1.0-pos*pos));
                    }

                    #pragma omp critical
                    {
                        this -> MomentVector()[m]+=localcoef;
                    }
                }
                
            }
            this -> MomentVector()[0]*=0.5;
        }
        break;

        case hole:{
            std::cout<<"Selected Hole filter with chemical potential "<<this->_filterpar[0]<<" and temperature "<<this -> _filterpar[1]<<std::endl;
            this -> _Sample = 8*this -> MomentVector().size();
            Float dx = 2.0*chebyshev::CUTOFF/(Float)_Sample;
            const Float emax = 0.5*(this -> BandWidth() +  this -> BandCenter());
            const Float emin = 0.5*(-this -> BandWidth() +  this -> BandCenter());
            for(size_t m = 0; m<this -> MomentVector().size(); m++){
                #pragma omp parallel
                {
                    scalar localcoef = 0.;
                    #pragma omp for schedule(static), firstprivate(emax, emin, dx)
                    for(size_t n =0; n<=_Sample;n++){
                        const scalar pos = -chebyshev::CUTOFF+(Float)n*dx;
                        localcoef+=2.0*dx*HoleFilter(pos,_filterpar[0],_filterpar[1],emax,emin,chebyshev::CUTOFF)*pol1(pos,m)/((Float)M_PI*std::sqrt(1.0-pos*pos));
                    }

                    #pragma omp critical
                    {
                        this -> MomentVector()[m]+=localcoef;
                    }
                }
                
            }
            this -> MomentVector()[0]*=0.5;

        }
        break;
        default:
            std::cerr<<"Not a valid option"<<std::endl;
            assert(false);
            break;
    }
}

void chebyshev::VectorFilter::PrintFilterParameters(){
    std::cout<<"the chemical potential in the filter is "<<this ->getParams()[0]<<std::endl;
    std::cout<<"The temperature in the filter is "<<this -> getParams()[1]<<std::endl;
    return;
}

void chebyshev::VectorFilter::PrepareOutVec(){
    const scalar zero =0;
    assert(this -> MomentVector().size()!=0);
    if(this->OPV().size()!=this->ChebV0().size()){
        this ->OPV()=std::move(std::vector<scalar>(this->ChebV0().size()));
        assert(this ->OPV().size() == this -> ChebV0().size());
    }
    //Now we proceed to evaluate the chebyshev products
    linalg::scal(this -> OPV().size(), &zero, this -> OPV());
    for(size_t m =0; m< this ->MomentVector().size(); m++){
        linalg::axpy(this -> ChebV0().size(), &(this -> MomentVector(m)), this ->ChebV0(), this -> OPV());
        this ->Iterate();
    }
    return;
};

void chebyshev::VectorFilter::ApplyJacksonKernel(const Float broad){
    assert(broad>0);
    const Float eta = (Float)2*broad/(Float)1000/ this-> BandWidth();
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