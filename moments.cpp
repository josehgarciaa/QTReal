#include "moments.h"


void chebyshev::Moments::SetInitVectors(const std::vector<scalar> & T0){
    const size_t size = T0.size();
    assert(size == this -> SystemSize());
    const size_t dim = this -> SystemSize();
    if (this -> ChebV0().size() != dim){
        this -> ChebV0() = std::move(std::vector<scalar>(dim, 0));
        this -> BlockSize(1);
    }

    if(this -> ChebV1().size() != dim)
        this -> ChebV1() = std::move(std::vector<scalar>(dim,0));
    
    scalar * ptrV0 = this -> ChebV0().data();
    const scalar * ptrT0 = T0.data();
    #pragma omp parallel for schedule(static)
    for( auto i =0; i<T0.size(); i++)
        ptrV0[i] = ptrT0[i];

    
    this -> HamiltonianHoppings().Multiply(this ->ChebV0(), this ->ChebV1());
    if(this -> HamiltonianOnSite().NumNNZ()!=0)
        this -> HamiltonianOnSite().Multiply(scalar(1,0),this -> ChebV0(), scalar(1,0),this ->ChebV1());
    

};


void chebyshev::Moments::SetInitBlock(const std::vector<scalar> & T0){
    const size_t dim = this -> SystemSize();
    const size_t size = T0.size();
    assert(size% dim == 0);
    const size_t bsize = size/dim;
    if(this -> ChebV0().size() / bsize != dim){
        this -> ChebV0() = std::move(std::vector<scalar>(size,0));
        this -> BlockSize(bsize);
    }
    if(this -> ChebV1().size()/ bsize != dim)
        this -> ChebV1() = std::move(std::vector<scalar>(size,0));

    scalar * ptrV0 = this -> ChebV0().data();
    const scalar * ptrT0 = T0.data();
    #pragma omp parallel for schedule(static)
    for( auto i =0; i<T0.size(); i++)
        ptrV0[i] = ptrT0[i];

    this -> HamiltonianHoppings().BlockMultiply(this -> ChebV0(), this -> ChebV1(), this ->BlockSize());
    if(this -> HamiltonianOnSite().NumNNZ()!=0)
        this -> HamiltonianOnSite().BlockMultiply(scalar(1,0),this -> ChebV0(),this -> BlockSize(), scalar(1,0),this ->ChebV1());
    
};


void chebyshev::Moments::SetInitVectors(SparseMat & Op, const std::vector<scalar> & T0){
    const size_t size = T0.size();
    assert(size == this -> SystemSize());
    const size_t dim = this -> SystemSize();
    if (this -> ChebV0().size() != dim){
        this -> ChebV0() = std::move(std::vector<scalar>(dim, 0));
        this -> BlockSize(1);
    }

    if(this -> ChebV1().size() != dim)
        this -> ChebV1() = std::move(std::vector<scalar>(dim,0));
    

    scalar * ptrV1 = this -> ChebV1().data();
    const scalar * ptrT0 = T0.data();
    #pragma omp parallel for schedule(static)
    for( auto i =0; i<T0.size(); i++)
        ptrV1[i] = ptrT0[i];
    
    Op.Multiply(this -> ChebV1(), this -> ChebV0());



    //ahroa podemos hacer la multiplicacion para esto tenemos que estimar que el hamiltoniano onsite es diferente de cero

    
    this -> HamiltonianHoppings().Multiply(scalar(1,0),this ->ChebV0(),scalar(0,0),this ->ChebV1());
    if(this -> HamiltonianOnSite().NumNNZ()!=0)
        this -> HamiltonianOnSite().Multiply(scalar(1,0),this -> ChebV0(), scalar(1,0),this ->ChebV1());

};


void chebyshev::Moments::SetInitBlock(SparseMat & Op, const std::vector<scalar> & T0){
    const size_t dim = this -> SystemSize();
    const size_t size = T0.size();
    assert(size% dim == 0);
    const size_t bsize = size/dim;
    if(this -> ChebV0().size() / bsize != dim){
        this -> ChebV0() = std::move(std::vector<scalar>(size,0));
        this -> BlockSize(bsize);
    }
    if(this -> ChebV1().size()/ bsize != dim)
        this -> ChebV1() = std::move(std::vector<scalar>(size,0));


    scalar * ptrV1 = this -> ChebV1().data();
    const scalar * ptrT0 = T0.data();
    #pragma omp parallel for schedule(static)
    for( auto i =0; i<T0.size(); i++)
        ptrV1[i] = ptrT0[i];

    Op.BlockMultiply(this ->ChebV1(), this -> ChebV0(), bsize);

    
    this -> HamiltonianHoppings().BlockMultiply(this -> ChebV0(), this -> ChebV1(), this ->BlockSize());
    if(this -> HamiltonianOnSite().NumNNZ()!=0)
        this -> HamiltonianOnSite().BlockMultiply(scalar(1,0),this -> ChebV0(),this -> BlockSize(), scalar(1,0),this ->ChebV1());
    
};


size_t chebyshev::Moments::Iterate(){
    if(this -> BlockSize() >1){
        this->HamiltonianHoppings().BlockMultiply(2.0, this->ChebV1(), this->BlockSize(), -1.0, this->ChebV0());
        if(this -> HamiltonianOnSite().NumNNZ()!=0)
            this->HamiltonianOnSite().BlockMultiply(2.0, this->ChebV1(), this->BlockSize(), 1.0, this->ChebV0());
        this->ChebV0().swap(this->ChebV1());
        return 0;
    }
    
    if(this -> BlockSize() ==1){
        
        this->HamiltonianHoppings().Multiply(2.0, this->ChebV1(), -1.0, this->ChebV0());
        if(this -> HamiltonianOnSite().NumNNZ()!=0)
         this->HamiltonianOnSite().Multiply(2.0, this->ChebV1(), 1.0, this->ChebV0());
        this->ChebV0().swap(this->ChebV1());
        return 0;
        

    }
    else{
        std::cerr << "Error: The block size cannot be zero, this means that the moment vectors are not initialized... Exiting..." << std::endl;
        assert(false);
        return 0;
    }
}

size_t chebyshev::Moments::JacksonKernelMomCutOff(const Float broad)
{
    assert(broad > 0);
    const Float eta = (Float)2.0 * broad * this->BandWidth() / (Float)1000;
    return std::ceil((Float)M_PI / eta);
}

Float chebyshev::Moments::JacksonKernel(const Float m, const Float Mom)
{
    const Float phi_J = (Float)M_PI / (Float)(Mom + (Float)1);
    return ((Mom - m + 1) * cos(phi_J * m) + sin(phi_J * m) * cos(phi_J) / sin(phi_J)) * phi_J / (Float)M_PI;
};


