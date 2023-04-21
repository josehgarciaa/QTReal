#ifndef MKLBLAS
#define MKLBLAS
#include "typedef.h"
#include <cassert>
#include "mkl.h"


namespace linalg{
    void scal(const Integer size, const scalar * alpha, std::vector<scalar> & x);

    Float norm2(const Integer size, const std::vector<scalar> & x);

    void axpby( const Integer size, const scalar * alpha, const std::vector<scalar> & x, const scalar  * beta, std::vector<scalar> & y);

    void axpy( const Integer size, const scalar * alpha, const std::vector<scalar> & x, std::vector<scalar> & y);

    void dot( const Integer size, const std::vector<scalar> & x, const std::vector<scalar>& y, scalar * dotc);

    void swap(std::vector<scalar> & x, std::vector<scalar> & y, Integer size);

}

#endif