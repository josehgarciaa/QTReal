#include "mkl_blas.h"

namespace linalg{

    void scal(const Integer size, const scalar  * alpha, std::vector<scalar> & x){
        cblas_zscal(size, alpha, x.data(), 1);
        return;
    }

    Float norm2(const Integer size, const std::vector<scalar> & x){
        return cblas_dznrm2(size, x.data(), 1);
    }

    void axpby(const Integer size, const scalar * alpha, const std::vector<scalar> & x, const scalar * beta, std::vector<scalar> & y){
        cblas_zaxpby(size, alpha, x.data(), 1, beta, y.data(), 1);
        return;
    }

    void axpy(const Integer size, const scalar * alpha, const std::vector<scalar> & x, std::vector<scalar> & y){
        cblas_zaxpy(size, alpha, x.data(), 1, y.data(), 1);
        return;
    }

    void dot(const Integer size, const std::vector<scalar> & x, const std::vector<scalar> & y, scalar * dotc ){
        cblas_zdotc_sub(size, x.data(), 1, y.data(), 1, dotc);
        return;
    }

    void swap(std::vector<scalar> & x, std::vector<scalar> & y){
        std::swap(x,y);
        return;
    }


}