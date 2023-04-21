#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H
#define USE_EIGEN
#include <cassert>
#include <iostream>
#include "typedef.h"


#include <Eigen/Sparse>

class SparseMatBase{
    private:
        size_t _nrows=0;
        size_t _ncols=0;
        size_t _nnz=0;
        std::string _format;
        std::string _label;

    public:
    bool initialized = false;
    inline std::string Format() {return _format;};
    inline size_t NumRows() {return _nrows;};
    inline size_t NumCols() {return _ncols;};
    inline size_t NumNNZ() {return _nnz;};
    inline void SetDimensions(const size_t rows, const size_t cols){ 
            _nrows = rows;
            _ncols = cols;
        };
    inline void SetFormat(std::string format){_format = format;};
    inline bool IsIdMat(){return Format() == "id";};
    inline void SetNelem(const Integer n){ _nnz = n;};
    size_t rank() {return ((this->NumRows() > this->NumCols())? this->NumCols(): this->NumRows());};  

    void Multiply(const scalar a, const std::vector<scalar> & x, const scalar b, std::vector<scalar> & y);
    /*!
    * \fn void Multiply(const scalar a, const std::vector<scalar>& x, const scalar b, std::vector<scalar>& y)
    * \brief Performs a matrix-vector multiplication and scales the input and output vectors.
    *
    * This function multiplies the sparse matrix with the input vector `x`, scales the result by
    * scalar `a`, scales the output vector `y` by scalar `b`, and adds the two results, storing the
    * final result back in the output vector `y`.
    *
    * \param a A scalar value to scale the result of the matrix-vector multiplication.
    * \param x A std::vector representing the input vector.
    * \param b A scalar value to scale the output vector before addition.
    * \param y A std::vector representing the output vector, storing the result of the operation.
    */


    void BlockMultiply(const scalar a, const std::vector<scalar> & x, const Integer bsize, const scalar b, std::vector<scalar> & y);
    /*!
    \fn void SparseMat::BlockMultiply(const scalar a, const std::vector<scalar> &x, const Integer bsize, const scalar b, std::vector<scalar> &y)
    \brief Performs block matrix-vector multiplication for a sparse matrix.
    *
    *This member function takes in two scalar values, a and b, a vector x, and a block size (bsize).
    *It computes the product of the sparse matrix with the given vector x, and stores the result in
    *the output vector y. The block size parameter is used to optimize the multiplication for block
    *structures in the sparse matrix.
    */

    void Rescale(const scalar a, const scalar b);
    /*!
    * \fn void SparseMat::Rescale(const scalar a, const scalar b)
    * \brief Rescales the sparse matrix by applying a linear transformation.
    *
    * This function performs a linear transformation on the sparse matrix, scaling it by `a` and
    * adding an identity matrix scaled by `b`. The resulting matrix is stored back in the object.
    *
    * \param a A scalar value to scale the sparse matrix.
    * \param b A scalar value to scale the identity matrix before addition.
    */

    inline void Multiply(const std::vector<scalar>& x, std::vector<scalar> & y){Multiply(scalar(1,0), x, scalar(0,0),y);};
    inline void BlockMultiply(const std::vector<scalar> & x, std::vector<scalar> & y, Integer bsize){BlockMultiply(scalar(1,0),x, bsize, scalar(0,0), y);};

    void ConvertFromCOO(std::vector<Integer> & rows, std::vector<Integer> & cols, std::vector<scalar> & vals);
    void ConvertFromCOO(std::vector<Integer> & rows, std::vector<Integer> & cols , std::vector<scalar> & vals, Integer bsize);

    void ConvertFromCSR(std::vector<Integer> & rowIdx, std::vector<Integer> & cols, std::vector<scalar> & vals);
    void ConvertFromCSR(std::vector<Integer> & rowIdx, std::vector<Integer> & cols, std::vector<scalar> & vals, Integer bsize);
    inline void MakeIdentity(){ this -> SetFormat("id");};
    inline std::string GetLabel(){return this -> _label;};
    inline void SetLabel(std::string label){this -> _label = label;};
};



class SparseMat : public SparseMatBase {
private:
    bool initialized = false;
    std::string _label;
    Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> matrix;

public:
    SparseMat() {
        // Set matrix properties if required
    }

    ~SparseMat() {
        // No need to manually destroy the matrix, Eigen takes care of it
        _label = std::string();
    }

    std::string matrix_type() const { return "CSR MATRIX FROM EIGEN"; }

    inline Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &eigen_matrix() { return matrix; }

    inline void ReleaseMatrix() {
        if (this->initialized) {
            matrix.resize(0, 0); // Resizes matrix to 0x0, effectively releasing the memory
            this->initialized = false;
        } else {
        }
    }

    inline void SetSkewHermitian() {
        // Implement functionality if required
    }
};


#endif
