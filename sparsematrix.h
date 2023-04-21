#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <cassert>
#include <iostream>
#include "typedef.h"
#define MKL_Complex16 scalar
#define MKL_INT Integer
#include "mkl.h"
#include "mkl_spblas.h"


class SparseMatBase{
    private:
        size_t _nrows=0;
        size_t _ncols=0;
        size_t _nnz=0;
        std::string _format;
    public:
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
};



class SparseMat : public SparseMatBase{
    private:
        bool initialized = false;
        std::string _label;
        struct matrix_descr descr;
        sparse_matrix_t handleMat;
    public:
        SparseMat(){
            descr.type = SPARSE_MATRIX_TYPE_HERMITIAN;
            descr.mode = SPARSE_FILL_MODE_UPPER;
            descr.diag = SPARSE_DIAG_NON_UNIT;
        }
        ~SparseMat(){
            if(this -> initialized){
                auto status = mkl_sparse_destroy(handleMat);
                assert(status == SPARSE_STATUS_SUCCESS);
            }
            _label = std::string();
        }
        inline matrix_descr& mkl_descr() {return descr;};
        inline sparse_matrix_t & mkl_handle() {return handleMat;};
        std::string matrix_type() const {return "CSR MATRIX FROM MKL";};
        void Multiply(const scalar a, const std::vector<scalar> & x, const scalar b, std::vector<scalar> & y);
        void BlockMultiply(const scalar a, const std::vector<scalar> & x, const Integer bsize, const scalar b, std::vector<scalar> & y);
        void Rescale(const scalar a, const scalar b);
        inline void Multiply(const std::vector<scalar>& x, std::vector<scalar> & y){Multiply(scalar(1,0), x, scalar(0,0),y);};
        inline void BlockMultiply(const std::vector<scalar> & x, std::vector<scalar> & y, Integer bsize){BlockMultiply(scalar(1,0),x, bsize, scalar(0,0), y);};
        void ConvertFromCOO(std::vector<Integer> & rows, std::vector<Integer> & cols, std::vector<scalar> & vals);
        void ConvertFromCOO(std::vector<Integer> & rows, std::vector<Integer> & cols , std::vector<scalar> & vals, Integer bsize);
        void ConvertFromCSR(std::vector<Integer> & rowIdx, std::vector<Integer> & cols, std::vector<scalar> & vals);
        void ConvertFromCSR(std::vector<Integer> & rowIdx, std::vector<Integer> & cols, std::vector<scalar> & vals, Integer bsize);
        inline void MakeIdentity(){ this -> SetFormat("id");};
        inline std::string GetLabel(){return this -> _label;};
        inline void SetLabel(std::string label){this -> _label = label;};
        inline void ReleaseMatrix(){ 
            if(this -> initialized){
                mkl_sparse_destroy(this -> handleMat);
                this -> initialized = false;
            }
            else{ }
        };
        inline void SetSkewHermitian(){
            descr = matrix_descr();
            descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        };


};


#endif
