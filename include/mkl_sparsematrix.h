#include "typedef.h"
#define MKL_Complex16 scalar
#define MKL_INT Integer
#include "mkl.h"
#include "mkl_spblas.h"

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
