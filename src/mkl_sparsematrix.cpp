#include "sparsematrix.h"

const Integer SPMULT = 1024;

void SparseMat::ConvertFromCOO(std::vector<Integer> & rows, std::vector<Integer> & cols, std::vector<scalar> & vals){
    if(rows.size() == 0 && cols.size() == 0 && vals.size() == 0)
        return;
    this -> initialized = true;
    sparse_matrix_t temp;
    auto status = mkl_sparse_z_create_coo(&temp, SPARSE_INDEX_BASE_ZERO, NumRows(), NumCols(), NumNNZ(), rows.data(), cols.data(), vals.data());
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_convert_csr(temp, SPARSE_OPERATION_NON_TRANSPOSE, & handleMat);
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_destroy(temp);
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_set_mv_hint(handleMat, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPMULT);
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_optimize(handleMat);
    assert(status == SPARSE_STATUS_SUCCESS);
    return;

};


void SparseMat::ConvertFromCOO(std::vector<Integer> & rows, std::vector<Integer> & cols, std::vector<scalar> & vals,Integer bsize){
    if(rows.size() == 0 && cols.size() == 0 && vals.size() == 0)
        return;
    this -> initialized = true;
    sparse_matrix_t temp;
    auto status = mkl_sparse_z_create_coo(&temp, SPARSE_INDEX_BASE_ZERO, NumRows(), NumCols(), NumNNZ(), rows.data(), cols.data(), vals.data());
    assert(status == SPARSE_STATUS_SUCCESS);

    status = mkl_sparse_convert_csr(temp, SPARSE_OPERATION_NON_TRANSPOSE, & handleMat);
    assert(status == SPARSE_STATUS_SUCCESS);

    status = mkl_sparse_destroy(temp);
    assert(status == SPARSE_STATUS_SUCCESS);

    status = mkl_sparse_set_mm_hint(handleMat, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_COLUMN_MAJOR,bsize, SPMULT);
    assert(status == SPARSE_STATUS_SUCCESS);

    status = mkl_sparse_optimize(handleMat);
    assert(status == SPARSE_STATUS_SUCCESS);
    return;

};


void SparseMat::ConvertFromCSR(std::vector<Integer> & rowIdx, std::vector<Integer> & cols, std::vector<scalar> & vals){
    if(rowIdx.size() == 0 && cols.size() == 0 && vals.size() == 0)
        return;
    this -> initialized = true;
    auto status = mkl_sparse_z_create_csr(&handleMat, SPARSE_INDEX_BASE_ZERO, NumRows(), NumCols(), rowIdx.data(), rowIdx.data()+1, cols.data(), vals.data());
    assert(status == SPARSE_STATUS_SUCCESS);

    status = mkl_sparse_set_mv_hint(handleMat, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPMULT);
    assert(status == SPARSE_STATUS_SUCCESS);

    status = mkl_sparse_optimize(handleMat);
    assert(status == SPARSE_STATUS_SUCCESS);
    return ;

};




void SparseMat::ConvertFromCSR(std::vector<Integer> & rowIdx, std::vector<Integer> & cols, std::vector<scalar> & vals, Integer bsize){
    if(rowIdx.size() == 0 && cols.size() == 0 && vals.size() == 0)
        return;
    this -> initialized = true;
    auto status = mkl_sparse_z_create_csr(&handleMat, SPARSE_INDEX_BASE_ZERO, NumRows(), NumCols(), rowIdx.data(), rowIdx.data()+1, cols.data(), vals.data());
    assert(status == SPARSE_STATUS_SUCCESS);

    status = mkl_sparse_set_mm_hint(handleMat, SPARSE_OPERATION_NON_TRANSPOSE, descr,SPARSE_LAYOUT_COLUMN_MAJOR, bsize ,SPMULT);
    assert(status == SPARSE_STATUS_SUCCESS);

    status = mkl_sparse_optimize(handleMat);
    assert(status == SPARSE_STATUS_SUCCESS);
    return ;

};


void SparseMat::Multiply(const scalar a, const std::vector<scalar> & x, const scalar b, std::vector<scalar> & y){
    auto status = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, a, handleMat, descr, x.data(), b, y.data());
    assert(status == SPARSE_STATUS_SUCCESS);
    return;
}


void SparseMat::BlockMultiply(const scalar a, const std::vector<scalar> & x, const Integer bsize, const scalar b, std::vector<scalar> & y){
    auto status = mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE, a, handleMat, descr,SPARSE_LAYOUT_COLUMN_MAJOR, x.data(), bsize, NumCols(), b, y.data(), NumCols());
    assert(status == SPARSE_STATUS_SUCCESS);
    return;
}

void SparseMat::Rescale(const scalar a, const scalar b){
    if(NumNNZ() == 0)
        return;
    sparse_matrix_t bId;
    sparse_matrix_t A;
    sparse_matrix_t C;
    std::vector<Integer> rows (NumRows()+1);
    std::vector<scalar> vals(NumRows());
    for(auto v = 0; v<NumRows()+1; v ++){
        rows[v] = v;
        if(v<NumRows())
            vals[v] = b;
        else{ }
    }

    auto status = mkl_sparse_z_create_csr(&bId, SPARSE_INDEX_BASE_ZERO, NumRows(), NumCols(), rows.data(), rows.data()+1, rows.data(), vals.data());
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_copy(handleMat, descr, &A);
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_destroy(handleMat);
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_z_add(SPARSE_OPERATION_NON_TRANSPOSE,A, a, bId, &C);
    status = mkl_sparse_destroy(bId);
    assert(status==SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_destroy(A);
    assert(status ==SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_copy(C, descr, & this -> handleMat);
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_destroy(C);
    assert(status == SPARSE_STATUS_SUCCESS);
    return;
}





