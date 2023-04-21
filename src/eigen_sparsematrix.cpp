#include "sparsematrix.h"

    void SparseMat::ConvertFromCOO(std::vector<Integer>& rows, std::vector<Integer>& cols, std::vector<scalar>& vals) {
        assert(rows.size() == cols.size() && cols.size() == vals.size());

        std::vector<Eigen::Triplet<scalar>> triplets;
        triplets.reserve(vals.size());

        for (size_t i = 0; i < vals.size(); ++i) {
            triplets.emplace_back(rows[i], cols[i], vals[i]);
        }

        matrix.setFromTriplets(triplets.begin(), triplets.end());
        matrix.makeCompressed();
    };

    void SparseMat::Multiply(const scalar a, const std::vector<scalar>& x, const scalar b, std::vector<scalar>& y) 
    {
        /*This implementation of the Multiply function performs a matrix-vector multiplication with the sparse 
        **matrix and a given input vector x. The result is scaled by scalar a and added to the output vector y 
        ** scaled by scalar b. Eigen's Map class is used to create Eigen-compatible maps for input and output vectors.*/

        assert(x.size() == static_cast<size_t>(matrix.cols()));
        assert(y.size() == static_cast<size_t>(matrix.rows()));

        Eigen::Map<const Eigen::Matrix<scalar, Eigen::Dynamic, 1>> x_map(x.data(), x.size());
        Eigen::Map<Eigen::Matrix<scalar, Eigen::Dynamic, 1>> y_map(y.data(), y.size());

        y_map = b * y_map + a * matrix * x_map;
    };

    void SparseMat::Rescale(const scalar a, const scalar b)
    {
        /*This implementation of the Rescale function scales the current sparse matrix by multiplying it with scalar a and then adding 
        **the result to an identity matrix scaled by scalar b. It uses Eigen's SparseMatrix class to perform these operations.*/
        if (matrix.nonZeros() == 0)
            return;

        Eigen::SparseMatrix<scalar, Eigen::RowMajor> bId(matrix.rows(), matrix.cols());
        bId.setIdentity();

        Eigen::SparseMatrix<scalar, Eigen::RowMajor> A = matrix;
        Eigen::SparseMatrix<scalar, Eigen::RowMajor> C = a * A + b * bId;

        matrix = C;
    };

    void SparseMat::BlockMultiply(const scalar a, const std::vector<scalar> &x, const Integer bsize, const scalar b, std::vector<scalar> &y) 
    {
        /*The source file implements the BlockMultiply function, which performs block multiplication of a sparse matrix and a dense matrix
        **using the Eigen library. It converts the input matrices to Eigen formats, performs the multiplication, scales the result by 
        ** scalar factors, and adds the result to the input/output matrix y. The implementation efficiently handles conversions between 
        ** the custom SparseMat class, Eigen matrix formats, and std::vector containers.*/
        Eigen::Map<const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X(x.data(), matrix.cols(), bsize);
        Eigen::Map<Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> Y(y.data(), matrix.rows(), bsize);

        Y = a * (matrix * X) + b * Y;
    };

