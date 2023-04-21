#include <Eigen/Core>
#include <Eigen/Dense>
#include <complex>
#include <vector>

using scalar = std::complex<double>;
using Float = double;
using Integer = int;

namespace linalg {
    void scal(const Integer size, const scalar *alpha, std::vector<scalar> &x) {
        Eigen::Map<Eigen::VectorXcd> X(x.data(), size);
        X *= *alpha;
    }

    Float norm2(const Integer size, const std::vector<scalar> &x) {
        Eigen::Map<const Eigen::VectorXcd> X(x.data(), size);
        return X.norm();
    }

    void axpby(const Integer size, const scalar *alpha, const std::vector<scalar> &x, const scalar *beta, std::vector<scalar> &y) {
        Eigen::Map<const Eigen::VectorXcd> X(x.data(), size);
        Eigen::Map<Eigen::VectorXcd> Y(y.data(), size);
        Y = (*alpha) * X + (*beta) * Y;
    }

    void axpy(const Integer size, const scalar *alpha, const std::vector<scalar> &x, std::vector<scalar> &y) {
        Eigen::Map<const Eigen::VectorXcd> X(x.data(), size);
        Eigen::Map<Eigen::VectorXcd> Y(y.data(), size);
        Y += (*alpha) * X;
    }

    void dot(const Integer size, const std::vector<scalar> &x, const std::vector<scalar> &y, scalar *dotc) {
        Eigen::Map<const Eigen::VectorXcd> X(x.data(), size);
        Eigen::Map<const Eigen::VectorXcd> Y(y.data(), size);
        *dotc = X.dot(Y);
    }

    void swap(std::vector<scalar> &x, std::vector<scalar> &y) {
        std::swap(x, y);
    }
}