#ifndef TYPES
#define TYPES
#include <complex>
#include <Eigen/Core>
#include "omp.h"
#include <memory>


typedef std::complex<double> scalar;
typedef double Float;
typedef int Integer;
typedef Eigen::Matrix<std::complex<double>,Eigen::Dynamic,1> Vector;
typedef Eigen::Vector3d Position;
typedef Eigen::Vector3i Index;
typedef Eigen::Matrix<size_t,3,1> CellVector;
typedef Eigen::Matrix<std::complex<double>,Eigen::Dynamic, Eigen::Dynamic> Matrix;
#endif