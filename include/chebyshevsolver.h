#ifndef CHEBYSHEVSOLVER
#define CHEBYSHEVSOLVER

#include "typedef.h"
#include <cassert>
#include <array>
#include <memory>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <limits>
#include "sparsematrix.h"
#include "moments.h"
#include "chebcoef.h"
#include "mklblas.h"
#include <omp.h>
#include <chrono>
#include "statefactory.h"
#include "auxfunc.h"
#include <memory>

namespace chebyshev{

    //Calculation of the moments for the DOS

    size_t SpectralMoments(SparseMat & OP, chebyshev::Moments1D &chebMoms, factory::generator & gen, size_t numstates=1);
    size_t CorrelationExpansionMoments(SparseMat &OPL, SparseMat & OPR, chebyshev::Moments2D & chebmoms, factory::generator & gen, size_t num_states=1 );
    size_t CorrelationExpansionMoments(SparseMat & OP1, SparseMat &OPL, SparseMat & OPR, chebyshev::Moments2D & chebmoms, factory::generator & gen, size_t num_states=1 );
    
}




#endif
