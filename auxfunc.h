#ifndef AUXFUNC
#define AUXFUNC
#include "typedef.h"
#include <cstdlib>
#include <cmath>
#include <boost/math/special_functions/bessel.hpp>


float Bessel(Integer m, float x);
double Bessel(Integer m, double x);
scalar pol1(scalar x, Integer m);
scalar FermiD(scalar energy, scalar fermi, scalar temp,scalar emax, scalar emin, scalar alpha);
scalar FermiFilter(scalar energy, scalar fermi, scalar temp, scalar emax, scalar emin, scalar alpha);
scalar HoleFilter(scalar energy, scalar fermi, scalar temp, scalar emax, scalar emin, scalar alpha);
scalar FermiCoef(Integer steps, Integer order, scalar fermi, scalar temperature, scalar emax, scalar emin, scalar alpha);
scalar HoleCoef(Integer steps, Integer order, scalar fermi, scalar temperature, scalar emax, scalar emin, scalar alpha);



#endif