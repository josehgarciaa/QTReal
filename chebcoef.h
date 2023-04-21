#ifndef CHEBCOEF
#define CHEBCOEF

#include "typedef.h"

inline Float delta_chebF(Float x, const Float m){
    const Float f0 = std::sqrt((Float)1-x*x);
    const Float fm = std::cos(m*std::acos(x));
    return fm/f0/(Float)M_PI;
};

inline scalar greenR_chebF(Float x, const Float m){
    const scalar I(0,1);
    const Float f0 = std::sqrt((Float)1-x*x);
    const scalar fm = std::pow(x - I*f0,m);
    return -I*fm/f0;
};

inline scalar DgreenR_chebF(Float x, const Float m){
    const scalar I(0,1);
    const Float f0 = std::sqrt((Float)1-x*x);
    const scalar fm = std::pow(x-I*f0,m)*(x+I*m*f0);
    return -I*fm/f0/f0/f0;
};

inline scalar DGreenR_chebFAlt(Float x, const Float m){
    const scalar I(0,1);
    const Float f0 = std::sqrt((Float)1-x*x);
    const scalar fm = (x+I*m*sqrt(1-x*x))*std::exp(-I * m * std::acos(x));
    return -I*fm/f0/f0/f0;
}


#endif