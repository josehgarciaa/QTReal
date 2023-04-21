#include "auxfunc.h"


float Bessel(Integer m, float x){
    return boost::math::cyl_bessel_j(m,x);
}


double Bessel(Integer m, double x){
    return boost::math::cyl_bessel_j(m,x);
}


scalar pol1(scalar x, Integer m){
    return std::cos((scalar)m*std::acos(x));
}


scalar FermiD(scalar energy, scalar fermi, scalar temp){
    if(std::exp((energy - fermi)/(temp)).real()>1E200)//ver esto aca
            return (scalar)0;
        else
            return ((scalar)1.0/(std::exp((energy - fermi)/(temp)) +(Float) 1.0));
    
}

scalar FermiFilter(scalar x, scalar fermi, scalar temp, scalar emax, scalar emin, scalar alpha){
    scalar a, b;
    a = (scalar)2*alpha/(emax - emin);
    b = a*(emax + emin)*(scalar)0.5;
    scalar chebfermi, chebtemp;
    chebfermi = (a*fermi)-b;
    chebtemp = (a*temp) - b;
    return FermiD(x, chebfermi,chebtemp);
    
}


scalar HoleFilter(scalar x, scalar fermi, scalar temp, scalar emax, scalar emin, scalar alpha){
    scalar a, b;
    a = (scalar)2*alpha/(emax - emin);
    b = a*(emax + emin)*(scalar)0.5;
    scalar chebfermi, chebtemp;
    chebfermi = (a*fermi)-b;
    chebtemp = (a*temp) - b;
    return (scalar) 1- FermiD(x, chebfermi,chebtemp);
    
}


scalar FermiCoef(Integer steps, Integer order, scalar fermi, scalar temperature, scalar emax, scalar emin, scalar alpha){
 scalar delta = ((scalar)2*alpha/(scalar)(steps-1))*(scalar)2/(scalar)M_PI;
 scalar dx = ((scalar)2*alpha/(scalar)(steps-1));
 scalar result=0;
 #pragma omp parallel shared(result)
 {
     scalar localresult =0;
     #pragma omp for schedule(dynamic) firstprivate(fermi,temperature, emax, emin, alpha, order,delta,dx)
     for(Integer i=0; i<steps;i++){
         scalar x = -alpha+(scalar)i*dx;
        localresult += delta*FermiFilter(x, fermi, temperature, emax, emin, alpha)*pol1(x,order)/(std::sqrt((scalar)1-x*x));
        
     }
     #pragma omp critical
     {
        result += localresult;
     }
 }

 return result;
}

scalar HoleCoef(Integer steps, Integer order, scalar fermi, scalar temperature, scalar emax, scalar emin, scalar alpha){
 scalar delta = ((scalar)2*alpha/(scalar)(steps-1))*(scalar)2/(scalar)M_PI;
 scalar dx = ((scalar)2*alpha/(scalar)(steps-1));
 scalar result=0;
 #pragma omp parallel shared(result)
 {
     scalar localresult =0;
     #pragma omp for schedule(dynamic) firstprivate(fermi,temperature, emax, emin, alpha, order,delta,dx)
     for(Integer i=0; i<steps;i++){
         scalar x = -alpha+(scalar)i*dx;
        localresult += delta*HoleFilter(x, fermi, temperature, emax, emin, alpha)*pol1(x,order)/(std::sqrt((scalar)1-x*x));
        
     }
     #pragma omp critical
     {
        result += localresult;
     }
 }
 return result;
}