#ifndef PERIODICHAMILTONIAN_H
#define PERIODICHAMILTONIAN_H
#include "typedef.h"
#include "unitary.h"
class PeriodicHamiltonian{
    //In this method we will generate the periodic parts of the Hamiltonian
    //We will create the onsite potentials and the hoppings
    //Each of these will be treated separately to avoid complications
    //The velocities will be calculated separately and taking in consideration the case in which the z component is not given
    //The code will deal with H = H0 VacOp + HOnsite VacOp + HdisVacOp
    //VacOp will delete the entry of the vacancy on the random phase vector. This vacancy operator will be defined later
    private:
        std::array<Integer,3> Ncells;
        Unitary base;
        std::string operatorName = "";
    public:
        PeriodicHamiltonian(Unitary & , Integer n1, Integer n2, Integer n3 = 0);
        PeriodicHamiltonian();
        ~PeriodicHamiltonian();
        template <Integer Dim> void SetPeriodicHoppings(std::vector<Integer>  & rows, std::vector<Integer>  & cols, std::vector<scalar> & values);
        template <Integer Dim> void SetOnSitePotentials(std::vector<Integer>  & rows, std::vector<Integer>  & cols, std::vector<scalar> & values);
        void SetVelocities(std::vector<Integer>  & rows, std::vector<Integer>  & cols, std::vector<scalar> & VXvalues, std::vector<scalar> & VYvalues);
        void SetVelocities(std::vector<Integer>  & rows, std::vector<Integer>  & cols, std::vector<scalar> & VXvalues, std::vector<scalar> & VYvalues, std::vector<scalar> & VZvalues);
        template <Integer Dim> void SetOperator(std::vector<Integer> & rows, std::vector<Integer> & cols, std::vector<scalar>& values);
        inline std::array<Integer,3>& getCellNum() {return Ncells;};
        inline Unitary& getUnitCell(){return base;};
        inline void SetOpeartorName(std::string name){ this -> operatorName = name;};
        
};
    



#endif