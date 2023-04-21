#ifndef UNITARY_H
#define UNITARY_H
#include <iostream>
#include "typedef.h"
#include <unordered_map>
#include <json.hpp>
#include <vector>
#include <string>
#include <fstream>


class Unitary{
    private:
        Integer OrbitalNumber;
        std::string name;
        std::string path;
        Float Volume;
        std::vector<Float> v1;
        std::vector<Float> v2;
        std::vector<Float> v3;

        std::vector<std::string> OrbitalList;
        std::vector<std::string> FromHoppings;
        std::vector<std::string> ToHoppings;
        std::vector<std::string> HoppingKind;
        std::vector<std::vector<std::string> > FromOperators;
        std::vector<std::vector<std::string> > ToOperators;
        std::vector<Index> HoppingDirections;
        std::vector<Position> DeltaPos;
        std::vector<Position> OrbitalPosition;
        std::unordered_map<std::string, Integer> OrbitalMaps;
        std::unordered_map<std::string, Integer> OrbitalPosMaps;
        std::vector<Matrix> OnSiteEnergies;
        std::vector<Matrix> OperatorValues;
        std::vector<Matrix> HoppingValues;
        
        //Here I need some form to store the disorder
    public:
        Unitary(const std::string &, const std::string & );
        Unitary();
        ~Unitary();
        Unitary(const Unitary & A);
        Unitary(Unitary && A);
        Unitary & operator = (const Unitary & A);
        Unitary & operator = (Unitary && A);
        void ReadJson();
        void SetDeltas();
        void SetVolume();
        void RegisterOrbitals();
        void RegisterHoppings();
        void RegisterOperators();
        inline Integer getOrbitalNubmer() {return this ->OrbitalNumber;};
        inline std::string getName() {return this -> name;};
        inline Float getVolume() {return this -> Volume;};
        inline std::vector<Float> getVec1() {return this -> v1;};
        inline std::vector<Float> getVec2() {return this -> v2;};
        inline std::vector<Float> getVec3() {return this -> v3;};
        inline std::vector<std::string> getOrbitalList() {return this -> OrbitalList;};
        inline std::vector<std::string> getFromHoppings() {return this -> FromHoppings;};
        inline std::vector<std::string> getToHoppings() {return this -> ToHoppings;};
        inline std::vector<std::vector<std::string> > getFromOperators() {return this -> FromOperators;};
        inline std::vector<std::vector<std::string> > getToOperators() {return this -> ToOperators;};
        inline std::vector<Index> getHoppingDirections() {return this -> HoppingDirections;};
        inline std::vector<Position> getDeltaPos() {return this -> DeltaPos; };
        inline std::vector<Position> getOrbitalPositions() {return this -> OrbitalPosition;};
        inline std::unordered_map<std::string, Integer> getOrbitalMaps() {return this -> OrbitalMaps;};
        inline std::unordered_map<std::string, Integer> getOrbitalPosMaps() {return this -> OrbitalPosMaps;};
        inline std::vector<Matrix> getOnSiteEnergies() {return this -> OnSiteEnergies;};
        inline std::vector<Matrix> getOperatorVaues() {return this -> OperatorValues;};
        inline std::vector<Matrix> getHoppingValues() {return this -> HoppingValues;};
        inline std::string getPath() {return this -> path;};
};





#endif
