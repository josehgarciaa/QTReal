#include "unitary.h"

using json = nlohmann::json;

Unitary::Unitary(const std::string & spath, const std::string & prefix){
    this -> path = spath+"/"+prefix+".prop";
    std::ifstream f(this->path+"/LatticeBasics.json");
    if(!f){
        std::cout<<"Error: The file LatticeBasics.json cannot be found in the specified path... exiting\n"<<std::endl;
        exit(EXIT_FAILURE);
    }
    json latticebasics = json::parse(f);

    this -> name = (std::string) latticebasics["name"];
    this -> v1 = (std::vector<Float>)latticebasics["Vec1"];
    this -> v2 = (std::vector<Float>)latticebasics["Vec2"];
    if(!latticebasics["Vec3"].is_null()){
        this -> v3 = (std::vector<Float>)latticebasics["Vec3"];
        //now we check
        if(v1.size() != v3.size() || v2.size() != v3.size()){
            printf("Error: the lenghts of the vectors are not the same ... exiting\n");
            exit(EXIT_FAILURE);
        }
        
    }

    if(v1.size()!=v2.size()){
        printf("Error: the lenghts of the vectors are not the same ... exiting\n");
        exit(EXIT_FAILURE);
    }

    f.close();

}

Unitary::Unitary(){}

Unitary::~Unitary(){
    this -> HoppingValues.clear();
    this -> HoppingValues.shrink_to_fit();
    this -> OperatorValues.clear();
    this -> OperatorValues.shrink_to_fit();
    this -> OnSiteEnergies.clear();
    this -> OnSiteEnergies.shrink_to_fit();
    this -> OrbitalMaps.clear();
    this -> OrbitalPosition.clear();
    this -> OrbitalPosition.shrink_to_fit();
    this -> DeltaPos.clear();
    this -> DeltaPos.shrink_to_fit();
    this -> HoppingDirections.clear();
    this -> HoppingDirections.shrink_to_fit();
    this -> ToOperators.clear();
    this -> ToOperators.shrink_to_fit();
    this -> FromOperators.clear();
    this -> FromOperators.shrink_to_fit();
    this -> HoppingKind.clear();
    this -> HoppingKind.shrink_to_fit();
    this -> ToHoppings.clear();
    this -> ToHoppings.shrink_to_fit();
    this -> FromHoppings.clear();
    this -> FromHoppings.shrink_to_fit();
    this -> OrbitalList.clear();
    this -> OrbitalList.shrink_to_fit();
    this -> v3.clear();
    this -> v3.shrink_to_fit();
    this -> v2.clear();
    this -> v2.shrink_to_fit();
    this -> v1.clear();
    this -> v1.shrink_to_fit();
    this -> Volume = (Float)0;
    this -> path = "";
    this -> name = "";
    this -> OrbitalNumber = 0;
}


Unitary::Unitary(const Unitary & A){
    this ->OrbitalNumber = A.OrbitalNumber;
    this -> name = A.name;
    this -> path = A.path;
    this -> Volume = A.Volume;
    this -> v1 = A.v1;
    this -> v2 = A.v2;
    this -> v3 = A.v3;
    this -> OrbitalList = A.OrbitalList;
    this -> FromHoppings = A.FromHoppings;
    this -> ToHoppings = A.ToHoppings;
    this -> HoppingKind = A.HoppingKind;
    this -> FromOperators = A.FromOperators;
    this -> ToOperators = A.ToOperators;
    this -> HoppingDirections = A.HoppingDirections;
    this -> DeltaPos = A.DeltaPos;
    this -> OrbitalPosition = A.OrbitalPosition;
    this -> OrbitalMaps = A.OrbitalMaps;
    this -> OnSiteEnergies = A.OnSiteEnergies;
    this -> OperatorValues = A.OperatorValues;
    this -> HoppingValues = A.HoppingValues;
}



Unitary::Unitary(Unitary && A){
    this ->OrbitalNumber = std::move(A.OrbitalNumber);
    this -> name = std::move(A.name);
    this -> path = std::move(A.path);
    this -> Volume = std::move(A.Volume);
    this -> v1 = std::move(A.v1);
    this -> v2 = std::move(A.v2);
    this -> v3 = std::move(A.v3);
    this -> OrbitalList = std::move(A.OrbitalList);
    this -> FromHoppings = std::move(A.FromHoppings);
    this -> ToHoppings = std::move(A.ToHoppings);
    this -> HoppingKind = std::move(A.HoppingKind);
    this -> FromOperators = std::move(A.FromOperators);
    this -> ToOperators = std::move(A.ToOperators);
    this -> HoppingDirections = std::move(A.HoppingDirections);
    this -> DeltaPos = std::move(A.DeltaPos);
    this -> OrbitalPosition = std::move(A.OrbitalPosition);
    this -> OrbitalMaps = std::move(A.OrbitalMaps);
    this -> OnSiteEnergies = std::move(A.OnSiteEnergies);
    this -> OperatorValues = std::move(A.OperatorValues);
    this -> HoppingValues = std::move(A.HoppingValues);
}


Unitary & Unitary::operator = (const Unitary & A){
    if (&A == this )
        return *this;
    else{
        this ->OrbitalNumber = A.OrbitalNumber;
        this -> name = A.name;
        this -> path = A.path;
        this -> Volume = A.Volume;
        this -> v1 = A.v1;
        this -> v2 = A.v2;
        this -> v3 = A.v3;
        this -> OrbitalList = A.OrbitalList;
        this -> FromHoppings = A.FromHoppings;
        this -> ToHoppings = A.ToHoppings;
        this -> HoppingKind = A.HoppingKind;
        this -> FromOperators = A.FromOperators;
        this -> ToOperators = A.ToOperators;
        this -> HoppingDirections = A.HoppingDirections;
        this -> DeltaPos = A.DeltaPos;
        this -> OrbitalPosition = A.OrbitalPosition;
        this -> OrbitalMaps = A.OrbitalMaps;
        this -> OnSiteEnergies = A.OnSiteEnergies;
        this -> OperatorValues = A.OperatorValues;
        this -> HoppingValues = A.HoppingValues;
        return *this;
    }
}

Unitary & Unitary::operator = (Unitary && A){
    if(this != &A){
        this ->OrbitalNumber = std::move(A.OrbitalNumber);
        this -> name = std::move(A.name);
        this -> path = std::move(A.path);
        this -> Volume = std::move(A.Volume);
        this -> v1 = std::move(A.v1);
        this -> v2 = std::move(A.v2);
        this -> v3 = std::move(A.v3);
        this -> OrbitalList = std::move(A.OrbitalList);
        this -> FromHoppings = std::move(A.FromHoppings);
        this -> ToHoppings = std::move(A.ToHoppings);
        this -> HoppingKind = std::move(A.HoppingKind);
        this -> FromOperators = std::move(A.FromOperators);
        this -> ToOperators = std::move(A.ToOperators);
        this -> HoppingDirections = std::move(A.HoppingDirections);
        this -> DeltaPos = std::move(A.DeltaPos);
        this -> OrbitalPosition = std::move(A.OrbitalPosition);
        this -> OrbitalMaps = std::move(A.OrbitalMaps);
        this -> OnSiteEnergies = std::move(A.OnSiteEnergies);
        this -> OperatorValues = std::move(A.OperatorValues);
        this -> HoppingValues = std::move(A.HoppingValues);
    }
    return *this;
}



 void Unitary::RegisterOrbitals(){
    std::ifstream f(path+"/Sublattices.json");
    if(!f){
        std::cout<<"Error: The file LatticeBasics.json cannot be found in the specified path... exiting\n"<<std::endl;
        exit(EXIT_FAILURE);
    }
    Integer nrows, ncols;
    std::vector<Float> onsiteReal;
    std::vector<Float> onsiteImag;
    json sublattices = json::parse(f);
    Integer count =0;
    for(auto it = sublattices["Sublattices"].begin(); it!=sublattices["Sublattices"].end(); ++it){
        OrbitalList.push_back( (*it)["Label"]);
        nrows = (*it)["Nrows"];
        ncols = (*it)["Ncols"];
        onsiteReal = (std::vector<Float>)(*it)["OnsiteReal"];
        onsiteImag = (std::vector<Float>)(*it)["OnsiteImag"];
        Matrix H = Matrix::Zero(nrows,ncols);
        OrbitalMaps[(*it)["Label"]] = count;
        for(auto i = 0; i< nrows; i++){
            for(auto j = 0; j<ncols; j++){
                H(i,j) = scalar(onsiteReal[i*ncols+j],onsiteImag[i*ncols+j]);
            }
        }
        OnSiteEnergies.push_back(H);
        //Here we reuse the onsiteImag
        onsiteImag = (std::vector<Float>)(*it)["Position"];
        if(onsiteImag.size()!=v1.size()){
            printf("Error: Inconsistency in the definition of the position of the sublattices \n");
            exit(EXIT_FAILURE);
        }
        if(onsiteImag.size()==3){
            OrbitalPosition.push_back(Position({onsiteImag[0],onsiteImag[1],onsiteImag[2]}));
        }
        if(onsiteImag.size()==2){
            OrbitalPosition.push_back(Position({onsiteImag[0],onsiteImag[1],0.}));
        }
        count+=H.cols();
        OrbitalPosMaps[(*it)["Label"]] = it - sublattices["Sublattices"].begin();
        
    }
    f.close();       
}

void Unitary::RegisterHoppings(){
    std::ifstream f(path+"/Hoppings.json");
    if(!f){
        printf("Error: The file Hoppings.json cannot be found in the specified path... exiting\n");
        exit(EXIT_FAILURE);
    }
    Integer nrows, ncols;
    std::vector<Float> HoppingReal;
    std::vector<Float> HoppingImag;
    std::vector<Integer> RelIdx;
    Position DeltaX;
    json hoppings = json::parse(f);
    for(auto it = hoppings["Hoppings"].begin(); it != hoppings["Hoppings"].end(); it++){
        FromHoppings.push_back((*it)["From"]);
        ToHoppings.push_back((*it)["To"]);
        RelIdx = (std::vector<Integer>)(*it)["Rel_Idx"];
        
        if(RelIdx.size()!=v1.size()){
            printf("Error: Inconsistency in the definition of the relative index of the hoppings\n");
            exit(EXIT_FAILURE);
        }
        if(RelIdx.size()==3){
            HoppingDirections.push_back(Index({RelIdx[0],RelIdx[1],RelIdx[2]}));
        }
        else{
            HoppingDirections.push_back(Index({RelIdx[0],RelIdx[1],0}));
        }

        nrows = (*it)["Nrows"];
        ncols = (*it)["Ncols"];
        HoppingReal = (std::vector<Float>)(*it)["ValsReal"];
        HoppingImag = (std::vector<Float>)(*it)["ValsImag"];
        Matrix H = Matrix::Zero(nrows,ncols);
        for(auto i = 0; i< nrows; i++){
            for(auto j = 0; j<ncols; j++){
                H(i,j) = scalar(HoppingReal[i*ncols+j],HoppingImag[i*ncols+j]);
            }
        }
        HoppingValues.emplace_back(H);
    }
    f.close();
}




void Unitary::SetDeltas(){
    for(auto v = 0; v< FromHoppings.size(); v++){

       auto source = this ->getOrbitalPositions()[this ->getOrbitalPosMaps()[this ->getFromHoppings()[v]]];

       auto dest = this ->getOrbitalPositions()[this ->getOrbitalPosMaps()[this ->getToHoppings()[v]]];

       if(v3.size() == 0){
        const Position Delta {dest[0]+((Float) this ->HoppingDirections[v][0]*v1[0] +(Float) this ->HoppingDirections[v][1]*v2[0]) - source[0],
                                    dest[1]+((Float) this ->HoppingDirections[v][0]*v1[1] +(Float) this ->HoppingDirections[v][1]*v2[1]) - source[1],0.};

        DeltaPos.emplace_back(Delta);
       }

       else{

        const Position Delta {dest[0]+((Float) this ->HoppingDirections[v][0]*v1[0] +(Float) this ->HoppingDirections[v][1]*v2[0] + (Float) this ->HoppingDirections[v][2]*v3[0]) - source[0],
                                    dest[1]+((Float) this ->HoppingDirections[v][0]*v1[1] +(Float) this ->HoppingDirections[v][1]*v2[1] +(Float) this ->HoppingDirections[v][2]*v3[1]  ) - source[1],
                                    dest[2]+((Float) this ->HoppingDirections[v][0]*v1[2] +(Float) this ->HoppingDirections[v][1]*v2[2] +(Float) this ->HoppingDirections[v][1]*v3[2]  ) - source[2]};

        DeltaPos.emplace_back(Delta);




       }


        
    }
    return ;
}