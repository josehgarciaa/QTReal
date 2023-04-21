#include "periodichamiltonian.h"


using json = nlohmann::json;

PeriodicHamiltonian::PeriodicHamiltonian(Unitary & Ucell, Integer n1, Integer n2, Integer n3){
    Ncells = {n1,n2,n3};
    base = std::move(Ucell);
}

PeriodicHamiltonian::PeriodicHamiltonian(){
    Ncells = {0,0,0};
    base = Unitary();
}

PeriodicHamiltonian::~PeriodicHamiltonian(){
    Ncells = {0,0,0};
    base = Unitary();
}

template <> void PeriodicHamiltonian::SetPeriodicHoppings<2>(std::vector<Integer> & rows, std::vector<Integer>  &cols, std::vector<scalar> & vals ){
    Integer MacroDeparture, MicroDeparture, MacroArrival,MicroArrival;
    Integer Inrow,Incol,stride;
    stride = 0;
    for (auto i =0; i<base.getOnSiteEnergies().size(); i++)
        stride += base.getOnSiteEnergies()[i].cols();
    std::cout<<"the value of the stride is "<<stride<<std::endl;
    for(auto n1 =0 ; n1< Ncells[0];n1++){
        for(auto n2 = 0; n2<Ncells[1]; n2++){
            for(auto k = 0; k<base.getFromHoppings().size(); k++){
                //Here we must iterate over rows and cols
                for(auto l = 0; l<(base.getHoppingValues()[k].rows()*base.getHoppingValues()[k].cols()); l++){
                    MacroDeparture = ((n1+Ncells[0])%Ncells[0])*Ncells[1] + (n2+Ncells[1])%Ncells[1];
                    MacroArrival = ((n1+Ncells[0]+base.getHoppingDirections()[k][0])%Ncells[0])*Ncells[1] + (n2+Ncells[1]+base.getHoppingDirections()[k][1])%Ncells[1];
                    Inrow = l/base.getHoppingValues()[k].cols();
                    Incol = l%base.getHoppingValues()[k].cols();
                    if(std::fabs(base.getHoppingValues()[k](Inrow,Incol))<1e-10){ }
                    else{
                        MicroDeparture = MacroDeparture*stride + base.getOrbitalMaps()[base.getFromHoppings()[k]] + Incol; 
                        MicroArrival = MacroArrival*stride + base.getOrbitalMaps()[base.getToHoppings()[k]] + Inrow;
                        if(MicroArrival < MicroDeparture){
                            cols.emplace_back(MicroDeparture);
                            rows.emplace_back(MicroArrival);
                            vals.emplace_back(base.getHoppingValues()[k](Inrow,Incol));
                        }

                        else{
                            cols.emplace_back(MicroArrival);
                            rows.emplace_back(MicroDeparture);
                            vals.emplace_back(std::conj(base.getHoppingValues()[k](Inrow,Incol)));
                        }
                    
                    
                    }
                
                }
            }
        }
    }
}




void PeriodicHamiltonian::SetVelocities(std::vector<Integer>  & rows, std::vector<Integer>  & cols, std::vector<scalar> & VXvalues, std::vector<scalar> & VYvalues){
    Integer MacroDeparture, MicroDeparture, MacroArrival,MicroArrival;
    Integer Inrow,Incol,stride;
    scalar I(0,1);
    stride = 0;
    for (auto i =0; i<base.getOnSiteEnergies().size(); i++)
        stride += base.getOnSiteEnergies()[i].cols();
    std::cout<<"the value of the stride is "<<stride<<std::endl;
    for(auto n1 =0 ; n1< Ncells[0];n1++){
        for(auto n2 = 0; n2<Ncells[1]; n2++){
            for(auto k = 0; k<base.getFromHoppings().size(); k++){
                //Here we must iterate over rows and cols
                for(auto l = 0; l<(base.getHoppingValues()[k].rows()*base.getHoppingValues()[k].cols()); l++){
                    MacroDeparture = ((n1+Ncells[0])%Ncells[0])*Ncells[1] + (n2+Ncells[1])%Ncells[1];
                    MacroArrival = ((n1+Ncells[0]+base.getHoppingDirections()[k][0])%Ncells[0])*Ncells[1] + (n2+Ncells[1]+base.getHoppingDirections()[k][1])%Ncells[1];
                    Inrow = l/base.getHoppingValues()[k].cols();
                    Incol = l%base.getHoppingValues()[k].cols();
                    if(std::fabs(base.getHoppingValues()[k](Inrow,Incol))<1e-10 || std::sqrt(base.getDeltaPos()[k][0]*base.getDeltaPos()[k][0] + base.getDeltaPos()[k][1]*base.getDeltaPos()[k][1])<1e-10){ }
                    else{
                        MicroDeparture = MacroDeparture*stride + base.getOrbitalMaps()[base.getFromHoppings()[k]] + Incol; 
                        MicroArrival = MacroArrival*stride + base.getOrbitalMaps()[base.getToHoppings()[k]] + Inrow;
                        if(MicroArrival < MicroDeparture){
                            cols.emplace_back(MicroDeparture);
                            rows.emplace_back(MicroArrival);
                            VXvalues.emplace_back(-I*base.getHoppingValues()[k](Inrow,Incol)*base.getDeltaPos()[k][0]);
                            VYvalues.emplace_back(-I*base.getHoppingValues()[k](Inrow,Incol)*base.getDeltaPos()[k][1]);
                            
                        }

                        else{
                            cols.emplace_back(MicroArrival);
                            rows.emplace_back(MicroDeparture);
                            VXvalues.emplace_back(std::conj(-I*base.getHoppingValues()[k](Inrow,Incol)*base.getDeltaPos()[k][0]));
                            VYvalues.emplace_back(std::conj(-I*base.getHoppingValues()[k](Inrow,Incol)*base.getDeltaPos()[k][1]));
                            
                        }
                    
                    
                    }
                
                }
            }
        }
    }
}




template <> void PeriodicHamiltonian::SetPeriodicHoppings<3>(std::vector<Integer> & rows, std::vector<Integer>  &cols, std::vector<scalar> & vals ){

    std::cout<<"Sorry: Not implemented right now"<<std::endl;


}

template <> void PeriodicHamiltonian::SetOnSitePotentials<3>(std::vector<Integer>& rows, std::vector<Integer> & cols , std::vector<scalar> & vals){
    std::cout<<"Sorry: Not implemented right now"<<std::endl;

}

template <> void PeriodicHamiltonian::SetOperator<3>(std::vector<Integer> & rows, std::vector<Integer> & cols, std::vector<scalar> & vals){
    std::cout<<"Sorry: Not implemented right now"<<std::endl;
}





template <> void PeriodicHamiltonian::SetOnSitePotentials<2>(std::vector<Integer>& rows, std::vector<Integer> & cols , std::vector<scalar> & vals){
    Integer MacroDeparture, MicroDeparture, MacroArrival,MicroArrival;
    Integer Inrow,Incol,stride;
    stride = 0;
    for (auto i =0; i<base.getOnSiteEnergies().size(); i++)
        stride += base.getOnSiteEnergies()[i].cols();
    std::cout<<"the value of the stride is "<<stride<<std::endl;
    for(auto n1 =0 ; n1< Ncells[0];n1++){
        for(auto n2 = 0; n2<Ncells[1]; n2++){
            for(auto k = 0; k<base.getOnSiteEnergies().size(); k++){
                //Here we must iterate over rows and cols
                for(auto l = 0; l<(base.getHoppingValues()[k].rows()*base.getHoppingValues()[k].cols()); l++){
                    MacroDeparture = ((n1+Ncells[0])%Ncells[0])*Ncells[1] + (n2+Ncells[1])%Ncells[1];
                    MacroArrival = ((n1+Ncells[0])%Ncells[0])*Ncells[1] + (n2+Ncells[1])%Ncells[1];
                    Inrow = l/base.getOnSiteEnergies()[k].cols();
                    Incol = l%base.getOnSiteEnergies()[k].cols();
                    if(std::fabs(base.getOnSiteEnergies()[k](Inrow,Incol))<1e-10){ }
                    else{
                        MicroDeparture = MacroDeparture*stride + base.getOrbitalMaps()[base.getOrbitalList()[k]] + Incol; 
                        MicroArrival = MacroArrival*stride + base.getOrbitalMaps()[base.getOrbitalList()[k]] + Inrow;
                        if(MicroArrival <= MicroDeparture){
                            cols.emplace_back(MicroDeparture);
                            rows.emplace_back(MicroArrival);
                            vals.emplace_back(base.getOnSiteEnergies()[k](Inrow,Incol));
                        }

                        else{
                            
                        }
                    
                    
                    }
                
                }
            }
        }
    }
}


//tengo que seguir con esto aca

template <> void PeriodicHamiltonian::SetOperator<2>(std::vector<Integer> & rows, std::vector<Integer> & cols, std::vector<scalar> & vals){
    std::ifstream f(base.getPath()+"/"+operatorName+".json");
    std::cout<<base.getPath()<<std::endl;
    if(!f){
        std::cout<<"Error: The file "+operatorName+".json cannot be found in the specified path... exiting\n"<<std::endl;
        exit(EXIT_FAILURE);
    }
    json opfile = json::parse(f);
    Integer count = 0;
    Integer nrows, ncols;
    std::vector<Float> onsiteReal;
    std::vector<Float> onsiteImag;
    std::vector<Matrix> OperatorElements;
    std::vector<std::string> Operatorlabels;
    for(auto it = opfile["Sublattices"].begin(); it!=opfile["Sublattices"].end(); it++){
        nrows = (*it)["Nrows"];;
        ncols = (*it)["Ncols"];
        onsiteReal = (std::vector<Float>)(*it)["OnsiteReal"];
        onsiteImag = (std::vector<Float>)(*it)["OnsiteImag"];
        Matrix H = Matrix::Zero(nrows, ncols);
        for(auto i = 0; i<nrows; i++){
            for(auto j = 0; j<ncols; j++){
                H(i,j) = scalar(onsiteReal[i*ncols+j],onsiteImag[i*ncols+j]);
            }
        }
        OperatorElements.emplace_back(H);
        Operatorlabels.emplace_back((*it)["Label"]);

    }
    
    //now we create the indices for the loading of the opeators
    Integer MacroDeparture, MicroDeparture, MacroArrival, MicroArrival;
    Integer stride;
    stride=0;
    for(auto i=0; i< base.getOnSiteEnergies().size(); i++)
        stride += base.getOnSiteEnergies()[i].cols();
    for(auto n1 = 0; n1<Ncells[0]; n1++){
        for(auto n2 =0; n2<Ncells[1]; n2++){
            for(auto k = 0; k<OperatorElements.size(); k++){                
                for(auto l = 0; l<(OperatorElements[k].rows()*OperatorElements[k].cols()); l++){
                    MacroDeparture = ((n1+Ncells[0])%Ncells[0])*Ncells[1] + (n2+Ncells[1])%Ncells[1];
                    MacroArrival = ((n1+Ncells[0])%Ncells[0])*Ncells[1] + (n2+Ncells[1])%Ncells[1];
                    nrows = l/base.getOnSiteEnergies()[k].cols();
                    ncols = l%base.getOnSiteEnergies()[k].cols();
                    if(std::fabs(OperatorElements[k](nrows, ncols))<1e-10){ }
                    else{
                        MicroDeparture = MacroDeparture*stride + base.getOrbitalMaps()[Operatorlabels[k]] + ncols; 
                        MicroArrival = MacroArrival*stride + base.getOrbitalMaps()[Operatorlabels[k]] + nrows;
                        if(MicroArrival <= MicroDeparture){
                            
                            cols.emplace_back(MicroDeparture);
                            rows.emplace_back(MicroArrival);
                            vals.emplace_back(OperatorElements[k](nrows,ncols));
                        }

                        else{
                            
                        }
                    }
                }
            }
        }
    }
    


    f.close();
}