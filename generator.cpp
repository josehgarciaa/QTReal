#include "statefactory.h"

bool factory::generator::getQuantumState(){
    if(this ->Count() < this -> NumberOfStates()){
        switch(this -> _kind){
            case LOCAL_STATE:
                linalg::scal(this->Out().size(),&zero,this->Out());
                this ->Out(this->Spos(_count)) = 1.0;
                break;
            case USER_STATE:
                std::cout<<"Using vetor at "<<this->Spos(_count)<<std::endl;
                #pragma omp parallel for schedule(static)
                for(auto v = 0; v<this->Out().size(); v++)
                    this ->Out()[v] = this ->Data()[v+this ->Spos()[_count]];
                break;
            case RANDOM_STATE:
                {
                    std::cout<<"entering into the random vector"<<std::endl;
                    const Float Norm = std::sqrt(this ->Out().size());
                    for(size_t v = 0; v<this->Out().size(); v++){
                        const Float phi = (Float)2.0*(Float)M_PI*(Float)rand()/(Float)RAND_MAX;
                        this ->Out()[v] = scalar(cos(phi)/Norm, sin(phi)/Norm);
                    }
                }
                break;

            case RANDOM_BLOCK:
                {
                    const Float Norm = std::sqrt(this-> Out().size()/this->BlockSize());
                    //this is for testing
                    //std::cout<<"entering hte random block"<<std::endl;
                    std::cout<<"the norm is :"<<Norm<<std::endl;
                    std::cout<<"the size of the outvector is :"<<this ->Out().size()<<std::endl;
                    for(size_t v = 0; v<this ->Out().size(); v++){
                        const Float phi = (Float)2.0*(Float)M_PI*(Float)rand()/(Float)RAND_MAX;
                        this ->Out()[v] = scalar(cos(phi)/Norm, sin(phi)/Norm);
                    }
                }
                break;
            
        }
        //std::cout<<"Estoy en el loop"<<std::endl;
        std::cout<<"The value of the count is "<<_count<<std::endl;
        _count++;
        //std::cout<<this ->NumberOfStates()<<","<<this -> Count()<<std::endl;
        std::cout<<"The state of the system is "<<true<<std::endl;
        return true;
    }
    std::cout<<"The state of the count is "<<_count<<std::endl;
    std::cout<<"And the state of the funciton is "<<false<<std::endl;
    printf("salio\n");
    return false;
}


void factory::readLocalState(std::ifstream & file, factory::generator & data){
    size_t ibuff;
    file >>ibuff;
    std::cout<<"Number of States "<<ibuff<<std::endl;
    data.NumberOfStates(ibuff);
    file >>ibuff;
    std::cout<<" System Size "<<ibuff<<std::endl;
    data.SystemSize(ibuff);
    file >>ibuff;
    std::cout<<" Block Size "<<ibuff<<std::endl;
    data.BlockSize(ibuff);
    if(data.Out().size() != data.BlockSize()*data.SystemSize()){
        std::cout<<" OutSize "<<data.BlockSize()*data.SystemSize()<<std::endl;
        data.Out()=std::move(std::vector<scalar>(data.BlockSize()*data.SystemSize()));
    }
    for(size_t i = 0; i<data.NumberOfStates(); i++){
        size_t ibuff;
        file>>ibuff;
        data.Spos()[i] = ibuff;
    }
    assert(data.Spos().size() == data.NumberOfStates());
    return;
}



void factory::readCustomState(std::ifstream & file, factory::generator & data){
    size_t ibuff;
    file >>ibuff;
    std::cout<<"Number of States "<<ibuff<<std::endl;
    data.NumberOfStates(ibuff);
    file >>ibuff;
    std::cout<<" System Size "<<ibuff<<std::endl;
    data.SystemSize(ibuff);
    file >>ibuff;
    std::cout<<" Block Size "<<ibuff<<std::endl;
    data.BlockSize(ibuff);
    file >>ibuff;
    std::cout<<" OutSize "<<ibuff<<std::endl;
    data.Out() = std::move(std::vector<scalar>(ibuff));
    const size_t size = data.SystemSize();
    const size_t num = data.NumberOfStates();
    Float re, im;
    data.Data() = std::vector<scalar>(size*num);
    for(size_t i = 0; i<num; i++){
        for(size_t j = 0; j<size; j++){
            file >> re>>im;
            data.Data(i*size+j) = scalar(re,im);
        }
        data.Spos(i)=i*size;
        std::cout<<"Added a vector at "<<data.Spos()[i]<<" = "<<i*size<<std::endl;
    }

    return;
}

factory::generator CreateStateSet(std::ifstream & file, std::string kind){
    factory::generator data;
    data.Kind(factory::String2State(kind));
    switch(data.Kind()){
        case(LOCAL_STATE):
            factory::readLocalState(file, data);
            break;
        case(USER_STATE):
            factory::readCustomState(file, data);
            break;
    }
    return data;
}

factory::generator LoadState(std::string filename){
    std::ifstream infile(filename);
    std::string kind("random_phase");
    if(infile.good())
        infile>>kind;
    else
        std::cout<<"State File not found, using the random phase fallback"<<std::endl;
    std::cout<<"Building "<<kind<<" states"<<std::endl;
    auto states = CreateStateSet(infile, kind);
    std::cout<<"Success"<<std::endl;
    states.StateLabel(filename);
    return states;
}

size_t factory::FillWithRandomPhase(std::vector<scalar>& ptr){
    const Integer size = ptr.size();
    size_t kpm_seed = time(NULL);
    if(getenv("KPM_SEED"))
        kpm_seed = std::stoi(std::string(getenv("KPM_SEED")));
    srand(kpm_seed);
    std::cout<<"Current seed is "<<kpm_seed<<std::endl;
    const Float norm = std::sqrt(size);
    size_t i = 0;
    for(size_t v = 0; v<size; v++ ){
        const auto phi = (Float)2.0*(Float)M_PI*(Float)rand()/(Float)RAND_MAX;
        ptr[v] = scalar(cos(phi)/norm,sin(phi)/norm);
        if(i<10)
            std::cout<<ptr[v].real()<<" "<<ptr[v].imag()<<std::endl;
        i++;
    }
    return 0;

}




