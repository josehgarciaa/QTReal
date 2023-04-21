#ifndef STATEFACTORY
#define STATEFACTORY

#include "typedef.h"
#include <iostream>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include "mklblas.h"
#include "sparsematrix.h"

enum StateType{
    LOCAL_STATE =0, USER_STATE, RANDOM_STATE, RANDOM_BLOCK
};

namespace factory{

    class generator{
        private:
            size_t _count, _dim, _num_states, _bsize;
            StateType _kind;
            std::string _label;
            const scalar zero= scalar(0,0);
            std::vector<size_t> _spos;
            std::vector<scalar> _out;
            std::vector<scalar> _data;
        public:
            generator () :_count(0),_dim(0),_num_states(0),_bsize(0),_kind(RANDOM_STATE),_label("default"){srand(time(NULL));};

            inline size_t SystemSize() const {return _dim;};

            inline size_t NumberOfStates() const {return _num_states;};

            inline size_t BlockSize() const {return _bsize;};

            inline size_t Count() const {return _count;};

            inline std::string StateLabel() const {return _label;};

            inline StateType Kind() const {return _kind;};


            inline void NumberOfStates(const size_t n){
                this ->_num_states = n;
                this ->_spos = std::move(std::vector<size_t>(n,0));
                return;
            }

            inline void BlockSize(const size_t bsize){
                this -> _bsize = bsize;
                if(this ->_num_states==0){
                    std::cerr<<"The number of states must not be zero to get a reasonable result"<<std::endl;
                    assert(false);
                }
                if(_bsize>1)
                    _kind = RANDOM_BLOCK;
                else if(_bsize ==1)
                    _kind = RANDOM_STATE;
                else{
                    std::cerr<<"The random block cannot have size leq than zero"<<std::endl;
                    assert(false);
                }
                return ;
            }
            
            inline void Count(const size_t newcount){
                this -> _count = newcount;
                return;
            }

            inline void StateLabel(std::string label){
                this -> _label = label;
                return;
            }

            inline void ResetCount(){
                this -> _count = 0;
                return;
            }


            inline std::vector<scalar>& Out(){return _out;};
            inline std::vector<scalar>& Data(){return _data;};
            inline std::vector<size_t>& Spos(){return _spos;};

            inline scalar& Out(const size_t i){return _out[i];};
            inline scalar& Data(const size_t i){return _data[i];};
            inline size_t& Spos(const size_t i){return _spos[i];};

            inline void Kind(StateType kind) {_kind = kind;};
            //En esta funcion, deberia haber una forma para controlar como se crean los bloques, en el caso
            //del vector block lo que hay que hacer es crear un solo momento, pero mantener el contador pra 
            //llevar el track de todo
            void SystemSize(const size_t n){
                if(SystemSize() ==0){
                    this -> _dim = n;
                    
                    //Nueva adicion, pero entonces con esto podemos ver que pasa
                    if (this ->_bsize>1){
                        std::cout<<"We are entering into the block size addition"<<std::endl;
                        printf("clearing spos\n");
                        this ->Spos() = std::move(std::vector<size_t>(1));
                        this ->Spos(0) = 99999; //this is a flag to set everything to zero
                        printf("Finished clearing\n");
                        this -> Out() = std::move(std::vector<scalar>(this->BlockSize()*n));
                    }
                    
                    else{
                        std::cout<<"the block size is one, therefore we are falling back to the random vector mode"<<std::endl;
                         printf("clearing spos\n");
                        this ->Spos() = std::move(std::vector<size_t>(1));
                        this ->Spos(0) = 99999; //this is a flag to set everything to zero
                        printf("ended the clearance of spos\n");
                        _out = std::move(std::vector<scalar>(this->BlockSize()*n));

                    }
                    return ;
                }
                assert(this ->SystemSize() == this -> Out().size()/this->BlockSize());
                return;
            }

            bool getQuantumState();
            
            
    };

    inline StateType String2State(std::string kind){
        if(kind =="local")
           return LOCAL_STATE;
        if(kind =="vector")
            return USER_STATE;
        if(kind == "randomvec")
            return RANDOM_STATE;
        if(kind == "randomblock")
            return RANDOM_BLOCK;
        else
            return RANDOM_STATE;
    }

    void readLocalState(std::ifstream & file, factory::generator & data);
    void readCustomState(std::ifstream & file, factory::generator& data);
    generator CreateStateSet(std::ifstream& file, std::string kind);
    generator LoadStateFile(std::string filename);
    size_t FillWithRandomPhase(std::vector<scalar>& vec);
    
    
    

    
}


#endif