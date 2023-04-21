#ifndef MOMENTS
#define MOMENTS

#include "typedef.h"
#include <string>
#include <cstdlib>
#include <array>
#include <cassert>
#include <limits>
#include <fstream>
#include "sparsematrix.h"
#include "mklblas.h"
#include "auxfunc.h"

namespace chebyshev{
    const Float CUTOFF = 0.99;
    const Float Boltzmann = 8.617333262145E-5; //eV/K
    const  Float Hbar = 0.6582119624; //electronvolts . fs
    class Moments{
        private:
            size_t _system_size, _bsize;
            Float _band_width, _band_center;
            std::string _system_label;
            std::vector<scalar> _ChebV0;
            std::vector<scalar> _ChebV1;
            std::vector<scalar> _OPV;
            std::vector<scalar> _Mu;
            SparseMat * _ptrHam;
            SparseMat * _ptrHOnsite;
            //aca hay que poner un array par alas vacnacies
        public:
            Moments() : _ptrHam(NULL), _ptrHOnsite(NULL), _system_label(""), _system_size(0), _bsize(0), _band_width(0), _band_center(0){};
            inline size_t SystemSize() const {return _system_size;};
            inline std::string SystemLabel() const {return _system_label;};
            inline Float BandWidth() const {return _band_width;};
            inline Float HalfWidth() const {return _band_width*(Float)0.5;};
            inline Float BandCenter() const {return _band_center;};
            inline Float ScaleFactor() const {return chebyshev::CUTOFF/HalfWidth();};
            inline Float ShiftFactor() const {return -BandCenter()/HalfWidth()/chebyshev::CUTOFF;};
            inline size_t BlockSize() const {return _bsize;};
            inline std::vector<scalar> & MomentVector(){return _Mu;};
            inline scalar & MomentVector(const size_t i){return _Mu[i];};
            inline std::vector<scalar>& ChebV0() {return _ChebV0;};
            inline std::vector<scalar>& ChebV1() {return _ChebV1;};
            inline std::vector<scalar>& OPV() {return _OPV;};
            inline SparseMat& HamiltonianHoppings(){
                return *_ptrHam;
            };

            inline SparseMat& HamiltonianOnSite(){
                return *_ptrHOnsite;
            };

            inline void SystemSize(const size_t dim){_system_size = dim;};
            inline void SystemLabel(std::string label){_system_label = label;};
            inline void BandWidth(const Float x){_band_width = x;};
            inline void BandCenter(const Float x){_band_center = x;};
            inline void MomentVector(const std::vector<scalar> & mu){_Mu = mu;};
            inline void BlockSize(const size_t bsize){_bsize = bsize;};
            inline void SetHamiltonianHoppings(SparseMat& NHam){
                if(this -> SystemSize() == 0)
                    this -> SystemSize(NHam.rank());
                assert(NHam.rank() == this -> SystemSize());
                _ptrHam = &NHam;
            };

            inline void SetHamiltonianOnSite(SparseMat& NOnsite){
                if(this -> SystemSize() == 0 && NOnsite.NumNNZ()!=0){
                    this -> SystemSize(NOnsite.rank());
                    assert(NOnsite.rank() == this -> SystemSize());
                }
                else { }
                _ptrHOnsite = &NOnsite;
            };

            size_t Rescale2ChebysevDomain(){
                this -> HamiltonianHoppings().Rescale(this -> ScaleFactor(), this ->ShiftFactor());
                if(this -> HamiltonianOnSite().NumNNZ() !=0)
                    this -> HamiltonianOnSite().Rescale(this -> ScaleFactor(), 0.0);
                return 0;
                
            }

            inline void SetAndRescaleHamiltonian(SparseMat & Ham, SparseMat &HOnsite){
                this -> SetHamiltonianHoppings(Ham);
                this ->SetHamiltonianOnSite(HOnsite);
                this -> Rescale2ChebysevDomain();
                
            };

            void SetInitVectors(const std::vector<scalar> & T0);
            void SetInitBlock(const std::vector<scalar> & T0);
            void SetInitVectors(SparseMat & Op, const std::vector<scalar> &T0);
            void SetInitBlock(SparseMat & Op, const std::vector<scalar> & T0);

            size_t Iterate();
            size_t JacksonKernelMomCutOff(const Float broad);
            Float JacksonKernel(const Float m, const Float Mom);

            void getMomentsParams(Moments& mom){
                this -> SetHamiltonianHoppings(mom.HamiltonianHoppings());
                this -> SetHamiltonianOnSite(mom.HamiltonianOnSite());
                this -> SystemLabel(mom.SystemLabel());
                this -> BandWidth(mom.BandWidth());
                this -> BandCenter(mom.BandCenter());
                this -> BlockSize(mom.BlockSize());
            };

    };

    class Moments1D : public Moments{
        private:
            size_t _numMoms;
        public:
            Moments1D():_numMoms(0){};
            Moments1D(const size_t m0): _numMoms(m0){this -> MomentVector(std::vector<scalar>(_numMoms,0));};
            Moments1D(const size_t m0, const size_t bsize) :_numMoms(m0){
                this -> BlockSize(bsize);
                this -> MomentVector(std::vector<scalar> (_numMoms,0));
            };
            Moments1D(const std::string momfilename);
            inline size_t MomentNumber() const {return _numMoms;};
            inline size_t HighestMomentNumber() const {return _numMoms;};
            inline scalar & operator()(const size_t m0){return this -> MomentVector(m0);};
            void MomentNumber(const size_t numMoms);
            void saveIn(const std::string filename);
            void ApplyJacksonKernel(const Float broad);
            void Print();
    };

    class Moments2D : public Moments{
        private:
            std::array<size_t, 2> _numMoms;
        public:
            Moments2D(): _numMoms({0,0}){};
            Moments2D(const size_t m0, const size_t m1):_numMoms({m0,m1}){
                this -> BlockSize(1);
                this -> MomentVector(std::vector<scalar>(m0*m1,0));
            };

            Moments2D(const size_t m0, const size_t m1, const size_t bsize):_numMoms({m0,m1}){
                this -> BlockSize(bsize);
                this -> MomentVector(std::vector<scalar>(m0*m1,0));
            };

            Moments2D(std::string momfilename);
            Moments2D(Moments2D mom, const size_t m0, const size_t m1){
                this -> getMomentsParams(mom);
                this -> _numMoms = {m0,m1};
                this -> MomentVector(std::vector<scalar>(_numMoms[0]*_numMoms[1],0));
            };
            
            inline std::array<size_t, 2> MomentNumber() const {return _numMoms;};
            inline size_t HighestMomentNumber(const Integer i) const {return _numMoms[i];};
            inline size_t HighestMomentNumber()const {return (_numMoms[1]>_numMoms[0])?_numMoms[1]:_numMoms[0];};
            void MomentNumber(const size_t mom0, const size_t mom1);

            inline scalar& operator()(const size_t m0, const size_t m1){return this -> MomentVector(m0*_numMoms[1]+ m1);};

            void ApplyJacksonKernel(const Float b0, const Float b1);

            void saveIn(std::string filename);

            void AddSubMatrix(Moments2D& sub, const size_t mL, const size_t mR){
                for(size_t m0 = 0; m0<sub.HighestMomentNumber(0); m0++){
                    for(size_t m1 = 0; m1<sub.HighestMomentNumber(1); m1++)
                        this -> operator()(mL+m0, mR+m1) = sub(m0, m1);
                }
            }

            void Print();


    };


    class MomentsTD: public Moments{
        private:
            Float _dt;
            size_t _numMoms, _maxTimeStep, _timeStep;
        public:
            MomentsTD() : _numMoms(0), _maxTimeStep(0), _timeStep(0), _dt(0){this -> BlockSize(0);};
            MomentsTD(const size_t times, const size_t moments) : _numMoms(moments), _maxTimeStep(times), _timeStep(0), _dt(0){
                this ->BlockSize(1);
                this ->MomentVector(std::vector<scalar>(times*moments,0));
            };

            MomentsTD(const size_t times, const size_t moments, const size_t bsize) : _numMoms(moments), _maxTimeStep(times), _timeStep(0), _dt(0){
                this -> BlockSize(bsize);
                this ->MomentVector(std::vector<scalar>(times*moments,0));
            };

            MomentsTD(std::string momfilename);

            inline size_t MomentNubmer() const{ return _numMoms;};
            inline size_t HighestMomentNumber() const {return _numMoms;};
            inline size_t CurrentTimeStep() const {return _timeStep;};
            inline size_t MaxTimeStep() const {return _maxTimeStep;};
            inline Float TimeDiff() const {return _dt;};
            inline Float ChebyshevFreq() const { return this -> HalfWidth()/chebyshev::CUTOFF/chebyshev::Hbar;};
            inline Float ChebyshevFreq0() const {return this -> BandCenter()/chebyshev::Hbar;};
            void MomentNumber(const size_t mom);
            void MaxTimeStep(const size_t maxTimeStep){_maxTimeStep = maxTimeStep;};
            inline void IncreaseTimeStep(){_timeStep++;};
            inline void ResetTime(){_timeStep=0;};
            inline void TimeDiff(const Float dt){_dt = dt;};
            size_t Evolve(std::vector<scalar> & phi);
            inline scalar& operator()(const size_t timestep, const size_t moments){
                return this -> MomentVector(timestep*(this->HighestMomentNumber())+moments);
            };

            void ApplyJacksonKernel(const Float broad);
            void saveIn(std::string filename);
            void Print();
    };


    //Faltan agregar las classes de momentos neq y la clase del preprocessing
    class MomentsNEq: public Moments{
        private:
            SparseMat * _ptrForceMat;
            size_t _numMoms, _maxTimeStep, _timeStep, _numMeasurements, _evolutionsXMeasurement;
            Float _dt;
        public:
            MomentsNEq(): _numMoms(1), _maxTimeStep(1), _timeStep(0), _numMeasurements(0), _evolutionsXMeasurement(0), _dt(0){this -> BlockSize(1);};
            MomentsNEq(const size_t times, const size_t moments, const size_t evolutionsxmeasurement) : _numMoms(moments), _maxTimeStep(times),_evolutionsXMeasurement(evolutionsxmeasurement), _numMeasurements(times/evolutionsxmeasurement), _timeStep(0), _dt(0){
                assert(_maxTimeStep%_evolutionsXMeasurement == 0);
                this -> BlockSize(1);
                this -> MomentVector(std::vector<scalar>(_numMoms*(_numMeasurements+1),0));
            };
            MomentsNEq(const size_t times, const size_t moments, const size_t evolutionsxmeasurement, size_t bsize) : _numMoms(moments), _maxTimeStep(times),_evolutionsXMeasurement(evolutionsxmeasurement), _numMeasurements(times/evolutionsxmeasurement), _timeStep(0), _dt(0){
                assert(_maxTimeStep%_evolutionsXMeasurement == 0);
                this -> BlockSize(bsize);
                this -> MomentVector(std::vector<scalar>(_numMoms*(_numMeasurements+1),0));
            };

            MomentsNEq(std::string momfilename);

            inline size_t MomentNumber() const {return _numMoms;};
            inline size_t HighestMomentNumber() const {return _numMoms;};
            inline size_t CurrentTimeStep() const {return _timeStep;};
            inline size_t MaxTimeStep()  const {return _maxTimeStep;};
            inline size_t NumMeasurements() const {return _numMeasurements;};
            inline size_t EvolutionsxMeasurements() const {return _evolutionsXMeasurement;};
            inline Float TimeDiff() const {return _dt;};
            inline Float ChebyshevFreq() const {return this -> HalfWidth()/chebyshev::CUTOFF/chebyshev::Hbar;};
            inline Float ChebyshevFreq0() const {return this -> BandCenter()/chebyshev::Hbar;};
            void MomentNumber(const size_t mom);
            void MaxTimeStep(const size_t maxtimestep){_maxTimeStep = maxtimestep;};
            inline void IncreaseTimeStep(){_timeStep++;};
            inline void NumMeasurements(const size_t measurements){_numMeasurements = measurements;};
            inline void EvolutionsxMeasurements(const size_t evolutionsxmeasurements){_evolutionsXMeasurement = evolutionsxmeasurements;};
            inline void ResetTime(){_timeStep = 0;};
            inline void TimeDiff(const Float dt){_dt = dt;};
            size_t Evolve(std::vector<scalar> & phi);
            inline scalar & operator() (const size_t measurement, const size_t moments){
                return this -> MomentVector(measurement*HighestMomentNumber() + moments);
            };
            void ApplyJacksonKernel(const Float broad);
            void saveIn(std::string filename);
            void Print();
            inline void SetForceMat(SparseMat & FMat){
                if(this -> SystemSize() ==0)
                    this -> SystemSize(FMat.rank());
                    assert(FMat.rank() == this -> SystemSize() && FMat.mkl_descr().type == SPARSE_MATRIX_TYPE_GENERAL);
                    _ptrForceMat = & FMat;
            };
    };
enum FilterType{fermi=0, hole};

class VectorFilter : public Moments {
    private:
        size_t _Sample, _numMoms;
        FilterType _filter;
        std::array<Float,2> _filterpar; //First Mu, Temp

    public:
        VectorFilter():_filter(fermi),_filterpar({0,0}),_numMoms(0){};
        VectorFilter(FilterType filter, std::array<Float,2> par,size_t Mom): _filter(filter), _filterpar(par),_numMoms(Mom){
            this -> MomentVector() = std::move(std::vector<scalar>(Mom,0));
        };
        inline FilterType Filter() const {return _filter;};
        inline std::array<Float,2> & getParams(){return _filterpar;};
        inline void setFilterParams (const std::array<Float, 2> & par){ _filterpar = par;};
        inline void setFilterType(FilterType ftype){_filter = ftype;};
        inline void setMomentNumber(size_t M){_numMoms = M;};
        inline size_t MomentNumber() const {return _numMoms;};
        inline size_t HighestMomentNumber() const {return _numMoms;};
        void FillCoef();
        void PrepareOutVec();
        void PrintFilterParameters();
        void ApplyJacksonKernel(const Float broad);
        inline scalar& operator()(const size_t m0){return this -> MomentVector(m0);};
        void FinishVector(std::vector<scalar> & V){
            assert((V.size()==this->ChebV0().size()) && V.size() == this -> OPV().size());
            V.swap(this ->OPV());
            //We start the vector with zero and do everything with 2
        }




};





};




#endif
