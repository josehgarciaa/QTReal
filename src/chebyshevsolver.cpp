#include "chebyshevsolver.h"


namespace chebyshev{
    size_t SpectralMoments(SparseMat& OP, chebyshev::Moments1D& chebMoms, factory::generator& gen, size_t numstates){
        const size_t Dim = chebMoms.SystemSize();
        const size_t NumMoms = chebMoms.HighestMomentNumber();
        scalar dot;
        gen.NumberOfStates(numstates);
        gen.BlockSize(chebMoms.BlockSize());
        gen.SystemSize(Dim);
        
        while(gen.getQuantumState()){
            
            
            if(gen.BlockSize() ==1){
                if(OP.IsIdMat())
                    chebMoms.SetInitVectors(gen.Out());
                else
                    chebMoms.SetInitVectors(OP,gen.Out());
            }
            
            else if(gen.BlockSize() >1){
                if(OP.IsIdMat())
                    chebMoms.SetInitBlock(gen.Out());
                else
                    chebMoms.SetInitBlock(OP,gen.Out());
            }
            else{
                std::cerr<<"Error: The generator is giving an unitialized block variable... exiting..."<<std::endl;
                assert(false);
            }
            
            std::cout<<"The moment size is "<<chebMoms.BlockSize()<<std::endl;
            for(size_t m = 0; m<NumMoms; m++){
                Float scal = 2.0/(gen.NumberOfStates()*gen.BlockSize());
                if(m==0)
                    scal *= (Float)0.5;
                linalg::dot(gen.Out().size(),gen.Out(),chebMoms.ChebV0(), &dot);
                chebMoms(m) +=scal*dot;
                chebMoms.Iterate();
            }
          
        }
        return 0;
    };
    
    //Now we continue with the calculation of the moments. First we will implement the full calculation and then we will modify everything 
    size_t CorrelationExpansionMoments(SparseMat &OPL, SparseMat & OPR, chebyshev::Moments2D & chebmoms, factory::generator & gen, size_t num_states){
        const size_t Dim = chebmoms.SystemSize();
        const auto NumMoms= chebmoms.MomentNumber();
        
        gen.NumberOfStates(num_states);
        gen.BlockSize(chebmoms.BlockSize());
        gen.SystemSize(Dim);
        std::vector<scalar> Temp(chebmoms.BlockSize()*chebmoms.HamiltonianHoppings().rank(),0);
        chebyshev::Moments1D LeftOp;
        LeftOp.getMomentsParams(chebmoms);
        scalar dot;
        while(gen.getQuantumState()){
        //Now we initialize the vector to the left
           /* //Linea para debugar
            gen.Out() = std::vector<scalar>(gen.Out().size(),0);
            for(auto b = 0; b<gen.BlockSize(); b++)
                gen.Out()[0+b*chebmoms.HamiltonianHoppings().rank()] = 1.0;
            for(auto v : gen.Out())
                std::cout<<v<<std::endl;
            printf("Salio del print de debug\n");*/
            if(gen.BlockSize()>1){
                chebmoms.SetInitBlock(gen.Out());
                OPL.BlockMultiply(gen.Out(),Temp,gen.BlockSize());
                LeftOp.SetInitBlock(Temp);
                OPR.BlockMultiply(chebmoms.ChebV0(), Temp, gen.BlockSize());
            }
            else
                {
                    chebmoms.SetInitVectors(gen.Out());
                    OPL.Multiply(gen.Out(),Temp);
                    LeftOp.SetInitVectors(Temp);
                    OPR.Multiply(chebmoms.ChebV0(), Temp);
                }
            
            //m=0 n=0
            linalg::dot(Temp.size(), LeftOp.ChebV0(),Temp,&dot);
            chebmoms(0,0) += dot/(scalar)chebmoms.BlockSize();
            //m=1 n=0
            LeftOp.Iterate();
            linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot) ;
            chebmoms(1,0) += dot/(scalar)chebmoms.BlockSize();
            //Now we iterate over all the M's
            for(auto m=2; m<NumMoms[0]; m++){
                LeftOp.Iterate();
                linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot);
                chebmoms(m,0) += dot/(scalar)chebmoms.BlockSize();

            }
            
            chebmoms.Iterate();
            
            
            
            
            //Now we initialize the vectors to the left

            if(gen.BlockSize()>1){
                OPL.BlockMultiply(gen.Out(),Temp,gen.BlockSize());
                LeftOp.SetInitBlock(Temp);
                OPR.BlockMultiply(chebmoms.ChebV0(), Temp, gen.BlockSize());
            }
            else{
                OPL.Multiply(gen.Out(),Temp);
                LeftOp.SetInitVectors(Temp);
                OPR.Multiply(chebmoms.ChebV0(), Temp);
            }
                            
           
            
            //m=0 n=1
            linalg::dot(Temp.size(), LeftOp.ChebV0(),Temp,&dot);
            chebmoms(0,1) += dot/(scalar)chebmoms.BlockSize();
            //m=1 n=1
            LeftOp.Iterate();
            linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot) ;
            chebmoms(1,1) += dot/(scalar)chebmoms.BlockSize();
            //Now we iterate over all the M's
            for(auto m=2; m<NumMoms[0]; m++){
                LeftOp.Iterate();
                linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot);
                chebmoms(m,1) += dot/(scalar)chebmoms.BlockSize();
            }

            for(auto n2 = 2; n2<NumMoms[1]; n2++){
                chebmoms.Iterate();
                //Now we initialize the vectors to the left

                if(gen.BlockSize()>1){
                    OPL.BlockMultiply(gen.Out(),Temp,gen.BlockSize());
                    LeftOp.SetInitBlock(Temp);
                    OPR.BlockMultiply(chebmoms.ChebV0(), Temp, gen.BlockSize());
                }
                else{
                    OPL.Multiply(gen.Out(),Temp);
                    LeftOp.SetInitVectors(Temp);
                    OPR.Multiply(chebmoms.ChebV0(), Temp);
                }

                //m=0 n=n2
                linalg::dot(Temp.size(), LeftOp.ChebV0(),Temp,&dot);
                chebmoms(0,n2) += dot/(scalar)chebmoms.BlockSize();
                //m=1 n=n2
                LeftOp.Iterate();
                linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot) ;
                chebmoms(1,n2) += dot/(scalar)chebmoms.BlockSize();
                //Now we iterate over all the M's
                for(auto m=2; m<NumMoms[0]; m++){
                    LeftOp.Iterate();
                    linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot);
                    chebmoms(m,n2) += dot/(scalar)chebmoms.BlockSize();
                }
                
            }
                
        
        }
        
        
        for (int mL = 0 ; mL < chebmoms.MomentNumber()[0]; mL++)				  
	    for (int mR = mL; mR < chebmoms.MomentNumber()[1]; mR++)
	    {
            double scal=4.0*gen.BlockSize()/(gen.NumberOfStates()*gen.BlockSize());
		    
            if(mL==0)
                scal*=0.5;
            if(mR==0)
                scal*=0.5;
            const scalar tmp = scal*( chebmoms(mL,mR) + std::conj(chebmoms(mR,mL)) )	/2.0;
		chebmoms(mL,mR)= tmp;
		chebmoms(mR,mL)= std::conj(tmp);
	    }
	
        
        return 1;
    }



    size_t CorrelationExpansionMoments(SparseMat& OP1, SparseMat &OPL, SparseMat & OPR, chebyshev::Moments2D & chebmoms, factory::generator & gen, size_t num_states){
        const size_t Dim = chebmoms.SystemSize();
        const auto NumMoms= chebmoms.MomentNumber();
        
        gen.NumberOfStates(num_states);
        gen.BlockSize(chebmoms.BlockSize());
        gen.SystemSize(Dim);
        std::vector<scalar> Temp(chebmoms.BlockSize()*chebmoms.HamiltonianHoppings().rank(),0);
        std::vector<scalar> Temp2(chebmoms.BlockSize()*chebmoms.HamiltonianHoppings().rank(),0);
        chebyshev::Moments1D LeftOp;
        LeftOp.getMomentsParams(chebmoms);
        scalar dot;
        while(gen.getQuantumState()){
        //Now we initialize the vector to the left
            
            if(gen.BlockSize()>1){
                chebmoms.SetInitBlock(gen.Out());
                OPL.BlockMultiply(gen.Out(),Temp,gen.BlockSize());
                OP1.BlockMultiply(0.5,Temp,gen.BlockSize(),0.0,Temp2);
                OP1.BlockMultiply(gen.Out(),Temp,gen.BlockSize());
                OPL.BlockMultiply(0.5,Temp,gen.BlockSize(),1.0,Temp2);
                LeftOp.SetInitBlock(Temp2);
                OPR.BlockMultiply(chebmoms.ChebV0(), Temp, gen.BlockSize());
            }
            else
                {
                    chebmoms.SetInitVectors(gen.Out());
                    OPL.Multiply(gen.Out(),Temp);
                    OP1.Multiply(0.5,Temp,0.0,Temp2);
                    OP1.Multiply(gen.Out(),Temp);
                    OPL.Multiply(0.5,Temp,1.0,Temp2);
                    LeftOp.SetInitVectors(Temp2);
                    OPR.Multiply(chebmoms.ChebV0(), Temp);
                }
            
            //m=0 n=0
            linalg::dot(Temp.size(), LeftOp.ChebV0(),Temp,&dot);
            chebmoms(0,0) += dot/(scalar)chebmoms.BlockSize();
            //m=1 n=0
            LeftOp.Iterate();
            linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot) ;
            chebmoms(1,0) += dot/(scalar)chebmoms.BlockSize();
            //Now we iterate over all the M's
            for(auto m=2; m<NumMoms[0]; m++){
                LeftOp.Iterate();
                linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot);
                chebmoms(m,0) += dot/(scalar)chebmoms.BlockSize();

            }
            
            chebmoms.Iterate();
            
            
            
            
            //Now we initialize the vectors to the left

            if(gen.BlockSize()>1){
                OPL.BlockMultiply(gen.Out(),Temp,gen.BlockSize());
                OP1.BlockMultiply(0.5,Temp,gen.BlockSize(),0.0,Temp2);
                OP1.BlockMultiply(gen.Out(),Temp,gen.BlockSize());
                OPL.BlockMultiply(0.5,Temp,gen.BlockSize(),1.0,Temp2);
                LeftOp.SetInitBlock(Temp2);
                OPR.BlockMultiply(chebmoms.ChebV0(), Temp, gen.BlockSize());


            }
            else{
                OPL.Multiply(gen.Out(),Temp);
                OP1.Multiply(0.5,Temp,0.0,Temp2);
                OP1.Multiply(gen.Out(),Temp);
                OPL.Multiply(0.5,Temp,1.0,Temp2);
                LeftOp.SetInitVectors(Temp2);
                OPR.Multiply(chebmoms.ChebV0(), Temp);
            }
                            
           
            
            //m=0 n=1
            linalg::dot(Temp.size(), LeftOp.ChebV0(),Temp,&dot);
            chebmoms(0,1) += dot/(scalar)chebmoms.BlockSize();
            //m=1 n=1
            LeftOp.Iterate();
            linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot) ;
            chebmoms(1,1) += dot/(scalar)chebmoms.BlockSize();
            //Now we iterate over all the M's
            for(auto m=2; m<NumMoms[0]; m++){
                LeftOp.Iterate();
                linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot);
                chebmoms(m,1) += dot/(scalar)chebmoms.BlockSize();
            }

            for(auto n2 = 2; n2<NumMoms[1]; n2++){
                chebmoms.Iterate();
                //Now we initialize the vectors to the left

                if(gen.BlockSize()>1){
                    OPL.BlockMultiply(gen.Out(),Temp,gen.BlockSize());
                    OP1.BlockMultiply(0.5,Temp,gen.BlockSize(),0.0,Temp2);
                    OP1.BlockMultiply(gen.Out(),Temp,gen.BlockSize());
                    OPL.BlockMultiply(0.5,Temp,gen.BlockSize(),1.0,Temp2);
                    LeftOp.SetInitBlock(Temp2);
                    OPR.BlockMultiply(chebmoms.ChebV0(), Temp, gen.BlockSize());
                }
                else{
                    OPL.Multiply(gen.Out(),Temp);
                    OP1.Multiply(0.5,Temp,0.0,Temp2);
                    OP1.Multiply(gen.Out(),Temp);
                    OPL.Multiply(0.5,Temp,1.0,Temp2);
                    LeftOp.SetInitVectors(Temp2);
                    OPR.Multiply(chebmoms.ChebV0(), Temp);
                }

                //m=0 n=n2
                linalg::dot(Temp.size(), LeftOp.ChebV0(),Temp,&dot);
                chebmoms(0,n2) += dot/(scalar)chebmoms.BlockSize();
                //m=1 n=n2
                LeftOp.Iterate();
                linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot) ;
                chebmoms(1,n2) += dot/(scalar)chebmoms.BlockSize();
                //Now we iterate over all the M's
                for(auto m=2; m<NumMoms[0]; m++){
                    LeftOp.Iterate();
                    linalg::dot(Temp.size(),LeftOp.ChebV0(), Temp, &dot);
                    chebmoms(m,n2) += dot/(scalar)chebmoms.BlockSize();
                }
                
            }
                
        
        }
        for (int mL = 0 ; mL < chebmoms.MomentNumber()[0]; mL++)				  
	    for (int mR = mL; mR < chebmoms.MomentNumber()[1]; mR++)
	    {
            double scal=4.0*gen.BlockSize()/(gen.NumberOfStates()*gen.BlockSize());
		    
            if(mL==0)
                scal*=0.5;
            if(mR==0)
                scal*=0.5;
            const scalar tmp = scal*( chebmoms(mL,mR) + std::conj(chebmoms(mR,mL)) )	/2.0;
		chebmoms(mL,mR)= tmp;
		chebmoms(mR,mL)= std::conj(tmp);
	    }
	
        
        return 1;
    }



}


//Crear dentro de la clase de los momentos una nueva clase de vectores para poder manejar las iteraciones de forma mas eficiente