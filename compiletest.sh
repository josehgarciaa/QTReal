#g++ *.cpp -c -I ../json/include/ -I /usr/include/eigen3/ -I /usr/include/mkl -std=c++17 -fopenmp -O3 -Wall -flto
g++ *.cpp -c -I ../json/include/ -I /usr/include/eigen3/ -I /usr/include/mkl -std=c++17 -fopenmp -O3 -flto
g++ *.o  -o test.out -std=c++17 -fopenmp -O3 -lpthread -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -ldl -lm -flto
#g++ *.o  -o test.out -std=c++17 -fopenmp -O3 -Wall -lpthread -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -ldl -lm -flto
