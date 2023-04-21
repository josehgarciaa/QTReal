g++ *.cpp -c -I ../json/include/ -I /usr/include/eigen3/ -I /usr/include/mkl -std=c++17 -fopenmp -O3 -Wall -flto -pg
g++ *.o  -o test.out -std=c++17 -fopenmp -O3 -Wall -lpthread -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -ldl -lm -flto -pg
