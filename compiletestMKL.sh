icpc *.cpp -c -I ../json/include/ -I /usr/include/eigen3/ -I /usr/include/mkl -std=c++17 -qopenmp -mkl=parallel -Wall -march=core-avx2 -fma -ftz -fomit-frame-pointer -Wextra -flto -O3

icpc *.o  -o test.out -std=c++17 -qopenmp -mkl=parallel -Wall -march=core-avx2 -fma -ftz -fomit-frame-pointer -Wextra -flto -O3
