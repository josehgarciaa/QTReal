cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)


# Set the URL of the GitHub repository containing the header files
set(REPO_URL "https://github.com/nlohmann/json")
set(SUBDIR "include/nlohmann")
set(DESTINATION ${PROJECT_SOURCE_DIR}/external/nlohmann/)
set(HEADERS  "json.hpp")

# Download each header file using the file(DOWNLOAD ...) command
file(DOWNLOAD ${REPO_URL}/raw/develop/${SUBDIR}/${HEADERS} ${DESTINATION}/${HEADERS})


# Include the directory containing the downloaded header files in your project's include path
include_directories(${DESTINATION})





#Include different modules to provide additional functionalities
# for instance FFT
#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Modules)


project(QTReal)

# Set the default header search path
include_directories(${CMAKE_SOURCE_DIR}/include/)

# Set the default source directory
set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)


find_package(Eigen3 REQUIRED)
if(${EIGEN3_FOUND})
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Error: Eigen is required for the compilation of the code\n")
endif()

find_package(OpenMP)
message(STATUS "OpenMP: ${OPENMP_FOUND}")
if(${OPENMP_FOUND})
    set(CODING_FLAGS "-Wall -Wextra")
    set(CMAKE_C_FLAGS "${CODING_FLAGS} -O3 ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CODING_FLAGS} -O3 ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
    message(FATAL_ERROR "Error: OpenMP is requrired for the compilation\n")
endif()

#Setting the MKL libraries
#set(BLA_VENDOR Intel10_64lp)
#find_package(BLAS)
#message(STATUS "INTEL MKL: ${BLAS_FOUND}")
#if(${BLAS_FOUND})
#    set(MKL_INC "${MKLROOT}/include")
#    include_directories(${MKL_INC})
#    message(STATUS "The blas path is ${MKL_INC}")
#else()
#    message(FATAL_ERROR "Error: The MKL libraries are essential for the code... exiting \n")
#endif()
#include_directories("/usr/include/mkl")

##Libraries

#Build the Real-Space Hamiltonian library
set(SOURCES "unitary.cpp" "periodichamiltonian.cpp")
list(TRANSFORM SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set(RSHAM rsham)
add_library(${RSHAM} STATIC ${SOURCES})
target_include_directories(${RSHAM} PUBLIC external/json/include)

#Build the Sparse Matrix Library
set(SOURCES "sparsematrix.cpp")
list(TRANSFORM SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set(MKLSP mklsp)
add_library(${MKLSP} STATIC ${SOURCES})
target_link_libraries(${MKLSP} ${BLAS_LIBRARIES})

#Build the MKL blas library
set(SOURCES "mklblas.cpp")
list(TRANSFORM SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set(MKLBLAS mklblas)
add_library(${MKLBLAS} STATIC ${SOURCES})
target_link_libraries(${MKLBLAS} ${BLAS_LIBRARIES})

#Build the KPM library
set(SOURCES "auxfunc.cpp" "moments.cpp" "moments1D.cpp" "moments2D.cpp" "momentsNEq.cpp" "momentsTD.cpp" "chebyshevsolver.cpp" "vectorfilter.cpp")
list(TRANSFORM SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set(KPMLIB kpmlib)
add_library(${KPMLIB} STATIC ${SOURCES})

#Build the vector factory librar
set(SOURCES "generator.cpp" "mklblas.cpp")
list(TRANSFORM SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set(VECFACTORY vectorfactory)
add_library(${VECFACTORY} STATIC ${SOURCES})
target_link_libraries(${VECFACTORY} ${BLAS_LIBRARIES})













### Tests
set(DLLINK dl)
set(MAIN TestDOSGrapheneSpinful)
add_executable(${MAIN} src/testDOS.cpp)
target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})

