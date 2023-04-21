cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)


#Include different modules to provide additional functionalities
# for instance FFT
#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Modules)


project(QTReal)


# find_package(Eigen3 REQUIRED)
# message(STATUS "Eigen3: ${EIGEN3_FOUND}")
# if(${EIGEN3_FOUND})
#     include_directories(${EIGEN3_INCLUDE_DIR})
# else()
#     message(FATAL_ERROR "Error: Eigen is required for the compilation of the code\n")
# endif()

# set(DLLINK dl)
# find_package(OpenMP)
# message(STATUS "OpenMP: ${OPENMP_FOUND}")
# if(${OPENMP_FOUND})
#     set(CODING_FLAGS "-Wall -Wextra")
#     set(CMAKE_C_FLAGS "${CODING_FLAGS} -O3 ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CODING_FLAGS} -O3 ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# else()
#     message(FATAL_ERROR "Error: OpenMP is requrired for the compilation\n")
# endif()

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


#find_package(PNG REQUIRED)
#message(STATUS "PNG LIB: ${PNG_FOUND}")
#if(${PNG_FOUND})
#    include_directories(${PNG_INCLUDE_DIRS})
#    message(STATUS "The PNG path is ${PNG_INCLUDE_DIRS}")
#else()
#    message(FATAL_ERROR "Error: The PNG libraries are required... exiting\n")
#endif()

#find_package(JPEG REQUIRED)
#message(STATUS "JPEG LIB: ${JPEG_FOUND}")
#if(${JPEG_FOUND})
#    include_directories(${JPEG_INCLUDE_DIRS})
#    message(STATUS "The JPEG library is in ${JPEG_INCLUDE_DIRS}")
#else()
#    message(FATAL_ERROR "Error: The JPEG libraries are required... exiting \n")
#endif()

#find_package(FFTW3 REQUIRED)
#message(STATUS "The FFTW3 path is ${FFTW3_INCLUDE_DIR}")
#if(${FFTW3_FOUND})
#    include_directories(${FFTW3_INCLUDE_DIR})
#    message(STATUS "The FFTW3 path is ${FFTW3_INCLUDE_DIR}")
#else()
#    message(FATAL_ERROR "Error: The FFTW3 libraries are required... exiting\n")
#endif()




###Libraries
#set(RSHAM_SOURCES "unitary.cpp" "periodichamiltonian.cpp")
#set(RSHAM rsham)
#add_library(${RSHAM} STATIC ${RSHAM_SOURCES})
#target_include_directories(${RSHAM} PUBLIC external/json/include)


#set(MKLSP_SOURCES "sparsematrix.cpp")
#set(MKLSP mklsp)
#add_library(${MKLSP} STATIC ${MKLSP_SOURCES})
#target_link_libraries(${MKLSP} ${BLAS_LIBRARIES})

#set(MKLBLAS_SOURCES "mklblas.cpp")
#set(MKLBLAS mklblas)
#add_library(${MKLBLAS} STATIC ${MKLBLAS_SOURCES})
#target_link_libraries(${MKLBLAS} ${BLAS_LIBRARIES})

#set(KPMLIB_SOURCES "auxfunc.cpp" "moments.cpp" "moments1D.cpp" "moments2D.cpp" "momentsNEq.cpp" "momentsTD.cpp" "chebyshevsolver.cpp" "vectorfilter.cpp")
#set(KPMLIB kpmlib)
#add_library(${KPMLIB} STATIC ${KPMLIB_SOURCES})

#set(VECFACTORY_SOURCES "generator.cpp" "mklblas.cpp")
#set(VECFACTORY vectorfactory)
#add_library(${VECFACTORY} STATIC ${VECFACTORY_SOURCES})
#target_link_libraries(${VECFACTORY} ${BLAS_LIBRARIES})

#set(SPATIAL_SOURCES "sites.cpp" "spatial.cpp")
#set(SPATIAL spatialgrid)
#add_library(${SPATIAL} STATIC ${SPATIAL_SOURCES})

#Couple library
#add_subdirectory(lmp-ompicouple)

#message(STATUS "The home file is in $ENV{HOME}")

### Tests

#set(MAIN TestDOSGrapheneSpinful)
#add_executable(${MAIN} testDOS.cpp)
#target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})

#set(MAIN TestConductivityXYHaldane)
#add_executable(${MAIN} testConductivityXY.cpp)
#target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})

#set(MAIN TestSpinConductivityXYKaneMele)
#add_executable(${MAIN} testOperatorConductivityXY.cpp)
#target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})


#set(MAIN TestOrbitalRelaxationTMD)
#add_executable(${MAIN} testTimeEvolutionOrbitalTMD.cpp)
#target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})


#set(MAIN TestSpinRelaxationGrapheneNeq)
#add_executable(${MAIN} testTimeEvolutionSpinGrapheneNEQ.cpp)
#target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})


#set(MAIN TestSpatialGrid)
#add_executable(${MAIN} testSPATIALDOS.cpp ${CMAKE_SOURCE_DIR}/grapheneGenerator.h)
#target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})

#set(MAIN TestSpatialFiniteGrid)
#add_executable(${MAIN} testSPATIALFiniteDOS.cpp ${CMAKE_SOURCE_DIR}/grapheneGenerator.h)
#target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})

#set(MAIN TestSpatialGModeGraphene)
#add_executable(${MAIN} testSPATIALTimeEv.cpp ${CMAKE_SOURCE_DIR}/grapheneGenerator.h)
#target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})


#set(MAIN TestSpatialOpticalGraphene)
#add_executable(${MAIN} testSPATIALTimeEvOptical.cpp ${CMAKE_SOURCE_DIR}/grapheneGenerator.h)
#target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})


#set(MAIN TestSpatialPseudoOpticalGraphene)
#add_executable(${MAIN} testSPATIALTimeEvOpticalPseudoSpin.cpp ${CMAKE_SOURCE_DIR}/grapheneGenerator.h)
#target_link_libraries(${MAIN} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL} ${DLLINK})

#set(MAIN TestCoupleLMP)
#add_executable(${MAIN} testcouplelammps.cpp)
#target_link_libraries(${MAIN} PUBLIC lmp-ompicouple ${LMPLIB} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${FFTW3_LIBRARIES} ${DLLINK})
#target_include_directories(${MAIN} PUBLIC ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR} ${CMAKE_DL_LIBS} ${FFTW3_INCLUDE_DIR})


#set(MAIN TestLMPDOS)
#add_executable(${MAIN} testlammpsDOS.cpp)
#target_link_libraries(${MAIN} PUBLIC lmp-ompicouple ${LMPLIB} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${FFTW3_LIBRARIES} ${DLLINK} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL})
#target_include_directories(${MAIN} PUBLIC ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR} ${CMAKE_DL_LIBS} ${FFTW3_INCLUDE_DIR})



#set(MAIN TestLMPNEQ)
#add_executable(${MAIN} testlammpsNeq.cpp)
#target_link_libraries(${MAIN} PUBLIC lmp-ompicouple ${LMPLIB} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${FFTW3_LIBRARIES} ${DLLINK} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL})
#target_include_directories(${MAIN} PUBLIC ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR} ${CMAKE_DL_LIBS} ${FFTW3_INCLUDE_DIR})


#set(MAIN TestLMPForceNEQ)
#add_executable(${MAIN} testlammpsForcesNeq.cpp)
#target_link_libraries(${MAIN} PUBLIC lmp-ompicouple ${LMPLIB} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${FFTW3_LIBRARIES} ${DLLINK} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL})
#target_include_directories(${MAIN} PUBLIC ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR} ${CMAKE_DL_LIBS} ${FFTW3_INCLUDE_DIR})

#set(MAIN TestLMPNEQDeparture)
#add_executable(${MAIN} testlammpsForcesNeqEqDeparture.cpp)
#target_link_libraries(${MAIN} PUBLIC lmp-ompicouple ${LMPLIB} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${FFTW3_LIBRARIES} ${DLLINK} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL})
#target_include_directories(${MAIN} PUBLIC ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR} ${CMAKE_DL_LIBS} ${FFTW3_INCLUDE_DIR})




#set(MAIN TestLMPNEQErel)
#add_executable(${MAIN} testlammpsERelWForce.cpp)
#target_link_libraries(${MAIN} PUBLIC lmp-ompicouple ${LMPLIB} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${FFTW3_LIBRARIES} ${DLLINK} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL})
#target_include_directories(${MAIN} PUBLIC ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR} ${CMAKE_DL_LIBS} ${FFTW3_INCLUDE_DIR})


#set(MAIN TestNewIter)
#add_executable(${MAIN} testlammpsERelNewIter.cpp)
#target_link_libraries(${MAIN} PUBLIC lmp-ompicouple ${LMPLIB} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${FFTW3_LIBRARIES} ${DLLINK} ${RSHAM} ${MKLSP} ${MKLBLAS} ${KPMLIB} ${VECFACTORY} ${SPATIAL})
#target_include_directories(${MAIN} PUBLIC ${CMAKE_SOURCE_DIR} ${PROJECT_BINARY_DIR} ${CMAKE_DL_LIBS} ${FFTW3_INCLUDE_DIR})

