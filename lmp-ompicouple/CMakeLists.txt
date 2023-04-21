set(LMPCOUPLE_LIB lmp-ompicouple)
set(LMPCOUPLE_SOURCES "error.cpp" "files.cpp" "irregular.cpp" "lammps_data_write.cpp" "many2many.cpp" "many2one.cpp" "memory.cpp" "one2many.cpp" "send2one.cpp" )
add_library(${LMPCOUPLE_LIB} STATIC ${LMPCOUPLE_SOURCES})
target_include_directories(${LMPCOUPLE_LIB} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

