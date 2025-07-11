cmake_minimum_required(VERSION 3.13)

project(SPC CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_definitions(-w)

find_package(CUDA)
include(FindCUDA/select_compute_arch)

# CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
# string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
# string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
# string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
# message("-- nvcc generates code for arch ${CUDA_ARCH_LIST}")
# SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
# cmake_policy(SET CMP0104 OLD)

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CUDA_ARCHITECTURES "80;90")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(glog_DIR /scratch/zt1/project/bhatele-lab/shared/spmm/glog/install/lib64/cmake/glog)
include(cmake/Dependencies.cmake)
find_package(glog REQUIRED)

list(APPEND SPC_LIBS glog::glog) 
include_directories(${GLOG_INCLUDE_DIRS})
message(STATUS "Glog found: ${Glog_FOUND}")
message(STATUS "Glog include dirs: ${GLOG_INCLUDE_DIRS}")
message(STATUS "Glog libraries: ${GLOG_LIBRARIES}")
set(UTILS_LIBS ${PROJECT_SOURCE_DIR}/utils)
set(Sputnik_SPMM_LIBS ${PROJECT_SOURCE_DIR}/Sputnik_SpMM)
set(Sputnik_SDDMM_LIBS ${PROJECT_SOURCE_DIR}/Sputnik_SDDMM)
set(CUSPARSE_SPMM_LIBS ${PROJECT_SOURCE_DIR}/cuSparse_SpMM)

set(RoDe_SPMM_LIBS ${PROJECT_SOURCE_DIR}/RoDe_SpMM)
set(RoDe_SDDMM_LIBS ${PROJECT_SOURCE_DIR}/RoDe_SDDMM)



include_directories(${PROJECT_SOURCE_DIR})


include_directories("${PROJECT_SOURCE_DIR}")
MESSAGE( STATUS "this var key = ${CMAKE_CUDA_FLAGS}.")

add_subdirectory(utils)
add_subdirectory(Sputnik_SpMM)
add_subdirectory(Sputnik_SDDMM)

add_subdirectory(cuSparse_SpMM)


add_subdirectory(RoDe_SDDMM)
add_subdirectory(RoDe_SpMM)

add_subdirectory(mat_prof)
add_subdirectory(eval)
add_subdirectory(Preprocess_opt)
