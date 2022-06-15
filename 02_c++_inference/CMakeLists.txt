cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

set(CMAKE_PREFIX_PATH /media/tianru/Rane/CODE/04_huagong_proj/06_Pytorch_inference/libtorch)
find_package(Torch REQUIRED)

add_executable(test 02_test_libtorch.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")

set_property(TARGET test PROPERTY CXX_STANDARD 14)




# cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
# project(libtorch_test)


# set(CMAKE_PREFIX_PATH /media/tianru/Rane/CODE/04_huagong_proj/06_Pytorch_inference/libtorch)
# # set(Torch_DIR /media/tianru/Rane/CODE/04_huagong_proj/06_Pytorch_inference/libtorch/share/cmake/Torch/TorchConfig.cmake)
# find_package(Torch)
# message("pytorch libs: " ${TORCH_LIBRARIES})
# include_directories(/media/tianru/Rane/CODE/04_huagong_proj/06_Pytorch_inference/libtorch/include)
# include_directories(/media/tianru/Rane/CODE/04_huagong_proj/06_Pytorch_inference/libtorch/libtorch/include/torch/csrc/api/include)


# add_executable(libtorch_test 02_test_libtorch.cpp)
# target_link_libraries(libtorch_test ${TORCH_LIBRARIES})

# set_property(TARGET libtorch_test PROPERTY CXX_STANDARD 11)




