cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

set(CMAKE_PREFIX_PATH /media/tianru/Rane/CODE/04_huagong_proj/06_Pytorch_inference/libtorch)
find_package(Torch REQUIRED)

# opencv
set(OpenCV_DIR /media/tianru/Rane/Package/opencv/opencv-3.4.7/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
MESSAGE("opencv libs:" ${OpenCV_LIBS})
MESSAGE("opencv includes:" ${OpenCV_INCLUDE_DIRS})
# set(cv_lib1 "/mnt/wangran/Package/opencv/opencv-3.4.7/build/lib/libopencv_core.so.3.4.7")
# set(cv_lib2 "/mnt/wangran/Package/opencv/opencv-3.4.7/build/lib/libopencv_highgui.so.3.4.7")
# set(cv_lib3 "/mnt/wangran/Package/opencv/opencv-3.4.7/build/lib/libopencv_imgcodecs.so.3.4.7")

# add_executable(test 01_test_libtorch.cpp)
add_executable(test 02_pred_dumei_model.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
# target_link_libraries(test ${cv_lib1} ${cv_lib2} ${cv_lib3})
target_link_libraries(test ${OpenCV_LIBS})

set_property(TARGET test PROPERTY CXX_STANDARD 14)

