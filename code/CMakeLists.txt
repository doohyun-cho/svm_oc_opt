cmake_minimum_required(VERSION 2.8)

project(CamPoseCorr)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_MODULE_PATH "/path")
set(CMAKE_CXX_FLAGS "-std=c++14 -O0")
# set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "~/dev/g2o/cmake_modules")
SET( G2O_DIR ~/dev/g2o/cmake_modules )

find_package( OpenCV REQUIRED )
find_package( Sophus REQUIRED )
FIND_PACKAGE( G2O REQUIRED )


include_directories( 
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    "/usr/include/eigen3/"
	${Sophus_INCLUDE_DIRS}
)

add_library(initKDT initKDT.cpp)
add_library(initKDTShared SHARED initKDT.cpp)
target_link_libraries(initKDT 
	${Sophus_LIBRARIES})
add_library(surroundView surroundView.cpp)
add_library(surroundViewShared SHARED surroundView.cpp)
target_link_libraries(surroundView 
	${OpenCV_LIBS}
	${Sophus_LIBRARIES})

	

add_executable(poseAdjust poseAdjust.cpp)
target_link_libraries(poseAdjust
	${OpenCV_LIBS}
	${G2O_LIBS}
	g2o_core
	initKDT
	surroundView
	${Sophus_LIBRARIES})
	
add_executable(PAV2 poseAdjustV2.cpp)
target_link_libraries(PAV2
	${OpenCV_LIBS}
	${G2O_LIBS}
	g2o_core
	initKDT
	surroundView
	${Sophus_LIBRARIES})





