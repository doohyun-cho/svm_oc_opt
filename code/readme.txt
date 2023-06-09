code/
    CMakeLists.txt      Cmake file.
    initKDT.cpp         Initiallize internal parameters (K), distortion parameters (D),
    initKDT.h           and camera poses (T) of the cameras in the given surround-view system.
    poseAdjust.cpp      Implementation of the Ground model.
    poseAdjustV2.cpp    Implementation of the Ground-Camera model.
    surroundView.cpp    Functions for adjusting the camera poses.
    surroundView.h        

test_cases/        
    ALL/                The images captured by the Front, Left, Back, Right camera respectively.
                        Same index means the images are captured at the same time.
    surround view/      The surround-view images corresponded to the images in ALL/.

result_examples/        Results of the test cases before and after being processed by our algorithm.