## The Android demo of dbnet/dbnet++ infer by ncnn  

## Please enjoy the dbnet hand demo on ncnn

You can try this APK demo https://pan.baidu.com/s/1ArAMH7uAic0cQJgOn-P-RQ pwd: jnrw  

https://github.com/Tencent/ncnn  
https://github.com/nihui/opencv-mobile
## db model support:  
1.det-sim-op 
2.pdocrv2.0_det-op 
3.ch_PP-OCRv3_det(bug)
4.dbnet++(bug)
 

## how to build and run
### step1
https://github.com/Tencent/ncnn/releases

* Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step2
https://github.com/nihui/opencv-mobile

* Download opencv-mobile-XYZ-android.zip
* Extract opencv-mobile-XYZ-android.zip into **app/src/main/jni** and change the **OpenCV_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step3
* Open this project with Android Studio, build it and enjoy!

## screenshot  
![](result.gif)  

## reference  
1.https://github.com/FeiGeChuanShu/ncnn-Android-mediapipe_hand
2.https://github.com/PaddlePaddle/PaddleOCR
