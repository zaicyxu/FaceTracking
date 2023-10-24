import cv2

cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数: propId - 设置的视频参数, value - 设置的参数值
"""
0. cv2.CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
1. cv2.CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
2. cv2.CAP_PROP_POS_AVI_RATIO Relative position of the video file
3. cv2.CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
4. cv2.CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
5. cv2.CAP_PROP_FPS Frame rate.
6. cv2.CAP_PROP_FOURCC 4-character code of codec.
7. cv2.CAP_PROP_FRAME_COUNT Number of frames in the video file.
8. cv2.CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
9. cv2.CAP_PROP_MODE Backend-specific value indicating the current capture mode.
10. cv2.CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
11. cv2.CAP_PROP_CONTRAST Contrast of the image (only for cameras).
12. cv2.CAP_PROP_SATURATION Saturation of the image (only for cameras).
13. cv2.CAP_PROP_HUE Hue of the image (only for cameras).
14. cv2.CAP_PROP_GAIN Gain of the image (only for cameras).
15. cv2.CAP_PROP_EXPOSURE Exposure (only for cameras).
16. cv2.CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
17. cv2.CAP_PROP_WHITE_BALANCE Currently unsupported
18. cv2.CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
"""

# cap.isOpened() 返回 true/false, 检查摄像头初始化是否成功
print(cap.isOpened())
while cap.isOpened():
    ret_flag, img_camera = cap.read()
    print("height: ", img_camera.shape[0])
    print("width:  ", img_camera.shape[1])
    print('\n')
    cv2.imshow("camera", img_camera)

    # 每帧数据延时 1ms, 延时为0, 读取的是静态帧
    k = cv2.waitKey(1)
    # Press 's' to save faces into local images
    if k == ord('s'):
        cv2.imwrite("test.jpg", img_camera)
    # Press 'q' to exit
    if k == ord('q'):
        break

# 释放所有摄像头
cap.release()

# Delete all Windows
cv2.destroyAllWindows()