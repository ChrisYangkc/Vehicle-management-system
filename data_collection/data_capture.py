import cv2
import threading
import os
import time

# 创建用于保存图像的文件夹
output_folder = 'camera'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def capture_video(camera_index, window_name):
    # 初始化摄像头
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    image_counter = 0
    while True:
        # 捕获帧
        ret, frame = cap.read()
        if not ret:
            print(f"Can't receive frame from camera {camera_index}. Exiting ...")
            break

        # 显示帧
        cv2.imshow(window_name, frame)

        # 每隔一定时间保存一次图像
        if image_counter % 30 == 0:  # 每30帧保存一次图像
            image_name = os.path.join(output_folder, f"camera_{camera_index}_{int(time.time())}.jpg")
            cv2.imwrite(image_name, frame)

        image_counter += 1

        # 按 'q' 退出
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()

# 创建两个线程分别从不同摄像头捕获视频
thread_in = threading.Thread(target=capture_video, args=(0, 'Camera IN')) # 0 通常是第一个摄像头
thread_out = threading.Thread(target=capture_video, args=(1, 'Camera OUT')) # 1 通常是第二个摄像头

# 启动线程
thread_in.start()
thread_out.start()

# 等待线程结束
thread_in.join()
thread_out.join()
