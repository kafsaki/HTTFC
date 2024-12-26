import os

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# 定义一个简单的全连接神经网络
class GestureClassifier(nn.Module):
    def __init__(self):
        super(GestureClassifier, self).__init__()
        self.fc1 = nn.Linear(21 * 3, 128)  # 21 keypoints, 3 coordinates each
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # Output layer with 4 classes

    def forward(self, x):
        x = x.view(-1, 21 * 3)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#初始化mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 加载训练好的模型
model = torch.load('mediaPipe_models/gesture_classifier.pth')
model.eval()

# 读取视频文件数据
# video_path = 'data/CE-CSL/video/train/A/train-00001.mp4'  # 视频文件路径
video_path = 'data/CE-CSL/video/dev/A/dev-00001.mp4'  # 视频文件路径

cap = cv2.VideoCapture(video_path)  # 使用视频文件路径代替相机索引


# 在循环中读取摄像头的每一帧图像，使用MediaPipe检测手势，并绘制手部关键点
mp_drawing = mp.solutions.drawing_utils
while True:
    success, frame = cap.read()
    if not success:
        break
    start_time = time.time()  # 记录处理开始时间
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    if results.multi_hand_landmarks:
        # 区分左右手
        for handedness in results.multi_handedness:
            print(handedness)
            for classification in handedness.classification:
                print(f'{classification.index} {classification.label}')

        handedness_info=[]
        for handedness in results.multi_handedness:
            handedness_info.append((handedness.classification[0].index, handedness.classification[0].label))
        print(f"handedness_info: \n{handedness_info}")
        # 将handedness_info按照index升序排序
        handedness_info = sorted(handedness_info, key=lambda x: x[0])
        print(f"handedness_info sorted by index: \n{handedness_info}")

        for hand_landmarks in results.multi_hand_landmarks:
            # # 提取关键点数据
            # keypoints = []
            # for landmark in hand_landmarks.landmark:
            #     keypoints.append([landmark.x, landmark.y, landmark.z])
            #
            # keypoints = torch.Tensor(keypoints)
            # # 模型预测
            # with torch.no_grad():
            #     outputs = model(keypoints)
            #     _, predicted = torch.max(outputs, 1)
            # label = predicted.item()
            # # 显示预测结果
            # gesture = 'two' if label==3 else 'you' if label == 2 else 'yes' if label == 1 else 'no'
            # cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print(f"hands:\n{hand_landmarks}")


    results_pose = pose.process(frameRGB)
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        print(f"pose:\n{results_pose.pose_landmarks}")



    elapsed_time = time.time() - start_time  # 计算处理耗时
    cv2.putText(frame, f"Detection Time: {elapsed_time:.2f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# vid = imageio.get_reader(videoPath)  # 读取视频
                # # nframes = vid.get_meta_data()['nframes']
                # nframes = vid.count_frames()
                #
                #
                # for i in range(nframes):
                #     try:
                #         image = vid.get_data(i)
                #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #         # image = cv2.resize(image, (256, 256))
                #
                #         # 使用MediaPipe检测手部关键点
                #         results = hands.process(image)
                #         if results.multi_hand_landmarks:
                #             # 获取左右手信息
                #             handedness_info = []
                #             for handedness in results.multi_handedness:
                #                 handedness_info.append(
                #                     (handedness.classification[0].index, handedness.classification[0].label))
                #             # 将handedness_info按照index升序排序
                #             handedness_info = sorted(handedness_info, key=lambda x: x[0])
                #
                #             # 处理双手情况
                #             left_landmarks = []
                #             right_landmarks = []
                #             for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                #                 for handedness_index, handedness_label in handedness_info:
                #                     if handedness_index == index:
                #                         if handedness_label == 'Left':
                #                             left_landmarks = hand_landmarks
                #                         elif handedness_label == 'Right':
                #                             right_landmarks = hand_landmarks
                #                         break  # 找到匹配的handedness后跳出循环,判断下一个hand_landmarks
                #
                #             mp_drawing.draw_landmarks(
                #                 image, left_landmarks, mp_hands.HAND_CONNECTIONS)
                #             mp_drawing.draw_landmarks(
                #                 image, right_landmarks, mp_hands.HAND_CONNECTIONS)
                #             cv2.imshow("Frame", image)
                #             if cv2.waitKey(1) & 0xFF == ord('q'):
                #                 break
                #             # 将关键点存入txt文件
                #             # 根据handedness信息，index代表multi_hand_landmarks中的hand_landmarks排序
                #             # label代表对应index的hand_landmarks对应左右手
                #             # 如果根据handedness信息有2个手，如果2个手一左一右，则先写入左手landmarks，再写入右手landmarks；如果是2个左或者2个右，则当作只有1个手，选取第1个作为手，进入只有1个手的判断
                #             # 如果根据handedness信息有1个手，如果有左手，则先写入左手landmarks，再写入21*3个0；如果有右手，则先写入21*3个0，再写入右手landmarks
                #             # 将关键点存入txt文件
                #             txtPath = os.path.join(saveImagePath, f"{i:05d}.txt")
                #             with open(txtPath, 'w') as f:
                #                 # 一左一右，则先写入左手landmarks，再写入右手landmarks；
                #                 if left_landmarks and right_landmarks:
                #                     for landmark in left_landmarks.landmark:
                #                         x = int(landmark.x * 256)
                #                         y = int(landmark.y * 256)
                #                         z = landmark.z
                #                         f.write(f"{x},{y},{z}\n")
                #                     for landmark in right_landmarks.landmark:
                #                         x = int(landmark.x * 256)
                #                         y = int(landmark.y * 256)
                #                         z = landmark.z
                #                         f.write(f"{x},{y},{z}\n")
                #
                #                 elif left_landmarks:  # 只有左手
                #                     for landmark in left_landmarks.landmark:
                #                         x = int(landmark.x * 256)
                #                         y = int(landmark.y * 256)
                #                         z = landmark.z
                #                         f.write(f"{x},{y},{z}\n")
                #                     for _ in range(21):  # 写入21*3个0，代表右手的坐标
                #                         f.write("0,0,0\n")
                #
                #                 elif right_landmarks:  # 只有右手
                #                     for _ in range(21):  # 写入21*3个0，代表左手的坐标
                #                         f.write("0,0,0\n")
                #                     for landmark in right_landmarks.landmark:
                #                         x = int(landmark.x * 256)
                #                         y = int(landmark.y * 256)
                #                         z = landmark.z
                #                         f.write(f"{x},{y},{z}\n")
                #         else:
                #             # 如果根据handedness信息有0个手，写入42*3个0
                #             txtPath = os.path.join(saveImagePath, f"{i:05d}.txt")
                #             with open(txtPath, 'w') as f:
                #                 for _ in range(42):  # 写入42个000
                #                     f.write("0,0,0\n")
                #         # print(f"OK: {videoPath}")
                #
                #
                #     except Exception as e:
                #         print(f"Error processing frame {i}: {e}")
                #         print(nframes)
                #         print(videoPath)
                #
                # vid.close()