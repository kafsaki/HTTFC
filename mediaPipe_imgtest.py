import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os

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

# 初始化mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 加载训练好的模型
model = torch.load('mediaPipe_models/gesture_classifier.pth')
model.eval()

# 读取图像序列的文件夹路径
# image_folder_path = 'data/CE-CSL/images/dev/A/dev-00001'
image_folder_path = 'data/CE-CSL/images/train/A/train-00001'

image_files = sorted(os.listdir(image_folder_path))

# 在循环中读取图像序列的每一帧图像，使用MediaPipe检测手势，并绘制手部关键点
mp_drawing = mp.solutions.drawing_utils
for image_file in image_files:
    if image_file.endswith(".jpg"):  # 确保是JPEG图像文件
        image_path = os.path.join(image_folder_path, image_file)
        frame = cv2.imread(image_path)  # 读取图像

        if frame is None:
            continue

        start_time = time.time()  # 记录处理开始时间
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 提取关键点数据
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.z])

                keypoints = torch.Tensor(keypoints)
                # 模型预测
                with torch.no_grad():
                    outputs = model(keypoints)
                    _, predicted = torch.max(outputs, 1)
                label = predicted.item()
                # 显示预测结果
                gesture = 'two' if label==3 else 'you' if label == 2 else 'yes' if label == 1 else 'no'
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        elapsed_time = time.time() - start_time  # 计算处理耗时
        cv2.putText(frame, f"Detection Time: {elapsed_time:.2f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()