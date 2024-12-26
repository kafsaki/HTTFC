import random
import os
import numpy as np
from tqdm import tqdm
import cv2
import mediapipe as mp
import time


def convertVideoToTxt(dataPath, saveDataPath):
    # 初始化mediapipe
    # 手势
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    # 姿态
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    videoPath = dataPath
    saveImagePath = saveDataPath

    if not os.path.exists(saveImagePath):
        os.makedirs(saveImagePath)

    cap = cv2.VideoCapture(videoPath)  # 使用视频文件路径代替相机索引


    i = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        if results.multi_hand_landmarks:
            # # 获取左右手信息
            # handedness_info = []
            # for handedness in results.multi_handedness:
            #     handedness_info.append(
            #         (handedness.classification[0].index, handedness.classification[0].label))
            # # 将handedness_info按照index升序排序
            # handedness_info = sorted(handedness_info, key=lambda x: x[0])

            # 处理双手情况
            left_landmarks = []
            right_landmarks = []
            # 根据handedness_info判断左右手
            # for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            #     for handedness_index, handedness_label in handedness_info:
            #         if handedness_index == index:
            #             if handedness_label == 'Left':
            #                 left_landmarks = hand_landmarks
            #             elif handedness_label == 'Right':
            #                 right_landmarks = hand_landmarks
            #             break  # 找到匹配的handedness后跳出循环,判断下一个hand_landmarks
            # 根据手腕坐标判断左右手
            multi_hand_landmarks = results.multi_hand_landmarks
            if multi_hand_landmarks.__len__() == 1:
                # 一个手的情况，根据mediapipe info的label判断
                if results.multi_handedness[0].classification[0].label == 'Left':
                    left_landmarks = multi_hand_landmarks[0]
                else:
                    right_landmarks = multi_hand_landmarks[0]
            elif multi_hand_landmarks.__len__() == 2:
                # 两个手的情况，根据手腕的x坐标来判断左右手
                left_wrist_idx = mp_hands.HandLandmark.WRIST
                wrist_x_coords = [hand_landmark.landmark[left_wrist_idx].x for hand_landmark in
                                  multi_hand_landmarks]
                left_idx, right_idx = (0, 1) if wrist_x_coords[0] < wrist_x_coords[1] else (1, 0)
                left_landmarks = multi_hand_landmarks[left_idx]
                right_landmarks = multi_hand_landmarks[right_idx]
                # 打印手腕x坐标以供验证
                # print(
                #     f"Left wrist x: {wrist_x_coords[left_idx]}, Right wrist x: {wrist_x_coords[right_idx]}")
            '''
            # 分左右绘制手
            mp_drawing.draw_landmarks(frame, left_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, right_landmarks, mp_hands.HAND_CONNECTIONS)
            # 定义函数计算边界框
            def get_hand_bbox(hand_landmarks, frame_width, frame_height):
                if not hand_landmarks:
                    return None
                xmin = ymin = float('inf')
                xmax = ymax = float('-inf')
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame_width + 0.5)
                    y = int(landmark.y * frame_height + 0.5)
                    xmin = min(xmin, x)
                    ymin = min(ymin, y)
                    xmax = max(xmax, x)
                    ymax = max(ymax, y)
                return (xmin, ymin, xmax, ymax)
            # 计算左手和右手的边界框
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            if left_landmarks:
                left_hand_bbox = get_hand_bbox(left_landmarks, frame_width, frame_height)
                if left_hand_bbox:
                    cv2.rectangle(frame, (left_hand_bbox[0], left_hand_bbox[1]),
                                  (left_hand_bbox[2], left_hand_bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, 'Left', (left_hand_bbox[0], left_hand_bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            if right_landmarks:
                right_hand_bbox = get_hand_bbox(right_landmarks, frame_width, frame_height)
                if right_hand_bbox:
                    cv2.rectangle(frame, (right_hand_bbox[0], right_hand_bbox[1]),
                                  (right_hand_bbox[2], right_hand_bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, 'Right', (right_hand_bbox[0], right_hand_bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            '''


            # # 绘制所有手，不分左右
            # for hand_landmarks in results.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(
            #         frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # 将关键点存入txt文件
            # 根据handedness信息，index代表multi_hand_landmarks中的hand_landmarks排序
            # label代表对应index的hand_landmarks对应左右手
            # 如果根据handedness信息有2个手，如果2个手一左一右，则先写入左手landmarks，再写入右手landmarks；如果是2个左或者2个右，则当作只有1个手，选取第1个作为手，进入只有1个手的判断
            # 如果根据handedness信息有1个手，如果有左手，则先写入左手landmarks，再写入21*3个0；如果有右手，则先写入21*3个0，再写入右手landmarks
            # 将关键点存入txt文件
            txtPath = os.path.join(saveImagePath, f"{i:05d}.txt")
            with open(txtPath, 'w') as f:
                # 一左一右，则先写入左手landmarks，再写入右手landmarks；
                if left_landmarks and right_landmarks:
                    for landmark in left_landmarks.landmark:
                        x = int(min(max(0, landmark.x), 1) * 255)  # 确保x在[0, 1]范围内
                        y = int(min(max(0, landmark.y), 1) * 255)  # 确保y在[0, 1]范围内
                        f.write(f"{x},{y}\n")
                    for landmark in right_landmarks.landmark:
                        x = int(min(max(0, landmark.x), 1) * 255)  # 确保x在[0, 1]范围内
                        y = int(min(max(0, landmark.y), 1) * 255)  # 确保y在[0, 1]范围内
                        f.write(f"{x},{y}\n")

                elif left_landmarks:  # 只有左手
                    for landmark in left_landmarks.landmark:
                        x = int(min(max(0, landmark.x), 1) * 255)  # 确保x在[0, 1]范围内
                        y = int(min(max(0, landmark.y), 1) * 255)  # 确保y在[0, 1]范围内
                        f.write(f"{x},{y}\n")
                    for _ in range(21):  # 写入21*3个0，代表右手的坐标
                        f.write("0,0\n")

                elif right_landmarks:  # 只有右手
                    for _ in range(21):  # 写入21*3个0，代表左手的坐标
                        f.write("0,0\n")
                    for landmark in right_landmarks.landmark:
                        x = int(min(max(0, landmark.x), 1) * 255)  # 确保x在[0, 1]范围内
                        y = int(min(max(0, landmark.y), 1) * 255)  # 确保y在[0, 1]范围内
                        f.write(f"{x},{y}\n")
        else:# 這裏應該是無效的，就算沒有手也會有multi結果。。。
            # 如果根据handedness信息有0个手，写入42*3个0
            txtPath = os.path.join(saveImagePath, f"{i:05d}.txt")
            with open(txtPath, 'w') as f:
                for _ in range(42):  # 写入42个000
                    f.write("0,0\n")

        i = i + 1

        # 姿态 暂不考虑
        # results_pose = pose.process(frameRGB)
        # if results_pose.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #     print(f"pose:\n{results_pose.pose_landmarks}")


        '''
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    '''
    print("done")

