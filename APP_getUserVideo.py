import os
from datetime import datetime
import cv2
import configparser
import csv

from APP_convertVideoToTxt import convertVideoToTxt



def getVideo(videoPath):
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 获取视频的宽度和高度
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # 定义视频编码和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定编码格式
    out = cv2.VideoWriter(videoPath, fourcc, 20.0, (frame_width, frame_height))
    # 记录开始时间
    start_time = datetime.now()
    # 循环录取视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 写入帧
        out.write(frame)
        # 显示帧
        cv2.imshow('frame', frame)
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 如果已经录制了10秒，则退出循环
        if (datetime.now() - start_time).seconds >= 5 :
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def update_config_file(currentUser, videoName):
    # 读取 config 文件
    config = configparser.ConfigParser()
    config.read('params/appConfig.ini')

    # 替换 testDataPath
    new_path = f"data/mydataset/mediapipe/test/{currentUser}/{videoName}"
    config.set('Path', 'testDataPath', new_path)

    # 将修改后的内容写回文件
    with open('params/appConfig.ini', 'w') as configfile:
        config.write(configfile)


def update_csv_file(videoName):
    # 打开 CSV 文件
    with open('data/mydataset/label/test.csv', 'r+',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        # 创建新的最后一行
        new_row = [videoName, 'L', '你好。', '你/好/。,']
        # 替换最后一行
        rows[-1] = new_row
        # 将修改后的内容写回文件
        csvfile.seek(0)  # 回到文件开头
        writer = csv.writer(csvfile)
        writer.writerows(rows)

def main():

    currentUser = "xkf"
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 构建包含用户名和时间戳的视频路径
    videoRootPath = os.path.join("data/mydataset/video")
    # 构建包含用户名和时间戳的文本文件路径
    txtRootPath = os.path.join("data/mydataset/mediapipe")
    # 带时间戳和用户名的视频文件名
    videoName = f"{currentUser}_{current_time}"

    videoPath = os.path.join(videoRootPath, "test", currentUser, f"{videoName}.mp4")
    txtPath = os.path.join(txtRootPath, "test", currentUser, f"{videoName}")
    # 录制视频
    getVideo(videoPath)
    # 提取特征点保存为txt
    # convertVideoToTxt(videoRootPath, txtRootPath)
    convertVideoToTxt(videoPath, txtPath)

    # 更新配置文件和 CSV 文件
    update_config_file(currentUser, videoName)
    update_csv_file(videoName)


if __name__ == '__main__':
    main()

