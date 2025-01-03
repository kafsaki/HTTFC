import os
from datetime import datetime
import configparser
import csv
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import requests
import json

from APP_convertVideoToTxt import convertVideoToTxt
from APP_runModel import runModel

app = Flask(__name__)

# 用于保存上传视频的路径
videoRootPath = os.path.join("data", "mydataset", "video")
txtRootPath = os.path.join("data", "mydataset", "mediapipe")


# 更新配置文件中的路径
def update_config_file(currentUser, videoName):
    config = configparser.ConfigParser()
    config.read('params/appConfig.ini')
    new_path = f"data/mydataset/mediapipe/test/{currentUser}/{videoName}"
    config.set('Path', 'testDataPath', new_path)
    with open('params/appConfig.ini', 'w') as configfile:
        config.write(configfile)


def update_csv_file(videoName):
    # 打开 CSV 文件
    with open('data/mydataset/label/test.csv', 'r+', encoding='utf-8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        # 创建新的最后一行
        new_row = [videoName, 'L', '没有。', '没有/。', '']
        # 替换最后一行
        rows[-1] = new_row
        # 将修改后的内容写回文件
        csvfile.seek(0)  # 回到文件开头
        writer = csv.writer(csvfile)
        writer.writerows(rows)


# 保存视频
def save_video_from_stream(currentUser, videoName, video_data):
    uservideo_path = os.path.join(videoRootPath, "test", currentUser)
    if not os.path.exists(uservideo_path):
        os.makedirs(uservideo_path)
    video_path = os.path.join(videoRootPath, "test", currentUser, f"{videoName}.mp4")

    with open(video_path, 'wb') as f:
        f.write(video_data)  # 将视频文件直接写入磁盘


# 主路由，展示前端页面
@app.route('/')
def index():
    return render_template('index.html')


# 接收前端视频流
@app.route('/upload_video', methods=['POST'])
def upload_video():
    currentUser = request.form.get('currentUser')
    videoName = f"{currentUser}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    video_data = request.files['video'].read()

    save_video_from_stream(currentUser, videoName, video_data)

    videoPath = os.path.join(videoRootPath, "test", currentUser, f"{videoName}.mp4")
    txtPath = os.path.join(txtRootPath, "test", currentUser, f"{videoName}")
    # 提取特征点保存为txt
    convertVideoToTxt(videoPath, txtPath)

    # 更新配置文件和CSV
    update_config_file(currentUser, videoName)
    update_csv_file(videoName)

    recognition = runModel() # 返回["str"]

    package = {
        "recognition": recognition
    }
    # 发送POST请求到后台接口
    url = 'http://ctnn0kcp420c739acjog-5000.agent.damodel.com/translate'  # 后台地址(在damodel控制台申请)
    print("\033[32mClient: waiting for response...\033[0m")
    response = requests.post(url, json=package, headers={"Content-Type": "application/json"})
    # 获取后台返回的响应
    if response.status_code == 200:
        data = response.json()
        translation = data.get('translation', None)
        print("Translation received from backend:", translation)
    else:
        translation = f"服务器响应错误，返回代码：{response.status_code}"
        print("Error in response:", response.status_code)

    return jsonify({"status": "success", "videoName": videoName, "recognition":recognition, "translation":translation})


if __name__ == '__main__':
    app.run(debug=True)
