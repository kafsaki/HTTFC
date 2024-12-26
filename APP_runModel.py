import Net
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from WER import WerScore
import os
import videoAugmentation
import numpy as np
import decode
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from evaluation import evaluteMode
from evaluationT import evaluteModeT
import random
from datetime import datetime
import cv2

import APP_DataProcessMoudle
from APP_ReadConfig import readConfig

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def test(configParams, isTrain=False, isCalc=True):
    hypotheses = []

    # 参数初始化
    # 读入数据路径
    trainDataPath = configParams["trainDataPath"]
    validDataPath = configParams["validDataPath"]
    testDataPath = configParams["testDataPath"]
    # 读入标签路径
    trainLabelPath = configParams["trainLabelPath"]
    validLabelPath = configParams["validLabelPath"]
    testLabelPath = configParams["testLabelPath"]
    # 读入模型参数
    bestModuleSavePath = configParams["bestModuleSavePath"]
    currentModuleSavePath = configParams["currentModuleSavePath"]
    # 读入参数
    device = configParams["device"]
    hiddenSize = int(configParams["hiddenSize"])
    lr = float(configParams["lr"])
    batchSize = int(configParams["batchSize"])
    numWorkers = int(configParams["numWorkers"])
    pinmMemory = bool(int(configParams["pinmMemory"]))
    moduleChoice = configParams["moduleChoice"]
    dataSetName = configParams["dataSetName"]
    max_num_states = 1

    # 预处理语言序列
    word2idx, wordSetNum, idx2word = APP_DataProcessMoudle.Word2Id(trainLabelPath, validLabelPath, testLabelPath,
                                                               dataSetName)
    # 定义模型
    moduleNet = Net.moduleNet(hiddenSize, wordSetNum * max_num_states + 1, moduleChoice, device, dataSetName, True)
    moduleNet = moduleNet.to(device)
    logSoftMax = nn.LogSoftmax(dim=-1)
    # 损失函数定义
    PAD_IDX = 0
    ctcLoss = nn.CTCLoss(blank=PAD_IDX, reduction='none', zero_infinity=True)
    # 解码参数
    decoder = decode.Decode(word2idx, wordSetNum + 1, 'beam')

    # txt关键点预处理
    transformTxt = videoAugmentation.Compose([
        videoAugmentation.TemporalRescale(0.2),
    ])

    # 加载数据
    testData = APP_DataProcessMoudle.MyDataset(testDataPath, testLabelPath, word2idx, dataSetName,
                                           transform=None)
    testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False, num_workers=numWorkers,
                            pin_memory=pinmMemory, collate_fn=APP_DataProcessMoudle.collate_fn, drop_last=True)


    lastEpoch = -1
    if os.path.exists(currentModuleSavePath):
        checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'))
        moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
        print("已加载预训练模型")
        moduleNet.eval()
        print("开始验证模型")
        # 验证模型
        werScoreSum = 0

        for Dict in tqdm(testLoader):
            data = Dict["video"].to(device)
            label = Dict["label"]
            dataLen = Dict["videoLength"]
            ##########################################################################
            targetData = [torch.tensor(yi).to(device) for yi in label]
            targetLengths = torch.tensor(list(map(len, targetData)))
            batchSize = len(targetLengths)

            with torch.no_grad():
                logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = moduleNet(data, dataLen, False)

                logProbs1 = logSoftMax(logProbs1)


            pred, targetOutDataCTC = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

            werScore, hypotheses, references = WerScore([targetOutDataCTC], targetData, idx2word, batchSize)
            werScoreSum = werScoreSum + werScore

            print(f"hypotheses: {hypotheses}");
            print(f"references: {references}");

            torch.cuda.empty_cache()


        werScore = werScoreSum / len(testLoader)

        print(f"werScore: {werScore:.2f}")

    else:
        print("未加载预训练模型 ")

    return hypotheses


def runModel():
    seed_torch(10)
    # 读取配置文件
    configParams = readConfig()

    # isTrain为True是训练模式，isTrain为False是验证模式
    return test(configParams, isTrain=False, isCalc=True)


