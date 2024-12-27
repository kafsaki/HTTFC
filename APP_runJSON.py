import Net
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from WER import WerScore
import os
import DataProcessMoudle
import videoAugmentation
import numpy as np
import decode
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from evaluation import evaluteMode
from evaluationT import evaluteModeT
import random
import json
from ReadConfig import readConfig  # 这里和runmodel不同，因为是对整个ce-csl数据集进行验证


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def stable(dataloader, seed):
    seed_torch(seed)
    return dataloader

def test(configParams, isTrain=True, isCalc=False):
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
    word2idx, wordSetNum, idx2word = DataProcessMoudle.Word2Id(trainLabelPath, validLabelPath, testLabelPath, dataSetName)

    testData = DataProcessMoudle.MyDataset(testDataPath, testLabelPath, word2idx, dataSetName,
                                               transform=None)

    print(f"testData length: {testData.__len__()}")

    testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False, num_workers=numWorkers,
                            pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)

    # 定义模型
    moduleNet = Net.moduleNet(hiddenSize, wordSetNum * max_num_states + 1, moduleChoice, device, dataSetName, True)
    moduleNet = moduleNet.to(device)

    # 损失函数定义
    PAD_IDX = 0
    if "MSTNet" == moduleChoice:
        ctcLoss = nn.CTCLoss(blank=PAD_IDX, reduction='mean', zero_infinity=True)
    elif "VAC" == moduleChoice or "CorrNet" == moduleChoice or "MAM-FSD" == moduleChoice \
         or "SEN" == moduleChoice or "TFNet" == moduleChoice or "KFNet" == moduleChoice:
        ctcLoss = nn.CTCLoss(blank=PAD_IDX, reduction='none', zero_infinity=True)
        kld = DataProcessMoudle.SeqKD(T=8)
        if "MAM-FSD" == moduleChoice:
            mseLoss = nn.MSELoss(reduction="mean")

    logSoftMax = nn.LogSoftmax(dim=-1)

    # 解码参数
    decoder = decode.Decode(word2idx, wordSetNum + 1, 'beam')


    # 验证模式
    if os.path.exists(bestModuleSavePath):
        checkpoint = torch.load(bestModuleSavePath, map_location=torch.device('cpu'))
        moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
        moduleNet.eval()
        print("开始验证模型")
        # 验证模型
        werScoreSum = 0
        loss_value = []

        print("正在加载json")
        with open("output/CE-CSL.json", "r", encoding="utf-8") as f:
            new_json = json.load(f)
        print("成功加载json")

        for Dict in tqdm(testLoader):
            data = Dict["video"].to(device)
            label = Dict["label"]
            dataLen = Dict["videoLength"]
            ##########################################################################
            targetOutData = [torch.tensor(yi).to(device) for yi in label]
            targetLengths = torch.tensor(list(map(len, targetOutData)))
            targetData = targetOutData
            targetOutData = torch.cat(targetOutData, dim=0).to(device)
            batchSize = len(targetLengths)

            with torch.no_grad():
                logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = moduleNet(data, dataLen, False)

                logProbs1 = logSoftMax(logProbs1)

                loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths).mean()

                loss = loss1

            loss_value.append(loss.item())

            pred, targetOutDataCTC = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

            werScore, hypotheses, references = WerScore([targetOutDataCTC], targetData, idx2word, batchSize)
            werScoreSum = werScoreSum + werScore

            for hypothese, reference in zip(hypotheses, references):
                hypothese= hypothese.replace(" ", "")
                reference = reference.replace(" ", "")
                print(f"hypothese: {hypothese}")
                print(f"reference: {reference}")
                # 用识别结果和groundtruth构建新json元素
                new_element = {
                    "recognition": hypothese,
                    "groundtruth": reference,
                    "translation": ""
                }
                new_json.append(new_element)

            torch.cuda.empty_cache()

        # 将更新后的数据写回到 JSON 文件
        with open("output/CE-CSL.json", "w", encoding="utf-8") as f:
            json.dump(new_json, f, ensure_ascii=False, indent=2)

        werScore = werScoreSum / len(testLoader)

        print(f"werScore: {werScore:.2f}")


def runJSON():
    seed_torch(10)
    # 读取配置文件
    configParams = readConfig()

    # isTrain为True是训练模式，isTrain为False是验证模式
    return test(configParams, isTrain=False, isCalc=True)

if __name__ == '__main__':
    runJSON()