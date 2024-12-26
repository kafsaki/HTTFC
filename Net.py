import torch.nn as nn
import torch
import Transformer
import torchvision.models as models
import numpy as np
import Module
from BiLSTM import BiLSTMLayer
import SEN

class moduleNet(nn.Module):
    def __init__(self, hiddenSize, wordSetNum, moduleChoice="Seq2Seq", device=torch.device("cuda:0"), dataSetName='RWTH', isFlag=False):
        super().__init__()
        self.device = device
        self.moduleChoice = moduleChoice
        self.outDim = wordSetNum
        self.dataSetName = dataSetName
        self.logSoftMax = nn.LogSoftmax(dim=-1)
        self.softMax = nn.Softmax(dim=-1)
        self.isFlag = isFlag
        self.probs_log = []

        if "TFNet" == self.moduleChoice:
            hidden_size = hiddenSize
            self.conv2d = Module.resnet34MAM()
            self.conv2d.fc = Module.Identity()

            self.conv1d = Module.TemporalConv(input_size=512,
                                           hidden_size=hidden_size,
                                           conv_type=2)

            self.conv1d1 = Module.TemporalConv(input_size=512,
                                                 hidden_size=hidden_size,
                                                 conv_type=2)

            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.temporal_model1 = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.classifier1 = Module.NormLinear(hidden_size, self.outDim)
            self.classifier2 = self.classifier1

            self.classifier3 = Module.NormLinear(hidden_size, self.outDim)
            self.classifier4 = self.classifier3

            self.classifier5 = Module.NormLinear(hidden_size, self.outDim)

            self.reLU = nn.ReLU(inplace=True)
        elif "KFNet" == self.moduleChoice:
            hidden_size = hiddenSize

            self.conv1d = Module.TemporalConv(input_size=84,
                                           hidden_size=hidden_size,
                                           conv_type=2)

            self.conv1d1 = Module.TemporalConv(input_size=84,
                                                 hidden_size=hidden_size,
                                                 conv_type=2)

            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.temporal_model1 = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.classifier1 = Module.NormLinear(hidden_size, self.outDim)
            self.classifier2 = self.classifier1

            self.classifier3 = Module.NormLinear(hidden_size, self.outDim)
            self.classifier4 = self.classifier3

            self.classifier5 = Module.NormLinear(hidden_size, self.outDim)

            self.reLU = nn.ReLU(inplace=True)

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

    def read_txt_data(self, file_path):
        """
        从 .txt 文件读取数据，并转换为张量。
        文件结构为 42 行，2 列，列之间用 ',' 分割，每个数据为 0~255 的整数。
        """
        data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)
        return torch.tensor(data, dtype=torch.uint8).view(1, -1)  # (1, 84)

    def forward(self, seqData, dataLen=None, isTrain=True):
        outData1 = None
        outData2 = None
        outData3 = None
        logProbs1 = None
        logProbs2 = None
        logProbs3 = None
        logProbs4 = None
        logProbs5 = None
        if "TFNet" == self.moduleChoice:
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape
            x = seqData.transpose(1, 2)

            framewise, outData1, outData2, outData3 = self.conv2d(x)

            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

            # 傅里叶变换
            framewise1 = framewise.transpose(1, 2).float()
            X = torch.fft.fft(framewise1, dim=-1, norm="forward")
            X = torch.abs(X)
            framewise1 = X.transpose(1, 2)

            # print(f"framewise shape:{framewise.shape}")
            # print(f"framewise1 shape:{framewise1.shape}")

            conv1d_outputs = self.conv1d(framewise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            lgt = torch.cat(lgt, dim=0)

            conv1d_outputs1 = self.conv1d1(framewise1, len_x)
            # x: T, B, C
            x1 = conv1d_outputs1['visual_feat']
            x1 = x1.permute(2, 0, 1)

            # print(f"x shape:{x.shape} lgt shape:{lgt.shape}")
            # print(f"x1 shape:{x1.shape} lgt1 shape:{lgt1.shape}")

            outputs = self.temporal_model(x, lgt)
            outputs1 = self.temporal_model1(x1, lgt)

            # print(f"outputs shape:{outputs['hidden'].shape}")
            # print(f"outputs1 shape:{outputs1['predictions'].shape}")

            encoderPrediction = self.classifier1(outputs['predictions'])
            logProbs1 = encoderPrediction

            encoderPrediction = self.classifier2(x)
            logProbs2 = encoderPrediction

            encoderPrediction = self.classifier3(outputs1['predictions'])
            logProbs3 = encoderPrediction

            encoderPrediction = self.classifier4(x1)
            logProbs4 = encoderPrediction

            x2 = outputs['predictions'] + outputs1['predictions']
            logProbs5 = self.classifier5(x2)

            if not isTrain:
                logProbs1 = logProbs5
        elif "KFNet" == self.moduleChoice:
            len_x = dataLen
            # batch, temp, channel, height, width = seqData.shape
            batch, temp, one, dim = seqData.shape  # [batch, temp, 1, 84]
            # x = seqData.transpose(1, 2)

            # framewise, outData1, outData2, outData3 = self.conv2d(x)
            framewise = seqData  # [batch, temp, 1, 84]
            # [batch, temp, 512]->[batch, 512, temp]
            # framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            # [batch, temp, 84]->[batch, 84, temp] 注意要轉float不然和lenx類型不一樣conv1d會報錯
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2).float()

            # 傅里叶变换
            framewise1 = framewise.transpose(1, 2).float()  # [batch, temp, 84]
            X = torch.fft.fft(framewise1, dim=-1, norm="forward")
            X = torch.abs(X)
            framewise1 = X.transpose(1, 2)
            # 1D 卷积处理
            conv1d_outputs = self.conv1d(framewise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            lgt = torch.cat(lgt, dim=0)

            conv1d_outputs1 = self.conv1d1(framewise1, len_x)
            # x: T, B, C
            x1 = conv1d_outputs1['visual_feat']
            x1 = x1.permute(2, 0, 1)
            # 双向 LSTM 处理
            outputs = self.temporal_model(x, lgt)
            outputs1 = self.temporal_model1(x1, lgt)

            # 分类器层
            encoderPrediction = self.classifier1(outputs['predictions'])
            logProbs1 = encoderPrediction

            encoderPrediction = self.classifier2(x)
            logProbs2 = encoderPrediction

            encoderPrediction = self.classifier3(outputs1['predictions'])
            logProbs3 = encoderPrediction

            encoderPrediction = self.classifier4(x1)
            logProbs4 = encoderPrediction
            # 最终分类结果
            x2 = outputs['predictions'] + outputs1['predictions']
            logProbs5 = self.classifier5(x2)

            if not isTrain:
                logProbs1 = logProbs5

        return logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, outData1, outData2, outData3