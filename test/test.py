# import torch
# import numpy as np
# import os
# import videoAugmentation
#
# def read_txt_data(file_path):
#     """
#     从 .txt 文件读取数据，并转换为张量。
#     文件结构为 42 行，2 列，列之间用 ',' 分割，每个数据为 0~255 的整数。
#     """
#     data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)  # 形状为 (42, 2)
#     # 转换为形状 (1, 84)
#     data = data.flatten().reshape(1, -1)
#     return data  # (1, 84)
#
# def sample_indices(n):
#     indices = np.linspace(0, n - 1, num=int(n // 1), dtype=int)
#     return indices
#
# transformTxt = videoAugmentation.Compose([
#         videoAugmentation.TemporalRescale(0.2),
#     ])
#
# imageSeqPath = 'data/CE-CSL/mediapipe/dev/A/dev-00002'
#
# ImageSeq = sorted(os.listdir(imageSeqPath))
#
# indices = sample_indices(len(ImageSeq))
#
# frames = [os.path.join(imageSeqPath, i) for i in ImageSeq]
# frames = [frames[i] for i in indices]
#
# # 读取并处理每帧的 .txt 数据
# # imgSeq = torch.stack([self.read_txt_data(img_path) for img_path in frames], dim=0) # (batch, 1, 84)
# imgSeq = [read_txt_data(img_path) for img_path in frames] # (n, 1, 84)
# # print(imgSeq)
# # for i, arr in enumerate(imgSeq):
# #     print(f"Index {i}: shape {arr.shape}")
# shapes = {arr.shape for arr in imgSeq}
# print("Unique shapes in imgSeq:", shapes)
#
# # 對應圖片的transform
# # 将 imgSeq 转换为 NumPy 数组
# imgSeq = np.stack(imgSeq, axis=0)  # 将列表堆叠成一个 numpy 数组(n, 1, 84)
# print(imgSeq.shape)
#
# # 将 NumPy 数组转换为 PyTorch 张量
# imgSeq = torch.from_numpy(imgSeq)  # 转为 Torch 张量，形状 (n, 1, 84)
# print(imgSeq.shape)
#
# # 對齊
# imgSeq = transformTxt(imgSeq)
# print(imgSeq.shape)
hypotheses = " 依靠 回来 。 "
hypotheses = hypotheses.replace(" ", "")
print(f"hypotheses: {hypotheses}")