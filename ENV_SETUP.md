

```
pip install tqdm

pip install imageio[ffmpeg]

pip install einops

```

数据处理 视频转图片集合（花费约4个小时）

```
F:\tmp_HAND_GESTURE\tmp_HAND_GESTURE\.venv\Scripts\python.exe F:\tmp_HAND_GESTURE\TFNet\CE-CSLDataPreProcess.py 
100%|██████████| 55/55 [02:33<00:00,  2.79s/it]
100%|██████████| 53/53 [01:26<00:00,  1.63s/it]
100%|██████████| 54/54 [02:05<00:00,  2.33s/it]
100%|██████████| 46/46 [02:07<00:00,  2.78s/it]
100%|██████████| 51/51 [01:09<00:00,  1.36s/it]
100%|██████████| 50/50 [02:29<00:00,  3.00s/it]
100%|██████████| 54/54 [02:07<00:00,  2.36s/it]
100%|██████████| 9/9 [00:22<00:00,  2.51s/it]
100%|██████████| 9/9 [00:25<00:00,  2.81s/it]
100%|██████████| 29/29 [01:47<00:00,  3.70s/it]
100%|██████████| 53/53 [03:10<00:00,  3.60s/it]
100%|██████████| 52/52 [02:25<00:00,  2.80s/it]
100%|██████████| 55/55 [02:40<00:00,  2.92s/it]
100%|██████████| 57/57 [01:30<00:00,  1.58s/it]
100%|██████████| 42/42 [01:50<00:00,  2.63s/it]
100%|██████████| 47/47 [02:17<00:00,  2.92s/it]
100%|██████████| 54/54 [01:15<00:00,  1.40s/it]
100%|██████████| 58/58 [02:52<00:00,  2.97s/it]
100%|██████████| 56/56 [03:14<00:00,  3.48s/it]
100%|██████████| 9/9 [00:27<00:00,  3.00s/it]
100%|██████████| 9/9 [00:26<00:00,  2.99s/it]
100%|██████████| 38/38 [01:46<00:00,  2.80s/it]
100%|██████████| 46/46 [02:53<00:00,  3.78s/it]
100%|██████████| 29/29 [01:25<00:00,  2.93s/it]
100%|██████████| 491/491 [23:32<00:00,  2.88s/it]
100%|██████████| 492/492 [13:43<00:00,  1.67s/it]
100%|██████████| 504/504 [22:00<00:00,  2.62s/it]
100%|██████████| 507/507 [09:36<00:00,  1.14s/it]
100%|██████████| 495/495 [11:08<00:00,  1.35s/it]
100%|██████████| 492/492 [31:13<00:00,  3.81s/it]
100%|██████████| 490/490 [15:19<00:00,  1.88s/it]
100%|██████████| 101/101 [04:52<00:00,  2.90s/it]
100%|██████████| 66/66 [02:37<00:00,  2.38s/it]
100%|██████████| 330/330 [16:54<00:00,  3.07s/it]
100%|██████████| 501/501 [15:05<00:00,  1.81s/it]
100%|██████████| 507/507 [26:30<00:00,  3.14s/it]
Max Frames Number:520
Min Frames Number:39
Max Video Time:17.33
Min Video Time:1.3
Fps Set:{20.83, 20.75, 20.92, 21.0, 24.08, 24.58, 24.67, 24.42, 28.0, 20.0, 23.0, 24.0, 23.67, 24.75, 30.0, 29.97, 60.0, 22.0, 22.33, 22.08, 22.17, 22.58}
Resolution Set:{(1920, 1080), (3840, 2160), (960, 544), (1008, 752), (568, 320), (1280, 720), (480, 352), (576, 320), (640, 368), (996, 748), (1908, 1080), (1906, 1080)}


Process finished with exit code 0
```

配置文件在params/config.ini

可以修改数据路径 使用模型。。。



# 安装ctcdecode

## windows10安装ctcdecode（不可行，可跳过）

0.4.1 readme

> # ctcdecode
>
> ctcdecode is an implementation of CTC (Connectionist Temporal Classification) beam search decoding for PyTorch.
> C++ code borrowed liberally from Paddle Paddles' [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech).
> It includes swappable scorer support enabling standard beam search, and KenLM-based decoding.
>
> ## Installation
>
> The library is largely self-contained and requires only PyTorch and CFFI. Building the C++ library requires gcc or clang. KenLM language modeling support is also optionally included, and enabled by default.
>
> ```bash
> # get the code
> git clone --recursive https://github.com/parlance/ctcdecode.git
> cd ctcdecode
> pip install .
> ```
>
> 

```
https://github.com/parlance/ctcdecode/releases/tag/0.4.1

git clone --recursive https://github.com/parlance/ctcdecode.git
git fetch --all --tags 
git checkout tags/0.4.1 
pip install .
pip install --upgrade setuptools
```

```
setup.py

from setuptools import setup, find_packages, disutils
```

```
AttributeError: module 'disutils' has no attribute 'ccompiler'
```

![image-20241206225335459](C:\Users\233\AppData\Roaming\Typora\typora-user-images\image-20241206225335459.png)

```
setup.py
from setuptools import setup, find_packages, disutils
from disutils import ccompiler
```

```
Requirements should be satisfied by a PEP 517 installer.
```

```
pip install --use-pep517 .
```

```
error in ctcdecode setup command .....ffi_variable
```

```sh
pip install --upgrade pip setuptools wheel
```

```
F:\tmp_HAND_GESTURE\tmp_HAND_GESTURE\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

应该是cffi包版本太新了？

pip install失败cffi==1.11.4

1.11.5失败

一直到1.13.0都不行

1.15.1可以

依然报错

```
error in ctcdecode setup command .....ffi_variable
```

要是不用0.4.1，用1.0.2

```
pip install ctcdecode
```

```
设置镜像
set http_proxy=
set https_proxy=
```

```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

装1.11.0+cu113，然后装ctcdecode==0.4，因为看到cffi用的是pytorch的



最后还是不行，搜索发现：**Windows暂时不能安装ctcdecode**





## wsl2安装ctcdecode

先按照docker.md卸载10.01

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-wsl-ubuntu-11-3-local_11.3.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-3-local_11.3.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

cudnn

```
cudnn-local-repo-ubuntu1804-8.9.7.29_1.0-1_amd64.deb
```

> 1. ~~sudo apt-get install tar~~
>
> 2. ~~tar -xzvf~~
>
> 3. cp cudnn解压目录下子文件夹`include/`的所有文件到`/usr/local/cuda-x.x/include`
>
> 4. cp cudnn解压目录下子文件夹`lib/`的所有文件到`/usr/local/cuda-x.x/lib`
>
>    sudo cp /tmp/cuda/include/* /usr/local/cuda-11.3/include
>    sudo cp /tmp/cuda/lib64/* /usr/local/cuda-11.3/lib64

> deb安装
>
> cuDNN同样提供了多种安装方式。这里与CUDA一样选择deb方式。注意cuDNN的版本需要与CUDA版本匹配。如果你需要编译TensorFlow或者PyTorch，安装开发时库，否则安装运行时库即可。
>
> https://yinguobing.com/install-cuda113-cudnn8-ubuntu20-04/
>
> [Download cuDNN v8.2.1 (June 7th, 2021), for CUDA 11.x](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse821-113)
>
> ```bash
> sudo dpkg -i libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
> ```
>
> wrong
>
> 需要先安装runtime库？
>
> ```
> sudo dpkg -i libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
> 
> sudo dpkg -i libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
> ```
>
> cuda/include和lib下找不到cudnn，还是用tgz把
>
> cudnn-11.3-linux-x64-v8.2.1.32.tgz



pytorch

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```



在wsl安装ctcdecode1.0.2成功

```
先装https://blog.csdn.net/weixin_50578022/article/details/129546132
不能直接pip
用git装 python install .
```





windows pycharm链接wsl conda

删除tran-00026（1）。mp4（video和images下）删除train-00848(1) train-00845(1)

```
except: imageName:train-00845(1) imageSeqPath:data/CE-CSL/images/train/B/train-00845(1)
except: imageName:train-00848(1) imageSeqPath:data/CE-CSL/images/train/B/train-00848(1)


```

config.ini  之前path填成video导致读取mp4而产生异常，data长度一直报错为0

```
# CE-CSL
trainDataPath = data/CE-CSL/images/train
validDataPath = data/CE-CSL/images/dev
testDataPath = data/CE-CSL/images/test
trainLabelPath = data/CE-CSL/label/train.csv
validLabelPath = data/CE-CSL/label/dev.csv
testLabelPath = data/CE-CSL/label/test.csv

bestModuleSavePath = module/bestMoudleNet.pth
currentModuleSavePath = module/currentMoudleNet.pth
```

```
报错outofmemory。。。
```

要修改pkg的源代码需要取wsl修改，pycharm中改不了

警告cuda memory为0

在config。ini中设置pinmMemory = 0 禁用pinmemory（加速cpu和gpu数据传输的）

> 锁页内存理解（pinned memory or page locked memory）：https://blog.csdn.net/dgh_dean/article/details/53130871
> What is the disadvantage of using pin_memory： https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
>
> pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
>
> 主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。
>
> 而显卡中的显存全部是锁页内存！
>
> 当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确保每一个炼丹玩家都有高端设备，因此pin_memory默认为False。

开启pinmemory。pycharm使用wsl运行训练模型，显存使用1.多G会爆cuda显存不够

关掉pinmemory后显存使用几乎满了，有新的报错

查了也是因为显存爆了





### 使用抽取的少量数据训练

训了一晚上

```

开始训练模型
  0%|          | 0/6 [00:00<?, ?it/s]/mnt/f/tmp_HAND_GESTURE/TFNet/Train.py:179: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  targetOutData = [torch.tensor(yi).to(device) for yi in label]
100%|██████████| 6/6 [08:36<00:00, 86.15s/it]
epoch: 0, trainLoss: 835.65448, lr: 0.000100
开始验证模型
  0%|          | 0/12 [00:00<?, ?it/s]/mnt/f/tmp_HAND_GESTURE/TFNet/Train.py:261: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  targetOutData = [torch.tensor(yi).to(device) for yi in label]
100%|██████████| 12/12 [01:36<00:00,  8.04s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 161.48443, werScore: 335.56
bestLoss: 161.48443, beatEpoch: 0, bestWerScore: 335.56, bestWerScoreEpoch: 0
100%|██████████| 6/6 [08:06<00:00, 81.07s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 1, trainLoss: 765.20094, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:35<00:00, 12.99s/it]
validLoss: 151.75026, werScore: 285.59
bestLoss: 151.75026, beatEpoch: 1, bestWerScore: 285.59, bestWerScoreEpoch: 1
100%|██████████| 6/6 [08:10<00:00, 81.77s/it]
epoch: 2, trainLoss: 685.87828, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:09<00:00, 10.77s/it]
validLoss: 144.21304, werScore: 267.78
bestLoss: 144.21304, beatEpoch: 2, bestWerScore: 267.78, bestWerScoreEpoch: 2
100%|██████████| 6/6 [07:12<00:00, 72.03s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 3, trainLoss: 639.70395, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:10<00:00, 10.85s/it]
validLoss: 139.33148, werScore: 248.35
bestLoss: 139.33148, beatEpoch: 3, bestWerScore: 248.35, bestWerScoreEpoch: 3
100%|██████████| 6/6 [08:01<00:00, 80.32s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 4, trainLoss: 668.19236, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:05<00:00, 10.47s/it]
validLoss: 116.76685, werScore: 161.41
bestLoss: 116.76685, beatEpoch: 4, bestWerScore: 161.41, bestWerScoreEpoch: 4
100%|██████████| 6/6 [07:28<00:00, 74.68s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 5, trainLoss: 603.98242, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:06<00:00, 10.52s/it]
validLoss: 114.61937, werScore: 157.16
bestLoss: 114.61937, beatEpoch: 5, bestWerScore: 157.16, bestWerScoreEpoch: 5
100%|██████████| 6/6 [07:51<00:00, 78.55s/it]
epoch: 6, trainLoss: 587.65658, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:29<00:00, 12.46s/it]
validLoss: 113.00956, werScore: 151.86
bestLoss: 113.00956, beatEpoch: 6, bestWerScore: 151.86, bestWerScoreEpoch: 6
100%|██████████| 6/6 [07:40<00:00, 76.74s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 7, trainLoss: 570.11069, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:11<00:00, 11.00s/it]
validLoss: 87.89896, werScore: 91.57
bestLoss: 87.89896, beatEpoch: 7, bestWerScore: 91.57, bestWerScoreEpoch: 7
100%|██████████| 6/6 [07:42<00:00, 77.12s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 8, trainLoss: 504.99503, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:20<00:00, 11.73s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 64.96075, werScore: 86.78
bestLoss: 64.96075, beatEpoch: 8, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:49<00:00, 78.26s/it]
epoch: 9, trainLoss: 449.34340, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:10<00:00, 10.90s/it]
validLoss: 64.89002, werScore: 86.78
bestLoss: 64.89002, beatEpoch: 9, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:34<00:00, 85.72s/it]
epoch: 10, trainLoss: 452.43743, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:09<00:00, 10.77s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 64.42114, werScore: 90.53
bestLoss: 64.42114, beatEpoch: 10, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:52<00:00, 78.80s/it]
epoch: 11, trainLoss: 472.10279, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [01:40<00:00,  8.39s/it]
validLoss: 42.69316, werScore: 88.87
bestLoss: 42.69316, beatEpoch: 11, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:04<00:00, 80.69s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 12, trainLoss: 395.73341, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:13<00:00, 11.09s/it]
validLoss: 30.39930, werScore: 92.62
bestLoss: 30.39930, beatEpoch: 12, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:56<00:00, 79.40s/it]
epoch: 13, trainLoss: 328.77748, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:14<00:00, 11.20s/it]
validLoss: 29.14506, werScore: 95.74
bestLoss: 29.14506, beatEpoch: 13, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:08<00:00, 71.47s/it]
epoch: 14, trainLoss: 358.21482, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:20<00:00, 11.73s/it]
validLoss: 25.39813, werScore: 100.00
bestLoss: 25.39813, beatEpoch: 14, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:42<00:00, 77.14s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 15, trainLoss: 287.65969, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [01:40<00:00,  8.41s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 26.34323, werScore: 100.00
bestLoss: 26.34323, beatEpoch: 15, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:54<00:00, 79.16s/it]
epoch: 16, trainLoss: 301.12249, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:02<00:00, 10.19s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 28.39087, werScore: 100.00
bestLoss: 28.39087, beatEpoch: 16, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:56<00:00, 79.43s/it]
epoch: 17, trainLoss: 272.85161, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:06<00:00, 10.52s/it]
validLoss: 30.39493, werScore: 100.00
bestLoss: 30.39493, beatEpoch: 17, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:26<00:00, 74.47s/it]
epoch: 18, trainLoss: 258.29917, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:53<00:00, 14.45s/it]
validLoss: 30.55381, werScore: 100.00
bestLoss: 30.55381, beatEpoch: 18, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:27<00:00, 74.51s/it]
epoch: 19, trainLoss: 251.44445, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:29<00:00, 12.46s/it]
validLoss: 31.63986, werScore: 100.00
bestLoss: 31.63986, beatEpoch: 19, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:28<00:00, 74.71s/it]
epoch: 20, trainLoss: 283.13517, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [01:57<00:00,  9.76s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 31.88916, werScore: 100.00
bestLoss: 31.88916, beatEpoch: 20, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:27<00:00, 74.54s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 21, trainLoss: 299.97713, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:07<00:00, 10.60s/it]
validLoss: 31.41565, werScore: 100.00
bestLoss: 31.41565, beatEpoch: 21, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:30<00:00, 75.01s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 22, trainLoss: 294.00463, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:12<00:00, 11.05s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 31.68893, werScore: 100.00
bestLoss: 31.68893, beatEpoch: 22, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:26<00:00, 84.39s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 23, trainLoss: 274.15168, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:12<00:00, 11.06s/it]
validLoss: 30.36517, werScore: 100.00
bestLoss: 30.36517, beatEpoch: 23, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:08<00:00, 81.42s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 24, trainLoss: 287.60315, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:00<00:00, 10.05s/it]
validLoss: 28.70852, werScore: 100.00
bestLoss: 28.70852, beatEpoch: 24, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:07<00:00, 81.21s/it]
epoch: 25, trainLoss: 291.49273, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:18<00:00, 11.52s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 26.84230, werScore: 100.00
bestLoss: 26.84230, beatEpoch: 25, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:27<00:00, 74.55s/it]
epoch: 26, trainLoss: 240.69461, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:08<00:00, 10.69s/it]
validLoss: 25.50509, werScore: 100.00
bestLoss: 25.50509, beatEpoch: 26, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:48<00:00, 78.16s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 27, trainLoss: 254.04739, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:32<00:00, 12.75s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 24.81076, werScore: 100.00
bestLoss: 24.81076, beatEpoch: 27, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:10<00:00, 81.82s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 28, trainLoss: 262.84143, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:29<00:00, 12.45s/it]
validLoss: 24.83436, werScore: 100.00
bestLoss: 24.83436, beatEpoch: 28, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:41<00:00, 76.88s/it]
epoch: 29, trainLoss: 251.74366, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:19<00:00, 11.63s/it]
validLoss: 24.71437, werScore: 98.33
bestLoss: 24.71437, beatEpoch: 29, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:19<00:00, 73.23s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 30, trainLoss: 262.65554, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:12<00:00, 11.01s/it]
validLoss: 25.13032, werScore: 92.62
bestLoss: 25.13032, beatEpoch: 30, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:31<00:00, 75.27s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 31, trainLoss: 264.72740, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:21<00:00, 11.78s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 25.47516, werScore: 88.87
bestLoss: 25.47516, beatEpoch: 31, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:34<00:00, 75.73s/it]
epoch: 32, trainLoss: 257.36278, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:21<00:00, 11.77s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 25.72166, werScore: 88.87
bestLoss: 25.72166, beatEpoch: 32, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:20<00:00, 83.45s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 33, trainLoss: 253.81935, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:02<00:00, 10.17s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 25.87929, werScore: 88.87
bestLoss: 25.87929, beatEpoch: 33, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:10<00:00, 81.67s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 34, trainLoss: 245.77248, lr: 0.000100
开始验证模型
100%|██████████| 12/12 [02:07<00:00, 10.65s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 25.69526, werScore: 88.87
bestLoss: 25.69526, beatEpoch: 34, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:41<00:00, 77.00s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 35, trainLoss: 247.92678, lr: 0.000020
开始验证模型
100%|██████████| 12/12 [02:19<00:00, 11.60s/it]
validLoss: 25.74614, werScore: 88.87
bestLoss: 25.74614, beatEpoch: 35, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:55<00:00, 79.21s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 36, trainLoss: 257.00435, lr: 0.000020
开始验证模型
100%|██████████| 12/12 [02:06<00:00, 10.58s/it]
validLoss: 25.39394, werScore: 88.87
bestLoss: 25.39394, beatEpoch: 36, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:38<00:00, 76.45s/it]
epoch: 37, trainLoss: 232.46864, lr: 0.000020
开始验证模型
100%|██████████| 12/12 [02:09<00:00, 10.78s/it]
validLoss: 25.35107, werScore: 88.87
bestLoss: 25.35107, beatEpoch: 37, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:10<00:00, 81.72s/it]
epoch: 38, trainLoss: 240.99702, lr: 0.000020
开始验证模型
100%|██████████| 12/12 [02:02<00:00, 10.25s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 25.35597, werScore: 88.87
bestLoss: 25.35597, beatEpoch: 38, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:19<00:00, 83.20s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 39, trainLoss: 238.14763, lr: 0.000020
开始验证模型
100%|██████████| 12/12 [02:06<00:00, 10.53s/it]
validLoss: 25.22559, werScore: 88.87
bestLoss: 25.22559, beatEpoch: 39, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:11<00:00, 71.84s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 40, trainLoss: 226.37475, lr: 0.000020
开始验证模型
100%|██████████| 12/12 [02:01<00:00, 10.11s/it]
validLoss: 25.22299, werScore: 88.87
bestLoss: 25.22299, beatEpoch: 40, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:30<00:00, 75.06s/it]
epoch: 41, trainLoss: 226.83960, lr: 0.000020
开始验证模型
100%|██████████| 12/12 [02:11<00:00, 10.99s/it]
validLoss: 25.30340, werScore: 88.87
bestLoss: 25.30340, beatEpoch: 41, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:06<00:00, 81.08s/it]
epoch: 42, trainLoss: 237.26000, lr: 0.000020
开始验证模型
100%|██████████| 12/12 [02:21<00:00, 11.76s/it]
validLoss: 25.17463, werScore: 88.87
bestLoss: 25.17463, beatEpoch: 42, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:56<00:00, 79.41s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 43, trainLoss: 243.67443, lr: 0.000020
开始验证模型
100%|██████████| 12/12 [02:03<00:00, 10.26s/it]
validLoss: 24.85764, werScore: 92.62
bestLoss: 24.85764, beatEpoch: 43, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:52<00:00, 78.79s/it]
epoch: 44, trainLoss: 216.23798, lr: 0.000020
开始验证模型
100%|██████████| 12/12 [02:10<00:00, 10.86s/it]
validLoss: 24.76623, werScore: 93.66
bestLoss: 24.76623, beatEpoch: 44, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:01<00:00, 70.32s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 45, trainLoss: 222.73826, lr: 0.000004
开始验证模型
100%|██████████| 12/12 [02:09<00:00, 10.82s/it]
validLoss: 24.79736, werScore: 93.66
bestLoss: 24.79736, beatEpoch: 45, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:14<00:00, 82.48s/it]
epoch: 46, trainLoss: 248.74166, lr: 0.000004
开始验证模型
100%|██████████| 12/12 [02:05<00:00, 10.45s/it]
validLoss: 24.76104, werScore: 93.66
bestLoss: 24.76104, beatEpoch: 46, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:56<00:00, 79.43s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 47, trainLoss: 228.74486, lr: 0.000004
开始验证模型
100%|██████████| 12/12 [02:17<00:00, 11.43s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 24.76564, werScore: 93.66
bestLoss: 24.76564, beatEpoch: 47, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:30<00:00, 75.16s/it]
epoch: 48, trainLoss: 217.57686, lr: 0.000004
开始验证模型
100%|██████████| 12/12 [01:48<00:00,  9.05s/it]
validLoss: 24.77337, werScore: 93.66
bestLoss: 24.77337, beatEpoch: 48, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:55<00:00, 79.17s/it]
epoch: 49, trainLoss: 216.89782, lr: 0.000004
开始验证模型
100%|██████████| 12/12 [02:19<00:00, 11.62s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 24.82139, werScore: 92.62
bestLoss: 24.82139, beatEpoch: 49, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:07<00:00, 81.33s/it]
  0%|          | 0/12 [00:00<?, ?it/s]epoch: 50, trainLoss: 217.47339, lr: 0.000004
开始验证模型
100%|██████████| 12/12 [01:57<00:00,  9.83s/it]
validLoss: 24.68719, werScore: 93.66
bestLoss: 24.68719, beatEpoch: 50, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [07:42<00:00, 77.05s/it]
epoch: 51, trainLoss: 229.34007, lr: 0.000004
开始验证模型
100%|██████████| 12/12 [01:45<00:00,  8.76s/it]
validLoss: 24.68892, werScore: 93.66
bestLoss: 24.68892, beatEpoch: 51, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:01<00:00, 80.23s/it]
epoch: 52, trainLoss: 247.92719, lr: 0.000004
开始验证模型
100%|██████████| 12/12 [02:07<00:00, 10.61s/it]
  0%|          | 0/6 [00:00<?, ?it/s]validLoss: 24.63358, werScore: 93.66
bestLoss: 24.63358, beatEpoch: 52, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [06:43<00:00, 67.18s/it]
epoch: 53, trainLoss: 230.56486, lr: 0.000004
开始验证模型
100%|██████████| 12/12 [02:00<00:00, 10.03s/it]
validLoss: 24.68841, werScore: 93.66
bestLoss: 24.68841, beatEpoch: 53, bestWerScore: 86.78, bestWerScoreEpoch: 8
100%|██████████| 6/6 [08:37<00:00, 86.19s/it]
epoch: 54, trainLoss: 253.30039, lr: 0.000004
开始验证模型
100%|██████████| 12/12 [02:21<00:00, 11.80s/it]
validLoss: 24.65093, werScore: 93.66
bestLoss: 24.65093, beatEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8

Process finished with exit code 0

```

原代码(TFNet)这里没有判断。。。(一处训练，一处验证)

```
if currentLoss < bestLoss:
	bestLoss = currentLoss
	bestLossEpoch = i + offset -1
```

原代码训练时，是看wescore，将最好的werscore模型放入最好模型文件

因为训练时记录的最好loss和最好loss的epoch有点问题，所以验证的时候有点问题。

验证时看werscore就好了

```

已加载预训练模型 epoch: 55, bestLoss: 24.65093, bestEpoch: 54, werScore: 86.78241, bestEpoch: 8
开始验证模型
  0%|          | 0/12 [00:00<?, ?it/s]/mnt/f/tmp_HAND_GESTURE/TFNet/Train.py:370: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  targetOutData = [torch.tensor(yi).to(device) for yi in label]
100%|██████████| 12/12 [00:17<00:00,  1.44s/it]
testLoss: 161.48443, werScore: 335.56
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 335.56, bestWerScoreEpoch: 0
开始验证模型
100%|██████████| 12/12 [00:51<00:00,  4.31s/it]
testLoss: 151.75026, werScore: 285.59
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 285.59, bestWerScoreEpoch: 1
开始验证模型
100%|██████████| 12/12 [00:38<00:00,  3.19s/it]
testLoss: 144.21304, werScore: 267.78
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 267.78, bestWerScoreEpoch: 2
开始验证模型
100%|██████████| 12/12 [00:47<00:00,  3.93s/it]
testLoss: 139.33148, werScore: 248.35
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 248.35, bestWerScoreEpoch: 3
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [00:51<00:00,  4.27s/it]
testLoss: 116.76685, werScore: 161.41
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 161.41, bestWerScoreEpoch: 4
开始验证模型
100%|██████████| 12/12 [00:36<00:00,  3.08s/it]
testLoss: 114.61937, werScore: 157.16
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 157.16, bestWerScoreEpoch: 5
开始验证模型
100%|██████████| 12/12 [00:49<00:00,  4.10s/it]
testLoss: 113.00956, werScore: 151.86
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 151.86, bestWerScoreEpoch: 6
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [00:42<00:00,  3.56s/it]
testLoss: 87.89896, werScore: 91.57
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 91.57, bestWerScoreEpoch: 7
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [00:51<00:00,  4.27s/it]
testLoss: 64.96075, werScore: 86.78
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:21<00:00,  6.81s/it]
testLoss: 64.89002, werScore: 86.78
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [00:50<00:00,  4.17s/it]
testLoss: 64.42114, werScore: 90.53
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [00:58<00:00,  4.91s/it]
testLoss: 42.69316, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:10<00:00,  5.85s/it]
testLoss: 30.39930, werScore: 92.62
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:00<00:00,  5.04s/it]
testLoss: 29.14506, werScore: 95.74
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:16<00:00,  6.34s/it]
testLoss: 25.39813, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [00:49<00:00,  4.12s/it]
testLoss: 26.34323, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [00:41<00:00,  3.43s/it]
testLoss: 28.39087, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [00:48<00:00,  4.06s/it]
testLoss: 30.39493, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:47<00:00,  8.93s/it]
testLoss: 30.55381, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [02:06<00:00, 10.57s/it]
testLoss: 31.63986, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [01:25<00:00,  7.14s/it]
testLoss: 31.88916, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [00:57<00:00,  4.77s/it]
testLoss: 31.41565, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:29<00:00,  7.43s/it]
testLoss: 31.68893, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:12<00:00,  6.08s/it]
testLoss: 30.36517, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:47<00:00,  8.98s/it]
testLoss: 28.70852, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:13<00:00,  6.10s/it]
testLoss: 26.84230, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [01:24<00:00,  7.08s/it]
testLoss: 25.50509, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [01:09<00:00,  5.76s/it]
testLoss: 24.81076, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:11<00:00,  5.99s/it]
testLoss: 24.83436, werScore: 100.00
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [01:16<00:00,  6.34s/it]
testLoss: 24.71437, werScore: 98.33
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [00:55<00:00,  4.66s/it]
testLoss: 25.13032, werScore: 92.62
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:16<00:00,  6.37s/it]
testLoss: 25.47516, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:04<00:00,  5.36s/it]
testLoss: 25.72166, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [01:15<00:00,  6.33s/it]
testLoss: 25.87929, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [01:15<00:00,  6.31s/it]
testLoss: 25.69526, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [00:54<00:00,  4.51s/it]
testLoss: 25.74614, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:33<00:00,  7.81s/it]
testLoss: 25.39394, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:30<00:00,  7.53s/it]
testLoss: 25.35107, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:02<00:00,  5.24s/it]
testLoss: 25.35597, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:16<00:00,  6.37s/it]
testLoss: 25.22559, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:00<00:00,  5.01s/it]
testLoss: 25.22299, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:02<00:00,  5.19s/it]
testLoss: 25.30340, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [00:49<00:00,  4.16s/it]
testLoss: 25.17463, werScore: 88.87
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [01:01<00:00,  5.15s/it]
testLoss: 24.85764, werScore: 92.62
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [01:15<00:00,  6.30s/it]
testLoss: 24.76623, werScore: 93.66
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [00:59<00:00,  4.98s/it]
testLoss: 24.79736, werScore: 93.66
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:08<00:00,  5.71s/it]
testLoss: 24.76104, werScore: 93.66
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:11<00:00,  5.93s/it]
testLoss: 24.76564, werScore: 93.66
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:03<00:00,  5.32s/it]
testLoss: 24.77337, werScore: 93.66
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:01<00:00,  5.14s/it]
testLoss: 24.82139, werScore: 92.62
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:05<00:00,  5.50s/it]
testLoss: 24.68719, werScore: 93.66
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
  0%|          | 0/12 [00:00<?, ?it/s]开始验证模型
100%|██████████| 12/12 [01:10<00:00,  5.83s/it]
testLoss: 24.68892, werScore: 93.66
bestLoss: 24.65093, bestEpoch: 54, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:30<00:00,  7.58s/it]
testLoss: 24.63358, werScore: 93.66
bestLoss: 24.63358, bestEpoch: 52, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:32<00:00,  7.68s/it]
testLoss: 24.68841, werScore: 93.66
bestLoss: 24.63358, bestEpoch: 52, bestWerScore: 86.78, bestWerScoreEpoch: 8
开始验证模型
100%|██████████| 12/12 [01:19<00:00,  6.64s/it]
testLoss: 24.65093, werScore: 93.66
bestLoss: 24.63358, bestEpoch: 52, bestWerScore: 86.78, bestWerScoreEpoch: 8

Process finished with exit code 0

```





#  mediapipe

```
multi_hand_landmarks  一个图像的多个手
```



```
hand_landmarks in results.multi_hand_landmarks  一个图像多个手的一个手
```

```
landmark in hand_landmarks.landmark:   
一个手的1个点，包含([landmark.x, landmark.y, landmark.z])
```

得到的是屏幕坐标，如何得到归一化坐标

原始1920x1080 各种分辨率都有，源代码是都处理为256x256，这里因为直接用256x256的mediapipe检测效果不好，很多检测不出的，所以先在原分辨率下进行检测，然后归一化到256x256的坐标。（或者1比1的相对坐标（0.5,0.5））

但是我覺得TFNEt對複雜環境的魯棒性並不好，查看代碼可以發現，作者對原圖進行128閾值的二值化。

那麽背景裏面白色的很可能為1，輸入整張圖片不僅增加了運算複雜度，還降低了魯棒性

這裏想了兩種方法，一種使用mediapipe提取關鍵點坐標代替cnn特徵；一種是用bodypix提取左右胳膊和左右手的像素（去掉其他部分，因爲手語識別只關注手部），然後將非手部凃黑色。

前者可以去除環境干擾，而且加快運算速率（3\*256\*256）-》（42*2）也就是（1\*84）這裏不考慮z坐標，因爲值本身很小，而且變化不大。

後者也可以去除環境干擾，但無法加快運算速率。



```
hands:
landmark {
  x: 0.359004796
  y: 0.841688156
  z: 1.363601e-007
}
landmark {
  x: 0.378225654
  y: 0.825019717
  z: -0.00762806972
}
landmark {
  x: 0.396989167
  y: 0.824872851
  z: -0.0144234728
}
landmark {
  x: 0.413537711
  y: 0.826220691
  z: -0.0196237341
}
landmark {
  x: 0.426838607
  y: 0.820527673
  z: -0.0248847362
}
landmark {
  x: 0.400595725
  y: 0.862684488
  z: -0.0191460755
}
landmark {
  x: 0.423701197
  y: 0.898436189
  z: -0.0262732711
}
landmark {
  x: 0.437036842
  y: 0.915997326
  z: -0.0289250948
}
landmark {
  x: 0.447005749
  y: 0.928076446
  z: -0.0297155
}
landmark {
  x: 0.390554816
  y: 0.889089942
  z: -0.018686587
}
landmark {
  x: 0.417742521
  y: 0.923398614
  z: -0.0250277873
}
landmark {
  x: 0.433483809
  y: 0.939809144
  z: -0.0269385669
}
landmark {
  x: 0.445319
  y: 0.950073957
  z: -0.0275971089
}
landmark {
  x: 0.38180694
  y: 0.908302665
  z: -0.0180329587
}
landmark {
  x: 0.408368289
  y: 0.939082861
  z: -0.0230580438
}
landmark {
  x: 0.424037457
  y: 0.952567518
  z: -0.0241835695
}
landmark {
  x: 0.435727209
  y: 0.960674465
  z: -0.0242071413
}
landmark {
  x: 0.375757575
  y: 0.921020627
  z: -0.0176066589
}
landmark {
  x: 0.396830678
  y: 0.942823946
  z: -0.0211991
}
landmark {
  x: 0.409857154
  y: 0.953528762
  z: -0.0211419947
}
landmark {
  x: 0.420426428
  y: 0.960379
  z: -0.0201784689
}

pose:
landmark {
  x: 0.415647328
  y: 0.198396444
  z: -0.654564261
  visibility: 0.999891281
}
landmark {
  x: 0.432775021
  y: 0.164731324
  z: -0.628566384
  visibility: 0.999777853
}
landmark {
  x: 0.442325145
  y: 0.165789902
  z: -0.628494143
  visibility: 0.99967
}
landmark {
  x: 0.450020254
  y: 0.167324901
  z: -0.628495932
  visibility: 0.99968338
}
landmark {
  x: 0.403202623
  y: 0.164758086
  z: -0.624168038
  visibility: 0.999867678
}
landmark {
  x: 0.394965976
  y: 0.165715575
  z: -0.624219835
  visibility: 0.99984
}
landmark {
  x: 0.389428347
  y: 0.1670627
  z: -0.624417126
  visibility: 0.999889374
}
landmark {
  x: 0.46472311
  y: 0.18927747
  z: -0.426819891
  visibility: 0.999643564
}
landmark {
  x: 0.383298814
  y: 0.190018773
  z: -0.401501864
  visibility: 0.999863863
}
landmark {
  x: 0.435700417
  y: 0.243445933
  z: -0.575503945
  visibility: 0.999970078
}
landmark {
  x: 0.399464667
  y: 0.240038
  z: -0.56866622
  visibility: 0.999981642
}
landmark {
  x: 0.527920425
  y: 0.418890178
  z: -0.289927214
  visibility: 0.999665737
}
landmark {
  x: 0.326101035
  y: 0.415323198
  z: -0.236421451
  visibility: 0.999856591
}
landmark {
  x: 0.592568815
  y: 0.665109336
  z: -0.24457328
  visibility: 0.991444409
}
landmark {
  x: 0.269563496
  y: 0.673306465
  z: -0.191393897
  visibility: 0.994168043
}
landmark {
  x: 0.473075539
  y: 0.809295833
  z: -0.39038679
  visibility: 0.779109895
}
landmark {
  x: 0.360799402
  y: 0.832422
  z: -0.378983349
  visibility: 0.87541765
}
landmark {
  x: 0.436569512
  y: 0.880468607
  z: -0.454602718
  visibility: 0.502907336
}
landmark {
  x: 0.388519704
  y: 0.897592127
  z: -0.435722411
  visibility: 0.686738372
}
landmark {
  x: 0.427178174
  y: 0.848899662
  z: -0.452397019
  visibility: 0.51774025
}
landmark {
  x: 0.401683509
  y: 0.862992346
  z: -0.454338
  visibility: 0.703485787
}
landmark {
  x: 0.432254672
  y: 0.822800159
  z: -0.390338242
  visibility: 0.539714038
}
landmark {
  x: 0.39509353
  y: 0.840719879
  z: -0.38650468
  visibility: 0.697405398
}
landmark {
  x: 0.494734168
  y: 0.924396396
  z: -0.0442407653
  visibility: 0.941738904
}
landmark {
  x: 0.362955928
  y: 0.938235879
  z: 0.0451873206
  visibility: 0.945709407
}
landmark {
  x: 0.489959359
  y: 1.31276345
  z: 0.03415557
  visibility: 0.393456042
}
landmark {
  x: 0.373027027
  y: 1.32114077
  z: 0.210402295
  visibility: 0.35201022
}
landmark {
  x: 0.495662361
  y: 1.60522842
  z: 0.537524462
  visibility: 0.0378696695
}
landmark {
  x: 0.376626551
  y: 1.62078333
  z: 0.635824
  visibility: 0.0544293337
}
landmark {
  x: 0.500703156
  y: 1.65202975
  z: 0.574997187
  visibility: 0.0244327076
}
landmark {
  x: 0.379050583
  y: 1.66880798
  z: 0.673434317
  visibility: 0.0754769
}
landmark {
  x: 0.477574766
  y: 1.73200572
  z: 0.33100757
  visibility: 0.0207985211
}
landmark {
  x: 0.378341377
  y: 1.74636984
  z: 0.414049178
  visibility: 0.0668645799
}
```

第一帧只有一只手



要处理没有手的情况

就是没有手的话填充什么数据

缺一个手填 21x3x‘0’ 两个手填两个 21x3x‘0’ 

要保证左手右手顺序

multi_hand_landmarks检测的左手在前还是右手在前

> 将关键点存入txt文件
>
> 根据handedness信息，index代表multi_hand_landmarks中的hand_landmarks排序
>
> label代表对应index的hand_landmarks对应左右手
>
> 如果根据handedness信息有2个手，如果2个手一左一右，则先写入左手landmarks，再写入右手landmarks；
>
> 需要先排序
>
> 根据升序排序的handedness_info，此时遍历handedness_info的index顺序为0，1，与遍历multi_hand_landmarks的hand_landmarks的顺序一致，判断当前hand_landmarks的index，查询handedness_info的label，如果为左就设置left_landmarks，如果为右就设置right_landmarks
>
> 将 `left_landmarks` 和 `right_landmarks` 初始化为包含21个landmark的列表，其中每个landmark的坐标都是 `(0, 0, 0)`，
>
> 
>
> 如果是2个左或者2个右，则当作只有1个手，选取第1个作为手，进入只有1个手的判断
>
> 
>
> 如果根据handedness信息有1个手，如果有左手，则先写入左手landmarks，再写入21x*3个0；如果有右手，则先写入21*3个0，再写入右手landmarks
>
> 



不知道为什么用作者的imageio的检测效果不好，很多用cv2读取能检测的都不能检测到

有时候是两只手，但是都识别为同一手，那么就会少一只手，这种情况会造成不连续，可能准确率降低，所以

只绘制左手，可以看到，左手不是很稳定的待在左边，而是有时候再两只手反复横跳

所以根据手腕区别左右手

【坐标系图片】

```
# 一个手的情况，根据mediapipe info的label判断
# 两个手的情况，根据手腕的x坐标来判断左右手
```



最后还要归一化！！！！

变为1：1的坐标

因为为相对坐标，只要x和y都乘以256就可以划到256 * 256的坐标（uint8）

x和y坐标，可以扩大256，加大它们之间的区别，为什么不直接用浮点呢？因为为了加速运算嘛。但是为了快速计算，还是用int8

因为z坐标变化比较小，而且z坐标本来就比较小，所以不考虑z坐标



x和y有可能大于1！！！！！！！！！如果大于1要变为1

也有可能小于0！！！！！！！！！如果小于0要变为0





20:53 开始提取特征

我服了，預處理的時候，沒有手的情況應該也算results。multihand裏了，我是在判斷results。multihand沒有的話就填充全0，看來沒有填充

```
/mnt/f/tmp_HAND_GESTURE/TFNet/mediaPipeDataCheck.py:10: UserWarning: loadtxt: input contained no data: "data/CE-CSL/mediapipe/dev/A/dev-00001/00170.txt"
  data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)  # 尝试读取文件
/mnt/f/tmp_HAND_GESTURE/TFNet/mediaPipeDataCheck.py:10: UserWarning: loadtxt: input contained no data: "data/CE-CSL/mediapipe/dev/A/dev-00008/00056.txt"
  data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)  # 尝试读取文件
/mnt/f/tmp_HAND_GESTURE/TFNet/mediaPipeDataCheck.py:10: UserWarning: loadtxt: input contained no data: "data/CE-CSL/mediapipe/dev/A/dev-00008/00057.txt"
  data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)  # 尝试读取文件
/mnt/f/tmp_HAND_GESTURE/TFNet/mediaPipeDataCheck.py:10: UserWarning: loadtxt: input contained no data: "data/CE-CSL/mediapipe/dev/A/dev-00011/00038.txt"
  data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)  # 尝试读取文件
/mnt/f/tmp_HAND_GESTURE/TFNet/mediaPipeDataCheck.py:10: UserWarning: loadtxt: input contained no data: "data/CE-CSL/mediapipe/dev/A/dev-00014/00138.txt"
  data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)  # 尝试读取文件
/mnt/f/tmp_HAND_GESTURE/TFNet/mediaPipeDataCheck.py:10: UserWarning: loadtxt: input contained no data: "data/CE-CSL/mediapipe/dev/A/dev-00018/00097.txt"
  data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)  # 尝试读取文件
/mnt/f/tmp_HAND_GESTURE/TFNet/mediaPipeDataCheck.py:10: UserWarning: loadtxt: input contained no data: "data/CE-CSL/mediapipe/dev/A/dev-00020/00183.txt"
  data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)  # 尝试读取文件
/mnt/f/tmp_HAND_GESTURE/TFNet/mediaPipeDataCheck.py:10: UserWarning: loadtxt: input contained no data: "data/CE-CSL/mediapipe/dev/A/dev-00021/00131.txt"
  data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)  # 尝试读取文件
```



```
Found (1, 0) shape in file: data/CE-CSL/mediapipe/dev/A/dev-00002/00275.txt
```

```
Total files padded: 1877
```

耗時：1臺電腦3小時 + 2臺電腦*4小時 共 11小時



# 修改網絡結構



序列前後填充，為同一長度



修改後主要用到是cpu，訓練明顯加快，之前gpu占用基本是滿的



驗證時報錯

```

未加载预训练模型 epoch: 0, bestLoss: 65535, bestEpoch: 0, werScore: 65535.00000, bestEpoch: 0
开始训练模型
  0%|          | 0/2486 [00:00<?, ?it/s]/mnt/f/tmp_HAND_GESTURE/TFNet/Train.py:187: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  targetOutData = [torch.tensor(yi).to(device) for yi in label]
100%|██████████| 2486/2486 [38:47<00:00,  1.07it/s]
epoch: 0, trainLoss: 197.79173, lr: 0.000100
开始验证模型
  0%|          | 0/515 [00:00<?, ?it/s]torch.Size([1, 84])
torch.Size([1, 84])
torch.Size([1, 84])
torch.Size([1, 84])
torch.Size([1, 84])
torch.Size([1, 84])
torch.Size([1, 84])
  0%|          | 0/515 [00:05<?, ?it/s]
Traceback (most recent call last):
  File "/mnt/f/tmp_HAND_GESTURE/TFNet/SLR.py", line 13, in <module>
    main()
  File "/mnt/f/tmp_HAND_GESTURE/TFNet/SLR.py", line 8, in main
    train(configParams, isTrain=True, isCalc=False)
  File "/mnt/f/tmp_HAND_GESTURE/TFNet/Train.py", line 243, in train
    for Dict in tqdm(validLoader):
  File "/home/yuki/anaconda3/envs/TFNet/lib/python3.9/site-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
  File "/home/yuki/anaconda3/envs/TFNet/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 530, in __next__
    data = self._next_data()
  File "/home/yuki/anaconda3/envs/TFNet/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1224, in _next_data
    return self._process_data(data)
  File "/home/yuki/anaconda3/envs/TFNet/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1250, in _process_data
    data.reraise()
  File "/home/yuki/anaconda3/envs/TFNet/lib/python3.9/site-packages/torch/_utils.py", line 457, in reraise
    raise exception
UnboundLocalError: Caught UnboundLocalError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/yuki/anaconda3/envs/TFNet/lib/python3.9/site-packages/torch/utils/data/_utils/worker.py", line 287, in _worker_loop
    data = fetcher.fetch(index)
  File "/home/yuki/anaconda3/envs/TFNet/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 49, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/yuki/anaconda3/envs/TFNet/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 49, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/mnt/f/tmp_HAND_GESTURE/TFNet/DataProcessMoudle.py", line 197, in __getitem__
    imgSeq = self.transform(imgSeq)
  File "/mnt/f/tmp_HAND_GESTURE/TFNet/videoAugmentation.py", line 21, in __call__
    image = t(image)
  File "/mnt/f/tmp_HAND_GESTURE/TFNet/videoAugmentation.py", line 158, in __call__
    new_h = im_h if new_h >= im_h else new_h
UnboundLocalError: local variable 'im_h' referenced before assignment

ERROR conda.cli.main_run:execute(41): `conda run python /mnt/f/tmp_HAND_GESTURE/TFNet/SLR.py` failed. (See above for error)

Process finished with exit code 1

```

dataprocess的transform，忘記將validdata的transform設置爲transformtxt了

注意test的transform比train的transform少TemporalRescale，所以txttest的transform為none





# 复现不可行

是我的3060太菜了



注意到之前复现时用的是testdataset，只有12张。

现在改用完整dataset，一个epoch要五十多小时。。。。



只有使用关键点坐标在cpu上训练可以，明显加快训练和推理：

​	**训练时间/epoch 55h -> 1h**

​	**推理  -90%**



