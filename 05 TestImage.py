# encoding: utf-8
"""
@author: Dianlei Zhang
@contact: dianlei.zhang@qq.com
@time: 2020/8/31 11:24 下午
"""
import torch
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import os

batch_size = 200  # 分批训练数据、每批数据量
learning_rate = 1e-2  # 学习率
num_epoches = 10  # 训练次数
DOWNLOAD_MNIST = True  # 是否下载数据

# Mnist digits dataset
if not (os.path.exists('./data/')) or not os.listdir('./data/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_dataset = datasets.MNIST(
    root='./data/',
    train=True,  # download train data
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,  # download test data
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# 该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入
# 按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
# shuffle 是否打乱加载数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, kernel_size=3, stride=1, padding=1),
            # input shape(1*28*28),(28+1*2-3)/1+1=28 卷积后输出（6*28*28）
            # 输出图像大小计算公式:(n*n像素的图）(n+2p-k)/s+1
            nn.ReLU(True),  # 激活函数
            nn.MaxPool2d(2, 2),  # 28/2=14 池化后（6*14*14）
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # (14-5)/1+1=10 卷积后（16*10*10）
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # 池化后（16*5*5）=400，the input of full connection
        )
        self.fc = nn.Sequential(  # full connection layers.
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out = self.conv(x)  # out shape(batch,16,5,5)
        out = out.view(out.size(0), -1)  # out shape(batch,400)
        out = self.fc(out)  # out shape(batch,10)
        return out


cnn = CNN(1, 10)
#  加载参数
ckpt_dir = './model/'
load_path = os.path.join(ckpt_dir, 'CNN_model_weight2.pth.tar')

ckpt = torch.load(load_path)
cnn.load_state_dict(ckpt['state_dict'])            #参数加载到指定模型cnn
#  要识别的图片
input_image = './image/1_black.png'

im = Image.open(input_image).resize((28, 28))     #取图片数据
im = im.convert('L')      #灰度图
im_data = np.array(im)

im_data = torch.from_numpy(im_data).float()

im_data = im_data.view(1, 1, 28, 28)
out = cnn(im_data)
_, pred = torch.max(out, 1)

print('预测为:数字{}。'.format(pred.item()))
