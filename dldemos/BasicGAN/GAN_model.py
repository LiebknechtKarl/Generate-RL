import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# 定义生成器（Generator）
# Generator 类定义了一个生成器网络，它从随机噪声（z）中生成图像。
# block 函数是生成器的基本构建模块，包括一个全连接层、批归一化（可选）和 LeakyReLU 激活。
# 最后一个全连接层将输出调整为图像的大小，并通过 Tanh 激活函数将像素值限制在 -1 到 1 之间。
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Discriminator 类定义了一个判别器网络，用于判断输入图像是真实的还是生成的。
# 输入图像首先被展平为一个向量，然后通过几层全连接层和激活函数，最终输出一个概率值（通过 Sigmoid 激活），表示图像的真实性。
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


if __name__ == "__main__":
    img_shape = (1, 28, 28)
    latent_dim = 100

    # Initialize generator and discriminator
    generator = Generator(latent_dim, img_shape)
    discriminator = Discriminator(img_shape)
