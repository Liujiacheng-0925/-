# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    model.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了CustomNet类，用于定义神经网络模型
#               ★★★请在空白处填写适当的语句，将CustomNet类的定义补充完整★★★
# -----------------------------------------------------------------------

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):  ## 最重要的都是forword函数了
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        ## 对tensor张量分块 x :1 197 1024   qkv 最后 是一个元组，tuple，长度是3，每个元素形状：1 197 1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # 分成多少个Head,与TRM生成qkv 的方式不同， 要更简单，不需要区分来自Encoder还是Decoder

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# VIT整体架构
class CustomNet(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # 初始化函数内，是将输入的图片，得到 img_size ，patch_size 的宽和高
        image_height, image_width = pair(image_size)  ## 224*224 *3
        patch_height, patch_width = pair(patch_size)  ## 16 * 16  *3
        # 图像尺寸必须能被patch大小整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  ## 步骤1.一个图像 分成 N 个patch
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),  # 步骤2.1将patch 铺开
            nn.Linear(patch_dim, dim),  # 步骤2.2 然后映射到指定的embedding的维度
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, img):
        x = self.to_patch_embedding(img)  ## img 1 3 224 224  输出形状x : 1 196 1024
        b, n, _ = x.shape  ##
        # 将cls 复制 batch_size 份
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # 将cls token在维度1 扩展到输入上
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        # 输入TRM
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


# -----------------------------------------------------------------------------
class CustomNet_2(nn.Module):
    def __init__(self, pretrained=False):
        super(CustomNet_2, self).__init__()
        resnet = resnet50(pretrained=pretrained)  # 是否加载预训练的权重
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.toplayer_3 = nn.Conv2d(256, 40, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.leakyrelu,
                                    resnet.maxpool)  # Sequential是一个容器，用于按顺序组织神经网络的模块。
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        # Smooth layers
        # self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # 全连接
        self.linear1 = nn.Linear(65536, 4096)
        self.linear2 = nn.Linear(4096, 10)

    # 这个函数的作用是初始化神经网络模型的权重和偏置。它遍历了模型的所有层，并根据层的类型进行不同的初始化操作。
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)  # 经过一个1x1的卷积层进行降维
        p4 = self._upsample_add(p5, self.latlayer1(c4))  # p5与c4连接并上采样得到特征图p4
        p3 = self._upsample_add(p4, self.latlayer2(c3))  # p4与c3连接并上采样得到特征图p3
        p2 = self._upsample_add(p3, self.latlayer3(c2))  # p3与c2连接并上采样得到特征图p2
        # Smooth
        p2 = self.smooth3(p2)
        # Attention
        p2 = p2.view(-1, 256 * 16 * 16)
        p2 = self.linear1(p2)
        p2 = self.linear2(p2)
        return p2


# 残差网络（ResNet）
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(3).mean(2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model Encoder"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth"))
    return model


# 创建一个3x3的卷积层
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# 基本的残差块（BasicBlock）
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.leakyrelu(out)
        return out


# 瓶颈残差块（Bottleneck）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.leakyrelu(out)

        return out


if __name__ == "__main__":
    # 测试
    from dataset import CustomDataset
    from torchvision.transforms import ToTensor

    c = CustomDataset('./images/train.txt', './images/train', ToTensor)
    x = torch.unsqueeze(c[10]['image'], 0)  # 模拟一个模型的输入数据

    vit = CustomNet(
        image_size=64,  # 图像大小，以像素为单位。
        # 如果图像是矩形的，则应该选择宽度和高度中较大的那个作为图像大小。
        patch_size=8,  # 补丁大小，指图像被切分成的小块大小。
        # 图像大小必须能被补丁大小整除。补丁的数量由 (image_size // patch_size) ** 2 计算得到，同时这个数量必须大于 16。
        num_classes=10,  # 分类的类别数量，即模型需要将图像分为多少个类别。
        dim=1024,  # 线性转换后输出张量的最后一个维度大小，
        # 通常用于指定 nn.Linear(..., dim) 中的输出维度。
        depth=12,  # Transformer 模块的堆叠层数，
        # 即模型中包含多少个 Transformer 块
        heads=12,  # 多头注意力（Multi-head Attention）层中的头的数量，
        # 用于增加模型对不同特征的关注度。
        mlp_dim=3072,  # MLP（全连接前馈）层的维度大小，用于提取特征。
        dropout=0.1,  # 在模型训练过程中随机丢弃神经元的比例，用于防止过拟合。
        # 取值范围为 [0, 1]，表示丢弃的比例。
        emb_dropout=0.1  # 嵌入（Embedding）层的 dropout 比例，
        # 用于在输入嵌入时进行随机丢弃。
    )  # 实例化
    print(vit.forward(x))  # 测试forward方法

    fpn = CustomNet_2()
    fpn.forward(x)
