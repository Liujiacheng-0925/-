# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    train.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了模型的训练流程
#               ★★★请在空白处填写适当的语句，将模型训练流程补充完整★★★
# -----------------------------------------------------------------------

import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import CustomNet, CustomNet_2
from test import test


def train_loop(epoch, dataloader, model, loss_fn, optimizer, device):
    """定义训练流程。
    :param epoch: 定义训练的总轮次
    :param dataloader: 数据加载器
    :param model: 模型，需在model.py文件中定义好
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 训练设备，即使用哪一块CPU、GPU进行训练
    """
    # 将模型置为训练模式
    model.train()

    # 记录每个epoch的损失值
    all_losses = []

    for e in range(epoch):
        print(f"Epoch {e + 1}/{epoch}")
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            inputs, targets = batch['image'].to(device), batch['label'].to(device)

            # 前向传播 + 计算损失
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            # 将梯度清零
            optimizer.zero_grad()
            # 反向传播 + 更新模型参数
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(dataloader)
        all_losses.append(average_loss)

        print(f"Epoch {e + 1}/{epoch} -> Loss: {average_loss:.4f}")
        # 保存模型
        torch.save(model, rf'D:\pycham\pythonProject20\nndl_project-master\models\model_epoch_{e + 1}.pkl')

        # 测试数据加载器
        test_dataloader = DataLoader(CustomDataset('./images/test.txt', './images/test', ToTensor),
                                     batch_size=32)

        test(test_dataloader, model, device)

    plt.figure()
    plt.plot(range(1, epoch + 1), all_losses, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig('training_loss.png')  # 保存图像文件
    plt.show()  # 显示图像


if __name__ == "__main__":
    # 定义模型超参数
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-2
    EPOCH = 100

    model = CustomNet(
        image_size=64,
        patch_size=8,
        num_classes=10,
        dim=128,
        depth=3,
        heads=4,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )

    # fpn = CustomNet_2()

    # 实例化
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model.to(device)

    # 训练数据加载器
    train_dataloader = DataLoader(CustomDataset('./images/train.txt', './images/train', ToTensor),
                                  batch_size=BATCH_SIZE)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 学习率和优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # 调用训练方法
    train_loop(EPOCH, train_dataloader, model, loss_fn, optimizer, device)
