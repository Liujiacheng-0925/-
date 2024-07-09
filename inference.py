# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    inference.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了用于在模型应用端进行推理，返回模型输出的流程
#               ★★★请在空白处填写适当的语句，将模型推理应用流程补充完整★★★
# -----------------------------------------------------------------------

import torch
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from model import CustomNet, CustomNet_2,Bottleneck
def inference(image_path, model, device):
    """定义模型推理应用的流程。
    :param image_path: 输入图片的路径
    :param model: 训练好的模型
    :param device: 模型推理使用的设备，即使用哪一块CPU、GPU进行模型推理
    """
    # 将模型置为评估（测试）模式
    model.eval()

    # START----------------------------------------------------------
    # 打开图片并转换为张量
    image = Image.open(image_path)
    image_tensor = ToTensor()(image).unsqueeze(0)  # 增加一个batch的维度

    # 将图片张量移到指定设备上
    image_tensor = image_tensor.to(device)

    # 禁用梯度计算，进行推理
    with torch.no_grad():
        output = model(image_tensor)

    # 输出处理（假设输出是分类结果）
    _, predicted = torch.max(output, 1)
    # END------------------------------------------------------------

    # 显示图片
    plt.imshow(image)
    plt.title(f'Predicted: {predicted.item()}')
    plt.axis('off')  # 隐藏坐标轴
    plt.show()

    # 返回预测结果
    print(predicted)
    return predicted.item()

if __name__ == "__main__":
    # 指定图片路径
    image_path = "./images/test/signs/img_0006.png"

    # 加载训练好的模型
    model = torch.load(r'D:\pycham\pythonProject20\nndl_project-master\models\较好模型\models.circ')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 显示图片，输出预测结果
    inference(image_path, model, device)
