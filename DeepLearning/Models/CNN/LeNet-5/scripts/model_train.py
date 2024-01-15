import copy
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import MNIST

# 导入模型
from model import LeNet5


# 处理数据集，划分为训练集和验证集
def train_valid_split():
    # 下载数据集
    Train_Dataset = MNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()]),
                          download=True)

    # 随机划分训练集和验证集
    train_size = int(0.8 * len(Train_Dataset))
    valid_size = len(Train_Dataset) - train_size
    train_data, valid_data = Data.random_split(Train_Dataset, [train_size, valid_size])

    # 将训练集和验证集转换为可迭代的 DataLoader 对象
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=2)

    valid_dataloader = Data.DataLoader(dataset=valid_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=2)

    return train_dataloader, valid_dataloader


# 训练模型
# noinspection PyTypeChecker
def train_model(model, train_dataloader, valid_dataloader, num_epochs, learning_rate):
    """
    @param model: 模型名称
    @param train_dataloader: 训练集数据
    @param valid_dataloader: 验证集数据
    @param num_epochs: 训练轮数
    @param learning_rate: 学习率
    @return: None
    """

    # 决定使用 GPU 还是 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 将模型加载到 device 当中
    model = model.to(device)
    # 复制当前模型参数作为最优模型参数
    best_model_params = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最佳验证集准确率
    best_valid_acc = 0.0
    # 分轮次训练集损失函数列表
    train_loss_list = []
    # 分轮次验证集损失函数列表
    valid_loss_list = []
    # 分轮次训练集准确率列表
    train_acc_list = []
    # 分轮次验证集准确率列表
    valid_acc_list = []
    # 当前时间
    train_start_time = time.time()

    # 分轮次训练模型
    for epoch in range(num_epochs):
        # 记录本轮次开始的的时间
        epoch_start_time = time.time()

        # 打印训练轮次
        print("=" * 70)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # 初始化每轮训练的损失值和准确率
        train_loss = 0.0
        train_corrects = 0.0
        # 初始化每轮验证的损失值和准确率
        valid_loss = 0.0
        valid_corrects = 0.0

        # 每轮训练、验证的样本数
        train_sample_num = 0
        valid_sample_num = 0

        # 对每一个 mini-batch 进行分批次训练和计算
        print("Training Progress:")
        for step, (batch_image, batch_label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # 将特征图和标签数据加载到 device 上
            batch_image = batch_image.to(device)
            batch_label = batch_label.to(device)
            # 将模型设置为训练模式
            model.train()

            # 前向传播
            # 输入为一个 batch 的四维张量，大小为 (batch_size, 1, 28, 28)
            # 输出为一个 batch 的二维张量，大小为 (batch_size, 10)，表示每个样本属于 10 个类别的概率
            output = model(batch_image)
            # 计算当前训练批次的损失值
            batch_loss = criterion(output, batch_label)
            # 对于 batch_size 中每个样本找到最大的 Softmax 概率值的行号对应的标签作为预测标签
            predict_label = torch.argmax(output, dim=1)

            # 对每一批次数据，将梯度初始化为 0 再训练，防止上一批次的梯度影响当前批次的训练
            optimizer.zero_grad()
            # 反向传播
            batch_loss.backward()
            # 更新参数
            optimizer.step()

            # 将当前训练批次的损失值按照 batch_size 加权累加到当前轮次的总损失 train_loss 上
            train_loss += batch_loss.item() * batch_image.size(0)
            # 将当前训练批次的准确数量按照 batch_size 加权累加到当前轮次的总准确数 train_corrects 上
            train_corrects += torch.sum(torch.eq(predict_label, batch_label.data)).item()

            # 更新当前训练样本数
            train_sample_num += batch_image.size(0)

        # 对每一个 mini-batch 进行分批次验证和计算
        print("Validating Progress:")
        for step, (batch_image, batch_label) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            # 将特征图和标签数据加载到 device 上
            batch_image = batch_image.to(device)
            batch_label = batch_label.to(device)

            # 将模型设置为验证模式
            model.eval()

            # 前向传播计算结果，输出为一个 batch 的二维张量，大小为 (batch_size, 10)，表示每个样本属于 10 个类别的概率
            output = model(batch_image)
            # 计算当前验证批次的损失值
            batch_loss = criterion(output, batch_label)
            # 对于 batch_size 中每个样本找到最大的 Softmax 概率值的行号对应的标签作为预测标签
            predict_label = torch.argmax(output, dim=1)

            # 将当前验证批次的损失值按照 batch_size 加权累加到当前轮次的验证总损失 valid_loss 上
            valid_loss += batch_loss.item() * batch_image.size(0)
            # 将当前验证批次的准确数量按照 batch_size 加权累加到当前轮次的总验证准确数 valid_corrects 上
            valid_corrects += torch.sum(torch.eq(predict_label, batch_label.data)).item()

            # 更新当前验证样本数
            valid_sample_num += batch_image.size(0)

        # 计算当前轮次训练的平均损失值并添加到 train_loss_list 中
        train_loss = train_loss / train_sample_num
        train_loss = round(train_loss, 4)
        train_loss_list.append(train_loss)
        # 计算当前轮次训练的平均准确率并添加到 train_acc_list 中
        train_acc = train_corrects / train_sample_num
        train_acc = round(train_acc, 4)
        train_acc_list.append(train_acc)

        # 计算当前轮次验证的平均损失值并添加到 valid_loss_list 中
        valid_loss = valid_loss / valid_sample_num
        valid_loss = round(valid_loss, 4)
        valid_loss_list.append(valid_loss)
        # 计算当前轮次验证的平均准确率并添加到 valid_acc_list 中
        valid_acc = valid_corrects / valid_sample_num
        valid_acc = round(valid_acc, 4)
        valid_acc_list.append(valid_acc)

        # 打印当前轮次训练和验证的损失值和准确率
        print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(train_loss, train_acc))
        print('Valid Loss: {:.4f} Valid Acc: {:.4f}'.format(valid_loss, valid_acc))

        # 如果当前轮次验证准确率更高，则更新最佳验证准确率和最佳模型参数
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_params = copy.deepcopy(model.state_dict())

        # 打印当前轮次训练时间
        time_elapsed = time.time() - epoch_start_time
        print('Epoch {} complete in {:.0f}mim {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60))

    # 训练结束, 保存模型
    torch.save(best_model_params, '../best_model.pth')

    # 将训练过程中的损失值和准确率保存为 DataFrame
    train_process = pd.DataFrame(
        data={
            'Epoch': np.arange(1, num_epochs + 1),
            'Train_Loss': train_loss_list,
            'Valid_Loss': valid_loss_list,
            'Train_Acc': train_acc_list,
            'Valid_Acc': valid_acc_list
        }
    )
    train_process.to_csv('./train_process.csv', index=False)

    # 打印训练总时间
    print("=" * 70)
    time_elapsed = time.time() - train_start_time
    print('Train complete in {:.0f}mim {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 返回训练过程的 DataFrame
    return train_process


# 绘制训练过程中的损失值和准确率曲线
def plot_train_process(train_process: pd.DataFrame):
    """
    @param train_process: 训练过程的 DataFrame
    @return: None
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['Epoch'], train_process['Train_Loss'], 'ro-', label='Train Loss')
    plt.plot(train_process['Epoch'], train_process['Valid_Loss'], 'bs', label='Valid Loss')
    plt.xlabel('Epoch')
    # 设置横坐标刻度从 0 开始，步长为 2
    plt.xticks(np.arange(0, train_process['Epoch'].max() + 1, 2))
    plt.ylabel('Loss')
    # 设置纵坐标刻度从 0 开始, 步长为 0.5
    plt.yticks(np.arange(0, train_process['Train_Loss'].max() + 0.5, 0.5))
    plt.legend()

    # 绘制训练集和验证集的准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process['Epoch'], train_process['Train_Acc'], 'ro-', label='Train Acc')
    plt.plot(train_process['Epoch'], train_process['Valid_Acc'], 'bs', label='Valid Acc')
    plt.xlabel('Epoch')
    # 设置横坐标刻度从 0 开始，步长为 2
    plt.xticks(np.arange(0, train_process['Epoch'].max() + 1, 2))
    plt.ylabel('Acc')
    # 设置纵坐标刻度从 0% 到 100%，步长为 20%，要求格式化显示为百分数
    plt.yticks(np.arange(0, 1.2, 0.2), ['{}%'.format(int(x * 100)) for x in np.arange(0, 1.2, 0.2)])
    plt.legend()

    plt.savefig('./train_process.png')
    plt.show()


if __name__ == '__main__':
    # 处理数据集，划分为训练集和验证集
    train_dataloader, valid_dataloader = train_valid_split()

    # 实例化模型
    model = LeNet5()

    # 训练模型
    train_process = train_model(model, train_dataloader, valid_dataloader, num_epochs=20, learning_rate=0.001)

    # 绘制训练过程中的损失值和准确率曲线
    plot_train_process(train_process)
