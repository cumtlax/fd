"""flower-tutorial: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.transforms as transforms

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 用于缓存 FederatedDataset 对象，避免每个客户端重复初始化。
fds = None  # Cache FederatedDataset

# 图像预处理流程
# ToTensor()：将 PIL 图像转为 Tensor，并归一化到 [0,1]
# Normalize(mean, std)：标准化为 x = (x - mean) / std
# 这里用 (0.5, 0.5, 0.5) 是为了把 [0,1] 映射到 [-1,1]
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 图像预处理（和原来一致）
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

def load_data(partition_id: int, num_partitions: int):
    """
    加载并划分CIFAR-10数据集，为联邦学习中的某个客户端返回其本地的训练和测试数据加载器。

    参数:
        partition_id (int): 当前客户端的ID（从0开始）。
        num_partitions (int): 将训练集划分为多少个部分（即客户端总数）。

    返回:
        tuple: 包含训练数据加载器(trainloader)和测试数据加载器(testloader)的元组。
    """

    # 1. 下载并加载完整的CIFAR-10训练集和测试集
    # `root="./data"` 指定数据存储目录
    # `train=True` 表示加载训练集
    # `download=True` 如果本地没有数据，则自动下载
    # `transform=transform_train` 应用预定义的数据增强和归一化变换
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # 2. 计算每个数据分区（客户端）应分配到的样本数量
    total_size = len(trainset)  # 获取训练集总样本数 (50000 for CIFAR-10)
    partition_size = total_size // num_partitions  # 计算每个分区的基础大小
    lengths = [partition_size] * num_partitions  # 创建一个列表，每个元素代表一个分区的大小

    # 3. 处理无法整除的情况：将余下的样本分配给最后一个分区
    # 例如，50000个样本分给3个客户端，前两个各16666，最后一个16668
    lengths[-1] += total_size - sum(lengths)

    # 4. 将训练集随机划分为num_partitions个不相交的子集
    # 使用`random_split`函数进行划分
    # `generator=torch.Generator().manual_seed(42)` 确保划分结果可复现（相同的随机种子）
    partitions = random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))

    # 5. 为当前客户端（由partition_id指定）选择对应的训练数据分区
    train_partition = partitions[partition_id]

    # 6. 测试集不分割，所有客户端共享同一个全局测试集
    # 这样可以统一评估模型在相同测试数据上的性能
    test_partition = testset

    # 7. 创建数据加载器(DataLoader)，用于批量读取数据
    # 训练数据加载器：使用客户端的训练分区，batch_size=32，并打乱顺序(shuffle=True)以提高训练效果
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)

    # 测试数据加载器：使用完整的测试集，batch_size=32，不打乱顺序(shuffle=False)
    # 通常测试时不需打乱，且保持一致的顺序有助于结果比较
    testloader = DataLoader(test_partition, batch_size=32, shuffle=False)

    # 8. 返回该客户端的训练和测试数据加载器
    return trainloader, testloader

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    # 把模型放到指定设备
    # .to(device) 的作用是：
    # 如果有 GPU，就把模型参数从 CPU 拷贝到 GPU 显存中
    # 这样后续计算（卷积、矩阵乘等）会用 GPU 加速，快很多！
    net.to(device)  # move model to GPU if available
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 告诉模型：“我现在要开始训练了！”
    # 影响某些层的行为，比如：
    # Dropout 层：训练时随机丢弃神经元（防过拟合）
    # BatchNorm 层：使用当前 batch 的均值和方差
    # 如果忘记写这句，Dropout 可能不生效，导致训练和测试行为不一致。
    net.train()
    # 用于累计每个 batch 的损失值
    # 最后除以 batch 数量，得到平均损失（用于返回）
    running_loss = 0.0
    for _ in range(epochs):
        # trainloader 是前面 DataLoader 创建的迭代器
        # 每次返回一个 batch，格式如：
        # python
        # 编辑
        # {
        #   "img": [img1, img2, ..., img32],  # 32张图像
        #   "label": [3, 8, 0, ..., 5]        # 32个标签
        # }
        for images, labels in trainloader:
            # 把图像和标签从 CPU 移到 GPU（如果 device='cuda'）
            # 必须和模型在同一设备上，否则会报错！
            # 📌 注意：batch["img"] 是一个 Tensor，形状为 (32, 3, 32, 32)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy





# def load_data(partition_id: int, num_partitions: int):
#     """Load partition CIFAR10 data."""
#     # Only initialize `FederatedDataset` once
#
#     # 使用了全局变量 fds 来存储一个 FederatedDataset 实例。这是为了避免多次初始化 FederatedDataset 对象，
#     # 从而节省资源并保证数据的一致性。如果 fds 不存在，则进行初始化。
#     global fds
#     if fds is None:
#         # IID = Independent and Identically Distributed
#         # Independent	独立	每个客户端的数据是独立的，互不影响
#         # Identically Distributed	同分布	每个客户端的数据分布相似（比如都有猫、狗、车等）
#         #  举个例子（CIFAR-10 数据集）：
#         #
#         # 总共 10 类：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
#         # 如果是 IID 分割：
#         # 客户端 0：有 10% 飞机、10% 猫、10% 车……
#         # 客户端 1：也有 10% 飞机、10% 猫、10% 车……
#         # ……
#         # 👉 所有客户端看到的“世界”是相似的。
#         # IidPartitioner 是一个“公平分数据”的工具，它把整个数据集平均、随机地分成 num_partitions 份，确保每一份的数据分布都差不多
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         # 初始化 FederatedDataset，指定了数据集名称 "uoft-cs/cifar10" 和训练集的分区策略。
#         fds = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={"train": partitioner},
#         )
#     # 根据传入的 partition_id 加载对应的数据分区。这意味着每个节点只会看到整个数据集的一个子集。
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     # Construct dataloaders
#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader