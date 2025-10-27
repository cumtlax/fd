"""flower-tutorial: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.datasets as datasets
# Flower 提供的联邦数据集加载工具
from flwr_datasets import FederatedDataset
# 将数据划分为多个“独立同分布”的客户端分区
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
# 图像预处理工具
from torchvision.transforms import Compose, Normalize, ToTensor


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



import torchvision.transforms as transforms

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


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch



def load_data(partition_id: int, num_partitions: int):
    """
    Load CIFAR-10 data from local disk and partition it into `num_partitions`.
    Each client gets a non-overlapping subset.
    """
    # Step 1: Download or load CIFAR-10 locally
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # Step 2: Split training set into `num_partitions` equal parts (non-IID 模式可后续扩展)
    total_size = len(trainset)
    partition_size = total_size // num_partitions
    lengths = [partition_size] * num_partitions
    # 处理不能整除的情况
    lengths[-1] += total_size - sum(lengths)

    # 随机划分数据（模拟 IID 分布）
    partitions = random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))

    # Step 3: Select the partition for this client
    train_partition = partitions[partition_id]

    # Step 4: Use same test set for all clients (or optionally also split)
    # 如果你也想让每个客户端有自己的测试集，可以类似划分 testset
    test_partition = testset

    # Step 5: Create DataLoaders
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=32, shuffle=False)  # 全局测试集不打乱

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




# partition_id: 我是第几个客户端
# num_partitions: 把整个数据集平均分成多少份，每一份给一个“客户端”（client）使用。
# num_partitions = 10 → 把数据分成 10 份，发给 10 个客户端
# num_partitions = 100 → 分成 100 份，发给 100 个客户端
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