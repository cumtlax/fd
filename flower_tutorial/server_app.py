"""flower-tutorial: A Flower / PyTorch app."""

import torch

# ArrayRecord	Flower 中用于存储模型参数的数据结构（类似字典）
# ConfigRecord	存储配置信息（如超参数）
# Context	包含当前运行环境的信息（比如用户传入的配置）
from flwr.app import ArrayRecord, ConfigRecord, Context

# ServerApp	Flower 的服务器端主类，定义服务器行为
# Grid	表示所有客户端提交的结果集合（来自客户端的模型更新）
from flwr.serverapp import Grid, ServerApp

# FedAvg	经典的联邦平均算法（Federated Averaging）
from flwr.serverapp.strategy import FedAvg

# Net	你自己定义的神经网络模型（比如 CNN）
from flower_tutorial.task import Net

# 创建一个 Flower 服务器应用实例
# Create ServerApp
app = ServerApp()

#  main 函数是 联邦学习服务器的“指挥中心”，它：
# 读取训练配置（如轮数、学习率）
# 初始化全局模型
# 启动 FedAvg 策略，协调多个客户端进行多轮训练
# 最终聚合出一个全局模型并保存到磁盘

#  @app.main()
# 这是一个 装饰器（decorator）
# 它告诉 Flower 框架：“这个函数是 ServerApp 启动时的主入口”
# 类似于程序的 main() 函数
# grid: 表示所有客户端节点的集合（逻辑上的“网格”）
# 你可以把它理解为“所有注册的客户端列表”
# 在 strategy.start() 中用于调度客户端
# context: 当前运行的上下文环境
# 包含用户传入的配置参数（如 num-server-rounds）
# 是一个字典-like 对象，可通过 context.run_config["key"] 访问
# -> None: 这个函数不返回有意义的值（最终结果通过 result 获取）
@app.main()
def main(grid: Grid, context: Context) -> None:
    """这是服务器的“主程序”，就像电脑的开机启动项"""

    # 第一步：读取用户设置的参数
    # 比如：每轮选 30% 的客户端，总共训练 5 轮，学习率是 0.001
    fraction_train = context.run_config["fraction-train"]      # 每轮选多少比例的客户端
    num_rounds     = context.run_config["num-server-rounds"]   # 总共训练几轮
    learning_rate  = context.run_config["lr"]                  # 学习率

    # 第二步：创建一个全新的模型（比如 CNN）
    # 就像发一张白纸给第一个学生，让他开始画图
    model = Net()  # 创建模型（结构已知，但参数是随机的）

    # 把模型的“参数”打包成 Flower 能传输的格式
    initial_weights = ArrayRecord(model.state_dict())

    # 第三步：设置联邦学习规则 —— 使用 FedAvg（联邦平均）
    # 就像老师说：“每轮随机叫几个同学上来分享作业，我取平均值更新全班答案”
    strategy = FedAvg(
        fraction_train=fraction_train,  # 每轮选 fraction_train 比例的客户端
    )

    # 第四步：开始联邦学习！
    # 让客户端们轮流训练，共 num_rounds 轮
    # 服务器会自动：
    #   - 下发模型 → 客户端训练 → 上传更新 → 聚合 → 更新全局模型
    result = strategy.start(
        grid=grid,                # 所有客户端的集合（Flower 自动管理）
        initial_arrays=initial_weights,  # 初始模型参数
        train_config=ConfigRecord({"lr": learning_rate}),  # 发给客户端的配置
        num_rounds=num_rounds,    # 总共训练多少轮
    )

    # 第五步：训练结束，保存最终模型
    print("✅ 联邦学习已完成！正在保存最终模型...")

    # 把 Flower 格式的模型转回 PyTorch 格式
    final_state_dict = result.arrays.to_torch_state_dict()

    # 保存到文件 "final_model.pt"
    torch.save(final_state_dict, "final_model.pt")

    print("🎉 模型已保存：final_model.pt")