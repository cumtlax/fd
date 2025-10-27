"""flower-tutorial: A Flower / PyTorch app."""

import torch
# ArrayRecord	Flower 中用于存储模型参数的数据结构
# Context	包含当前节点的运行环境信息（如配置）
# Message	客户端与服务器通信的“信封”，包含请求和响应
# MetricRecord	存储评估指标（如 loss, accuracy）
# RecordDict	类似字典，用于组织 ArrayRecord 和 MetricRecord
# ClientApp	Flower 的客户端主类，定义客户端行为
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_tutorial.task import Net, load_data
from flower_tutorial.task import test as test_fn
from flower_tutorial.task import train as train_fn

# 创建一个客户端应用实例
app = ClientApp()

# 所有通信通过 Message 对象完成，实现了隐私保护的分布式训练。

# 接收全局模型 → 在本地数据上训练 → 返回更新后的模型
# @app.train()：告诉 Flower “当收到训练任务时，调用这个函数”
# msg: 服务器发来的消息（包含模型参数、配置等）
# context: 当前客户端的上下文（包含配置，如 partition-id）
# 返回值：一个新的 Message，作为响应发回服务器
@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    # msg.content["arrays"]：服务器发来的模型参数（ArrayRecord 格式）
    # .to_torch_state_dict()：转为 PyTorch 原生的 state_dict
    # load_state_dict(...)：把服务器的权重加载到本地模型中
    # ✅ 相当于：“我现在拿到了最新的全局模型，准备开始本地训练。”
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # 把训练后的模型参数包装成 ArrayRecord
    # 准备发回给服务器
    model_record = ArrayRecord(model.state_dict())
    # 记录训练损失和样本数量
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    # 包装成 MetricRecord，用于监控
    metric_record = MetricRecord(metrics)
    # 把模型和指标打包成一个字典结构
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    # 构造一个 Message 对象
    # reply_to=msg：表示这是对服务器请求的回复
    # 发送回服务器
    return Message(content=content, reply_to=msg)

# 接收模型 → 在本地测试集上评估 → 返回准确率和损失
# 通常用于全局评估（服务器每轮结束后让部分客户端评估当前全局模型）
@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
