import matplotlib.pyplot as plt
import pandas as pd

action = "compare"

if action == "train":
    # 读取CSV文件
    input_file = "logs/training_log2.csv"
    data = pd.read_csv(input_file)

    # 提取数据
    epochs = data["epoch"]
    train_loss = data["trainLoss"]
    valid_loss = data["validLoss"]
    wer_score = data["werScore"]

    # 创建曲线图
    plt.figure(figsize=(10, 6))

    # 绘制 trainLoss 曲线
    plt.plot(epochs, train_loss, label="Train Loss", marker='o', color='blue')

    # 绘制 validLoss 曲线
    plt.plot(epochs, valid_loss, label="Valid Loss", marker='s', color='green')

    # 绘制 werScore 曲线
    plt.plot(epochs, wer_score, label="WER Score", marker='^', color='red')

    # 添加图例、标题和坐标轴标签
    plt.title("Training Metrics over Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Metrics", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # 保存图片（可选）
    output_image = "logs/training_metrics2.png"
    plt.savefig(output_image, dpi=300)

    # 显示图表
    plt.show()

elif action == "test":
    # 读取CSV文件
    input_file = "logs/testing_log2.csv"  # 替换为你的CSV文件名
    data = pd.read_csv(input_file)

    # 提取数据
    epochs = data["epoch"]
    test_loss = data["testLoss"]
    wer_score = data["werScore"]

    # 创建曲线图
    plt.figure(figsize=(10, 6))

    # 绘制 testLoss 曲线
    plt.plot(epochs, test_loss, label="TestLoss", marker='o', color='blue')

    # 绘制 werScore 曲线
    plt.plot(epochs, wer_score, label="WER Score", marker='^', color='red')

    # 添加图例、标题和坐标轴标签
    plt.title("Testing Metrics over Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Metrics", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # 保存图片（可选）
    output_image = "logs/testing_metrics1.png"
    plt.savefig(output_image, dpi=300)

    # 显示图表
    plt.show()

elif action == "compare":
    # 读取CSV文件
    input_file1 = "logs/training_log1.csv"
    input_file2 = "logs/training_log2.csv"

    data1 = pd.read_csv(input_file1)
    data2 = pd.read_csv(input_file2)

    # 提取数据
    epochs1 = data1["epoch"]
    train_loss1 = data1["trainLoss"]
    valid_loss1 = data1["validLoss"]
    wer_score1 = data1["werScore"]

    epochs2 = data2["epoch"]
    train_loss2 = data2["trainLoss"]
    valid_loss2 = data2["validLoss"]
    wer_score2 = data2["werScore"]

    # 创建曲线图
    plt.figure(figsize=(12, 7))

    # 绘制 logs/training_log1.csv 的曲线（半透明）
    plt.plot(epochs1, train_loss1, label="Train Loss (Log1)", marker='o', color='blue', alpha=0.2)
    plt.plot(epochs1, valid_loss1, label="Valid Loss (Log1)", marker='s', color='green', alpha=0.2)
    plt.plot(epochs1, wer_score1, label="WER Score (Log1)", marker='^', color='red', alpha=0.2)

    # 绘制 logs/training_log2.csv 的曲线
    plt.plot(epochs2, train_loss2, label="Train Loss (Log2)", marker='o', color='blue')
    plt.plot(epochs2, valid_loss2, label="Valid Loss (Log2)", marker='s', color='green')
    plt.plot(epochs2, wer_score2, label="WER Score (Log2)", marker='^', color='red')

    # 添加图例、标题和坐标轴标签
    plt.title("Training Metrics Comparison over Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Metrics", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # 保存图片（可选）
    output_image = "logs/training_comparison_metrics.png"
    plt.savefig(output_image, dpi=300)

    # 显示图表
    plt.show()