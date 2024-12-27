import re
import csv

action = "train"

if action == "train":
    # 定义正则表达式模式
    pattern = re.compile(
        r"epoch:\s*(\d+),\s*trainLoss:\s*([\d.]+),\s*lr:\s*([\d.]+).*?validLoss:\s*([\d.]+),\s*werScore:\s*([\d.]+)",
        re.DOTALL
    )

    # 读取文件内容
    input_file = "training_log2.txt"
    output_file = "training_log2.csv"

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 匹配内容
    matches = pattern.findall(content)

    # 将结果写入 CSV
    header = ["epoch", "trainLoss", "lr", "validLoss", "werScore"]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # 写入表头
        writer.writerows(matches)  # 写入匹配数据

    print(f"提取完成，结果已保存到 {output_file}")

elif action == "test":
    # 定义正则表达式模式
    pattern = re.compile(
        r"testLoss:\s*([\d.]+),\s*werScore:\s*([\d.]+)",
        re.DOTALL
    )

    # 读取文件内容
    input_file = "testing_log2.txt"
    output_file = "testing_log2.csv"

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 匹配内容
    matches = pattern.findall(content)

    # 自动添加 epoch 列
    matches_with_epoch = [(i, *match) for i, match in enumerate(matches)]

    # 将结果写入 CSV
    header = ["epoch", "testLoss", "werScore"]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # 写入表头
        writer.writerows(matches_with_epoch)  # 写入匹配数据

    print(f"提取完成，结果已保存到 {output_file}")