import json
import csv

# 文件路径
json_file_path = 'output/CE-CSL.json'  # 替换为你的 JSON 文件路径
csv_file_path = 'data/CE-CSL/label/test.csv'   # 替换为你的 CSV 文件路径
output_file_path = 'output/CE-CSL.json'

# 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载 CSV 文件
def load_csv(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['Number']] = row['Chinese Sentences']
    return data

# 更新 JSON 数据
def update_json(json_data, csv_data):
    for entry in json_data:
        id_value = entry.get('id')
        if id_value and id_value in csv_data:
            entry['origin'] = csv_data[id_value]
    return json_data

# 保存更新后的 JSON 文件
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 主程序
if __name__ == '__main__':
    json_data = load_json(json_file_path)
    csv_data = load_csv(csv_file_path)
    updated_json = update_json(json_data, csv_data)
    save_json(updated_json, output_file_path)
    print(f'JSON 文件已更新并保存到 {output_file_path}')
