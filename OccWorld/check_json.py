import json

try:
    with open('/nas/vlm_driving/OccWorld/data/lf_train.json', 'r') as f:
        data = json.load(f)
    print(f"数据类型: {type(data)}")
    if isinstance(data, list):
        print(f"JSON 文件包含 {len(data)} 条数据")
    elif isinstance(data, dict):
        print(f"JSON 文件是字典，键数量: {len(data.keys())}")
    else:
        print("无法计算长度，数据不是列表或字典")
except FileNotFoundError:
    print("错误：文件 '/nas/vlm_driving/OccWorld/data/lf_train.json' 不存在")
except json.JSONDecodeError:
    print("错误：文件不是有效的 JSON 格式")
except Exception as e:
    print(f"其他错误：{e}")