import json
import os

def parse_time_to_seconds(time_str: str) -> int:
    """
    将 'MM:SS' 格式转换为总秒数。
    例如: "1:30" -> 90
    """
    try:
        parts = time_str.strip().split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1:
            return int(parts[0])
        return 0
    except ValueError:
        return 0

def load_bo_data(file_path: str) -> list:
    """
    加载 JSON 文件，计算秒数并按时间排序。
    """
    if not os.path.exists(file_path):
        print(f"[BOReader] Warning: File not found at {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 预处理：增加 'seconds' 字段
        for item in data:
            item['seconds'] = parse_time_to_seconds(item.get('time', '0:00'))

        # 按时间排序
        data.sort(key=lambda x: x['seconds'])
        return data
    except Exception as e:
        print(f"[BOReader] Error loading data: {e}")
        return []

def check_and_get_bo_hints(current_time_seconds: float, bo_data: list, notified_indices: set, tolerance: int = 5) -> list:
    """
    检查当前时间是否有临近的 BO 操作。

    Args:
        current_time_seconds: 当前游戏时间（秒）。
        bo_data: 已加载的 BO 数据列表。
        notified_indices: 一个集合 (set)，用于记录已经提示过的条目索引。函数会直接修改这个集合。
        tolerance: 提前多少秒开始提示。

    Returns:
        new_hints: 需要展示的新提示列表。
    """
    new_hints = []

    for idx, item in enumerate(bo_data):
        # 如果这个条目已经提示过了，跳过
        if idx in notified_indices:
            continue

        bo_time = item['seconds']

        # 逻辑：当 (当前时间 >= BO时间 - 提前量) 时触发
        if current_time_seconds >= (bo_time - tolerance):
            new_hints.append(item)
            notified_indices.add(idx) # 标记为已提示，直接修改传入的集合

    return new_hints