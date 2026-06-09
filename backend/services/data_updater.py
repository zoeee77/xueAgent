"""知识库增量更新脚本。支持单条新增、批量导入、自动备份。"""

import json
import shutil
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _backup_file(filepath: Path) -> None:
    """备份文件到带时间戳的副本。"""
    backup = filepath.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(filepath, backup)


def add_or_update_data(filename: str, data: dict, merge: bool = True) -> None:
    """新增或更新单条数据。

    Args:
        filename: 数据文件名 (如 majors.json)
        data: 要写入的字典
        merge: True=增量合并, False=完全替换
    """
    filepath = DATA_DIR / filename
    if filepath.exists():
        _backup_file(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    if merge:
        existing.update(data)
    else:
        existing = data

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def validate_major(data: dict) -> list[str]:
    """校验专业数据格式，返回错误列表。"""
    errors = []
    required = ["employment_rate", "avg_salary", "top_directions", "resource_threshold", "description"]
    for field in required:
        if field not in data:
            errors.append(f"缺少必填字段: {field}")
    if "employment_rate" in data and not (0 <= data["employment_rate"] <= 1):
        errors.append("employment_rate 必须在 0-1 之间")
    return errors
