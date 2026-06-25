from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """读取YAML配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """创建目录并返回Path对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
