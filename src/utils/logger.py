import json
from datetime import datetime
from pathlib import Path


class JsonlLogger:
    """将训练日志逐行写入JSONL文件。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, data: dict) -> None:
        """追加一条日志记录。"""
        record = {"time": datetime.now().isoformat(timespec="seconds"), **data}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
