import json
from pathlib import Path
from typing import Union, Dict, List


def load_json(file_path: Union[str, Path]) -> Union[Dict, List]:
    with open(file_path, "r") as f:
        return json.load(f)


def export_json(data: Union[Dict, List], file_path: Union[str, Path]) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_markdown_report(report: str, output_path: Path) -> None:
    """Save the generated report to the specified path.

    Args:
        report: The generated report text
        output_path: Path where the report should be saved
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
