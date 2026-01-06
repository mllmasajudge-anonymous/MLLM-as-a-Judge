import json
import os
from typing import Dict, Any, List, Tuple
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(PROJECT_ROOT, "results", "traditional_evaluations.jsonl")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "normalized_traditional_evaluations.jsonl")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def extract_task_from_image_id(image_id: str) -> str:
    if image_id.startswith("HumanEdit_"):
        parts = image_id.split("_")
        if len(parts) >= 2:
            return parts[1]
    return "Unknown"


def collect_metric_ranges_by_group(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    ranges: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    categories = ["pixel_level_fidelity", "content_preservation"]

    for entry in entries:
        image_id = entry.get("image_id", "")
        if image_id.startswith("task_metrics"):
            continue
        
        setting = entry.get("setting", "unknown")
        task = extract_task_from_image_id(image_id)
        group_key = f"{setting}_{task}"
        
        if group_key not in ranges:
            ranges[group_key] = {}
        
        for cat in categories:
            cat_data = entry.get(cat)
            if not isinstance(cat_data, dict):
                continue
            
            if cat not in ranges[group_key]:
                ranges[group_key][cat] = {}
            
            for metric, value in cat_data.items():
                if not isinstance(value, (int, float)):
                    continue
                
                if metric not in ranges[group_key][cat]:
                    ranges[group_key][cat][metric] = (value, value)
                else:
                    current_min, current_max = ranges[group_key][cat][metric]
                    if value < current_min:
                        current_min = value
                    if value > current_max:
                        current_max = value
                    ranges[group_key][cat][metric] = (current_min, current_max)

    return ranges


def normalize_entries_by_group(entries: List[Dict[str, Any]], ranges: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    categories = ["pixel_level_fidelity", "content_preservation"]

    for entry in entries:
        image_id = entry.get("image_id", "")
        if image_id.startswith("task_metrics"):
            normalized.append(entry)
            continue

        setting = entry.get("setting", "unknown")
        task = extract_task_from_image_id(image_id)
        group_key = f"{setting}_{task}"
        
        new_entry: Dict[str, Any] = dict(entry)
        for cat in categories:
            cat_data = entry.get(cat)
            if not isinstance(cat_data, dict):
                continue
            
            new_cat: Dict[str, Any] = {}
            for metric, value in cat_data.items():
                if not isinstance(value, (int, float)):
                    new_cat[metric] = value
                    continue
                
                group_ranges = ranges.get(group_key, {})
                cat_ranges = group_ranges.get(cat, {})
                min_val, max_val = cat_ranges.get(metric, (value, value))
                
                if max_val > min_val:
                    norm = (value - min_val) / (max_val - min_val)
                else:
                    norm = 0.0
                new_cat[metric] = float(norm)
            new_entry[cat] = new_cat

        normalized.append(new_entry)

    return normalized


def write_jsonl(path: str, entries: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    entries = read_jsonl(INPUT_PATH)
    ranges = collect_metric_ranges_by_group(entries)
    normalized = normalize_entries_by_group(entries, ranges)
    write_jsonl(OUTPUT_PATH, normalized)


if __name__ == "__main__":
    main()


