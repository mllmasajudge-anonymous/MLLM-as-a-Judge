import json
import os
from typing import Dict, Any, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(PROJECT_ROOT, "results", "judge_evaluations_all_v3_gemini.jsonl")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "normalized_judge_evaluations_all_v3_gemini.jsonl")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def get_task_from_image_id(image_id: str) -> str:
    parts = image_id.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return "Unknown"


def collect_score_ranges(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    ranges: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    settings = ["offline_factor_results", "online_factor_results"]

    for entry in entries:
        image_id = entry.get("image_id", "")
        task = get_task_from_image_id(image_id)
        
        if task not in ranges:
            ranges[task] = {}
        
        for setting in settings:
            setting_data = entry.get(setting)
            if not isinstance(setting_data, dict):
                continue
            
            if setting not in ranges[task]:
                ranges[task][setting] = {}
            
            for factor, factor_data in setting_data.items():
                if not isinstance(factor_data, dict):
                    continue
                
                score = factor_data.get("score")
                if not isinstance(score, (int, float)):
                    continue
                
                if factor not in ranges[task][setting]:
                    ranges[task][setting][factor] = (score, score)
                else:
                    current_min, current_max = ranges[task][setting][factor]
                    if score < current_min:
                        current_min = score
                    if score > current_max:
                        current_max = score
                    ranges[task][setting][factor] = (current_min, current_max)

    return ranges


def normalize_entries(entries: List[Dict[str, Any]], ranges: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    settings = ["offline_factor_results", "online_factor_results"]

    for entry in entries:
        image_id = entry.get("image_id", "")
        task = get_task_from_image_id(image_id)
        
        new_entry: Dict[str, Any] = {"image_id": image_id}
        
        for setting in settings:
            setting_data = entry.get(setting)
            if not isinstance(setting_data, dict):
                continue
            
            new_setting: Dict[str, Any] = {}
            for factor, factor_data in setting_data.items():
                if not isinstance(factor_data, dict):
                    new_setting[factor] = factor_data
                    continue
                
                score = factor_data.get("score")
                if not isinstance(score, (int, float)):
                    new_setting[factor] = factor_data
                    continue
                
                task_ranges = ranges.get(task, {})
                setting_ranges = task_ranges.get(setting, {})
                min_val, max_val = setting_ranges.get(factor, (score, score))
                
                if max_val > min_val:
                    norm_score = (score - min_val) / (max_val - min_val)
                else:
                    norm_score = 0.0
                
                new_factor_data = dict(factor_data)
                new_factor_data["score"] = float(norm_score)
                new_setting[factor] = new_factor_data
            
            new_entry[setting] = new_setting

        normalized.append(new_entry)

    return normalized


def write_jsonl(path: str, entries: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    entries = read_jsonl(INPUT_PATH)
    ranges = collect_score_ranges(entries)
    normalized = normalize_entries(entries, ranges)
    write_jsonl(OUTPUT_PATH, normalized)


if __name__ == "__main__":
    main()