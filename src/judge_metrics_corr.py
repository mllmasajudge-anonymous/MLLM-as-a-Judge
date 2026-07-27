import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

JUDGE_FILE = "results/normalized_judge_evaluations_all.jsonl"
TRADITIONAL_FILE = "results/normalized_traditional_evaluations.jsonl"
OUTPUT_FILE = "results/judge_traditional_correlations_all.jsonl"

def load_judge_evaluations(file_path, setting='offline'):
    judge_data = {}
    skipped_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            image_id = data['image_id']
            
            if setting == 'offline':
                if 'offline_factor_results' not in data:
                    skipped_count += 1
                    continue
                results = data['offline_factor_results']
            elif setting == 'online':
                if 'online_factor_results' not in data:
                    skipped_count += 1
                    continue
                results = data['online_factor_results']
            else:
                raise ValueError(f"Invalid setting: {setting}. Must be 'offline' or 'online'")
            
            factor_scores = {}
            for factor, result in results.items():
                factor_scores[factor] = result['score']
            
            judge_data[image_id] = factor_scores
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} entries without {setting}_factor_results")
    
    return judge_data

def load_traditional_evaluations(file_path, setting='offline'):
    traditional_data = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            image_id = data['image_id']
            
            if image_id.startswith('task_metrics'):
                continue
            
            data_setting = data['setting']
            
            if data_setting == setting:
                metrics = {}
                
                if 'pixel_level_fidelity' in data:
                    for metric, value in data['pixel_level_fidelity'].items():
                        metrics[metric] = value
                
                if 'content_preservation' in data:
                    for metric, value in data['content_preservation'].items():
                        metrics[metric] = value
                
                if 'perceptual_alignment' in data:
                    for metric, value in data['perceptual_alignment'].items():
                        metrics[metric] = value
                
                traditional_data[image_id] = metrics
    
    return traditional_data

def extract_task_from_image_id(image_id):
    if not image_id.startswith('HumanEdit_'):
        return None
    
    parts = image_id.split('_')
    if len(parts) >= 2:
        return parts[1]
    return None

def calculate_correlations(judge_scores, traditional_scores, setting='offline'):
    correlations = {}
    
    all_factors = set()
    all_metrics = set()
    
    for image_id in judge_scores:
        if image_id in traditional_scores:
            all_factors.update(judge_scores[image_id].keys())
            all_metrics.update(traditional_scores[image_id].keys())
    
    for factor in all_factors:
        factor_correlations = {}
        
        for metric in all_metrics:
            judge_values = []
            traditional_values = []
            
            for image_id in judge_scores:
                if (image_id in traditional_scores and 
                    factor in judge_scores[image_id] and 
                    metric in traditional_scores[image_id]):
                    judge_values.append(judge_scores[image_id][factor])
                    traditional_values.append(traditional_scores[image_id][metric])
            
            if len(judge_values) > 1:
                judge_array = np.array(judge_values)
                traditional_array = np.array(traditional_values)
                
                mse = mean_squared_error(judge_array, traditional_array)
                mae = mean_absolute_error(judge_array, traditional_array)
                
                try:
                    pearson_corr, pearson_p = pearsonr(judge_array, traditional_array)
                    spearman_corr, spearman_p = spearmanr(judge_array, traditional_array)
                    kendall_corr, kendall_p = kendalltau(judge_array, traditional_array)
                    
                    factor_correlations[metric] = {
                        'mse': float(mse),
                        'mae': float(mae),
                        'pearson': {
                            'correlation': float(pearson_corr),
                            'p_value': float(pearson_p)
                        },
                        'spearman': {
                            'correlation': float(spearman_corr),
                            'p_value': float(spearman_p)
                        },
                        'kendall': {
                            'correlation': float(kendall_corr),
                            'p_value': float(kendall_p)
                        },
                        'sample_size': len(judge_values)
                    }
                except Exception as e:
                    print(f"Error calculating correlations for {factor} vs {metric} ({setting}): {e}")
                    continue
        
        correlations[factor] = factor_correlations
    
    return correlations

def calculate_correlations_by_task(judge_scores, traditional_scores, setting='offline'):
    correlations = {}
    
    task_data = {}
    task_traditional_data = {}
    
    for image_id, factor_scores in judge_scores.items():
        task = extract_task_from_image_id(image_id)
        if task is None:
            continue
        if task not in task_data:
            task_data[task] = {}
        task_data[task][image_id] = factor_scores
    
    for image_id, metrics in traditional_scores.items():
        task = extract_task_from_image_id(image_id)
        if task is None:
            continue
        if task not in task_traditional_data:
            task_traditional_data[task] = {}
        task_traditional_data[task][image_id] = metrics
    
    for task in task_data:
        if task not in task_traditional_data:
            continue
        
        task_judge_scores = task_data[task]
        task_traditional_scores = task_traditional_data[task]
        
        all_factors = set()
        all_metrics = set()
        
        for image_id in task_judge_scores:
            if image_id in task_traditional_scores:
                all_factors.update(task_judge_scores[image_id].keys())
                all_metrics.update(task_traditional_scores[image_id].keys())
        
        task_correlations = {}
        for factor in all_factors:
            factor_correlations = {}
            
            for metric in all_metrics:
                judge_values = []
                traditional_values = []
                
                for image_id in task_judge_scores:
                    if (image_id in task_traditional_scores and 
                        factor in task_judge_scores[image_id] and 
                        metric in task_traditional_scores[image_id]):
                        judge_values.append(task_judge_scores[image_id][factor])
                        traditional_values.append(task_traditional_scores[image_id][metric])
                
                if len(judge_values) > 1:
                    judge_array = np.array(judge_values)
                    traditional_array = np.array(traditional_values)
                    
                    mse = mean_squared_error(judge_array, traditional_array)
                    mae = mean_absolute_error(judge_array, traditional_array)
                    
                    try:
                        pearson_corr, pearson_p = pearsonr(judge_array, traditional_array)
                        spearman_corr, spearman_p = spearmanr(judge_array, traditional_array)
                        kendall_corr, kendall_p = kendalltau(judge_array, traditional_array)
                        
                        factor_correlations[metric] = {
                            'mse': float(mse),
                            'mae': float(mae),
                            'pearson': {
                                'correlation': float(pearson_corr),
                                'p_value': float(pearson_p)
                            },
                            'spearman': {
                                'correlation': float(spearman_corr),
                                'p_value': float(spearman_p)
                            },
                            'kendall': {
                                'correlation': float(kendall_corr),
                                'p_value': float(kendall_p)
                            },
                            'sample_size': len(judge_values)
                        }
                    except Exception as e:
                        print(f"Error calculating correlations for {factor} vs {metric} (task={task}, setting={setting}): {e}")
                        continue
            
            if factor_correlations:
                task_correlations[factor] = factor_correlations
        
        if task_correlations:
            correlations[task] = task_correlations
    
    return correlations

def process_setting(judge_file, traditional_file, setting, output_file, write_mode='w'):
    print(f"\n=== Processing {setting.upper()} setting ===")
    
    print(f"Loading {setting} judge evaluations...")
    judge_data = load_judge_evaluations(judge_file, setting)
    print(f"Loaded {len(judge_data)} {setting} judge evaluations")
    
    print(f"Loading {setting} traditional evaluations...")
    traditional_data = load_traditional_evaluations(traditional_file, setting)
    print(f"Loaded {len(traditional_data)} {setting} traditional evaluations")
    
    print(f"Calculating {setting} correlations...")
    correlations = calculate_correlations(judge_data, traditional_data, setting)
    
    print(f"Calculating {setting} correlations by task...")
    task_correlations = calculate_correlations_by_task(judge_data, traditional_data, setting)
    
    print(f"Saving {setting} results...")
    with open(output_file, write_mode, encoding='utf-8') as f:
        for factor, factor_correlations in correlations.items():
            cleaned_correlations = {}
            for metric, correlation_data in factor_correlations.items():
                cleaned_metric = metric.replace('_norm', '') if metric.endswith('_norm') else metric
                cleaned_correlations[cleaned_metric] = correlation_data
            
            result = {
                'setting': setting,
                'factor': factor,
                'correlations': cleaned_correlations
            }
            f.write(json.dumps(result) + '\n')
        
        for task, task_factor_correlations in task_correlations.items():
            for factor, factor_correlations in task_factor_correlations.items():
                cleaned_correlations = {}
                for metric, correlation_data in factor_correlations.items():
                    cleaned_metric = metric.replace('_norm', '') if metric.endswith('_norm') else metric
                    cleaned_correlations[cleaned_metric] = correlation_data
                
                result = {
                    'setting': setting,
                    'factor': factor,
                    'task': task,
                    'correlations': cleaned_correlations
                }
                f.write(json.dumps(result) + '\n')
    
    print(f"{setting.capitalize()} results saved to {output_file}")
    
    print(f"\n{setting.capitalize()} Summary:")
    print(f"  Overall correlations: {len(correlations)} factors")
    for factor, factor_correlations in correlations.items():
        print(f"\n  Factor: {factor}")
        print(f"    Number of traditional metrics: {len(factor_correlations)}")
        if factor_correlations:
            sample_sizes = [data['sample_size'] for data in factor_correlations.values()]
            print(f"    Sample sizes: {min(sample_sizes)}-{max(sample_sizes)}")
    
    print(f"\n  Task-specific correlations: {len(task_correlations)} tasks")
    for task, task_factor_correlations in task_correlations.items():
        print(f"\n  Task: {task}")
        print(f"    Number of factors: {len(task_factor_correlations)}")
        for factor, factor_correlations in task_factor_correlations.items():
            if factor_correlations:
                sample_sizes = [data['sample_size'] for data in factor_correlations.values()]
                print(f"      Factor {factor}: {len(factor_correlations)} metrics, sample sizes: {min(sample_sizes)}-{max(sample_sizes)}")
    
    return correlations, task_correlations

def main():
    offline_correlations, offline_task_correlations = process_setting(JUDGE_FILE, TRADITIONAL_FILE, 'offline', OUTPUT_FILE, 'w')
    
    online_correlations, online_task_correlations = process_setting(JUDGE_FILE, TRADITIONAL_FILE, 'online', OUTPUT_FILE, 'a')
    
    print("\n=== Overall Summary ===")
    print(f"Offline correlations: {len(offline_correlations)} factors")
    print(f"Online correlations: {len(online_correlations)} factors")
    print(f"Offline task-specific correlations: {len(offline_task_correlations)} tasks")
    print(f"Online task-specific correlations: {len(online_task_correlations)} tasks")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("File contains:")
    print("  - Overall correlations (setting + factor)")
    print("  - Task-specific correlations (setting + factor + task)")
    print("  Use 'task' field to distinguish task-specific results (missing for overall results).")

if __name__ == "__main__":
    main()
