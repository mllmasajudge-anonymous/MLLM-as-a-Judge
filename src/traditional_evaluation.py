import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torchvision.transforms as T
from scipy import linalg
from torchvision.models import inception_v3
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d
import glob
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
API_IMG_DIR = os.path.join(PROJECT_ROOT, "HumanEdit", "api_img_400")
GT_IMG_DIR = os.path.join(PROJECT_ROOT, "HumanEdit", "gt_img_400")
INPUT_IMG_DIR = os.path.join(PROJECT_ROOT, "HumanEdit", "input_img_400")
MASK_IMG_DIR = os.path.join(PROJECT_ROOT, "HumanEdit", "mask_img_400")
INSTRUCTIONS_DIR = os.path.join(PROJECT_ROOT, "HumanEdit", "instructions_400")
OUTPUT_JSONL_PATH = os.path.join(PROJECT_ROOT, "results", "traditional_evaluations.jsonl")

os.makedirs(os.path.dirname(OUTPUT_JSONL_PATH), exist_ok=True)

MAX_WORKERS = min(4, os.cpu_count())
print(f"Using {MAX_WORKERS} threads for parallel processing")
print(f"Available CPU cores: {os.cpu_count()}")
print(f"Note: Adjust MAX_WORKERS in the code if you want different threading behavior")

file_lock = threading.Lock()

import time
start_time = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading LPIPS model...")
lpips_model = lpips.LPIPS(net='alex').to("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Inception model...")
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.eval()
if torch.cuda.is_available():
    inception_model = inception_model.cuda()

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, inception_model):
        super().__init__()
        self.inception = inception_model
        
    def forward(self, x):
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        x = adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

inception_feature_extractor = InceptionFeatureExtractor(inception_model)
if torch.cuda.is_available():
    inception_feature_extractor = inception_feature_extractor.cuda()

def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image)

def calculate_l1_l2_metrics(img1_path, img2_path):
    img1 = load_image_as_tensor(img1_path)
    img2 = load_image_as_tensor(img2_path)
    
    if img1.shape != img2.shape:
        img2 = F.interpolate(img2.unsqueeze(0), size=img1.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
    
    l1_error = torch.mean(torch.abs(img1 - img2)).item()
    l2_error = torch.mean((img1 - img2) ** 2).item()
    
    return l1_error, l2_error

def calculate_clip_similarity(img1_path, img2_path):
    
    return None

def calculate_dino_similarity(img1_path, img2_path):

    return None

def calculate_psnr_ssim_lpips(img1_path, img2_path):
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        
        psnr_value = psnr(img1_np, img2_np, data_range=255)
        
        ssim_value = ssim(img1_np, img2_np, channel_axis=2, data_range=255)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img1_tensor = transform(img1).unsqueeze(0)
        img2_tensor = transform(img2).unsqueeze(0)
        
        if torch.cuda.is_available():
            img1_tensor = img1_tensor.cuda()
            img2_tensor = img2_tensor.cuda()
        
        with torch.no_grad():
            lpips_value = lpips_model(img1_tensor, img2_tensor).item()
        
        return psnr_value, ssim_value, lpips_value
    except Exception as e:
        print(f"Error calculating PSNR/SSIM/LPIPS: {e}")
        return None, None, None

def create_binary_mask_from_color_mask(mask_path, threshold=200):
    try:
        mask = Image.open(mask_path).convert('RGB')
        mask_np = np.array(mask)
        
        mask_gray = np.mean(mask_np, axis=2)
        
        binary_mask = (mask_gray > threshold).astype(np.float32)
        
        print(f"Mask processing: {mask_path}")
        print(f"  - Original mask shape: {mask_np.shape}")
        print(f"  - Binary mask shape: {binary_mask.shape}")
        print(f"  - Edit regions (white/blank): {binary_mask.sum():.0f} pixels")
        print(f"  - Preserved regions (colored): {(binary_mask == 0).sum():.0f} pixels")
        
        return binary_mask
    except Exception as e:
        print(f"Error creating binary mask: {e}")
        return None

def calculate_mask_ssim_lpips(img1_path, img2_path, mask_path):
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        binary_mask = create_binary_mask_from_color_mask(mask_path)
        if binary_mask is None:
            return None, None
        
        if binary_mask.shape[:2] != img1.size[::-1]:
            mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(img1.size, Image.Resampling.LANCZOS)
            binary_mask = np.array(mask_pil) / 255.0
        
        img1_np = np.array(img1)
        img2_np = np.array(img2)

        ssim_value = ssim(img1_np, img2_np, channel_axis=2, data_range=255)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img1_tensor = transform(img1).unsqueeze(0)
        img2_tensor = transform(img2).unsqueeze(0)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0)
        
        if torch.cuda.is_available():
            img1_tensor = img1_tensor.cuda()
            img2_tensor = img2_tensor.cuda()
            mask_tensor = mask_tensor.cuda()
        
        with torch.no_grad():
            lpips_value = lpips_model(img1_tensor, img2_tensor)
            if mask_tensor.sum() > 0:
                lpips_value = (lpips_value * mask_tensor).sum() / mask_tensor.sum()
            else:
                lpips_value = torch.tensor(0.0)
            lpips_value = lpips_value.item()
        
        return ssim_value, lpips_value
    except Exception as e:
        print(f"Error calculating Mask-SSIM/Mask-LPIPS: {e}")
        return None, None

def calculate_background_consistency(img1_path, img2_path):
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

        img1_np = np.array(img1)
        img2_np = np.array(img2)

        consistency = ssim(img1_np, img2_np, channel_axis=2, data_range=255)
        
        return consistency
    except Exception as e:
        print(f"Error calculating Background Consistency: {e}")
        return None

def get_inception_features(image_paths, batch_size=32):
    features = []
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(image_paths), batch_size), 
                desc="Extracting features", 
                unit="batch", 
                total=num_batches)
    
    for i in pbar:
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                batch_tensors.append(img_tensor)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue
        
        if batch_tensors:
            batch = torch.cat(batch_tensors, dim=0)
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            with torch.no_grad():
                batch_features = inception_feature_extractor(batch)
                features.append(batch_features.cpu().numpy())
            
            pbar.set_postfix_str(f"Processed {len(features)} batches")
    
    pbar.close()
    
    if features:
        return np.concatenate(features, axis=0)
    else:
        return np.array([])

def calculate_fid_for_task_offline(api_img_dir, input_img_dir, task_name):
    try:
        api_paths = []
        input_paths = []
        
        for f in os.listdir(api_img_dir):
            if f.endswith('.png') and extract_task_from_filename(f) == task_name:
                api_paths.append(os.path.join(api_img_dir, f))
                input_path = os.path.join(input_img_dir, f)
                if os.path.exists(input_path):
                    input_paths.append(input_path)
        
        if len(api_paths) == 0:
            print(f"No images found for task: {task_name}")
            return None
            
        print(f"Calculating OFFLINE FID for task '{task_name}' with {len(api_paths)} API images and {len(input_paths)} input images")
        
        api_features = get_inception_features(api_paths)
        input_features = get_inception_features(input_paths)
        
        if len(api_features) == 0 or len(input_features) == 0:
            print(f"No valid features extracted for FID calculation for task: {task_name}")
            return None
        
        mu1, sigma1 = np.mean(api_features, axis=0), np.cov(api_features, rowvar=False)
        mu2, sigma2 = np.mean(input_features, axis=0), np.cov(input_features, rowvar=False)
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print(f"FID calculation produced singular matrix for task {task_name}, adding regularization")
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return float(fid)
    except Exception as e:
        print(f"Error calculating OFFLINE FID for task {task_name}: {e}")
        return None

def calculate_fid_for_task_online(gt_img_dir, input_img_dir, task_name):
    try:
        gt_paths = []
        input_paths = []
        
        for f in os.listdir(gt_img_dir):
            if f.endswith('.png') and extract_task_from_filename(f) == task_name:
                gt_paths.append(os.path.join(gt_img_dir, f))
                input_path = os.path.join(input_img_dir, f)
                if os.path.exists(input_path):
                    input_paths.append(input_path)
        
        if len(gt_paths) == 0:
            print(f"No images found for task: {task_name}")
            return None
            
        print(f"Calculating ONLINE FID for task '{task_name}' with {len(gt_paths)} GT images and {len(input_paths)} input images")
        
        gt_features = get_inception_features(gt_paths)
        input_features = get_inception_features(input_paths)
        
        if len(gt_features) == 0 or len(input_features) == 0:
            print(f"No valid features extracted for FID calculation for task: {task_name}")
            return None
        
        mu1, sigma1 = np.mean(gt_features, axis=0), np.cov(gt_features, rowvar=False)
        mu2, sigma2 = np.mean(input_features, axis=0), np.cov(input_features, rowvar=False)
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print(f"FID calculation produced singular matrix for task {task_name}, adding regularization")
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return float(fid)
    except Exception as e:
        print(f"Error calculating ONLINE FID for task {task_name}: {e}")
        return None

def calculate_fid(api_img_dir, input_img_dir):
    try:
        api_paths = [os.path.join(api_img_dir, f) for f in os.listdir(api_img_dir) if f.endswith('.png')]
        input_paths = [os.path.join(input_img_dir, f) for f in os.listdir(input_img_dir) if f.endswith('.png')]
        
        print(f"Calculating FID with {len(api_paths)} API images and {len(input_paths)} input images")
        
        api_features = get_inception_features(api_paths)
        input_features = get_inception_features(input_paths)
        
        if len(api_features) == 0 or len(input_features) == 0:
            print("No valid features extracted for FID calculation")
            return None
        
        mu1, sigma1 = np.mean(api_features, axis=0), np.cov(api_features, rowvar=False)
        mu2, sigma2 = np.mean(input_features, axis=0), np.cov(input_features, rowvar=False)
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print("FID calculation produced singular matrix, adding regularization")
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return float(fid)
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return None

def calculate_inception_score_for_task_offline(api_img_dir, task_name):
    try:
        api_paths = []
        for f in os.listdir(api_img_dir):
            if f.endswith('.png') and extract_task_from_filename(f) == task_name:
                api_paths.append(os.path.join(api_img_dir, f))
        
        if len(api_paths) == 0:
            print(f"No images found for task: {task_name}")
            return None
            
        print(f"Calculating OFFLINE IS for task '{task_name}' with {len(api_paths)} API images")
        
        features = get_inception_features(api_paths)
        
        if len(features) == 0:
            print(f"No valid features extracted for IS calculation for task: {task_name}")
            return None
        
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        probs = []
        prob_pbar = tqdm(api_paths, desc="Calculating OFFLINE probabilities", unit="image", leave=False)
        
        for path in prob_pbar:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                
                with torch.no_grad():
                    logits = inception_model(img_tensor)
                    prob = F.softmax(logits, dim=1)
                    probs.append(prob.cpu().numpy())
                
                prob_pbar.set_postfix_str(f"Processed {len(probs)}/{len(api_paths)}")
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                continue
        
        prob_pbar.close()
        
        if not probs:
            print(f"No valid probabilities calculated for IS for task: {task_name}")
            return None
        
        probs = np.concatenate(probs, axis=0)
        
        py = np.mean(probs, axis=0)
        scores = []
        for i in range(probs.shape[0]):
            pyx = probs[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py)))
        
        is_score = np.exp(np.mean(scores))
        return float(is_score)
    except Exception as e:
        print(f"Error calculating OFFLINE Inception Score for task {task_name}: {e}")
        return None

def calculate_inception_score_for_task_online(gt_img_dir, task_name):
    try:
        gt_paths = []
        for f in os.listdir(gt_img_dir):
            if f.endswith('.png') and extract_task_from_filename(f) == task_name:
                gt_paths.append(os.path.join(gt_img_dir, f))
        
        if len(gt_paths) == 0:
            print(f"No images found for task: {task_name}")
            return None
            
        print(f"Calculating ONLINE IS for task '{task_name}' with {len(gt_paths)} GT images")
        
        features = get_inception_features(gt_paths)
        
        if len(features) == 0:
            print(f"No valid features extracted for IS calculation for task: {task_name}")
            return None
        
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        probs = []
        prob_pbar = tqdm(gt_paths, desc="Calculating ONLINE probabilities", unit="image", leave=False)
        
        for path in prob_pbar:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                
                with torch.no_grad():
                    logits = inception_model(img_tensor)
                    prob = F.softmax(logits, dim=1)
                    probs.append(prob.cpu().numpy())
                
                prob_pbar.set_postfix_str(f"Processed {len(probs)}/{len(gt_paths)}")
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                continue
        
        prob_pbar.close()
        
        if not probs:
            print(f"No valid probabilities calculated for IS for task: {task_name}")
            return None
        
        probs = np.concatenate(probs, axis=0)
        
        py = np.mean(probs, axis=0)
        scores = []
        for i in range(probs.shape[0]):
            pyx = probs[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py)))
        
        is_score = np.exp(np.mean(scores))
        return float(is_score)
    except Exception as e:
        print(f"Error calculating ONLINE Inception Score for task {task_name}: {e}")
        return None

def calculate_inception_score(api_img_dir):
    try:
        api_paths = [os.path.join(api_img_dir, f) for f in os.listdir(api_img_dir) if f.endswith('.png')]
        
        print(f"Calculating IS with {len(api_paths)} API images")
        
        features = get_inception_features(api_paths)
        
        if len(features) == 0:
            print("No valid features extracted for IS calculation")
            return None
        
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        probs = []
        for path in api_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                
                with torch.no_grad():
                    logits = inception_model(img_tensor)
                    prob = F.softmax(logits, dim=1)
                    probs.append(prob.cpu().numpy())
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                continue
        
        if not probs:
            print("No valid probabilities calculated for IS")
            return None
        
        probs = np.concatenate(probs, axis=0)
        
        py = np.mean(probs, axis=0)
        scores = []
        for i in range(probs.shape[0]):
            pyx = probs[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py)))
        
        is_score = np.exp(np.mean(scores))
        return float(is_score)
    except Exception as e:
        print(f"Error calculating Inception Score: {e}")
        return None

def get_instruction_from_file(instruction_path):
    with open(instruction_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def extract_task_from_filename(filename):
    try:
        base_name = filename.replace('.png', '')
        parts = base_name.split('_')
        if len(parts) >= 2:
            return parts[1]
        else:
            return "Unknown"
    except:
        return "Unknown"

def get_all_api_images():
    return [f for f in os.listdir(API_IMG_DIR) if f.endswith('.png')]

def get_all_tasks():
    tasks = set()
    for f in os.listdir(API_IMG_DIR):
        if f.endswith('.png'):
            task = extract_task_from_filename(f)
            if task != "Unknown":
                tasks.add(task)
    return sorted(list(tasks))

def process_single_evaluation(api_image_name):
    base_name = api_image_name.replace('.png', '')
    
    ground_truth_path = os.path.join(GT_IMG_DIR, api_image_name)
    api_path = os.path.join(API_IMG_DIR, api_image_name)
    input_path = os.path.join(INPUT_IMG_DIR, api_image_name)
    mask_path = os.path.join(MASK_IMG_DIR, api_image_name)
    instruction_path = os.path.join(INSTRUCTIONS_DIR, f"{base_name}.txt")
    
    required_files = [ground_truth_path, api_path, input_path, instruction_path, mask_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            return None
    
    instruction = get_instruction_from_file(instruction_path)
    
    print(f"Processing evaluation for: {base_name}")
    print(f"Instruction: {instruction}")
    
    results = []
    
    print("Calculating OFFLINE Pixel-level Fidelity metrics (API vs GT)...")
    l1_error_offline, l2_error_offline = calculate_l1_l2_metrics(api_path, ground_truth_path)
    psnr_value_offline, ssim_value_offline, lpips_value_offline = calculate_psnr_ssim_lpips(api_path, ground_truth_path)
    
    print("Calculating OFFLINE Content Preservation metrics...")
    mask_ssim_value_offline, mask_lpips_value_offline = calculate_mask_ssim_lpips(api_path, ground_truth_path, mask_path)
    background_consistency_offline = calculate_background_consistency(api_path, ground_truth_path)
    
    offline_result = {
        "image_id": base_name,
        "instruction": instruction,
        "setting": "offline",
        "pixel_level_fidelity": {
            "l1_error": l1_error_offline,
            "l2_error": l2_error_offline,
            "psnr": psnr_value_offline,
            "ssim": ssim_value_offline,
            "lpips": lpips_value_offline
        },
        "content_preservation": {
            "mask_ssim": mask_ssim_value_offline,
            "mask_lpips": mask_lpips_value_offline,
            "background_consistency": background_consistency_offline
        }
    }
    results.append(offline_result)
    
    print("Calculating ONLINE Pixel-level Fidelity metrics (Input vs API)...")
    l1_error_online, l2_error_online = calculate_l1_l2_metrics(input_path, api_path)
    psnr_value_online, ssim_value_online, lpips_value_online = calculate_psnr_ssim_lpips(input_path, api_path)
    
    print("Calculating ONLINE Content Preservation metrics...")
    mask_ssim_value_online, mask_lpips_value_online = calculate_mask_ssim_lpips(input_path, api_path, mask_path)
    background_consistency_online = calculate_background_consistency(input_path, api_path)
    
    online_result = {
        "image_id": base_name,
        "instruction": instruction,
        "setting": "online",
        "pixel_level_fidelity": {
            "l1_error": l1_error_online,
            "l2_error": l2_error_online,
            "psnr": psnr_value_online,
            "ssim": ssim_value_online,
            "lpips": lpips_value_online
        },
        "content_preservation": {
            "mask_ssim": mask_ssim_value_online,
            "mask_lpips": mask_lpips_value_online,
            "background_consistency": background_consistency_online
        }
    }
    results.append(online_result)
    
    print(f"Successfully evaluated {base_name}")
    print(f"OFFLINE SETTING:")
    print(f"  Pixel-level Fidelity:")
    print(f"    L1 Error: {l1_error_offline:.6f}")
    print(f"    L2 Error: {l2_error_offline:.6f}")
    print(f"    PSNR: {psnr_value_offline:.6f}")
    print(f"    SSIM: {ssim_value_offline:.6f}")
    print(f"    LPIPS: {lpips_value_offline:.6f}")
    print(f"  Content Preservation:")
    print(f"    Mask-SSIM: {mask_ssim_value_offline:.6f}")
    print(f"    Mask-LPIPS: {mask_lpips_value_offline:.6f}")
    print(f"    Background Consistency: {background_consistency_offline:.6f}")
    print(f"ONLINE SETTING (Input vs API):")
    print(f"  Pixel-level Fidelity:")
    print(f"    L1 Error: {l1_error_online:.6f}")
    print(f"    L2 Error: {l2_error_online:.6f}")
    print(f"    PSNR: {psnr_value_online:.6f}")
    print(f"    SSIM: {ssim_value_online:.6f}")
    print(f"    LPIPS: {lpips_value_online:.6f}")
    print(f"  Content Preservation:")
    print(f"    Mask-SSIM: {mask_ssim_value_online:.6f}")
    print(f"    Mask-LPIPS: {mask_lpips_value_online:.6f}")
    print(f"    Background Consistency: {background_consistency_online:.6f}")
    print("-" * 80)
    
    return results

def save_to_jsonl(evaluations, output_path, mode='a'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with file_lock:
        with open(output_path, mode, encoding='utf-8') as f:
            for evaluation in evaluations:
                if evaluation is not None:
                    f.write(json.dumps(evaluation, ensure_ascii=False) + '\n')
    
    if mode == 'a':
        print(f"Results appended to: {output_path}")
    else:
        print(f"Results saved to: {output_path}")

def process_single_evaluation_threaded(api_image_name, processed_ids):
    base_name = api_image_name.replace('.png', '')
    
    if base_name in processed_ids:
        return None, f"Skipped: {api_image_name} (already processed)"
    
    try:
        evaluation_results = process_single_evaluation(api_image_name)
        
        if evaluation_results is not None:
            save_to_jsonl(evaluation_results, OUTPUT_JSONL_PATH)
            return evaluation_results, f"Saved: {api_image_name}"
        else:
            return None, f"Failed: {api_image_name}"
    except Exception as e:
        return None, f"Error processing {api_image_name}: {str(e)}"

def main():
    print("Starting traditional evaluation process...")
    
    api_images = get_all_api_images()
    print(f"Found {len(api_images)} API images to evaluate")
    
    processed_ids = set()
    if os.path.exists(OUTPUT_JSONL_PATH):
        with open(OUTPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'image_id' in data:
                        processed_ids.add(data['image_id'])
                except:
                    continue
        print(f"Found {len(processed_ids)} already processed images")
    
    evaluations = []
    skipped = 0
    
    images_to_process = []
    for api_image in api_images:
        base_name = api_image.replace('.png', '')
        if base_name in processed_ids:
            skipped += 1
        else:
            images_to_process.append(api_image)
    
    print(f"Processing {len(images_to_process)} images with {MAX_WORKERS} threads...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_image = {
            executor.submit(process_single_evaluation_threaded, api_image, processed_ids): api_image 
            for api_image in images_to_process
        }
        
        pbar = tqdm(total=len(images_to_process), desc="Processing images", unit="image")
        
        for future in as_completed(future_to_image):
            api_image = future_to_image[future]
            try:
                evaluation_results, status = future.result()
                
                if evaluation_results is not None:
                    evaluations.extend(evaluation_results)
                    pbar.set_postfix_str(f"{api_image}")
                else:
                    pbar.set_postfix_str(f"Failed: {api_image}")
                    
            except Exception as e:
                pbar.set_postfix_str(f"Error: {api_image}")
                print(f"Error processing {api_image}: {e}")
            
            pbar.update(1)
        
        pbar.close()
    
    print("\n" + "="*80)
    print("Calculating Perceptual Quality metrics per task...")
    print("="*80)
    
    all_tasks = get_all_tasks()
    print(f"Found tasks: {all_tasks}")
    
    task_metrics = []
    
    def calculate_task_metrics(task):
        try:
            fid_offline_value = calculate_fid_for_task_offline(API_IMG_DIR, INPUT_IMG_DIR, task)
            
            fid_online_value = calculate_fid_for_task_offline(API_IMG_DIR, INPUT_IMG_DIR, task)
            
            is_offline_value = calculate_inception_score_for_task_offline(API_IMG_DIR, task)
            
            is_online_value = calculate_inception_score_for_task_offline(API_IMG_DIR, task)
            
            task_metric_offline = {
                "image_id": f"task_metrics_{task}_offline",
                "instruction": f"Perceptual quality metrics for task: {task} (OFFLINE setting)",
                "task": task,
                "setting": "offline",
                "perceptual_quality": {
                    "fid": fid_offline_value,
                    "inception_score": is_offline_value
                }
            }
            
            task_metric_online = {
                "image_id": f"task_metrics_{task}_online",
                "instruction": f"Perceptual quality metrics for task: {task} (ONLINE setting)",
                "task": task,
                "setting": "online",
                "perceptual_quality": {
                    "fid": fid_online_value,
                    "inception_score": is_online_value
                }
            }
            
            save_to_jsonl([task_metric_offline, task_metric_online], OUTPUT_JSONL_PATH)
            
            return [task_metric_offline, task_metric_online], f"Saved: {task} (FID: {fid_offline_value:.2f}/{fid_online_value:.2f}, IS: {is_offline_value:.2f}/{is_online_value:.2f})"
            
        except Exception as e:
            return None, f"Error processing {task}: {str(e)}"
    
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(all_tasks))) as executor:
        future_to_task = {
            executor.submit(calculate_task_metrics, task): task 
            for task in all_tasks
        }
        
        task_pbar = tqdm(total=len(all_tasks), desc="Calculating task metrics", unit="task")
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                task_results, status = future.result()
                
                if task_results is not None:
                    task_metrics.extend(task_results)
                    task_pbar.set_postfix_str(f"{task}")
                else:
                    task_pbar.set_postfix_str(f"Failed: {task}")
                    
            except Exception as e:
                task_pbar.set_postfix_str(f"Error: {task}")
                print(f"Error processing task {task}: {e}")
            
            task_pbar.update(1)
        
        task_pbar.close()
    
    successful = sum(1 for e in evaluations if e is not None)
    offline_count = sum(1 for e in evaluations if e is not None and e.get('setting') == 'offline')
    online_count = sum(1 for e in evaluations if e is not None and e.get('setting') == 'online')
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Total images found: {len(api_images)}")
    print(f"Already processed: {skipped}")
    print(f"Newly processed: {len(evaluations)}")
    print(f"Successfully evaluated: {successful}")
    print(f"  - Offline setting results: {offline_count}")
    print(f"  - Online setting results: {online_count}")
    print(f"Failed: {len(evaluations) - successful}")
    print(f"Tasks processed: {len(all_tasks)}")
    print(f"Perceptual Quality Metrics per Task:")
    for task_metric in task_metrics:
        task = task_metric['task']
        setting = task_metric['setting']
        fid = task_metric['perceptual_quality']['fid']
        is_score = task_metric['perceptual_quality']['inception_score']
        print(f"  {task} ({setting}): FID={fid}, IS={is_score}")
    print(f"Results saved to: {OUTPUT_JSONL_PATH}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n=== Performance Statistics ===")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / max(len(api_images), 1):.2f} seconds")
    print(f"Threading efficiency: {MAX_WORKERS} workers used")
    if len(images_to_process) > 0:
        print(f"Processing rate: {len(images_to_process) / total_time:.2f} images/second")

if __name__ == "__main__":
    main()
