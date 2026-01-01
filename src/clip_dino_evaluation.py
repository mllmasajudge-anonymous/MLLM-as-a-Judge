
import os
import io
import json
import math
import glob
import time
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

# Robust __file__ handling (works if run as script or in notebooks)
try:
    SCRIPT_DIR = ""
except NameError:
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = SCRIPT_DIR  # adjusted for this environment; repo root == current dir

HUMANEDIT_DIR = os.path.join(PROJECT_ROOT, "HumanEdit")
API_IMG_DIR = os.path.join(HUMANEDIT_DIR, "api_img")
GT_IMG_DIR = os.path.join(HUMANEDIT_DIR, "gt_img")
INPUT_IMG_DIR = os.path.join(HUMANEDIT_DIR, "input_img")
MASK_IMG_DIR = os.path.join(HUMANEDIT_DIR, "mask_img")
INSTRUCTIONS_DIR = os.path.join(HUMANEDIT_DIR, "instructions")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_JSONL = os.path.join(RESULTS_DIR, "clip_dino_evaluations.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def list_api_images() -> List[str]:
    if not os.path.isdir(API_IMG_DIR):
        return []
    return sorted([f for f in os.listdir(API_IMG_DIR) if f.lower().endswith(".png")])


def read_instruction(image_basename: str) -> Optional[str]:
    path = os.path.join(INSTRUCTIONS_DIR, f"{image_basename}.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None
    return None


def load_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return float((a * b).sum().item())


def extract_task_from_filename(filename: str) -> str:
    """
    Matches your traditional_evaluation.py:
    Format: HumanEdit_{task}_{number}_{id}.png  -> returns {task}
    """
    try:
        base = filename.replace(".png", "")
        parts = base.split("_")
        return parts[1] if len(parts) >= 2 else "Unknown"
    except Exception:
        return "Unknown"


class CLIPWrapper:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenize = None
        self.loaded = False
        self.kind = "unavailable"
        self.load()

    def load(self):
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE)
            self.model = model.eval()
            self.preprocess = preprocess
            self.tokenize = open_clip.get_tokenizer("ViT-B-32")
            self.kind = "open_clip ViT-B/32 laion2b_s34b_b79k"
            self.loaded = True
            return
        except Exception:
            pass
        try:
            import clip as openai_clip
            model, preprocess = openai_clip.load("ViT-B/32", device=DEVICE, jit=False)
            self.model = model.eval()
            self.preprocess = preprocess
            self.tokenize = openai_clip.tokenize
            self.kind = "openai-clip ViT-B/32"
            self.loaded = True
            return
        except Exception:
            self.loaded = False

    def encode_image(self, img: Image.Image) -> Optional[torch.Tensor]:
        if not self.loaded:
            return None
        with torch.no_grad():
            t = self.preprocess(img).unsqueeze(0).to(DEVICE)
            return self.model.encode_image(t).float().cpu().squeeze(0)

    def encode_text(self, text: str) -> Optional[torch.Tensor]:
        if not self.loaded:
            return None
        with torch.no_grad():
            tok = self.tokenize([text]).to(DEVICE)
            return self.model.encode_text(tok).float().cpu().squeeze(0)


class DINOv2Wrapper:
    def __init__(self):
        self.model = None
        self.size = 518
        self.loaded = False
        self.kind = "unavailable"
        self.load()

    def load(self):
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.model.eval().to(DEVICE)
            self.loaded = True
            self.kind = "dinov2_vits14 (torch.hub)"
            return
        except Exception:
            pass
        try:
            import timm
            self.model = timm.create_model("vit_small_patch14_dinov2", pretrained=True)
            self.model.eval().to(DEVICE)
            self.loaded = True
            self.kind = "timm vit_small_patch14_dinov2"
            return
        except Exception:
            self.loaded = False

    def encode_image(self, img: Image.Image) -> Optional[torch.Tensor]:
        if not self.loaded:
            return None
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats = self.model.forward_features(x)
            if isinstance(feats, dict) and "x_norm_clstoken" in feats:
                emb = feats["x_norm_clstoken"]
            elif isinstance(feats, dict) and "pre_logits" in feats:
                emb = feats["pre_logits"]
            else:
                try:
                    z = self.model.forward_head(feats, pre_logits=True)
                except Exception:
                    z = feats if isinstance(feats, torch.Tensor) else None
                emb = z
        return emb.float().cpu().squeeze(0) if emb is not None else None


def compute_scores_for_image(fname: str, clip: CLIPWrapper, dino: DINOv2Wrapper) -> Dict[str, Any]:
    base = fname.replace(".png", "")
    api_path = os.path.join(API_IMG_DIR, fname)
    gt_path = os.path.join(GT_IMG_DIR, fname)
    input_path = os.path.join(INPUT_IMG_DIR, fname)

    img_api = load_image(api_path)
    img_ref = load_image(gt_path) if os.path.exists(gt_path) else (load_image(input_path) if os.path.exists(input_path) else None)
    ref_kind = "gt_img" if os.path.exists(gt_path) else ("input_img" if os.path.exists(input_path) else "")

    instruction = read_instruction(base)
    task = extract_task_from_filename(fname)

    out: Dict[str, Any] = {
        "image_id": base,
        "dataset": "HumanEdit",
        "task": task,
        "paths": {
            "api_img": os.path.relpath(api_path, PROJECT_ROOT) if os.path.exists(api_path) else "",
            "ref_img": os.path.relpath(gt_path if os.path.exists(gt_path) else (input_path if os.path.exists(input_path) else ""), PROJECT_ROOT) if ref_kind else "",
        },
        "reference_kind": ref_kind,
        "scores": {},
        "model_meta": {},
    }

    # CLIP
    if clip.loaded and img_api is not None:
        img_api_emb = clip.encode_image(img_api)
        if img_api_emb is not None:
            out["model_meta"]["clip"] = clip.kind
            if img_ref is not None:
                ref_emb = clip.encode_image(img_ref)
                if ref_emb is not None:
                    out["scores"]["clip_image_cosine"] = cosine_sim(img_api_emb, ref_emb)
            if instruction:
                txt_emb = clip.encode_text(instruction)
                if txt_emb is not None:
                    out["scores"]["clip_text_cosine"] = cosine_sim(img_api_emb, txt_emb)
    else:
        out["model_meta"]["clip"] = "unavailable"

    # DINO
    if dino.loaded and img_api is not None and img_ref is not None:
        out["model_meta"]["dino"] = dino.kind
        api_emb = dino.encode_image(img_api)
        ref_emb = dino.encode_image(img_ref)
        if api_emb is not None and ref_emb is not None:
            out["scores"]["dino_image_cosine"] = cosine_sim(api_emb, ref_emb)
    else:
        out["model_meta"]["dino"] = out["model_meta"].get("dino", "unavailable")

    return out


def summarize_by_task(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    agg = defaultdict(list)
    for r in records:
        agg[r["task"]].append(r)

    summary: List[Dict[str, Any]] = []
    for task, recs in sorted(agg.items()):
        all_keys = set()
        for r in recs:
            all_keys |= set(r["scores"].keys())
        stats = {}
        for k in sorted(all_keys):
            vals = [r["scores"][k] for r in recs if k in r["scores"]]
            if not vals:
                continue
            mean = float(sum(vals) / len(vals))
            var = float(sum((v-mean)**2 for v in vals) / max(1, (len(vals)-1)))
            sd = math.sqrt(var) if len(vals) > 1 else 0.0
            stats[k] = {"mean": mean, "std": sd, "n": len(vals)}
        summary.append({
            "task": task,
            "n_samples": len(recs),
            "metrics": stats,
        })
    return summary


def main():
    api_images = list_api_images()
    if not api_images:
        print("No images found under:", API_IMG_DIR)
        print("Please populate the HumanEdit/api_img (and optionally gt_img, input_img, instructions) folders.")
        return

    clip = CLIPWrapper()
    dino = DINOv2Wrapper()

    records: List[Dict[str, Any]] = []
    for i, fname in enumerate(api_images, 1):
        rec = compute_scores_for_image(fname, clip, dino)
        records.append(rec)
        if i % 25 == 0 or i == len(api_images):
            print(f"Processed {i}/{len(api_images)} images...")

    summary = summarize_by_task(records)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(records)} records + summary to:\n  {OUTPUT_JSONL}")
    print("Fields: scores.clip_image_cosine, scores.clip_text_cosine, scores.dino_image_cosine (when available).")

if __name__ == "__main__":
    main()
