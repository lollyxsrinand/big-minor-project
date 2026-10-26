import os
import sys
import json
import time
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_feature_extractor(device):
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    model.classifier = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)

def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_features_for_video(video_path, output_pt_path, model, device, transform, target_fps=config.SAMPLE_FPS, chunk_size=256, batch_size=32):
    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 30.0
        
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(round(native_fps / target_fps)))
    total_sampled_estimate = max(1, total_native_frames // frame_step) if total_native_frames > 0 else 0
    
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    feature_chunks = []
    frame_buffer = []
    sampled_count = 0
    frame_idx = 0
    
    def process_buffer(buffer):
        if not buffer:
            return None
        buf_tensor = torch.stack(buffer)
        feats_list = []
        with torch.no_grad():
            for b_idx in range(0, len(buf_tensor), batch_size):
                batch = buf_tensor[b_idx:b_idx+batch_size].to(device)
                feats = model(batch)  # [B, 960]
                feats_list.append(feats.cpu())
        return torch.cat(feats_list, dim=0)  # [N, 960]

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        if frame_idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            tensor_img = transform(pil_img)
            frame_buffer.append(tensor_img)
            sampled_count += 1
            
            # Process chunk if buffer reaches chunk_size (256 frames)
            if len(frame_buffer) >= chunk_size:
                chunk_feats = process_buffer(frame_buffer)
                feature_chunks.append(chunk_feats)
                frame_buffer.clear()
                
                pct = (sampled_count / total_sampled_estimate * 100.0) if total_sampled_estimate > 0 else 0.0
                print(f"  [EXTRACTING] {video_id}: {sampled_count:,} / {total_sampled_estimate:,} frames ({pct:.1f}%)...", flush=True)
                
        frame_idx += 1
        
    cap.release()
    
    # Process remaining frames in buffer
    if frame_buffer:
        chunk_feats = process_buffer(frame_buffer)
        feature_chunks.append(chunk_feats)
        frame_buffer.clear()
        
    if not feature_chunks:
        raise ValueError(f"No frames extracted from {video_path}")
        
    features = torch.cat(feature_chunks, dim=0)  # [T, 960]
    
    # Atomic save: write to .pt.tmp first, then rename to .pt
    tmp_path = output_pt_path + ".tmp"
    torch.save(features, tmp_path)
    os.replace(tmp_path, output_pt_path)
    
    file_size_mb = round(os.path.getsize(output_pt_path) / (1024 * 1024), 3)
    
    return {
        "video_path": video_path,
        "saved_path": output_pt_path,
        "native_fps": native_fps,
        "sampled_frames": features.shape[0],
        "feature_dim": features.shape[1],
        "shape": list(features.shape),
        "file_size_mb": file_size_mb
    }

def run_full_feature_extraction():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== FULL FEATURE EXTRACTION ===")
    print(f"Device: {device}")
    print(f"Target Sampling FPS: {config.SAMPLE_FPS}")
    print(f"Expected Feature Dim: {config.FEATURE_DIM}")
    
    # Load dataset_split.json
    with open(config.SPLIT_FILE, "r") as f:
        splits = json.load(f)
        
    all_videos = splits["train"] + splits["val"] + splits["test"]
    total_videos = len(all_videos)
    print(f"Total Videos in dataset_split.json: {total_videos}")
    
    model = get_feature_extractor(device)
    transform = get_transforms()
    
    start_time = time.time()
    
    extracted_count = 0
    skipped_count = 0
    failures = []
    
    for i, item in enumerate(all_videos, 1):
        v_id = item["video_id"]
        v_path = item["path"]
        pt_path = os.path.join(config.CACHE_DIR, f"{v_id}.pt")
        
        # Requirement 8: If valid cache exists, skip
        if os.path.exists(pt_path):
            try:
                existing_feats = torch.load(pt_path)
                if isinstance(existing_feats, torch.Tensor) and len(existing_feats.shape) == 2 and existing_feats.shape[1] == config.FEATURE_DIM:
                    skipped_count += 1
                    print(f"[{i}/{total_videos}] [SKIP] {v_id} -> {existing_feats.shape}")
                    continue
            except Exception:
                pass  # Corrupt cache, will re-extract
                
        # Extract features
        try:
            res = extract_features_for_video(v_path, pt_path, model, device, transform)
            extracted_count += 1
            print(f"[{i}/{total_videos}] [EXTRACTED] {v_id} -> Shape: {res['shape']} ({res['file_size_mb']} MB)")
        except Exception as e:
            failures.append((v_id, v_path, str(e)))
            print(f"[{i}/{total_videos}] [FAILED] {v_id}: {e}")
            
    elapsed_sec = time.time() - start_time
    print(f"\nExtraction completed in {elapsed_sec:.2f} seconds.")
    print(f"Extracted: {extracted_count} | Skipped: {skipped_count} | Failures: {len(failures)}")
    
    # Verification Pass
    print("\n=== RUNNING VERIFICATION PASS ===")
    cached_files = [f for f in os.listdir(config.CACHE_DIR) if f.endswith('.pt')]
    missing_caches = []
    invalid_caches = []
    dim_distribution = {}
    total_cache_size_bytes = 0
    
    for item in all_videos:
        v_id = item["video_id"]
        pt_path = os.path.join(config.CACHE_DIR, f"{v_id}.pt")
        if not os.path.exists(pt_path):
            missing_caches.append(v_id)
        else:
            try:
                sz = os.path.getsize(pt_path)
                total_cache_size_bytes += sz
                feats = torch.load(pt_path)
                if not isinstance(feats, torch.Tensor) or len(feats.shape) != 2:
                    invalid_caches.append((v_id, "Not 2D Tensor"))
                else:
                    dim = feats.shape[1]
                    dim_distribution[dim] = dim_distribution.get(dim, 0) + 1
                    if dim != config.FEATURE_DIM:
                        invalid_caches.append((v_id, f"Wrong dim {dim}"))
            except Exception as e:
                invalid_caches.append((v_id, str(e)))

    total_cache_size_mb = round(total_cache_size_bytes / (1024 * 1024), 2)
    total_cache_size_gb = round(total_cache_size_bytes / (1024 ** 3), 3)
    
    print(f"1. Total Videos Expected: {total_videos}")
    print(f"2. Successfully Cached: {len(cached_files)}")
    print(f"3. Extraction Failures: {len(failures)}")
    print(f"4. Missing Caches: {len(missing_caches)}")
    print(f"5. Invalid Caches: {len(invalid_caches)}")
    print(f"6. Total Cache Disk Usage: {total_cache_size_mb} MB ({total_cache_size_gb} GB)")
    print(f"7. Feature Dimension Distribution: {dim_distribution}")

if __name__ == "__main__":
    run_full_feature_extraction()
