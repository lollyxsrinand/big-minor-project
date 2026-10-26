import os
import sys
import json
import time
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.extract_features import get_feature_extractor, get_transforms, extract_features_for_video

EXPANDED_SPLIT_FILE = os.path.join(config.WORKSPACE_DIR, "dataset_split_expanded.json")

def run_phase_4c_extraction():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=== PHASE 4C: FULL NEW-NORMAL FEATURE EXTRACTION ===", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Sampling Rate: {config.SAMPLE_FPS} FPS", flush=True)
    print(f"Target Feature Dim: {config.FEATURE_DIM}", flush=True)
    print(f"Split File: {EXPANDED_SPLIT_FILE}", flush=True)
    
    with open(EXPANDED_SPLIT_FILE, "r") as f:
        splits = json.load(f)
        
    all_500_videos = splits["train"] + splits["val"] + splits["test"]
    print(f"Total Dataset Videos: {len(all_500_videos)}", flush=True)
    
    model = get_feature_extractor(device)
    transform = get_transforms()
    print("MobileNetV3-Large loaded on device. Starting extraction loop...\n", flush=True)
    
    extracted_count = 0
    skipped_count = 0
    failures = []
    
    start_time = time.time()
    
    for i, item in enumerate(all_500_videos, 1):
        v_id = item["video_id"]
        v_path = item["path"]
        pt_path = os.path.join(config.CACHE_DIR, f"{v_id}.pt")
        
        # Skip if valid cache already exists (> 1KB)
        if os.path.exists(pt_path) and os.path.getsize(pt_path) > 1000:
            skipped_count += 1
            continue
            
        # Extract features for missing video using chunked robust pipeline
        print(f"[{i:03d}/500] [STARTING EXTRACTION] {v_id} ({item['category']})...", flush=True)
        v_start = time.time()
        try:
            res = extract_features_for_video(v_path, pt_path, model, device, transform, chunk_size=256, batch_size=32)
            v_elapsed = time.time() - v_start
            extracted_count += 1
            print(f"[{i:03d}/500] [FINISHED] {v_id} -> Shape: {res['shape']} ({res['file_size_mb']} MB) in {v_elapsed:.2f}s\n", flush=True)
        except Exception as e:
            failures.append((v_id, v_path, str(e)))
            print(f"[{i:03d}/500] [FAILED] {v_id}: {e}\n", flush=True)
            
    total_elapsed = time.time() - start_time
    print(f"\nExtraction loop finished in {total_elapsed:.2f} seconds ({total_elapsed/60.0:.2f} min).", flush=True)
    print(f"Newly Extracted:      {extracted_count}", flush=True)
    print(f"Pre-existing Skipped: {skipped_count}", flush=True)
    print(f"Extraction Failures:  {len(failures)}", flush=True)

    # Full Verification Pass
    print("\n=== POST-EXTRACTION FULL 500-VIDEO VERIFICATION PASS ===", flush=True)
    cached_files = [f for f in os.listdir(config.CACHE_DIR) if f.endswith('.pt')]
    tmp_files = [f for f in os.listdir(config.CACHE_DIR) if f.endswith('.pt.tmp')]
    
    missing_caches = []
    invalid_caches = []
    dim_distribution = {}
    total_cache_bytes = 0
    
    for item in all_500_videos:
        v_id = item["video_id"]
        pt_path = os.path.join(config.CACHE_DIR, f"{v_id}.pt")
        
        if not os.path.exists(pt_path):
            missing_caches.append(v_id)
        else:
            try:
                sz = os.path.getsize(pt_path)
                total_cache_bytes += sz
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
                
    total_mb = round(total_cache_bytes / (1024 * 1024), 2)
    total_gb = round(total_cache_bytes / (1024 ** 3), 3)
    
    print(f"1. Total Videos Expected:         500", flush=True)
    print(f"2. Total .pt Cache Files Found:   {len(cached_files)}", flush=True)
    print(f"3. Remaining .pt.tmp Files:      {len(tmp_files)} (Expected: 0)", flush=True)
    print(f"4. Extraction Failures:           {len(failures)}", flush=True)
    print(f"5. Missing Caches:                {len(missing_caches)}", flush=True)
    print(f"6. Invalid Caches:                {len(invalid_caches)}", flush=True)
    print(f"7. Feature Dimension Distribution: {dim_distribution}", flush=True)
    print(f"8. Total Cache Disk Usage:        {total_mb} MB ({total_gb} GB)", flush=True)
    
    assert len(cached_files) == 500, f"Expected 500 cached files, found {len(cached_files)}"
    assert len(tmp_files) == 0, f"Temporary files found: {tmp_files}"
    assert len(failures) == 0, "Failures encountered during extraction!"
    assert len(missing_caches) == 0, "Missing caches found!"
    assert len(invalid_caches) == 0, "Invalid caches found!"
    assert dim_distribution == {960: 500}, f"Unexpected dimension distribution {dim_distribution}"
    print("\nALL 500 DATASET VIDEOS HAVE VALID FEATURE CACHES!", flush=True)

if __name__ == "__main__":
    run_phase_4c_extraction()
