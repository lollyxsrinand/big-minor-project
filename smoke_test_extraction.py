import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.extract_features import get_feature_extractor, get_transforms, extract_features_for_video

def run_smoke_test():
    print("=== PHASE 4B: EXTRACTION SMOKE TEST ===")
    
    # 1. Select a short new normal video: Normal_Videos_881_x264 (7.5 sec, ~37 sampled frames)
    test_video_id = "Normal_Videos_881_x264"
    video_path = os.path.join(config.NORMAL_DIR, f"{test_video_id}.mp4")
    output_pt_path = os.path.join(config.CACHE_DIR, f"{test_video_id}.pt")
    
    print(f"Smoke Test Target Video: {test_video_id}")
    print(f"Video File Path: {video_path}")
    print(f"Output Cache Path: {output_pt_path}")
    
    assert os.path.exists(video_path), f"Video file not found at {video_path}"
    
    # Track mtime of pre-existing baseline caches to verify non-modification
    baseline_cache_sample = os.path.join(config.CACHE_DIR, "Normal_Videos_704_x264.pt")
    mtime_before = os.path.getmtime(baseline_cache_sample)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = get_feature_extractor(device)
    transform = get_transforms()
    
    # Run robust extraction for single video
    res = extract_features_for_video(video_path, output_pt_path, model, device, transform, chunk_size=256, batch_size=32)
    print(f"Extraction result dict: {res}")
    
    # 2. Verify resulting .pt file loads successfully
    assert os.path.exists(output_pt_path), f"Cache file not found at {output_pt_path}"
    loaded_feats = torch.load(output_pt_path)
    
    print(f"Loaded Tensor Type:  {type(loaded_feats)}")
    print(f"Loaded Tensor Shape: {loaded_feats.shape}")
    print(f"Feature Dimension:   {loaded_feats.shape[1]} (Expected: 960)")
    
    assert isinstance(loaded_feats, torch.Tensor), "Loaded feature is not a Tensor!"
    assert len(loaded_feats.shape) == 2, f"Expected 2D tensor, got shape {loaded_feats.shape}"
    assert loaded_feats.shape[1] == 960, f"Expected feature dim 960, got {loaded_feats.shape[1]}"
    
    # 3. Verify existing pre-existing cache was NOT modified
    mtime_after = os.path.getmtime(baseline_cache_sample)
    assert mtime_before == mtime_after, "Pre-existing baseline cache was unexpectedly modified!"
    print(f"Verified pre-existing baseline cache ({os.path.basename(baseline_cache_sample)}) was UNTOUCHED!")
    
    print("\nALL SMOKE TEST VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_smoke_test()
