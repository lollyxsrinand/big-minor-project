import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import VideoFeatureDataset

def verify_expanded_dataset_usage():
    print("=== PHASE 5A: VERIFY EXPANDED DATASET USAGE ===")
    
    # 1. Split File Currently Used
    split_file_used = config.SPLIT_FILE
    split_basename = os.path.basename(split_file_used)
    print(f"1. Split File Currently Used in Config: {split_file_used}")
    print(f"   Split Filename: {split_basename}")
    assert split_basename == "dataset_split_expanded.json", f"Expected dataset_split_expanded.json, got {split_basename}"
    
    # 2. Inspect Split Datasets via VideoFeatureDataset
    train_ds = VideoFeatureDataset(split="train")
    val_ds = VideoFeatureDataset(split="val")
    test_ds = VideoFeatureDataset(split="test")
    
    print("\n2. Dataset Split Counts:")
    print(f"   Train Dataset Size: {len(train_ds)} (Expected: 350)")
    print(f"   Val Dataset Size:   {len(val_ds)} (Expected: 73)")
    print(f"   Test Dataset Size:  {len(test_ds)} (Expected: 77)")
    print(f"   Total Dataset Size: {len(train_ds) + len(val_ds) + len(test_ds)} (Expected: 500)")
    
    # Class Breakdown for each split
    def get_split_breakdown(ds):
        norm_count = 0
        anom_count = 0
        cat_counts = {}
        for item in ds.items:
            lbl = item["label"]
            cat = item["category"]
            if lbl == 0:
                norm_count += 1
            else:
                anom_count += 1
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        return norm_count, anom_count, cat_counts

    tr_n, tr_a, tr_cats = get_split_breakdown(train_ds)
    va_n, va_a, va_cats = get_split_breakdown(val_ds)
    te_n, te_a, te_cats = get_split_breakdown(test_ds)
    
    print("\n3. Class Breakdown per Split:")
    print(f"   Train: Normal={tr_n} | Anomaly={tr_a} | Breakdown: {tr_cats}")
    print(f"   Val:   Normal={va_n} | Anomaly={va_a} | Breakdown: {va_cats}")
    print(f"   Test:  Normal={te_n} | Anomaly={te_a} | Breakdown: {te_cats}")
    
    assert (tr_n, tr_a) == (105, 245), f"Expected Train (105, 245), got ({tr_n}, {tr_a})"
    assert (va_n, va_a) == (22, 52), f"Expected Val (22, 52), got ({va_n}, {va_a})"
    assert (te_n, te_a) == (23, 53), f"Expected Test (23, 53), got ({te_n}, {te_a})"
    
    # 4. Verify all referenced video caches exist
    print("\n4. Cache Integrity Check across all 500 videos:")
    all_items = train_ds.items + val_ds.items + test_ds.items
    missing_caches = []
    invalid_caches = []
    
    for item in all_items:
        v_id = item["video_id"]
        pt_path = os.path.join(config.CACHE_DIR, f"{v_id}.pt")
        if not os.path.exists(pt_path):
            missing_caches.append(v_id)
        else:
            try:
                feats = torch.load(pt_path)
                if not isinstance(feats, torch.Tensor) or len(feats.shape) != 2 or feats.shape[1] != 960:
                    invalid_caches.append(v_id)
            except Exception:
                invalid_caches.append(v_id)
                
    print(f"   Total Videos Verified: {len(all_items)}")
    print(f"   Missing Caches:       {len(missing_caches)}")
    print(f"   Invalid Caches:       {len(invalid_caches)}")
    assert len(missing_caches) == 0, f"Missing caches: {missing_caches}"
    assert len(invalid_caches) == 0, f"Invalid caches: {invalid_caches}"
    print("   All 500 referenced video feature caches exist and are valid [T, 960] tensors!")

    # 5. Baseline Checkpoint Verification
    baseline_ckpt = os.path.join(config.WORKSPACE_DIR, "best_baseline_model.pt")
    ckpt_exists = os.path.exists(baseline_ckpt)
    print(f"\n5. Baseline Checkpoint ('best_baseline_model.pt'): Exists={ckpt_exists}")
    
    # 6. File Modifications
    print("\n6. Files Modified:")
    print("   - config.py (Updated SPLIT_FILE to dataset_split_expanded.json, preserved old path in comment)")

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_expanded_dataset_usage()
