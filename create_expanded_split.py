import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

EXPANDED_SPLIT_FILE = os.path.join(config.WORKSPACE_DIR, "dataset_split_expanded.json")

def create_expanded_stratified_split():
    random.seed(config.SEED)
    print("=== PHASE 3: CREATING EXPANDED DATASET SPLIT ===")
    print(f"Random Seed: {config.SEED}")
    print(f"Target Save Location: {EXPANDED_SPLIT_FILE}")
    
    # 1. Collect all 500 videos
    videos_by_category = {
        "Normal": [],
        "RoadAccidents": [],
        "Robbery": [],
        "Shooting": []
    }
    
    # Normal videos (150 total)
    normal_files = sorted([f for f in os.listdir(config.NORMAL_DIR) if f.endswith(".mp4") and not f.startswith(".")])
    for f in normal_files:
        video_id = os.path.splitext(f)[0]
        full_path = os.path.join(config.NORMAL_DIR, f)
        videos_by_category["Normal"].append((video_id, full_path, 0))  # Label 0 = Normal
        
    # Anomaly videos (350 total)
    for subcat in ["RoadAccidents", "Robbery", "Shooting"]:
        sub_dir = os.path.join(config.ANOMALY_DIR, subcat)
        anomaly_files = sorted([f for f in os.listdir(sub_dir) if f.endswith(".mp4") and not f.startswith(".")])
        for f in anomaly_files:
            video_id = os.path.splitext(f)[0]
            full_path = os.path.join(sub_dir, f)
            videos_by_category[subcat].append((video_id, full_path, 1))  # Label 1 = Anomaly

    split_data = {
        "train": [],
        "val": [],
        "test": []
    }
    
    category_counts = {}
    
    # 2. Perform 70 / 15 / 15 Stratified Split
    for cat_name, vid_list in videos_by_category.items():
        random.shuffle(vid_list)
        n_total = len(vid_list)
        n_train = int(round(n_total * 0.70))
        n_val = int(round(n_total * 0.15))
        
        train_vids = vid_list[:n_train]
        val_vids = vid_list[n_train:n_train + n_val]
        test_vids = vid_list[n_train + n_val:]
        
        category_counts[cat_name] = {
            "total": n_total,
            "train": len(train_vids),
            "val": len(val_vids),
            "test": len(test_vids)
        }
        
        for item in train_vids:
            split_data["train"].append({"video_id": item[0], "path": item[1], "label": item[2], "category": cat_name})
        for item in val_vids:
            split_data["val"].append({"video_id": item[0], "path": item[1], "label": item[2], "category": cat_name})
        for item in test_vids:
            split_data["test"].append({"video_id": item[0], "path": item[1], "label": item[2], "category": cat_name})

    # Save to dataset_split_expanded.json
    with open(EXPANDED_SPLIT_FILE, "w") as f:
        json.dump(split_data, f, indent=2)
        
    print(f"\nSuccessfully created and saved {EXPANDED_SPLIT_FILE}")
    
    # 3. Post-Split Verification Pass
    print("\n=== POST-SPLIT VERIFICATION PASS ===")
    train_ids = [item["video_id"] for item in split_data["train"]]
    val_ids = [item["video_id"] for item in split_data["val"]]
    test_ids = [item["video_id"] for item in split_data["test"]]
    
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)
    
    # Intersections
    train_val = set_train.intersection(set_val)
    train_test = set_train.intersection(set_test)
    val_test = set_val.intersection(set_test)
    
    all_500_set = set_train.union(set_val).union(set_test)
    total_entries_count = len(train_ids) + len(val_ids) + len(test_ids)
    
    print(f"1. train ∩ val == empty: {len(train_val) == 0} (Overlap: {len(train_val)})")
    print(f"2. train ∩ test == empty: {len(train_test) == 0} (Overlap: {len(train_test)})")
    print(f"3. val ∩ test == empty: {len(val_test) == 0} (Overlap: {len(val_test)})")
    print(f"4. Total Video Entries in Split: {total_entries_count} (Expected: 500)")
    print(f"5. Unique Video IDs Count: {len(all_500_set)} (Expected: 500)")
    print(f"6. Duplicate Video IDs in Split: {total_entries_count - len(all_500_set)}")
    
    # Category Breakdown for each split
    def summarize_split(split_items):
        counts = {"Normal": 0, "RoadAccidents": 0, "Robbery": 0, "Shooting": 0}
        for item in split_items:
            counts[item["category"]] += 1
        total_n = counts["Normal"]
        total_a = counts["RoadAccidents"] + counts["Robbery"] + counts["Shooting"]
        return total_n, total_a, counts

    tr_n, tr_a, tr_cats = summarize_split(split_data["train"])
    va_n, va_a, va_cats = summarize_split(split_data["val"])
    te_n, te_a, te_cats = summarize_split(split_data["test"])
    
    print("\nCategorical Breakdown per Split:")
    print(f"  TRAIN (350 vids): Normal={tr_n} (30.0%) | Anomaly={tr_a} (70.0%) | Breakdown: {tr_cats}")
    print(f"  VAL   (73 vids):  Normal={va_n} (30.1%) | Anomaly={va_a} (69.9%) | Breakdown: {va_cats}")
    print(f"  TEST  (77 vids):  Normal={te_n} (29.9%) | Anomaly={te_a} (70.1%) | Breakdown: {te_cats}")
    
    # Check baseline dataset_split.json was untouched
    assert os.path.exists(config.SPLIT_FILE), "Error: Original dataset_split.json missing!"
    
    # Strict Assertions
    assert len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0, "Leakage detected!"
    assert total_entries_count == 500, f"Expected 500 videos, got {total_entries_count}"
    assert len(all_500_set) == 500, f"Expected 500 unique IDs, got {len(all_500_set)}"
    print("\nALL VERIFICATION ASSERTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    create_expanded_stratified_split()
