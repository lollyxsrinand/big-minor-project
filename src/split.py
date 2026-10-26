import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def create_stratified_split():
    random.seed(config.SEED)
    
    workspace = config.WORKSPACE_DIR
    normal_dir = config.NORMAL_DIR
    anomaly_dir = config.ANOMALY_DIR
    
    # Collect all video paths with categories
    videos_by_category = {
        "Normal": [],
        "RoadAccidents": [],
        "Robbery": [],
        "Shooting": []
    }
    
    # Normal videos
    for f in sorted(os.listdir(normal_dir)):
        if f.endswith(".mp4") and not f.startswith("."):
            video_id = os.path.splitext(f)[0]
            full_path = os.path.join(normal_dir, f)
            videos_by_category["Normal"].append((video_id, full_path, 0)) # Label 0 = Normal
            
    # Anomaly videos
    for subcat in ["RoadAccidents", "Robbery", "Shooting"]:
        sub_dir = os.path.join(anomaly_dir, subcat)
        for f in sorted(os.listdir(sub_dir)):
            if f.endswith(".mp4") and not f.startswith("."):
                video_id = os.path.splitext(f)[0]
                full_path = os.path.join(sub_dir, f)
                videos_by_category[subcat].append((video_id, full_path, 1)) # Label 1 = Anomaly

    split_data = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # Ratios: 0.70 train, 0.15 val, 0.15 test
    for cat_name, vid_list in videos_by_category.items():
        random.shuffle(vid_list)
        n_total = len(vid_list)
        n_train = int(round(n_total * 0.70))
        n_val = int(round(n_total * 0.15))
        
        train_vids = vid_list[:n_train]
        val_vids = vid_list[n_train:n_train + n_val]
        test_vids = vid_list[n_train + n_val:]
        
        for item in train_vids:
            split_data["train"].append({"video_id": item[0], "path": item[1], "label": item[2], "category": cat_name})
        for item in val_vids:
            split_data["val"].append({"video_id": item[0], "path": item[1], "label": item[2], "category": cat_name})
        for item in test_vids:
            split_data["test"].append({"video_id": item[0], "path": item[1], "label": item[2], "category": cat_name})

    with open(config.SPLIT_FILE, "w") as f:
        json.dump(split_data, f, indent=2)
        
    print(f"Created dataset split at {config.SPLIT_FILE}")
    print(f"Train: {len(split_data['train'])} videos")
    print(f"Val: {len(split_data['val'])} videos")
    print(f"Test: {len(split_data['test'])} videos")
    
    return split_data

if __name__ == "__main__":
    create_stratified_split()
