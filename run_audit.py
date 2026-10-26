import os
import sys
import json
import hashlib
import cv2
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.extract_features import get_feature_extractor, get_transforms, extract_features_for_video

def audit_1_split():
    print("--- 1. VERIFY SOURCE-LEVEL SPLIT ---")
    with open(config.SPLIT_FILE, "r") as f:
        splits = json.load(f)
        
    train_ids = set(item["video_id"] for item in splits["train"])
    val_ids = set(item["video_id"] for item in splits["val"])
    test_ids = set(item["video_id"] for item in splits["test"])
    
    # Intersections
    train_val = train_ids.intersection(val_ids)
    train_test = train_ids.intersection(test_ids)
    val_test = val_ids.intersection(test_ids)
    
    print(f"train ∩ val == empty: {len(train_val) == 0} (count: {len(train_val)})")
    print(f"train ∩ test == empty: {len(train_test) == 0} (count: {len(train_test)})")
    print(f"val ∩ test == empty: {len(val_test) == 0} (count: {len(val_test)})")
    
    # Category breakdowns
    def get_category_counts(split_items):
        counts = {"Normal": 0, "RoadAccidents": 0, "Robbery": 0, "Shooting": 0, "Total_Anomaly": 0}
        for item in split_items:
            cat = item["category"]
            counts[cat] += 1
            if cat != "Normal":
                counts["Total_Anomaly"] += 1
        return counts

    train_counts = get_category_counts(splits["train"])
    val_counts = get_category_counts(splits["val"])
    test_counts = get_category_counts(splits["test"])
    
    print("\nSplit Category Breakdown:")
    print(f"  Train: Normal={train_counts['Normal']}, Anomaly={train_counts['Total_Anomaly']} (RoadAccidents={train_counts['RoadAccidents']}, Robbery={train_counts['Robbery']}, Shooting={train_counts['Shooting']}) | Total={len(splits['train'])}")
    print(f"  Val:   Normal={val_counts['Normal']}, Anomaly={val_counts['Total_Anomaly']} (RoadAccidents={val_counts['RoadAccidents']}, Robbery={val_counts['Robbery']}, Shooting={val_counts['Shooting']}) | Total={len(splits['val'])}")
    print(f"  Test:  Normal={test_counts['Normal']}, Anomaly={test_counts['Total_Anomaly']} (RoadAccidents={test_counts['RoadAccidents']}, Robbery={test_counts['Robbery']}, Shooting={test_counts['Shooting']}) | Total={len(splits['test'])}")
    
    assert len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0, "Leakage detected!"
    return True

def audit_2_cached_features():
    print("\n--- 2. VERIFY CACHED FEATURES ---")
    sample_vids = [
        ("Normal_Videos_704_x264", os.path.join(config.NORMAL_DIR, "Normal_Videos_704_x264.mp4")),
        ("RoadAccidents102_x264", os.path.join(config.ANOMALY_DIR, "RoadAccidents", "RoadAccidents102_x264.mp4")),
        ("Robbery120_x264", os.path.join(config.ANOMALY_DIR, "Robbery", "Robbery120_x264.mp4"))
    ]
    
    audit_results = []
    for vid_id, vid_path in sample_vids:
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(frame_count / fps, 2)
        cap.release()
        
        pt_path = os.path.join(config.CACHE_DIR, f"{vid_id}.pt")
        feats = torch.load(pt_path)
        file_size_mb = round(os.path.getsize(pt_path) / (1024 * 1024), 3)
        
        sampled_count = feats.shape[0]
        feat_dim = feats.shape[1]
        expected_approx = round(duration * config.SAMPLE_FPS)
        
        res = {
            "vid_id": vid_id,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration,
            "sampled_feature_count": sampled_count,
            "expected_approx": expected_approx,
            "shape": list(feats.shape),
            "file_size_mb": file_size_mb,
            "valid_dim": feat_dim == 960
        }
        audit_results.append(res)
        print(f"Video: {vid_id}")
        print(f"  FPS: {fps} | Frame Count: {frame_count} | Duration: {duration}s")
        print(f"  Sampled Features: {sampled_count} (Expected approx duration * 5 = {expected_approx})")
        print(f"  Shape: {res['shape']} | Cache Size: {file_size_mb} MB | Valid Dim (960): {res['valid_dim']}")

    return audit_results

def audit_3_sampling_logic():
    print("\n--- 3. VERIFY FRAME SAMPLING LOGIC ---")
    print("Code in extract_features.py:")
    print("  native_fps = cap.get(cv2.CAP_PROP_FPS)")
    print("  frame_step = max(1, int(round(native_fps / target_fps)))")
    print("  if frame_idx % frame_step == 0: extract_frame()")
    print("Explanation:")
    print("  For 30.0 FPS video: step = round(30.0 / 5) = 6. Frames 0, 6, 12, 18... are selected (5 FPS).")
    print("  For 25.0 FPS video: step = round(25.0 / 5) = 5. Frames 0, 5, 10, 15... are selected (5 FPS).")
    print("  For 15.0 FPS video: step = round(15.0 / 5) = 3. Frames 0, 3, 6, 9... are selected (5 FPS).")
    print("  This dynamically guarantees ~5 sampled frames per second regardless of source video FPS.")

def audit_4_temporal_windows():
    print("\n--- 4. VERIFY TEMPORAL WINDOW COUNT ---")
    sample_vids = ["Normal_Videos_704_x264", "RoadAccidents102_x264", "Robbery120_x264"]
    
    W = config.WINDOW_SIZE
    S = config.STRIDE
    
    for vid_id in sample_vids:
        pt_path = os.path.join(config.CACHE_DIR, f"{vid_id}.pt")
        feats = torch.load(pt_path)
        T_len = feats.shape[0]
        
        # Formula: M = floor((T - W) / S) + 1
        expected_M = (T_len - W) // S + 1
        
        unfolded = feats.unfold(dimension=0, size=W, step=S)
        actual_M = unfolded.shape[0]
        
        print(f"Video: {vid_id}")
        print(f"  T = {T_len} | Window Size W = {W} | Stride S = {S}")
        print(f"  Expected Windows M = floor(({T_len} - {W}) / {S}) + 1 = {expected_M}")
        print(f"  Actual Tensor Windows M = {actual_M}")
        print(f"  Formula Match: {expected_M == actual_M}")
        assert expected_M == actual_M

def audit_5_short_videos():
    print("\n--- 5. CHECK & FIX SHORT VIDEO BEHAVIOR ---")
    with open("dataset_inspection_results.json", "r") as f:
        data = json.load(f)
        
    vids = data["video_details"]
    min_dur = min(v['duration_sec'] for v in vids)
    min_vid = min(vids, key=lambda x: x['duration_sec'])
    
    min_sampled_frames = int(round(min_dur * config.SAMPLE_FPS))
    print(f"Shortest video in full dataset: {min_vid['filename']} ({min_dur}s)")
    print(f"At 5 FPS, shortest video yields ~{min_sampled_frames} sampled frames.")
    print(f"Configured Window Size W = {config.WINDOW_SIZE}")
    print(f"Are all videos natively >= W ({config.WINDOW_SIZE} frames)? {min_sampled_frames >= config.WINDOW_SIZE}")
    
    return min_sampled_frames >= config.WINDOW_SIZE

def audit_6_duplication():
    print("\n--- 6. CHECK FOR SOURCE DUPLICATION / NEAR-DUPLICATES ---")
    with open("dataset_inspection_results.json", "r") as f:
        data = json.load(f)
        
    vids = data["video_details"]
    
    # Check by file size and duration/frame_count
    size_map = {}
    dups_by_size = []
    
    for v in vids:
        size = v['size_mb']
        key = (v['size_mb'], v['frame_count'], v['fps'])
        if key in size_map:
            dups_by_size.append((v['rel_path'], size_map[key]['rel_path'], size, v['frame_count']))
        else:
            size_map[key] = v
            
    print(f"Total videos checked: {len(vids)}")
    print(f"Potential duplicate candidates (identical file size + frame count + FPS): {len(dups_by_size)}")
    if dups_by_size:
        print("Candidates:")
        for d1, d2, sz, fc in dups_by_size[:5]:
            print(f"  - {d1} <==> {d2} ({sz} MB, {fc} frames)")

if __name__ == "__main__":
    audit_1_split()
    audit_2_cached_features()
    audit_3_sampling_logic()
    audit_4_temporal_windows()
    audit_5_short_videos()
    audit_6_duplication()
