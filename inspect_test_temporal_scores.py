import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import VideoFeatureDataset
from src.model import MLPAnomalyScorer

def inspect_temporal_scores():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = os.path.join(config.WORKSPACE_DIR, "best_mil_model.pt")
    
    model = MLPAnomalyScorer(in_channels=960, hidden_dim1=256, hidden_dim2=64, dropout=0.4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    test_dataset = VideoFeatureDataset(split="test")
    
    normal_items = []
    anomaly_items = []
    
    for idx, item in enumerate(test_dataset.items):
        v_id = item["video_id"]
        cat = item["category"]
        label = int(item["label"])
        pt_path = os.path.join(config.CACHE_DIR, f"{v_id}.pt")
        feats = torch.load(pt_path)
        T = feats.shape[0]
        M = 1 if T < 16 else (T - 16) // 8 + 1
        
        info = {
            "idx": idx,
            "video_id": v_id,
            "category": cat,
            "label": label,
            "T": T,
            "M": M,
            "duration_sec": T / 5.0
        }
        
        if label == 0:
            normal_items.append(info)
        else:
            anomaly_items.append(info)
            
    # Sort normal items by duration to select across different lengths (short to long)
    normal_items.sort(key=lambda x: x["duration_sec"])
    n_total = len(normal_items)
    selected_normals = [
        normal_items[0],                    # Shortest
        normal_items[n_total // 4],         # Medium-short
        normal_items[n_total // 2],         # Median
        normal_items[(3 * n_total) // 4],   # Medium-long
        normal_items[-1]                    # Longest
    ]
    
    # Select 5 anomaly items covering categories RoadAccidents, Robbery, Shooting, including a long video
    road_acc = [x for x in anomaly_items if x["category"] == "RoadAccidents"]
    robbery = [x for x in anomaly_items if x["category"] == "Robbery"]
    shooting = [x for x in anomaly_items if x["category"] == "Shooting"]
    
    road_acc.sort(key=lambda x: x["duration_sec"])
    robbery.sort(key=lambda x: x["duration_sec"])
    shooting.sort(key=lambda x: x["duration_sec"])
    
    selected_anomalies = [
        road_acc[0],        # RoadAccidents (Short/Medium)
        road_acc[-1],       # RoadAccidents (Longest)
        robbery[0],         # Robbery (Short/Medium)
        robbery[-1],        # Robbery (Longest)
        shooting[len(shooting) // 2]  # Shooting (Median)
    ]
    
    print("=========================================================================")
    print("        PHASE 6J: QUALITATIVE TEMPORAL WINDOW SCORE INSPECTION           ")
    print("=========================================================================\n")
    
    def analyze_video(item_info):
        idx = item_info["idx"]
        sample = test_dataset[idx]
        windows = sample["windows"].to(device)
        
        with torch.no_grad():
            logits = model(windows)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        M = probs.shape[0]
        max_score = float(np.max(probs))
        
        # Get top-5 highest scoring window indices
        top5_indices = np.argsort(probs)[::-1][:min(5, M)]
        
        top5_details = []
        for w_idx in top5_indices:
            score = probs[w_idx]
            # At 5 FPS, W=16, S=8: start frame = 8*w_idx, end frame = 8*w_idx + 15
            start_sec = (8 * w_idx) / 5.0
            end_sec = (8 * w_idx + 15) / 5.0
            center_sec = (8 * w_idx + 7.5) / 5.0
            top5_details.append({
                "w_idx": int(w_idx),
                "score": float(score),
                "timestamp_range": f"{start_sec:.1f}s - {end_sec:.1f}s",
                "center_sec": center_sec
            })
            
        # Check temporal concentration (max distance between top-5 window indices)
        if len(top5_indices) > 1:
            idx_range = np.max(top5_indices) - np.min(top5_indices)
            sec_range = idx_range * 1.6  # 8 frames at 5 FPS = 1.6 seconds per window step
            is_concentrated = sec_range <= 15.0  # Within 15 seconds
        else:
            idx_range = 0
            sec_range = 0.0
            is_concentrated = True
            
        return {
            "video_id": item_info["video_id"],
            "category": item_info["category"],
            "duration_sec": item_info["duration_sec"],
            "T": item_info["T"],
            "M": M,
            "video_score": max_score,
            "top5_details": top5_details,
            "index_span": int(idx_range),
            "time_span_sec": float(sec_range),
            "is_concentrated": is_concentrated
        }

    print("--- 1. NORMAL VIDEOS QUALITATIVE INSPECTION ---")
    for norm_info in selected_normals:
        res = analyze_video(norm_info)
        print(f"\nVideo ID:       {res['video_id']}")
        print(f"Category:       {res['category']}")
        print(f"Duration:       {res['duration_sec']:.1f}s ({res['T']} frames at 5 FPS)")
        print(f"Total Windows:  {res['M']}")
        print(f"Video Score:    {res['video_score']:.4f}")
        print("Top 5 Highest Scoring Windows:")
        for t in res['top5_details']:
            print(f"  - Window {t['w_idx']:3d} | Timestamp: {t['timestamp_range']:<15} | Score: {t['score']:.4f}")
            
    print("\n\n--- 2. ANOMALY VIDEOS QUALITATIVE INSPECTION ---")
    for anom_info in selected_anomalies:
        res = analyze_video(anom_info)
        print(f"\nVideo ID:       {res['video_id']}")
        print(f"Category:       {res['category']}")
        print(f"Duration:       {res['duration_sec']:.1f}s ({res['T']} frames at 5 FPS)")
        print(f"Total Windows:  {res['M']}")
        print(f"Video Score:    {res['video_score']:.4f}")
        print(f"Temporal Span:  Top-5 windows span {res['index_span']} window steps ({res['time_span_sec']:.1f} seconds)")
        print(f"Pattern:        {'TEMPORALLY CONCENTRATED (Clustered Anomaly Event)' if res['is_concentrated'] else 'SCATTERED / DISSIPATED'}")
        print("Top 5 Highest Scoring Windows:")
        for t in res['top5_details']:
            print(f"  - Window {t['w_idx']:3d} | Timestamp: {t['timestamp_range']:<15} | Score: {t['score']:.4f}")

if __name__ == "__main__":
    inspect_temporal_scores()
