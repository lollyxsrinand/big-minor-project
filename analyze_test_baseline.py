import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import VideoFeatureDataset, video_collate_fn
from src.model import VideoAnomalyDetector

CHECKPOINT_PATH = os.path.join(config.WORKSPACE_DIR, "best_baseline_model.pt")

def run_test_diagnostic():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Test Dataset & Loader
    test_dataset = VideoFeatureDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=video_collate_fn)
    
    # 2. Load Model Checkpoint
    model = VideoAnomalyDetector(
        in_channels=config.FEATURE_DIM,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.3,
        pooling_method=config.POOLING_METHOD,
        topk_percentage=config.TOPK_PERCENTAGE,
        topk_count=config.TOPK_COUNT
    ).to(device)
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    test_records = []
    
    with torch.no_grad():
        for batch in test_loader:
            windows_list = batch["windows_list"]
            labels = batch["labels"]
            categories = batch["categories"]
            v_ids = batch["video_ids"]
            
            for i, windows in enumerate(windows_list):
                out = model(windows.to(device))
                score = out["video_score"].item()
                lbl = int(labels[i].item())
                cat = categories[i]
                vid = v_ids[i]
                
                test_records.append({
                    "video_id": vid,
                    "category": cat,
                    "true_label": lbl,
                    "predicted_score": score
                })
                
    # Sort all test videos by score descending
    sorted_all = sorted(test_records, key=lambda x: x["predicted_score"], reverse=True)
    
    # Separate normal and anomaly records
    normal_records = sorted([r for r in test_records if r["true_label"] == 0], key=lambda x: x["predicted_score"], reverse=True)
    anomaly_records = sorted([r for r in test_records if r["true_label"] == 1], key=lambda x: x["predicted_score"], reverse=True)
    
    normal_scores = np.array([r["predicted_score"] for r in normal_records])
    anomaly_scores = np.array([r["predicted_score"] for r in anomaly_records])
    
    y_true = np.array([r["true_label"] for r in test_records])
    y_scores = np.array([r["predicted_score"] for r in test_records])
    
    roc_auc = roc_auc_score(y_true, y_scores)
    prec_array, rec_array, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec_array, prec_array)
    
    print("=========================================================================")
    print("                 BASELINE MODEL TEST DIAGNOSTIC ANALYSIS                 ")
    print("=========================================================================")
    
    print("\n--- 1. ALL TEST VIDEOS SORTED BY PREDICTED ANOMALY SCORE (DESCENDING) ---")
    print(f"{'Rank':<5} | {'Video ID':<30} | {'Category':<15} | {'True Label':<10} | {'Predicted Score':<15}")
    print("-" * 85)
    for rank, r in enumerate(sorted_all, 1):
        print(f"{rank:<5} | {r['video_id']:<30} | {r['category']:<15} | {r['true_label']:<10} | {r['predicted_score']:.6f}")
        
    print("\n--- 2. SEPARATE LISTING: NORMAL VIDEOS (6 TOTAL) ---")
    print(f"{'Video ID':<30} | {'Category':<15} | {'True Label':<10} | {'Predicted Score':<15}")
    print("-" * 80)
    for r in normal_records:
        print(f"{r['video_id']:<30} | {r['category']:<15} | {r['true_label']:<10} | {r['predicted_score']:.6f}")
        
    print("\n--- 3. SEPARATE LISTING: ANOMALY VIDEOS (53 TOTAL) ---")
    print(f"{'Video ID':<30} | {'Category':<15} | {'True Label':<10} | {'Predicted Score':<15}")
    print("-" * 80)
    for r in anomaly_records:
        print(f"{r['video_id']:<30} | {r['category']:<15} | {r['true_label']:<10} | {r['predicted_score']:.6f}")
        
    print("\n--- 4. SCORE STATISTICS: NORMAL VS ANOMALY ---")
    def calc_stats(arr):
        return {
            "min": np.min(arr),
            "max": np.max(arr),
            "mean": np.mean(arr),
            "median": np.median(arr),
            "std": np.std(arr)
        }
        
    n_stats = calc_stats(normal_scores)
    a_stats = calc_stats(anomaly_scores)
    
    print(f"{'Statistic':<12} | {'Normal Videos (N=6)':<25} | {'Anomaly Videos (N=53)':<25}")
    print("-" * 68)
    for stat in ["min", "max", "mean", "median", "std"]:
        print(f"{stat.capitalize():<12} | {n_stats[stat]:<25.6f} | {a_stats[stat]:<25.6f}")
        
    print("\n--- 5. RAW SCORE DISCRIMINATION METRICS (NO THRESHOLDING) ---")
    print(f"ROC-AUC (Area Under ROC Curve): {roc_auc:.6f}")
    print(f"PR-AUC  (Area Under Precision-Recall Curve): {pr_auc:.6f}")

if __name__ == "__main__":
    run_test_diagnostic()
