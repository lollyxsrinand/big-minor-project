import os
import sys
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import VideoFeatureDataset
from src.model import MLPAnomalyScorer

def evaluate_mil_test():
    print("=========================================================================")
    print("      PHASE 6I: PURE HELD-OUT TEST EVALUATION OF BEST MIL MODEL          ")
    print("=========================================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device Acceleration:   {device}")
    
    checkpoint_path = os.path.join(config.WORKSPACE_DIR, "best_mil_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    print(f"Loading Checkpoint:    {checkpoint_path}")
    model = MLPAnomalyScorer(in_channels=960, hidden_dim1=256, hidden_dim2=64, dropout=0.4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    test_dataset = VideoFeatureDataset(split="test")
    print(f"Test Split Size:       {len(test_dataset)} videos (Using ALL temporal windows per video)")
    
    labels = []
    scores = []
    categories = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            windows = sample["windows"].to(device)  # [M, 16, 960] (ALL windows)
            label = int(sample["label"].item())
            category = sample["category"]
            
            # Model outputs raw logits [M]
            logits = model(windows)
            probs = torch.sigmoid(logits)
            video_score = torch.max(probs).item()
            
            labels.append(label)
            scores.append(video_score)
            categories.append(category)
            
    labels = np.array(labels)
    scores = np.array(scores)
    categories = np.array(categories)
    
    # Threshold 0.5 predictions
    preds_05 = (scores >= 0.5).astype(int)
    
    # 1. ROC-AUC and PR-AUC
    test_roc_auc = roc_auc_score(labels, scores)
    test_pr_auc = average_precision_score(labels, scores)
    
    # 2. Threshold 0.5 Metrics
    acc_05 = accuracy_score(labels, preds_05)
    prec_05 = precision_score(labels, preds_05, zero_division=0)
    rec_05 = recall_score(labels, preds_05, zero_division=0)
    f1_05 = f1_score(labels, preds_05, zero_division=0)
    cm = confusion_matrix(labels, preds_05)  # [TN, FP / FN, TP]
    
    # 3. Class Accuracies
    normal_mask = (labels == 0)
    anomaly_mask = (labels == 1)
    
    normal_acc = accuracy_score(labels[normal_mask], preds_05[normal_mask]) if np.sum(normal_mask) > 0 else 0.0
    anomaly_acc = accuracy_score(labels[anomaly_mask], preds_05[anomaly_mask]) if np.sum(anomaly_mask) > 0 else 0.0
    
    # 4. Per-Category Breakdown
    unique_cats = sorted(list(set(categories)))
    cat_results = {}
    for cat in unique_cats:
        c_mask = (categories == cat)
        c_tot = np.sum(c_mask)
        c_corr = np.sum(preds_05[c_mask] == labels[c_mask])
        c_acc = c_corr / c_tot if c_tot > 0 else 0.0
        c_pred_norm = np.sum(preds_05[c_mask] == 0)
        c_pred_anom = np.sum(preds_05[c_mask] == 1)
        cat_results[cat] = {
            "total": c_tot,
            "correct": c_corr,
            "accuracy": c_acc,
            "pred_norm": c_pred_norm,
            "pred_anom": c_pred_anom
        }
        
    # 5. Score Distributions
    normal_scores = scores[normal_mask]
    anomaly_scores = scores[anomaly_mask]
    
    norm_dist = {
        "min": float(np.min(normal_scores)),
        "mean": float(np.mean(normal_scores)),
        "median": float(np.median(normal_scores)),
        "max": float(np.max(normal_scores))
    }
    
    anom_dist = {
        "min": float(np.min(anomaly_scores)),
        "mean": float(np.mean(anomaly_scores)),
        "median": float(np.median(anomaly_scores)),
        "max": float(np.max(anomaly_scores))
    }
    
    print("\n--- FINAL HELD-OUT TEST RESULTS ---")
    print(f"Test ROC-AUC:          {test_roc_auc:.4f}")
    print(f"Test PR-AUC:           {test_pr_auc:.4f}")
    print(f"\n--- METRICS AT THRESHOLD 0.5 ---")
    print(f"Accuracy:              {acc_05:.4f} ({acc_05*100:.2f}%)")
    print(f"Precision:             {prec_05:.4f}")
    print(f"Recall:                {rec_05:.4f}")
    print(f"F1-Score:              {f1_05:.4f}")
    print(f"\nConfusion Matrix [TN, FP / FN, TP]:")
    print(f"  [[{cm[0,0]:2d} {cm[0,1]:2d}]")
    print(f"   [{cm[1,0]:2d} {cm[1,1]:2d}]]")
    
    print(f"\nOverall Normal Accuracy:  {normal_acc*100:.2f}% ({cm[0,0]}/{np.sum(normal_mask)})")
    print(f"Overall Anomaly Accuracy: {anomaly_acc*100:.2f}% ({cm[1,1]}/{np.sum(anomaly_mask)})")
    
    print(f"\n--- PER-CATEGORY ACCURACY BREAKDOWN ---")
    for cat, res in cat_results.items():
        print(f"  Category: {cat:<15} | Correct: {res['correct']:2d}/{res['total']:2d} | Accuracy: {res['accuracy']*100:5.1f}% (Pred Normal: {res['pred_norm']:2d}, Pred Anomaly: {res['pred_anom']:2d})")
        
    print(f"\n--- VIDEO ANOMALY SCORE DISTRIBUTIONS ---")
    print(f"  Normal Videos  (N={len(normal_scores)}): Min={norm_dist['min']:.4f}, Mean={norm_dist['mean']:.4f}, Median={norm_dist['median']:.4f}, Max={norm_dist['max']:.4f}")
    print(f"  Anomaly Videos (N={len(anomaly_scores)}): Min={anom_dist['min']:.4f}, Mean={anom_dist['mean']:.4f}, Median={anom_dist['median']:.4f}, Max={anom_dist['max']:.4f}")

    print("\n=========================================================================")
    print("                  BASELINE COMPARISON SUMMARY                            ")
    print("=========================================================================")
    print(f"Original Baseline Test ROC-AUC (390 videos, 1D-CNN BCE):      0.6101")
    print(f"Expanded Baseline Test ROC-AUC (500 videos, 1D-CNN BCE):      0.5193")
    print(f"New MIL Model Test ROC-AUC (500 videos, LayerNorm MLP MIL):   {test_roc_auc:.4f}")
    print(f"Absolute Test ROC-AUC Improvement over Expanded Baseline:     +{test_roc_auc - 0.5193:.4f}")
    print(f"Absolute Test ROC-AUC Improvement over Original Baseline:     +{test_roc_auc - 0.6101:.4f}")
    print("=========================================================================")

if __name__ == "__main__":
    evaluate_mil_test()
