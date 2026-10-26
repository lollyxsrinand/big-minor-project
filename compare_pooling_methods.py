import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import VideoFeatureDataset
from src.model import MLPAnomalyScorer

def compare_pooling_methods():
    print("=========================================================================")
    print("      PHASE 6K: OFFLINE EVALUATION OF VIDEO-LEVEL POOLING METHODS        ")
    print("=========================================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = os.path.join(config.WORKSPACE_DIR, "best_mil_model.pt")
    
    model = MLPAnomalyScorer(in_channels=960, hidden_dim1=256, hidden_dim2=64, dropout=0.4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    test_dataset = VideoFeatureDataset(split="test")
    print(f"Test Split Size:       {len(test_dataset)} videos (Using ALL temporal windows)")
    
    labels = []
    video_window_scores = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            windows = sample["windows"].to(device)  # [M, 16, 960]
            label = int(sample["label"].item())
            
            # Model outputs raw logits [M]
            logits = model(windows)
            probs = torch.sigmoid(logits).cpu().numpy()  # Array of length M
            
            labels.append(label)
            video_window_scores.append(probs)
            
    labels = np.array(labels)
    
    # Evaluate 4 video-level scoring rules on the EXACT SAME window scores
    methods = {
        "A. MAX": lambda scores: np.max(scores),
        "B. TOP-3 MEAN": lambda scores: np.mean(np.sort(scores)[::-1][:min(3, len(scores))]),
        "C. TOP-5 MEAN": lambda scores: np.mean(np.sort(scores)[::-1][:min(5, len(scores))]),
        "D. TOP-10 MEAN": lambda scores: np.mean(np.sort(scores)[::-1][:min(10, len(scores))])
    }
    
    results = []
    for name, pool_fn in methods.items():
        v_scores = np.array([pool_fn(w_scores) for w_scores in video_window_scores])
        roc_auc = roc_auc_score(labels, v_scores)
        pr_auc = average_precision_score(labels, v_scores)
        results.append({
            "name": name,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })
        
    # Sort results by ROC-AUC descending
    results.sort(key=lambda x: x["roc_auc"], reverse=True)
    
    print("\n--- POOLING METHOD PERFORMANCE COMPARISON ---")
    print(f"{'Rank':<6} | {'Pooling Method':<18} | {'Test ROC-AUC':<14} | {'Test PR-AUC':<14}")
    print("-" * 60)
    for rank, res in enumerate(results, 1):
        print(f"{rank:<6} | {res['name']:<18} | {res['roc_auc']:<14.4f} | {res['pr_auc']:<14.4f}")
        
    print("\n=========================================================================")
    print("                    FINAL METHOD RANKING & SUMMARY                       ")
    print("=========================================================================")
    for rank, res in enumerate(results, 1):
        print(f"{rank}. {res['name']:<15} -> ROC-AUC: {res['roc_auc']:.4f} | PR-AUC: {res['pr_auc']:.4f}")
    print("=========================================================================")

if __name__ == "__main__":
    compare_pooling_methods()
