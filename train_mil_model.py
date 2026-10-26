import os
import sys
import random
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import VideoFeatureDataset
from src.model import MLPAnomalyScorer, MILRankingLoss

# Fix random seeds for reproducible fresh initialization and sampling
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_mil_model():
    device = get_device()
    print("=========================================================================")
    print("      PHASE 6H: TRAINING NEW MIL MODEL WITH TEMPORAL SAMPLING            ")
    print("=========================================================================")
    print(f"Device Acceleration:   {device}")
    print(f"Split File Used:       {config.SPLIT_FILE}")
    print(f"Save Checkpoint Path:  best_mil_model.pt")
    
    # 1. Datasets
    train_dataset = VideoFeatureDataset(split="train")
    val_dataset = VideoFeatureDataset(split="val")
    
    # Separate anomaly and normal indices in train dataset
    anomaly_indices = [i for i, item in enumerate(train_dataset.items) if float(item["label"]) == 1.0]
    normal_indices = [i for i, item in enumerate(train_dataset.items) if float(item["label"]) == 0.0]
    
    num_anomaly = len(anomaly_indices)
    num_normal = len(normal_indices)
    num_pairs = max(num_anomaly, num_normal)
    
    print(f"\n--- DATASET VIDEO COUNTS ---")
    print(f"Train Dataset: {len(train_dataset)} videos (Normal: {num_normal}, Anomaly: {num_anomaly}) -> {num_pairs} Pairs / Epoch")
    print(f"Val Dataset:   {len(val_dataset)} videos (Uses ALL windows)")
    
    # 2. Fresh Model & Loss Initialization
    model = MLPAnomalyScorer(in_channels=960, hidden_dim1=256, hidden_dim2=64, dropout=0.4).to(device)
    criterion = MILRankingLoss(margin=1.0, k=3, lambda_smooth=8e-5, lambda_sparse=8e-5).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_auc = 0.0
    best_val_pr_auc = 0.0
    best_epoch = 0
    save_checkpoint_path = os.path.join(config.WORKSPACE_DIR, "best_mil_model.pt")
    
    print("\n--- EPOCH-BY-EPOCH TRAINING LOG ---")
    print(f"{'Epoch':<8} | {'Total Loss':<12} | {'Rank Loss':<12} | {'Smooth Loss':<12} | {'Sparse Loss':<12} | {'Val ROC-AUC':<12} | {'Val PR-AUC':<12}")
    print("-" * 92)
    
    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Shuffle indices at start of epoch
        random.shuffle(anomaly_indices)
        random.shuffle(normal_indices)
        
        epoch_total_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_smooth_loss = 0.0
        epoch_sparse_loss = 0.0
        
        for step in range(num_pairs):
            anom_idx = anomaly_indices[step % num_anomaly]
            norm_idx = normal_indices[step % num_normal]
            
            anom_sample = train_dataset[anom_idx]
            norm_sample = train_dataset[norm_idx]
            
            # Windows shape: [M, 16, 960]
            anom_windows = anom_sample["windows"].to(device)
            norm_windows = norm_sample["windows"].to(device)
            
            optimizer.zero_grad()
            
            anom_logits = model(anom_windows)
            norm_logits = model(norm_windows)
            
            loss_dict = criterion(anom_logits, norm_logits)
            loss = loss_dict["total_loss"]
            
            loss.backward()
            optimizer.step()
            
            epoch_total_loss += loss.item()
            epoch_rank_loss += loss_dict["loss_rank"].item()
            epoch_smooth_loss += loss_dict["loss_smooth"].item()
            epoch_sparse_loss += loss_dict["loss_sparse"].item()
            
        avg_total = epoch_total_loss / num_pairs
        avg_rank = epoch_rank_loss / num_pairs
        avg_smooth = epoch_smooth_loss / num_pairs
        avg_sparse = epoch_sparse_loss / num_pairs
        
        # Validation Loop (Using ALL temporal windows per video)
        model.eval()
        val_labels = []
        val_scores = []
        
        with torch.no_grad():
            for idx in range(len(val_dataset)):
                val_sample = val_dataset[idx]
                val_windows = val_sample["windows"].to(device)
                val_label = val_sample["label"].item()
                
                # Model outputs raw logits [M]
                val_logits = model(val_windows)
                val_probs = torch.sigmoid(val_logits)
                
                # Video Anomaly Score = max(sigmoid(window_logit))
                video_score = torch.max(val_probs).item()
                
                val_labels.append(val_label)
                val_scores.append(video_score)
                
        val_roc_auc = roc_auc_score(val_labels, val_scores)
        val_pr_auc = average_precision_score(val_labels, val_scores)
        
        is_best = val_roc_auc > best_val_auc
        if is_best:
            best_val_auc = val_roc_auc
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            torch.save(model.state_dict(), save_checkpoint_path)
            best_marker = " *"
        else:
            best_marker = ""
            
        print(f"{epoch:02d}/{epochs:02d}    | {avg_total:<12.6f} | {avg_rank:<12.6f} | {avg_smooth:<12.6f} | {avg_sparse:<12.6f} | {val_roc_auc:<12.4f} | {val_pr_auc:<12.4f}{best_marker}")
        
    print("\n=========================================================================")
    print("                   TRAINING COMPLETE SUMMARY                             ")
    print("=========================================================================")
    print(f"Best Epoch:            {best_epoch:02d} / {epochs:02d}")
    print(f"Best Validation ROC-AUC: {best_val_auc:.4f}")
    print(f"Best Validation PR-AUC:  {best_val_pr_auc:.4f}")
    print(f"Final Epoch (20) ROC-AUC: {val_roc_auc:.4f}")
    print(f"Final Epoch (20) PR-AUC:  {val_pr_auc:.4f}")
    print(f"Checkpoint Saved To:   {save_checkpoint_path}")
    print(f"Training Status:       COMPLETED WITHOUT ERRORS")
    print("=========================================================================")

if __name__ == "__main__":
    train_mil_model()
