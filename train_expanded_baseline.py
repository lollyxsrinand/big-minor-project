import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import VideoFeatureDataset, video_collate_fn, get_balanced_sampler
from src.model import VideoAnomalyDetector

EXPANDED_CHECKPOINT_PATH = os.path.join(config.WORKSPACE_DIR, "best_expanded_baseline_model.pt")
OLD_BASELINE_CHECKPOINT_PATH = os.path.join(config.WORKSPACE_DIR, "best_baseline_model.pt")

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    all_scores = []
    all_labels = []
    all_categories = []
    all_video_ids = []
    
    with torch.no_grad():
        for batch in loader:
            windows_list = batch["windows_list"]
            labels = batch["labels"].to(device)
            categories = batch["categories"]
            v_ids = batch["video_ids"]
            
            video_scores = []
            for windows in windows_list:
                out = model(windows.to(device))
                video_scores.append(out["video_score"])
                
            scores_tensor = torch.stack(video_scores)
            loss = criterion(scores_tensor, labels)
            
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            
            all_scores.extend(scores_tensor.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_categories.extend(categories)
            all_video_ids.extend(v_ids)
            
    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, np.array(all_labels), np.array(all_scores), all_categories, all_video_ids

def find_best_threshold(y_true, y_scores):
    best_thresh = 0.5
    best_f1 = -1.0
    
    # Grid search over threshold from 0.05 to 0.95
    thresholds = np.linspace(0.05, 0.95, 91)
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh

def compute_full_metrics(y_true, y_scores, threshold, categories):
    y_pred = (y_scores >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        roc_auc = 0.5
        
    try:
        p_prec, p_rec, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(p_rec, p_prec)
    except Exception:
        pr_auc = 0.5
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Per-category metrics
    cat_metrics = {}
    unique_cats = sorted(list(set(categories)))
    
    for cat in unique_cats:
        indices = [i for i, c in enumerate(categories) if c == cat]
        cat_true = y_true[indices]
        cat_pred = y_pred[indices]
        
        total = len(indices)
        correct = np.sum(cat_true == cat_pred)
        cat_acc = correct / total if total > 0 else 0.0
        
        cat_metrics[cat] = {
            "total": total,
            "correct": int(correct),
            "accuracy": round(cat_acc, 4),
            "predicted_anomaly_count": int(np.sum(cat_pred == 1)),
            "predicted_normal_count": int(np.sum(cat_pred == 0))
        }
        
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm.tolist(),
        "per_category": cat_metrics
    }

def run_expanded_baseline_training(epochs=20):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=========================================================================")
    print("     PHASE 5B: TRAINING NEW EXPANDED-DATASET BASELINE MODEL             ")
    print("=========================================================================")
    print(f"Split File Used:       {config.SPLIT_FILE}")
    print(f"Save Checkpoint Path:  {EXPANDED_CHECKPOINT_PATH}")
    print(f"Old Baseline Path:     {OLD_BASELINE_CHECKPOINT_PATH} (UNTOUCHED)")
    print(f"Device Acceleration:   {device}")
    print(f"Epochs:                {epochs}")
    print(f"Batch Size:            {config.BATCH_SIZE}")
    print(f"Learning Rate:         {config.LEARNING_RATE}")
    print(f"Weight Decay:          {config.WEIGHT_DECAY}")
    print(f"Top-K Temporal Pool:   {config.POOLING_METHOD} ({config.TOPK_PERCENTAGE*100}%)")
    
    # Verify baseline checkpoint untouched
    assert os.path.exists(OLD_BASELINE_CHECKPOINT_PATH), "Old baseline model checkpoint missing!"
    old_ckpt_mtime = os.path.getmtime(OLD_BASELINE_CHECKPOINT_PATH)
    
    # 1. Prepare Datasets & Loaders
    train_dataset = VideoFeatureDataset(split="train")
    val_dataset = VideoFeatureDataset(split="val")
    test_dataset = VideoFeatureDataset(split="test")
    
    print("\n--- DATASET SPLIT VIDEO COUNTS ---")
    print(f"Train Dataset: {len(train_dataset)} videos (Normal: {sum(1 for x in train_dataset.items if x['label']==0)}, Anomaly: {sum(1 for x in train_dataset.items if x['label']==1)})")
    print(f"Val Dataset:   {len(val_dataset)} videos (Normal: {sum(1 for x in val_dataset.items if x['label']==0)}, Anomaly: {sum(1 for x in val_dataset.items if x['label']==1)})")
    print(f"Test Dataset:  {len(test_dataset)} videos (Normal: {sum(1 for x in test_dataset.items if x['label']==0)}, Anomaly: {sum(1 for x in test_dataset.items if x['label']==1)})")
    
    train_sampler = get_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler, collate_fn=video_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=video_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=video_collate_fn)
    
    # 2. Instantiate Model (FRESH Initialization)
    model = VideoAnomalyDetector(
        in_channels=config.FEATURE_DIM,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.3,
        pooling_method=config.POOLING_METHOD,
        topk_percentage=config.TOPK_PERCENTAGE,
        topk_count=config.TOPK_COUNT
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    best_val_auc = -1.0
    best_epoch = -1
    epoch_logs = []
    
    print("\n--- EPOCH-BY-EPOCH TRAINING LOG ---")
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val ROC-AUC':<12} | {'Val F1 (0.5)':<12}")
    print("-" * 65)
    
    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        
        for batch in train_loader:
            windows_list = batch["windows_list"]
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            scores = []
            for windows in windows_list:
                out = model(windows.to(device))
                scores.append(out["video_score"])
                
            scores_tensor = torch.stack(scores)
            loss = criterion(scores_tensor, labels)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * len(labels)
            train_samples += len(labels)
            
        train_loss = train_loss_sum / max(1, train_samples)
        
        # Validation Pass
        val_loss, val_true, val_scores, val_cats, _ = evaluate_loader(model, val_loader, criterion, device)
        val_metrics_05 = compute_full_metrics(val_true, val_scores, 0.5, val_cats)
        val_auc = val_metrics_05["roc_auc"]
        
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_auc": val_auc,
            "val_acc": val_metrics_05["accuracy"],
            "val_prec": val_metrics_05["precision"],
            "val_rec": val_metrics_05["recall"],
            "val_f1": val_metrics_05["f1"]
        }
        epoch_logs.append(log_entry)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | {train_loss:<12.6f} | {val_loss:<12.6f} | {val_auc:<12.4f} | {val_metrics_05['f1']:<12.4f}")
        
        # Save best model checkpoint based ONLY on Validation ROC-AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), EXPANDED_CHECKPOINT_PATH)
            
    print(f"\nTraining Complete. Best Validation ROC-AUC = {best_val_auc:.4f} achieved at Epoch {best_epoch}")
    
    # 4. Load Best Expanded Checkpoint
    model.load_state_dict(torch.load(EXPANDED_CHECKPOINT_PATH))
    
    # 5. Determine Classification Threshold using VALIDATION set ONLY
    val_loss, val_true, val_scores, val_cats, _ = evaluate_loader(model, val_loader, criterion, device)
    best_thresh = find_best_threshold(val_true, val_scores)
    val_final_metrics = compute_full_metrics(val_true, val_scores, best_thresh, val_cats)
    
    print("\n--- VALIDATION THRESHOLD SELECTION ---")
    print(f"Optimal Validation Threshold (maximizing F1): {best_thresh:.4f}")
    print(f"Validation Performance at Threshold {best_thresh:.4f}:")
    print(f"  Accuracy:  {val_final_metrics['accuracy']}")
    print(f"  Precision: {val_final_metrics['precision']}")
    print(f"  Recall:    {val_final_metrics['recall']}")
    print(f"  F1-Score:  {val_final_metrics['f1']}")
    print(f"  ROC-AUC:   {val_final_metrics['roc_auc']}")
    print(f"  PR-AUC:    {val_final_metrics['pr_auc']}")
    
    # 6. Evaluate Saved Best Checkpoint ONCE on TEST set
    test_loss, test_true, test_scores, test_cats, _ = evaluate_loader(model, test_loader, criterion, device)
    test_final_metrics = compute_full_metrics(test_true, test_scores, best_thresh, test_cats)
    
    print("\n--- FINAL TEST SET EVALUATION (SINGLE PASS ON EXPANDED TEST SET) ---")
    print(f"Test Loss:             {test_loss:.6f}")
    print(f"Validation Threshold:  {best_thresh:.4f}")
    print(f"Test ROC-AUC:          {test_final_metrics['roc_auc']}")
    print(f"Test PR-AUC:           {test_final_metrics['pr_auc']}")
    print(f"Test Accuracy:         {test_final_metrics['accuracy']}")
    print(f"Test Precision:        {test_final_metrics['precision']}")
    print(f"Test Recall:           {test_final_metrics['recall']}")
    print(f"Test F1-Score:         {test_final_metrics['f1']}")
    print(f"Confusion Matrix [TN, FP / FN, TP]:\n  {np.array(test_final_metrics['confusion_matrix'])}")
    
    print("\n--- TEST PER-CATEGORY BREAKDOWN ---")
    for cat_name, cat_res in test_final_metrics["per_category"].items():
        print(f"  Category: {cat_name:15s} | Correct: {cat_res['correct']}/{cat_res['total']} | Accuracy: {cat_res['accuracy']*100:.1f}% (Pred Normal: {cat_res['predicted_normal_count']}, Pred Anomaly: {cat_res['predicted_anomaly_count']})")

    # Verify old baseline checkpoint was untouched
    assert old_ckpt_mtime == os.path.getmtime(OLD_BASELINE_CHECKPOINT_PATH), "Old baseline model checkpoint was accidentally modified!"
    print("\nVerified 'best_baseline_model.pt' was UNTOUCHED!")
    
    # ROC-AUC Comparison
    old_test_auc = 0.6101
    new_test_auc = test_final_metrics['roc_auc']
    auc_diff = new_test_auc - old_test_auc
    
    print("\n=========================================================================")
    print("               TEST ROC-AUC BASELINE COMPARISON                          ")
    print("=========================================================================")
    print(f"Old Baseline Test ROC-AUC (390 videos):      {old_test_auc:.4f}")
    print(f"New Expanded Baseline Test ROC-AUC (500 vids): {new_test_auc:.4f}")
    print(f"ROC-AUC Absolute Change:                       {auc_diff:+.4f}")
    print("=========================================================================")

if __name__ == "__main__":
    run_expanded_baseline_training(epochs=20)
