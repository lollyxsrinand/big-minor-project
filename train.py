import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import VideoFeatureDataset, video_collate_fn, get_balanced_sampler
from src.model import VideoAnomalyDetector

CHECKPOINT_PATH = os.path.join(config.WORKSPACE_DIR, "best_baseline_model.pt")

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
    best_metrics = {}
    
    # Grid search over threshold from 0.05 to 0.95
    thresholds = np.linspace(0.05, 0.95, 91)
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
            
    return best_thresh, best_metrics

def compute_full_metrics(y_true, y_scores, threshold, categories):
    y_pred = (y_scores >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.5
        
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
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "per_category": cat_metrics
    }

def run_baseline_training(epochs=20):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=== STARTING BASELINE TRAINING ===")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Weight Decay: {config.WEIGHT_DECAY}")
    print(f"Top-K Pooling Method: {config.POOLING_METHOD} (Top {config.TOPK_PERCENTAGE*100}%)")
    
    # 1. Prepare Datasets & Loaders
    train_dataset = VideoFeatureDataset(split="train")
    val_dataset = VideoFeatureDataset(split="val")
    test_dataset = VideoFeatureDataset(split="test")
    
    print(f"Dataset Split Sizes -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    train_sampler = get_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler, collate_fn=video_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=video_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=video_collate_fn)
    
    # 2. Instantiate Model, Loss & Optimizer
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
        
        # Evaluate validation metrics at standard 0.5 threshold
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
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.4f} | Val F1 (0.5): {val_metrics_05['f1']:.4f}")
        
        # Save best model checkpoint based ONLY on Validation ROC-AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            
    print(f"\nTraining complete. Best Validation ROC-AUC = {best_val_auc:.4f} at Epoch {best_epoch}")
    
    # 4. Load Best Model Checkpoint for Final Evaluation
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    
    # 5. Determine Classification Threshold using VALIDATION set ONLY
    val_loss, val_true, val_scores, val_cats, _ = evaluate_loader(model, val_loader, criterion, device)
    best_thresh, best_val_thresh_metrics = find_best_threshold(val_true, val_scores)
    val_final_metrics = compute_full_metrics(val_true, val_scores, best_thresh, val_cats)
    
    print("\n=== VALIDATION THRESHOLD SELECTION ===")
    print(f"Selected Validation Threshold (maximizing F1): {best_thresh:.4f}")
    print(f"Validation Performance at Threshold {best_thresh:.4f}:")
    print(f"  Accuracy:  {val_final_metrics['accuracy']}")
    print(f"  Precision: {val_final_metrics['precision']}")
    print(f"  Recall:    {val_final_metrics['recall']}")
    print(f"  F1-Score:  {val_final_metrics['f1']}")
    print(f"  ROC-AUC:   {val_final_metrics['roc_auc']}")
    
    # 6. Evaluate Saved Best Model Checkpoint ONCE on TEST set
    test_loss, test_true, test_scores, test_cats, _ = evaluate_loader(model, test_loader, criterion, device)
    test_final_metrics = compute_full_metrics(test_true, test_scores, best_thresh, test_cats)
    
    print("\n=== FINAL TEST EVALUATION (SINGLE PASS ON HELD-OUT TEST SET) ===")
    print(f"Test Loss:      {test_loss:.6f}")
    print(f"Test Threshold: {best_thresh:.4f} (Derived strictly from Validation set)")
    print(f"Test Accuracy:  {test_final_metrics['accuracy']}")
    print(f"Test Precision: {test_final_metrics['precision']}")
    print(f"Test Recall:    {test_final_metrics['recall']}")
    print(f"Test F1-Score:  {test_final_metrics['f1']}")
    print(f"Test ROC-AUC:   {test_final_metrics['roc_auc']}")
    print(f"Test Confusion Matrix [TN, FP / FN, TP]:\n  {test_final_metrics['confusion_matrix']}")
    
    print("\n=== TEST PER-CATEGORY BREAKDOWN ===")
    for cat_name, cat_res in test_final_metrics["per_category"].items():
        print(f"  Category: {cat_name:15s} | Correct: {cat_res['correct']}/{cat_res['total']} | Accuracy: {cat_res['accuracy']*100:.1f}% (Pred Normal: {cat_res['predicted_normal_count']}, Pred Anomaly: {cat_res['predicted_anomaly_count']})")
        
    return {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "selected_threshold": best_thresh,
        "val_metrics": val_final_metrics,
        "test_metrics": test_final_metrics,
        "epoch_logs": epoch_logs
    }

if __name__ == "__main__":
    run_baseline_training(epochs=20)
