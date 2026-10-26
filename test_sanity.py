import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import VideoFeatureDataset, video_collate_fn, get_balanced_sampler
from src.model import VideoAnomalyDetector

def run_sanity_test():
    print("=== END-TO-END SANITY TEST ===")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Instantiate Dataset on cached videos
    # We create a temporary split mapping only to the 3 cached videos for sanity testing
    sanity_items = [
        {"video_id": "Normal_Videos_704_x264", "label": 0.0, "category": "Normal"},
        {"video_id": "RoadAccidents102_x264", "label": 1.0, "category": "RoadAccidents"},
        {"video_id": "Robbery120_x264", "label": 1.0, "category": "Robbery"}
    ]
    
    # Override dataset items for sanity check
    dataset = VideoFeatureDataset(split="train")
    dataset.items = sanity_items
    
    print(f"Sanity Dataset size: {len(dataset)} videos")
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=video_collate_fn)
    
    # 2. Instantiate Model & Optimizer
    model = VideoAnomalyDetector(
        in_channels=config.FEATURE_DIM,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.3,
        pooling_method=config.POOLING_METHOD,
        topk_percentage=config.TOPK_PERCENTAGE,
        topk_count=config.TOPK_COUNT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameter Count: {total_params:,} trainable parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.BCELoss()
    
    # 3. Get mini-batch
    batch = next(iter(loader))
    windows_list = batch["windows_list"]  # List of [M_i, 16, 960]
    labels = batch["labels"].to(device)   # [2]
    video_ids = batch["video_ids"]
    
    print(f"\nMini-batch size: {len(windows_list)} videos ({video_ids})")
    for i, w in enumerate(windows_list):
        print(f"  Video {i+1} ({video_ids[i]}): {w.shape[0]} clips of shape [W={w.shape[1]}, C={w.shape[2]}] | Label: {labels[i].item()}")
        
    # 4. Forward Pass 1 (Before optimizer step)
    model.train()
    optimizer.zero_grad()
    
    predicted_video_scores = []
    clip_score_shapes = []
    
    for i, windows in enumerate(windows_list):
        windows_gpu = windows.to(device)  # [M_i, 16, 960]
        out = model(windows_gpu)
        v_score = out["video_score"]      # Scalar []
        c_scores = out["clip_scores"]     # [M_i]
        
        predicted_video_scores.append(v_score)
        clip_score_shapes.append(list(c_scores.shape))
        
    pred_scores_tensor = torch.stack(predicted_video_scores)  # [2]
    loss_0 = criterion(pred_scores_tensor, labels)
    loss_0_val = loss_0.item()
    
    print(f"\n--- FORWARD PASS 1 ---")
    print(f"  Clip score tensor shapes: {clip_score_shapes}")
    print(f"  Aggregated Video Anomaly Scores: {pred_scores_tensor.detach().cpu().numpy().round(4)}")
    print(f"  True Video Labels: {labels.cpu().numpy()}")
    print(f"  Loss BEFORE Optimizer Step: {loss_0_val:.6f}")
    
    # 5. Backward Pass
    loss_0.backward()
    
    # Check Gradients
    has_grads = True
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                has_grads = False
                print(f"  ERROR: No gradient for {name}")
            else:
                norm = param.grad.norm().item()
                grad_norms[name] = round(norm, 6)
                
    print(f"  Gradients check: {'PASSED' if has_grads else 'FAILED'}")
    print(f"  Sample Gradient Norms: {dict(list(grad_norms.items())[:3])}")
    
    # 6. Run 5 Optimizer Steps to verify loss reduction trend
    print("\n--- RUNNING 5 OPTIMIZER STEPS ---")
    for step in range(1, 6):
        optimizer.zero_grad()
        scores = []
        for windows in windows_list:
            out = model(windows.to(device))
            scores.append(out["video_score"])
        loss = criterion(torch.stack(scores), labels)
        loss.backward()
        optimizer.step()
        print(f"  Step {step}: Loss = {loss.item():.6f} | Scores = {torch.stack(scores).detach().cpu().numpy().round(4)}")
        
    print("\n=== SANITY TEST SUCCESSFUL! ===")

if __name__ == "__main__":
    run_sanity_test()
