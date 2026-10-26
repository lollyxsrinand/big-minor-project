import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import MLPAnomalyScorer, MILRankingLoss

def run_model_smoke_test():
    print("=== PHASE 6G: SMOKE TEST FOR NEW MLP MODEL AND MIL LOSS ===")
    
    # 1. Instantiate Model & Loss
    model = MLPAnomalyScorer(in_channels=960, hidden_dim1=256, hidden_dim2=64, dropout=0.4)
    criterion = MILRankingLoss(margin=1.0, k=3, lambda_smooth=8e-5, lambda_sparse=8e-5)
    
    # Calculate Total Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Total Trainable Parameters: {total_params:,} ({total_params / 1e3:.2f}K)")
    
    # 2. Test Input Shapes
    M_a, M_n = 256, 120  # Anomaly video has 256 windows, Normal video has 120 windows
    anomaly_windows = torch.randn(M_a, 16, 960, requires_grad=False)
    normal_windows = torch.randn(M_n, 16, 960, requires_grad=False)
    
    # 3. Forward Pass
    logits_a = model(anomaly_windows)  # [M_a]
    logits_n = model(normal_windows)   # [M_n]
    
    print(f"Anomaly Video Forward Output Shape: {list(logits_a.shape)} (Expected: [{M_a}])")
    print(f"Normal Video Forward Output Shape:  {list(logits_n.shape)} (Expected: [{M_n}])")
    
    assert logits_a.shape == (M_a,), f"Expected shape ({M_a},), got {logits_a.shape}"
    assert logits_n.shape == (M_n,), f"Expected shape ({M_n},), got {logits_n.shape}"
    
    # 4. Compute MIL Ranking Loss
    loss_dict = criterion(logits_a, logits_n)
    total_loss = loss_dict["total_loss"]
    
    print(f"\nLoss Breakdown:")
    print(f"  S_a (Top-3 Mean Raw Logit): {loss_dict['S_a'].item():.4f}")
    print(f"  S_n (Top-3 Mean Raw Logit): {loss_dict['S_n'].item():.4f}")
    print(f"  Ranking Hinge Loss:         {loss_dict['loss_rank'].item():.4f}")
    print(f"  Smoothness Loss:            {loss_dict['loss_smooth'].item():.6f}")
    print(f"  Sparsity Loss:              {loss_dict['loss_sparse'].item():.6f}")
    print(f"  Total Loss Scalar:          {total_loss.item():.4f}")
    
    assert torch.isfinite(total_loss), "Loss is not a finite scalar!"
    
    # 5. Gradient Flow Check via Backward Pass
    total_loss.backward()
    
    has_grads = True
    for name, param in model.named_parameters():
        if param.grad is None or not torch.isfinite(param.grad).all():
            print(f"ERROR: Gradient for {name} is missing or non-finite!")
            has_grads = False
            
    assert has_grads, "Gradient backpropagation failed!"
    print("\nGradient Flow Check: Passed (Gradients successfully computed for all parameters)")
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_model_smoke_test()
