import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class TemporalPooler(nn.Module):
    """
    Aggregates clip-level anomaly scores [M] into a single video-level score scalar.
    Supports: 'topk_percentage', 'max', 'topk_count'.
    """
    def __init__(self, method="topk_percentage", topk_percentage=0.20, topk_count=3):
        super().__init__()
        self.method = method
        self.topk_percentage = topk_percentage
        self.topk_count = topk_count
        
    def forward(self, clip_scores):
        # clip_scores: 1D Tensor [M] with values in [0, 1]
        M = clip_scores.shape[0]
        if M == 0:
            return torch.tensor(0.0, device=clip_scores.device)
            
        if self.method == "max":
            return torch.max(clip_scores)
        elif self.method == "topk_count":
            k = max(1, min(self.topk_count, M))
            topk_vals = torch.topk(clip_scores, k=k).values
            return torch.mean(topk_vals)
        elif self.method == "topk_percentage":
            k = max(1, int(round(self.topk_percentage * M)))
            topk_vals = torch.topk(clip_scores, k=k).values
            return torch.mean(topk_vals)
        else:
            raise ValueError(f"Unsupported pooling method: {self.method}")

class TemporalCNNClassifier(nn.Module):
    """
    1D-CNN temporal classifier operating across the temporal dimension W=16.
    Input shape:  [M, W=16, C=960]
    Output shape: [M] (Clip anomaly probabilities)
    """
    def __init__(self, in_channels=config.FEATURE_DIM, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super().__init__()
        # Conv1d operates over (Batch, Channels, Length) -> [M, in_channels, W]
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim1, out_channels=hidden_dim2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout = nn.Dropout(dropout)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim2, 1)
        
    def forward(self, windows):
        # windows shape: [M, W=16, C=960]
        # Transpose for Conv1d: [M, C=960, W=16]
        x = windows.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Global temporal pooling per clip: [M, 64, 16] -> [M, 64, 1]
        x = self.pool(x).squeeze(-1)  # [M, 64]
        
        logits = self.classifier(x).squeeze(-1)  # [M]
        clip_scores = torch.sigmoid(logits)     # [M]
        return clip_scores

class VideoAnomalyDetector(nn.Module):
    """
    Full pipeline combining 1D-CNN clip classifier and TemporalPooler.
    """
    def __init__(self, in_channels=config.FEATURE_DIM, hidden_dim1=128, hidden_dim2=64, dropout=0.3,
                 pooling_method=config.POOLING_METHOD, topk_percentage=config.TOPK_PERCENTAGE, topk_count=config.TOPK_COUNT):
        super().__init__()
        self.clip_classifier = TemporalCNNClassifier(in_channels=in_channels, hidden_dim1=hidden_dim1,
                                                     hidden_dim2=hidden_dim2, dropout=dropout)
        self.pooler = TemporalPooler(method=pooling_method, topk_percentage=topk_percentage, topk_count=topk_count)
        
    def forward(self, windows):
        # windows shape: [M, W=16, C=960]
        clip_scores = self.clip_classifier(windows)  # [M]
        video_score = self.pooler(clip_scores)       # Scalar []
        return {
            "video_score": video_score,
            "clip_scores": clip_scores
        }

class MLPAnomalyScorer(nn.Module):
    """
    Phase 6G: LayerNorm MLP Anomaly Scorer for weakly-supervised video anomaly detection.
    Input shape:  [M, 16, 960] (M temporal windows, 16 frames per window, 960 features)
    Output shape: [M] (RAW UNCONSTRAINED ANOMALY LOGITS PER WINDOW)
    """
    def __init__(self, in_channels=config.FEATURE_DIM, hidden_dim1=256, hidden_dim2=64, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim2, 1)
        
    def forward(self, windows):
        # windows shape: [M, 16, 960]
        if len(windows.shape) == 3:
            x = windows.mean(dim=1)  # Mean-pool 16 frame features per window -> [M, 960]
        else:
            x = windows  # [M, 960]
            
        x = self.drop1(self.relu1(self.ln1(self.fc1(x))))
        x = self.drop2(self.relu2(self.ln2(self.fc2(x))))
        logits = self.fc3(x).squeeze(-1)  # [M] raw logits
        return logits

class MILRankingLoss(nn.Module):
    """
    Phase 6G: Multiple Instance Learning Hinge Ranking Loss with Temporal Smoothness and Sparsity.
    Operates on RAW LOGITS for ranking, and Sigmoid probabilities for regularization.
    """
    def __init__(self, margin=1.0, k=3, lambda_smooth=8e-5, lambda_sparse=8e-5):
        super().__init__()
        self.margin = margin
        self.k = k
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        
    def forward(self, anomaly_logits, normal_logits):
        """
        anomaly_logits: Raw logits for an anomaly video [M_a]
        normal_logits:  Raw logits for a normal video  [M_n]
        """
        M_a = anomaly_logits.shape[0]
        M_n = normal_logits.shape[0]
        
        k_a = max(1, min(self.k, M_a))
        k_n = max(1, min(self.k, M_n))
        
        # 1. Top-K Mean Raw Logits
        topk_a_logits = torch.topk(anomaly_logits, k=k_a).values
        topk_n_logits = torch.topk(normal_logits, k=k_n).values
        
        S_a = torch.mean(topk_a_logits)
        S_n = torch.mean(topk_n_logits)
        
        # 2. Ranking Hinge Loss on RAW LOGITS
        loss_rank = F.relu(self.margin - S_a + S_n)
        
        # 3. Temporal Smoothness Regularization on Anomaly Window Probabilities
        anomaly_probs = torch.sigmoid(anomaly_logits)
        if M_a > 1:
            diffs = anomaly_probs[1:] - anomaly_probs[:-1]
            loss_smooth = torch.mean(diffs ** 2)
        else:
            loss_smooth = torch.tensor(0.0, device=anomaly_logits.device)
            
        # 4. Sparsity Regularization on Anomaly Window Probabilities
        loss_sparse = torch.mean(anomaly_probs)
        
        # 5. Total Combined Loss
        total_loss = loss_rank + self.lambda_smooth * loss_smooth + self.lambda_sparse * loss_sparse
        
        return {
            "total_loss": total_loss,
            "loss_rank": loss_rank,
            "loss_smooth": loss_smooth,
            "loss_sparse": loss_sparse,
            "S_a": S_a,
            "S_n": S_n
        }
