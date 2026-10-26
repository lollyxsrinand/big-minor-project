import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def sample_window_indices(M, num_regions=128):
    """
    Selects actual window indices for a video with M temporal windows.
    If M <= num_regions: retains ALL M windows.
    If M > num_regions: divides M windows into 128 equal temporal regions,
                        selecting 2 actual windows per region using deterministic
                        offsets (25% and 75% of region length).
    Returns sorted, unique indices within [0, M-1].
    """
    if M <= num_regions:
        return list(range(M))
        
    indices = []
    for r in range(num_regions):
        start = int(r * M / num_regions)
        end = int((r + 1) * M / num_regions)
        region_len = max(1, end - start)
        
        idx1 = start + int(0.25 * region_len)
        idx2 = start + int(0.75 * region_len)
        
        # Clamp to valid range [0, M-1]
        idx1 = min(max(0, idx1), M - 1)
        idx2 = min(max(0, idx2), M - 1)
        
        indices.append(idx1)
        indices.append(idx2)
        
    # Ensure unique and sorted
    unique_sorted_indices = sorted(list(dict.fromkeys(indices)))
    
    # If deduplication dropped any indices, fill sequentially up to min(M, 256) unique indices
    target_count = min(M, num_regions * 2)
    if len(unique_sorted_indices) < target_count:
        existing_set = set(unique_sorted_indices)
        for cand in range(M):
            if cand not in existing_set:
                unique_sorted_indices.append(cand)
                existing_set.add(cand)
                if len(unique_sorted_indices) == target_count:
                    break
        unique_sorted_indices.sort()
        
    return unique_sorted_indices

class VideoFeatureDataset(Dataset):
    def __init__(self, split="train", split_file=config.SPLIT_FILE, cache_dir=config.CACHE_DIR,
                 window_size=config.WINDOW_SIZE, stride=config.STRIDE):
        super().__init__()
        self.split = split
        self.cache_dir = cache_dir
        self.window_size = window_size
        self.stride = stride
        
        with open(split_file, "r") as f:
            all_splits = json.load(f)
            
        self.items = all_splits.get(split, [])
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        v_id = item["video_id"]
        label = float(item["label"])
        
        pt_path = os.path.join(self.cache_dir, f"{v_id}.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Cached feature file not found: {pt_path}. Run feature extraction first.")
            
        features = torch.load(pt_path)  # [T, 960]
        T_len, C_dim = features.shape
        
        # Handle short videos (T < window_size) with explicit zero-padding and logging warning
        if T_len < self.window_size:
            pad_len = self.window_size - T_len
            import logging
            logging.warning(f"Video {v_id} feature length ({T_len}) < window_size ({self.window_size}). Zero-padding {pad_len} frames.")
            features = torch.nn.functional.pad(features, (0, 0, 0, pad_len))
            T_len = features.shape[0]
            
        # Extract temporal windows/clips: [M, W, 960]
        # unfold(dimension=0, size=W, step=S) -> [M, 960, W]
        unfolded = features.unfold(dimension=0, size=self.window_size, step=self.stride)
        windows = unfolded.permute(0, 2, 1)  # [M, W=16, C=960]
        
        M_total = windows.shape[0]
        
        # Apply 128-region x 2-window sampling ONLY during training for long videos (M > 128)
        if self.split == "train" and M_total > 128:
            sampled_idx = sample_window_indices(M_total, num_regions=128)
            windows = windows[sampled_idx]
            
        return {
            "windows": windows,         # Tensor [M, 16, 960]
            "label": torch.tensor(label, dtype=torch.float32),  # Tensor scalar 0.0 or 1.0
            "video_id": v_id,
            "category": item["category"]
        }

def video_collate_fn(batch):
    # Custom collate function to handle variable number of clips per video bag
    windows_list = [item["windows"] for item in batch]
    labels_list = [item["label"] for item in batch]
    video_ids = [item["video_id"] for item in batch]
    categories = [item["category"] for item in batch]
    
    return {
        "windows_list": windows_list,  # List of [M_i, 16, 960]
        "labels": torch.stack(labels_list),  # Tensor [B]
        "video_ids": video_ids,
        "categories": categories
    }

def get_balanced_sampler(dataset):
    labels = [int(item["label"]) for item in dataset.items]
    class_counts = {}
    for l in labels:
        class_counts[l] = class_counts.get(l, 0) + 1
        
    weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler
