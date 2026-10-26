import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.dataset import sample_window_indices, VideoFeatureDataset

def run_sampling_utility_tests():
    print("=== PHASE 6F: TESTING TEMPORAL SAMPLING UTILITY ===")
    
    # 1. Test M = 1, 16, 32, 128 -> All indices retained
    for m in [1, 16, 32, 128]:
        idx = sample_window_indices(m)
        print(f"Testing M={m:3d} -> Retained Count: {len(idx)} (Expected: {m})")
        assert idx == list(range(m)), f"Failed for M={m}"
        assert len(idx) == m, f"Expected length {m}, got {len(idx)}"
        
    # 2. Test M = 129 -> Exactly min(129, 256) = 129 indices
    idx_129 = sample_window_indices(129)
    print(f"Testing M=129 -> Retained Count: {len(idx_129)} (Expected: 129)")
    assert len(idx_129) == 129, f"Expected 129 indices for M=129, got {len(idx_129)}"
    assert idx_129 == sorted(idx_129), "Indices are not sorted for M=129!"
    assert len(set(idx_129)) == 129, "Indices are not unique for M=129!"
    assert all(0 <= i < 129 for i in idx_129), "Indices out of bounds for M=129!"
    
    # 3. Test M = 2249 -> Exactly 256 indices
    idx_2249 = sample_window_indices(2249)
    print(f"Testing M=2249 -> Retained Count: {len(idx_2249)} (Expected: 256)")
    assert len(idx_2249) == 256, f"Expected 256 indices for M=2249, got {len(idx_2249)}"
    assert idx_2249 == sorted(idx_2249), "Indices are not sorted for M=2249!"
    assert len(set(idx_2249)) == 256, "Indices are not unique for M=2249!"
    assert all(0 <= i < 2249 for i in idx_2249), "Indices out of bounds for M=2249!"
    
    # 4. Repeatability & Determinism Check
    idx_2249_repeat = sample_window_indices(2249)
    assert idx_2249 == idx_2249_repeat, "Function is not deterministic!"
    print("Determinism Check: Passed (Repeated calls produce identical indices)")

    # 5. Dataset Split Behavior Verification (Train vs Val/Test)
    print("\n--- DATASET SPLIT BEHAVIOR VERIFICATION ---")
    train_ds = VideoFeatureDataset(split="train")
    val_ds = VideoFeatureDataset(split="val")
    
    # Find long video in split
    long_video_id = "Normal_Videos_924_x264"
    
    # Verify train vs val behavior on dataset items
    for item in train_ds.items:
        if item["video_id"] == long_video_id:
            idx_in_train = train_ds.items.index(item)
            sample = train_ds[idx_in_train]
            w_shape = sample["windows"].shape
            print(f"Train Dataset Long Video ('{long_video_id}') Windows Shape: {list(w_shape)} (Expected: [256, 16, 960])")
            assert w_shape[0] == 256, f"Expected 256 windows in train split, got {w_shape[0]}"
            
    for item in val_ds.items:
        if item["video_id"] == long_video_id:
            idx_in_val = val_ds.items.index(item)
            sample = val_ds[idx_in_val]
            w_shape = sample["windows"].shape
            print(f"Val Dataset Long Video ('{long_video_id}') Windows Shape:   {list(w_shape)} (Expected: [2249, 16, 960])")
            assert w_shape[0] == 2249, f"Expected ALL 2249 windows in val split, got {w_shape[0]}"

    print("\nALL TEMPORAL SAMPLING UTILITY TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_sampling_utility_tests()
