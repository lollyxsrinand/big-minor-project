import os
import sys
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.extract_features import get_feature_extractor, get_transforms
from src.model import MLPAnomalyScorer

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"

def run_inference(video_path, checkpoint_path=None, top_k_regions=10):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.WORKSPACE_DIR, "best_mil_model.pt")
        
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # 1. Load Feature Extractor & Model
    feature_extractor = get_feature_extractor(device)
    transform = get_transforms()
    
    model = MLPAnomalyScorer(in_channels=960, hidden_dim1=256, hidden_dim2=64, dropout=0.4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. Open Video and Extract 5-FPS Features
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 30.0
        
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = config.SAMPLE_FPS  # 5 FPS
    frame_step = max(1, int(round(native_fps / target_fps)))
    
    frame_buffer = []
    sampled_count = 0
    frame_idx = 0
    batch_size = 32
    feature_chunks = []
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        if frame_idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            tensor_img = transform(pil_img)
            frame_buffer.append(tensor_img)
            sampled_count += 1
            
            if len(frame_buffer) >= 256:
                buf_tensor = torch.stack(frame_buffer)
                with torch.no_grad():
                    for b_idx in range(0, len(buf_tensor), batch_size):
                        batch = buf_tensor[b_idx:b_idx+batch_size].to(device)
                        feats = feature_extractor(batch)
                        feature_chunks.append(feats.cpu())
                frame_buffer.clear()
                
        frame_idx += 1
        
    cap.release()
    
    if frame_buffer:
        buf_tensor = torch.stack(frame_buffer)
        with torch.no_grad():
            for b_idx in range(0, len(buf_tensor), batch_size):
                batch = buf_tensor[b_idx:b_idx+batch_size].to(device)
                feats = feature_extractor(batch)
                feature_chunks.append(feats.cpu())
        frame_buffer.clear()
        
    if not feature_chunks:
        raise ValueError(f"No frames could be read or sampled from video: {video_path}")
        
    features = torch.cat(feature_chunks, dim=0)  # [T, 960]
    T_len, C_dim = features.shape
    
    # 3. Construct Temporal Windows (W=16, S=8)
    window_size = config.WINDOW_SIZE  # 16
    stride = config.STRIDE            # 8
    
    if T_len < window_size:
        pad_len = window_size - T_len
        features = F.pad(features, (0, 0, 0, pad_len))
        T_len = features.shape[0]
        
    unfolded = features.unfold(dimension=0, size=window_size, step=stride)  # [M, 960, 16]
    windows = unfolded.permute(0, 2, 1).to(device)                           # [M, 16, 960]
    M_count = windows.shape[0]
    
    # 4. Run Frozen MLP Model (Evaluates ALL windows, no training sampler)
    with torch.no_grad():
        logits = model(windows)
        probs = torch.sigmoid(logits).cpu().numpy()  # Array [M]
        
    video_score = float(np.max(probs))
    
    # 5. Extract Top Suspicious Regions
    k = min(top_k_regions, M_count)
    top_indices = np.argsort(probs)[::-1][:k]
    
    top_regions = []
    for w_idx in top_indices:
        score = float(probs[w_idx])
        start_sec = (w_idx * stride) / float(target_fps)
        end_sec = (w_idx * stride + window_size - 1) / float(target_fps)
        top_regions.append({
            "window_index": int(w_idx),
            "score": score,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "timestamp_str": f"{format_timestamp(start_sec)} - {format_timestamp(end_sec)}"
        })
        
    duration_sec = total_native_frames / native_fps if native_fps > 0 else sampled_count / target_fps
    
    # 6. Display Human-Readable Output
    video_basename = os.path.basename(video_path)
    print("\n=========================================================================")
    print("                    END-TO-END ANOMALY INFERENCE                         ")
    print("=========================================================================")
    print(f"Video File:          {video_basename}")
    print(f"Full Video Path:     {video_path}")
    print(f"Video Duration:      {format_timestamp(duration_sec)} ({total_native_frames} frames at {native_fps:.2f} FPS)")
    print(f"Sampled 5-FPS Frames: {sampled_count} frames")
    print(f"Temporal Windows:    {M_count} windows (W={window_size}, S={stride})")
    print(f"Loaded Checkpoint:   {checkpoint_path} (FROZEN)")
    print(f"\nOverall Video Score: {video_score:.2f} (Max Sigmoid Window Score: {video_score:.4f})")
    print(f"\nTop Suspicious Regions:")
    print("-" * 65)
    print(f"{'Rank':<5} | {'Timestamp Range':<20} | {'Window Index':<14} | {'Anomaly Score':<12}")
    print("-" * 65)
    for rank, reg in enumerate(top_regions, 1):
        print(f"{rank:<5} | {reg['timestamp_str']:<20} | Window {reg['window_index']:<7} | {reg['score']:.4f}")
    print("=========================================================================\n")
    
    # Full timeline details for visualization
    all_timeline = []
    for w_idx in range(M_count):
        s_sec = (w_idx * stride) / float(target_fps)
        e_sec = (w_idx * stride + window_size - 1) / float(target_fps)
        all_timeline.append({
            "window_index": w_idx,
            "center_sec": (s_sec + e_sec) / 2.0,
            "start_sec": s_sec,
            "end_sec": e_sec,
            "score": float(probs[w_idx])
        })

    return {
        "video_path": video_path,
        "video_score": video_score,
        "total_windows": M_count,
        "top_regions": top_regions,
        "all_timeline": all_timeline,
        "duration_sec": duration_sec
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Real-Time Video Anomaly Detection Inference")
    default_video = os.path.join(config.NORMAL_DIR, "Normal_Videos_003_x264.mp4") if os.path.exists(config.NORMAL_DIR) else ""
    parser.add_argument("--video_path", type=str, default=default_video,
                        help="Path to input video file for inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    
    run_inference(args.video_path, args.checkpoint)
