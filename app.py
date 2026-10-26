import os
import sys
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from infer_video import run_inference

# Streamlit Page Config
st.set_page_config(
    page_title="Real-Time Video Anomaly Detection",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Modern Rich Aesthetics
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8C00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #A0A0B0;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #313244;
        text-align: center;
    }
    .metric-score {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎬 Weakly-Supervised Video Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by MobileNetV3-Large & LayerNorm MLP Multiple Instance Learning (MIL)</div>', unsafe_allow_html=True)

# Sidebar Options & Upload Widget
st.sidebar.header("📹 Input Source")
uploaded_file = st.sidebar.file_uploader("Upload an MP4 Video", type=["mp4", "avi", "mov", "mkv"])

# Sample video option if no file uploaded
sample_video_path = None
if uploaded_file is None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Or Pick a Sample Video:")
    sample_options = {
        "None": None,
        "Normal Video 1 (Normal)": os.path.join(config.NORMAL_DIR, "Normal_Videos_006_x264.mp4"),
        "Normal Video 2 (Normal)": os.path.join(config.NORMAL_DIR, "Normal_Videos_003_x264.mp4"),
        "Road Accident (Anomaly)": os.path.join(config.ANOMALY_DIR, "RoadAccidents", "RoadAccidents145_x264.mp4"),
        "Robbery (Anomaly)": os.path.join(config.ANOMALY_DIR, "Robbery", "Robbery031_x264.mp4"),
        "Shooting (Anomaly)": os.path.join(config.ANOMALY_DIR, "Shooting", "Shooting024_x264.mp4")
    }
    selected_sample = st.sidebar.selectbox("Select Sample Video", list(sample_options.keys()))
    if selected_sample and selected_sample != "None":
        sample_video_path = sample_options[selected_sample]

video_path_to_process = None

# Determine input source
if uploaded_file is not None:
    temp_dir = os.path.join(config.WORKSPACE_DIR, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    video_path_to_process = temp_video_path
elif sample_video_path and os.path.exists(sample_video_path):
    video_path_to_process = sample_video_path

# Main UI Columns
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📺 Video Player")
    if video_path_to_process and os.path.exists(video_path_to_process):
        st.video(video_path_to_process)
    else:
        st.info("👆 Please upload an MP4 video or select a sample video from the sidebar to begin.")

with col_right:
    st.subheader("⚡ Inference Control")
    if video_path_to_process and os.path.exists(video_path_to_process):
        st.write(f"**Target Video:** `{os.path.basename(video_path_to_process)}`")
        run_btn = st.button("🚀 Analyze Video for Anomalies", type="primary", use_container_width=True)
    else:
        run_btn = False

if run_btn and video_path_to_process:
    with st.spinner("Extracting 5-FPS MobileNetV3-Large features & running MIL model..."):
        try:
            results = run_inference(video_path_to_process, top_k_regions=10)
            
            st.success("Inference completed successfully!")
            
            st.markdown("---")
            st.subheader("📊 Anomaly Detection Results")
            
            # Score Metric Cards
            score_col1, score_col2, score_col3 = st.columns(3)
            with score_col1:
                st.metric("Overall Video Score (MAX)", f"{results['video_score']:.4f}")
            with score_col2:
                st.metric("Score Percentage", f"{results['video_score'] * 100:.1f}%")
            with score_col3:
                st.metric("Evaluated Windows", f"{results['total_windows']} windows")
                
            # Timeline Score Visualization
            st.markdown("### 📈 Temporal Anomaly Score Timeline")
            timeline_data = results["all_timeline"]
            df_timeline = pd.DataFrame(timeline_data)
            
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(df_timeline["center_sec"], df_timeline["score"], color="#FF4B4B", linewidth=2.0, label="Window Anomaly Score")
            ax.fill_between(df_timeline["center_sec"], df_timeline["score"], color="#FF4B4B", alpha=0.2)
            
            # Highlight peak score
            max_row = df_timeline.loc[df_timeline["score"].idxmax()]
            ax.scatter([max_row["center_sec"]], [max_row["score"]], color="#FFD700", s=100, zorder=5, label=f"Peak Score ({max_row['score']:.2f})")
            
            ax.set_xlabel("Time (seconds)", fontsize=10)
            ax.set_ylabel("Anomaly Score [0 - 1]", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            fig.tight_layout()
            
            st.pyplot(fig)
            
            # Top 10 Suspicious Regions Table
            st.markdown("### 🔍 Top 10 Suspicious Temporal Regions")
            top_regions = results["top_regions"]
            df_top = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Timestamp Start": f"{reg['start_sec']:.1f}s",
                    "Timestamp End": f"{reg['end_sec']:.1f}s",
                    "Timestamp Range": reg['timestamp_str'],
                    "Window Index": reg['window_index'],
                    "Anomaly Score": round(reg['score'], 4)
                }
                for i, reg in enumerate(top_regions)
            ])
            st.dataframe(df_top[["Rank", "Timestamp Range", "Window Index", "Anomaly Score"]], use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Inference failed with error: {e}")
