# Surveillance Video Anomaly Detection

A weakly-supervised surveillance video anomaly detection system that scores video clips and localizes suspicious temporal segments.

The application operates locally and performs inference directly using existing pretrained feature extractors and trained checkpoint weights without requiring model retraining.

---

## 🔍 System Architecture & Inference Pipeline

The inference pipeline (`infer_video.py` and `app.py`) consists of five stages:

1. **Video Decoding & Subsampling:**
   * Reads video frames using OpenCV (`cv2.VideoCapture`).
   * Subsamples frames at **5 FPS** (`config.SAMPLE_FPS = 5`) to reduce compute while capturing motion dynamics.
2. **Spatial Feature Extraction:**
   * Resizes sampled frames to $224 \times 224$ and applies ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
   * Passes frames in batches through a frozen **MobileNetV3-Large** convolutional backbone (`torchvision.models.mobilenet_v3_large`).
   * Extracts a 960-dimensional feature embedding per sampled frame, forming a temporal sequence of shape $[T, 960]$.
3. **Temporal Windowing:**
   * Segments the feature sequence into sliding temporal windows of length **$W=16$ frames** (~3.2 seconds at 5 FPS) with a **stride of $S=8$ frames** (50% temporal overlap).
   * Short clips ($T < 16$) are zero-padded to 16 frames.
   * Produces a windowed tensor of shape $[M, 16, 960]$, where $M$ is the number of temporal windows.
4. **Anomaly Scoring (MLP Multiple Instance Learning):**
   * Uses `MLPAnomalyScorer` (`src/model.py`), which:
     * Mean-pools the 16 frame features per window to $[M, 960]$.
     * Passes through `Linear(960, 256)` $\rightarrow$ `LayerNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.4)`.
     * Passes through `Linear(256, 64)` $\rightarrow$ `LayerNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.4)`.
     * Passes through `Linear(64, 1)` $\rightarrow$ raw logits $[M]$.
   * Computes window anomaly probabilities via Sigmoid: $\text{probs} = \sigma(\text{logits}) \in [0, 1]$.
   * The overall video anomaly score is calculated as $\max(\text{probs})$ (the peak window anomaly score).
5. **Temporal Localization & Reporting:**
   * Ranks the top suspicious temporal windows by score.
   * Computes human-readable timestamp ranges (`MM:SS.S - MM:SS.S`) for each window based on sampling rate and stride.

---

## 📦 Model Checkpoints

The repository includes pre-trained model weights in the project root:

* **`best_mil_model.pt`** *(Primary)*: Trained LayerNorm MLP anomaly scorer used by default for standalone inference and the Streamlit web application.
* **`best_baseline_model.pt`**: 1D-CNN temporal baseline model checkpoint (`VideoAnomalyDetector`).
* **`best_expanded_baseline_model.pt`**: 1D-CNN temporal baseline model checkpoint trained on the expanded dataset split.

Inference runs out-of-the-box using `best_mil_model.pt` without retraining.

---

## 🚀 Setup & Installation

### Prerequisites
* **Python 3.10** (Recommended)
* macOS (Apple Silicon `arm64` hardware acceleration via Metal Performance Shaders / MPS supported) or Linux / Windows (CPU or CUDA).
* Standard `venv` and `pip` (no Conda required).

### 1. Create and Activate Virtual Environment
```bash
# Navigate to the project directory
cd "/path/to/project"

# Create a clean virtual environment using Python 3.10
python3.10 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Install Runtime Dependencies
```bash
python -m pip install -r requirements.txt
```

> [!NOTE]
> **Internet Access on First Run:** When running inference for the first time, TorchVision will download the official pretrained MobileNetV3-Large weights (`mobilenet_v3_large-5c1a4163.pth`, ~22 MB) from PyTorch to `~/.cache/torch/hub/checkpoints/`. An active internet connection is required for this initial download. Subsequent runs use the cached weights offline.

---

## 💻 Running Inference

### Option 1: Command-Line Interface (CLI)
Run end-to-end inference on any local video file using `infer_video.py`:

```bash
# Run inference on a video file
python infer_video.py --video_path path/to/video.mp4

# Run with an explicit checkpoint (optional, defaults to best_mil_model.pt)
python infer_video.py --video_path path/to/video.mp4 --checkpoint best_mil_model.pt
```

#### Verified CLI Test Example Output:
Below is the output from a verified test run on a sample normal video clip (`Normal_Videos_003_x264.mp4`):

```text
=========================================================================
                    END-TO-END ANOMALY INFERENCE                         
=========================================================================
Video File:          Normal_Videos_003_x264.mp4
Full Video Path:     Normal_Videos_for_Event_Recognition/Normal_Videos_003_x264.mp4
Video Duration:      01:34.1 (2822 frames at 30.00 FPS)
Sampled 5-FPS Frames: 471 frames
Temporal Windows:    57 windows (W=16, S=8)
Loaded Checkpoint:   .../best_mil_model.pt (FROZEN)

Overall Video Score: 0.21 (Max Sigmoid Window Score: 0.2092)

Top Suspicious Regions:
-----------------------------------------------------------------
Rank  | Timestamp Range      | Window Index   | Anomaly Score
-----------------------------------------------------------------
1     | 00:24.0 - 00:27.0    | Window 15      | 0.2092
2     | 00:22.4 - 00:25.4    | Window 14      | 0.2072
3     | 00:25.6 - 00:28.6    | Window 16      | 0.2012
4     | 00:20.8 - 00:23.8    | Window 13      | 0.1980
5     | 01:29.6 - 01:32.6    | Window 56      | 0.1917
6     | 01:16.8 - 01:19.8    | Window 48      | 0.1916
7     | 00:19.2 - 00:22.2    | Window 12      | 0.1892
8     | 01:15.2 - 01:18.2    | Window 47      | 0.1890
9     | 00:35.2 - 00:38.2    | Window 22      | 0.1889
10    | 01:12.0 - 01:15.0    | Window 45      | 0.1885
=========================================================================
```

### Option 2: Interactive Streamlit Web Application
Launch the local web dashboard:

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. The web UI provides:
* **Video Input (`app.py`):** Accepts video file uploads via `st.sidebar.file_uploader` (supported types: `.mp4`, `.avi`, `.mov`, `.mkv`), or selection from five built-in sample videos (`Normal_Videos_006_x264.mp4`, `Normal_Videos_003_x264.mp4`, `RoadAccidents145_x264.mp4`, `Robbery031_x264.mp4`, `Shooting024_x264.mp4`).
* **Video Player:** Embedded HTML5 video playback column.
* **Inference Trigger:** `🚀 Analyze Video for Anomalies` button executes `run_inference()`.
* **Metric Cards:** Displays overall video score, percentage, and total evaluated window count.
* **Score Timeline Plot:** A Matplotlib line chart plotting anomaly scores across time in seconds, with highlighted peak anomaly score.
* **Top 10 Suspicious Regions Table:** Tabular breakdown of the top 10 suspicious temporal windows with rank, timestamp range, and window score.

To stop the Streamlit server, press `Ctrl + C` in the running terminal.

---

## 📁 Repository Structure

```text
├── app.py                         # Streamlit web application
├── infer_video.py                 # Standalone CLI inference script
├── config.py                      # Global configuration settings & hyperparameters
├── best_mil_model.pt              # Trained LayerNorm MLP anomaly detection weights
├── best_baseline_model.pt         # Baseline 1D-CNN model checkpoint
├── best_expanded_baseline_model.pt# Expanded baseline 1D-CNN model checkpoint
├── src/
│   ├── extract_features.py        # 5-FPS frame sampling & MobileNetV3 feature extraction
│   ├── model.py                   # MLPAnomalyScorer, TemporalPooler, and VideoAnomalyDetector
│   ├── dataset.py                 # PyTorch Dataset for loading windowed feature tensors
│   └── split.py                   # Dataset partitioning utilities
├── dataset_split.json             # Baseline dataset split definitions
├── dataset_split_expanded.json    # Expanded dataset split definitions
├── requirements.txt               # Pinned runtime dependencies
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project documentation
```

---

## 💾 Datasets and Cache Notice

To keep the repository lightweight, large datasets and pre-extracted feature tensors are excluded from Git tracking via `.gitignore`:
* `Anomaly-Videos-Part-3/` (Local directory containing anomaly video clips under `RoadAccidents/`, `Robbery/`, and `Shooting/`)
* `Normal_Videos_for_Event_Recognition/` (Local directory containing normal surveillance video clips)
* `cached_features/` (Local directory of pre-extracted PyTorch `.pt` feature files used for training)
* `temp_uploads/` (Temporary directory for videos uploaded during Streamlit sessions)

These dataset directories are only needed if you re-extract features or retrain models from scratch using `train.py`. Real-time inference on individual videos via `app.py` or `infer_video.py` does not require the training dataset folders.

---

## ⚠️ Advisory Disclaimer

Anomaly scores generated by this model are probabilistic statistical predictions produced by machine learning models. They do not constitute a definitive guarantee, verified alarm, or legal confirmation of an actual security incident, crime, or accident.
