# Chest X-Ray Pneumonia Detection 🏥

This project implements an automated system for detecting pneumonia in Chest X-Ray images using the **YOLOv8** architecture. The model is trained to differentiate between **NORMAL** and **PNEUMONIA** cases with high precision and speed.

## 📊 Training Results

The model was trained on **500 Chest X-Ray samples** sourced from Roboflow. The performance metrics demonstrate state-of-the-art results:   

- **mAP50:** 0.9795 (97.95%) – High mean Average Precision.
- **Precision:** 0.965 (96.5%) – High accuracy in identifying pneumonia cases.
- **Recall:** 0.972 (97.2%) – Excellent sensitivity in capturing actual positive cases.
- **Inference Speed:** 2.26ms/image – Optimized for real-time clinical diagnostic support.

Training reports such as `confusion_matrix.png` and `results.png` can be found in the `reports/` directory.

## 📁 Project Structure

```
medical-lesion-detection/
│
├── data/                       # Dataset directory
│   ├── train/                  # Training set (Images & .txt labels)
│   ├── valid/                  # Validation set
│   └── test/                   # Test set
│
├── models/                     # Model weights and configuration
│   ├── yolov8_best.pt          # Best trained YOLOv8 model
│   └── metadata.yaml           # Dataset metadata (Roboflow source)
│
├── notebooks/                  # Training experiments
│   └── training_notebook.ipynb # Jupyter notebook for model training
│
├── reports/                    # Performance visualizations
│   ├── confusion_matrix.png    # Classification performance
│   └── results.png             # Training/Validation loss & metrics
│
├── src/                        # Core source code
│   ├── __init__.py
│   ├── preprocess.py           # CLAHE-based image enhancement
│   └── predict.py              # Inference logic
│
├── app.py                      # Interactive Web UI (Streamlit)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### 1. Installation

Install all necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Web Dashboard

Experience the detection system in your browser:

```bash

python -m streamlit run app.py
```

## 🧠 Technical Highlights

- **Architecture**: Powered by YOLOv8 for balance between speed and accuracy.
- **Image Enhancement**: Implements **CLAHE** (Contrast Limited Adaptive Histogram Equalization) in `preprocess.py` to highlight lung details in X-Rays.
- **Dataset**: Integrated with Roboflow for managing medical image annotations.
- **Web Interface**: Built with Streamlit for seamless user interaction and real-time visualization.
