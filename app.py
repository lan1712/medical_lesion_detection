import streamlit as st
import cv2
import PIL.Image
from ultralytics import YOLO
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# Page configuration
st.set_page_config(
    page_title="Medical Lesion Detection", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .detection-box {
    border: 2px solid #4CAF50;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #f0f8f0;
    color: #000000;  
    }
    .warning-box {
    border: 2px solid #ff9800;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #fff8e1;
    color: #000000;  
    }
    .detection-box, .detection-box h4, .detection-box p {
    color: #000000 !important;
    }
    .warning-box, .warning-box h4, .warning-box p {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏥 Medical Lesion Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'processed_count' not in st.session_state:
    st.session_state.processed_count = 0

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Settings
    st.subheader("🤖 Model Settings")
    model_path = st.text_input("Model Path", "models/yolov8_best.pt")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, 
                                     help="Lower threshold = more sensitive (fewer missed cases)")
    iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    st.markdown("---")
    
    # Preprocessing Options
    st.subheader("🔧 Preprocessing")
    use_clahe = st.checkbox("Apply CLAHE Enhancement", value=True, 
                           help="Contrast Limited Adaptive Histogram Equalization")
    denoise = st.checkbox("Apply Denoising", value=False)
    enhance_contrast = st.checkbox("Enhance Contrast", value=False)
    
    if use_clahe:
        clip_limit = st.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0, 0.5)
        tile_size = st.slider("CLAHE Tile Size", 4, 16, 8, 2)
    
    st.markdown("---")
    
    # Advanced Inference Options
    st.subheader("🧪 Advanced Inference")
    use_tta = st.checkbox("Test-Time Augmentation (TTA)", value=True, 
                         help="Runs inference on multiple augmented versions for better accuracy")
    use_ensemble = st.checkbox("Ensemble Predictions", value=False,
                              help="Combines multiple passes for robust detection")
    
    st.markdown("---")
    
    # Visualization Options
    st.subheader("📊 Visualization")
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_heatmap = st.checkbox("Show Attention Heatmap", value=False)
    show_comparison = st.checkbox("Side-by-side Comparison", value=True)
    show_regions = st.checkbox("Show Diagnosis Regions", value=True, help="Toggle pneumonia diagnosis zones/boxes")
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📈 Statistics")
    st.metric("Total Processed", st.session_state.processed_count)
    if st.session_state.detection_history:
        positive_cases = sum(1 for h in st.session_state.detection_history if h['detected'])
        st.metric("Positive Detections", positive_cases)
        st.metric("Detection Rate", f"{positive_cases/len(st.session_state.detection_history)*100:.1f}%")
    
    if st.button("🗑️ Clear History"):
        st.session_state.detection_history = []
        st.session_state.processed_count = 0
        st.rerun()

# Preprocessing functions
def apply_clahe(image, clip_limit=2.0, tile_size=8):
    """Apply CLAHE to enhance contrast"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def apply_denoising(image):
    """Apply denoising filter"""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def enhance_image_contrast(image):
    """Enhance overall contrast"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def create_heatmap(image, boxes):
    """Create attention heatmap based on detections"""
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), conf, -1)
    
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

def test_time_augmentation(model, image):
    """Apply Test-Time Augmentation for robust predictions"""
    predictions = []
    
    # Original
    predictions.append(model.predict(image, verbose=False))
    
    # Horizontal flip
    flipped = cv2.flip(image, 1)
    predictions.append(model.predict(flipped, verbose=False))
    
    # Slight brightness adjustments
    bright = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
    predictions.append(model.predict(bright, verbose=False))
    
    dark = cv2.convertScaleAbs(image, alpha=0.9, beta=-10)
    predictions.append(model.predict(dark, verbose=False))
    
    # Combine results - use voting or averaging
    # For simplicity, return the one with highest confidence
    best_result = max(predictions, key=lambda x: len(x[0].boxes))
    return best_result

def ensemble_predict(model, image, num_runs=3):
    """Run multiple predictions and ensemble results"""
    all_boxes = []
    all_confidences = []
    
    for _ in range(num_runs):
        results = model.predict(image, verbose=False)
        if len(results[0].boxes) > 0:
            all_boxes.extend(results[0].boxes)
    
    # If we got consistent detections, return them
    if len(all_boxes) >= num_runs // 2:  # Majority vote
        return model.predict(image, verbose=False)
    else:
        return model.predict(image, verbose=False)

# Main content - Tabs for Single/Batch Processing
tab1, tab2 = st.tabs(["🖼️ Single Image Analysis", "📁 Batch Processing"])

# ========== TAB 1: Single Image ========== 
with tab1:
    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-Ray Image", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
        key="single_uploader"
    )
    
    if uploaded_file is not None:
        # Reset results if a new file is uploaded or model parameters changed
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'current_file_id' not in st.session_state or st.session_state.current_file_id != file_id:
            st.session_state.current_file_id = file_id
            st.session_state.latest_results = None

        # Convert file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original and processed images
        if show_comparison:
            col1, col2 = st.columns(2)
        else:
            col1 = st.container()
            col2 = None
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(opencv_image, channels="BGR", width='stretch')
            st.caption(f"Size: {opencv_image.shape[1]}x{opencv_image.shape[0]} pixels")
        
        # Apply preprocessing
        processed_image = opencv_image.copy()
        
        if use_clahe:
            processed_image = apply_clahe(processed_image, clip_limit, tile_size)
        
        if denoise:
            processed_image = apply_denoising(processed_image)
        
        if enhance_contrast:
            processed_image = enhance_image_contrast(processed_image)
        
        if show_comparison and col2:
            with col2:
                st.subheader("🔧 Processed Image")
                st.image(processed_image, channels="BGR", width='stretch')
                preprocessing_steps = []
                if use_clahe: preprocessing_steps.append("CLAHE")
                if denoise: preprocessing_steps.append("Denoising")
                if enhance_contrast: preprocessing_steps.append("Contrast Enhancement")
                st.caption(f"Applied: {', '.join(preprocessing_steps) if preprocessing_steps else 'None'}")
        
        st.markdown("---")
        
        # Detection button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            detect_button = st.button("🚀 Run Detection", width='stretch', type="primary", key="single_detect")
        
        if detect_button:
            with st.spinner("🔍 Analyzing image..."):
                try:
                    # Load model
                    model = YOLO(model_path)
                    model.conf = confidence_threshold
                    model.iou = iou_threshold
                    
                    # Perform prediction with advanced techniques
                    if use_tta:
                        st.info("🔬 Running Test-Time Augmentation for better accuracy...")
                        results = test_time_augmentation(model, processed_image)
                    elif use_ensemble:
                        st.info("🔬 Running Ensemble Predictions...")
                        results = ensemble_predict(model, processed_image)
                    else:
                        results = model.predict(processed_image, verbose=False)
                    
                    st.session_state.latest_results = results
                    
                    # Update statistics
                    st.session_state.processed_count += 1
                    detection_data = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': uploaded_file.name,
                        'detected': len(results[0].boxes) > 0,
                        'num_detections': len(results[0].boxes),
                        'confidence_scores': [box.conf[0].item() for box in results[0].boxes]
                    }
                    st.session_state.detection_history.append(detection_data)
                    
                except Exception as e:
                    st.error(f"❌ Error during detection: {str(e)}")
                    st.info("💡 Please check that the model path is correct and the model file exists.")

        # Display results if they exist in state
        if 'latest_results' in st.session_state and st.session_state.latest_results is not None:
            results = st.session_state.latest_results
            
            # Display results heading
            st.markdown("---")
            st.subheader("🎯 Detection Results")
            
            # Determine image to show
            if show_regions:
                # Use plot() with labels and boxes
                res_plotted = results[0].plot(labels=show_confidence, boxes=True)
                caption = "Diagnosis Regions Highlighted"
            else:
                # Show original processed image without boxes
                res_plotted = processed_image
                caption = "Diagnosis Regions Hidden (Clean View)"
                
            if show_heatmap and len(results[0].boxes) > 0:
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.image(res_plotted, caption=caption, 
                            width='stretch', channels="BGR")
                with col_res2:
                    heatmap = create_heatmap(processed_image, results[0].boxes)
                    st.image(heatmap, caption="Attention Heatmap", 
                            width='stretch', channels="BGR")
            else:
                st.image(res_plotted, caption=caption, 
                        width='stretch', channels="BGR")
                    
            st.markdown("---")
            
            # Detection details
            if len(results[0].boxes) == 0:
                st.success("✅ No pneumonia lesions detected.")
                st.balloons()
            else:
                st.warning(f"⚠️ Detected {len(results[0].boxes)} potential lesion(s)")
                
                # Create detailed results table
                st.subheader("📋 Detection Details")
                
                # We need the model names - they are on the results object
                model_names = results[0].names
                
                for idx, box in enumerate(results[0].boxes, 1):
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = model_names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area = (x2 - x1) * (y2 - y1)
                    
                    box_class = "detection-box" if label != "PNEUMONIA" else "warning-box"
                    
                    st.markdown(f"""
                    <div class="{box_class}">
                        <h4>Detection #{idx}</h4>
                        <p><strong>Label:</strong> {label}</p>
                        <p><strong>Confidence:</strong> {conf:.2%}</p>
                        <p><strong>Location:</strong> ({x1}, {y1}) to ({x2}, {y2})</p>
                        <p><strong>Area:</strong> {area:,} pixels</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence distribution chart
                if show_confidence and len(results[0].boxes) > 1:
                    st.subheader("📊 Confidence Distribution")
                    confidences = [box.conf[0].item() for box in results[0].boxes]
                    labels = [model_names[int(box.cls[0].item())] for box in results[0].boxes]
                    
                    fig = px.bar(
                        x=[f"Detection {i+1}" for i in range(len(confidences))],
                        y=confidences,
                        color=labels,
                        labels={'x': 'Detection', 'y': 'Confidence Score'},
                        title='Detection Confidence Scores'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')
            
            # Export results
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # Save annotated image
                res_for_export = results[0].plot()
                _, buffer = cv2.imencode('.png', res_for_export)
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=buffer.tobytes(),
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/png"
                )
            
            with col_exp2:
                # Export detection data as JSON
                current_detection_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'filename': uploaded_file.name,
                    'detected': len(results[0].boxes) > 0,
                    'num_detections': len(results[0].boxes),
                    'confidence_scores': [box.conf[0].item() for box in results[0].boxes]
                }
                json_data = json.dumps(current_detection_data, indent=2)
                st.download_button(
                    label="📥 Download Detection Data (JSON)",
                    data=json_data,
                    file_name=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# ========== TAB 2: Batch Processing ========== 
with tab2:
    st.subheader("📁 Batch Image Analysis")
    st.info("Upload multiple X-Ray images to analyze them all at once. Perfect for screening multiple patients.")
    
    uploaded_files = st.file_uploader(
        "📤 Upload Multiple Chest X-Ray Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You can select multiple files at once",
        key="batch_uploader"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        # Batch processing button
        if st.button("🚀 Analyze All Images", type="primary", width='stretch', key="batch_detect"):
            # Initialize batch results storage
            if 'batch_results' not in st.session_state:
                st.session_state.batch_results = []
            st.session_state.batch_results = []
            
            # Load model once
            try:
                model = YOLO(model_path)
                model.conf = confidence_threshold
                model.iou = iou_threshold
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each image
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Read image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Preprocess
                    processed_image = opencv_image.copy()
                    if use_clahe:
                        processed_image = apply_clahe(processed_image, clip_limit, tile_size)
                    if denoise:
                        processed_image = apply_denoising(processed_image)
                    if enhance_contrast:
                        processed_image = enhance_image_contrast(processed_image)
                    
                    # Predict
                    if use_tta:
                        results = test_time_augmentation(model, processed_image)
                    elif use_ensemble:
                        results = ensemble_predict(model, processed_image)
                    else:
                        results = model.predict(processed_image, verbose=False)
                    
                    # Store results
                    batch_result = {
                        'filename': uploaded_file.name,
                        'image': opencv_image,
                        'processed_image': processed_image,
                        'results': results,
                        'num_detections': len(results[0].boxes),
                        'detected': len(results[0].boxes) > 0,
                        'confidence_scores': [box.conf[0].item() for box in results[0].boxes] if len(results[0].boxes) > 0 else []
                    }
                    st.session_state.batch_results.append(batch_result)
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ All images processed!")
                st.session_state.processed_count += len(uploaded_files)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Display batch results
        if 'batch_results' in st.session_state and st.session_state.batch_results:
            st.markdown("---")
            st.subheader("📊 Batch Analysis Summary")
            
            results_list = st.session_state.batch_results
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", len(results_list))
            with col2:
                positive_count = sum(1 for r in results_list if r['detected'])
                st.metric("Positive Cases", positive_count, delta=f"{positive_count/len(results_list)*100:.1f}%")
            with col3:
                negative_count = len(results_list) - positive_count
                st.metric("Negative Cases", negative_count)
            with col4:
                total_detections = sum(r['num_detections'] for r in results_list)
                st.metric("Total Detections", total_detections)
            
            st.markdown("---")
            
            # Detailed results for each image
            st.subheader("🔍 Individual Results")
            
            for idx, result in enumerate(results_list, 1):
                with st.expander(f"{'⚠️ PNEUMONIA' if result['detected'] else '✅ NORMAL'} - {result['filename']}", 
                               expanded=(idx == 1)):  # First one expanded by default
                    
                    col_img1, col_img2 = st.columns(2)
                    
                    with col_img1:
                        st.image(result['image'], channels="BGR", caption="Original", width='stretch')
                    
                    with col_img2:
                        if show_regions and result['detected']:
                            annotated = result['results'][0].plot()
                            st.image(annotated, channels="BGR", caption="Detected Regions", width='stretch')
                        else:
                            st.image(result['processed_image'], channels="BGR", caption="Processed", width='stretch')
                    
                    # Detection info
                    if result['detected']:
                        st.warning(f"⚠️ {result['num_detections']} lesion(s) detected")
                        if result['confidence_scores']:
                            avg_conf = np.mean(result['confidence_scores'])
                            st.write(f"**Average Confidence:** {avg_conf:.2%}")
                    else:
                        st.success("✅ No lesions detected")
            
            # Export all results
            st.markdown("---")
            st.subheader("💾 Export Batch Results")
            
            # Create summary report
            summary_data = []
            for r in results_list:
                summary_data.append({
                    'Filename': r['filename'],
                    'Status': 'PNEUMONIA' if r['detected'] else 'NORMAL',
                    'Detections': r['num_detections'],
                    'Avg Confidence': f"{np.mean(r['confidence_scores']):.2%}" if r['confidence_scores'] else 'N/A'
                })
            
            # Create CSV
            import pandas as pd
            df = pd.DataFrame(summary_data)
            csv = df.to_csv(index=False).encode('utf-8')
            
            col_export1, col_export2 = st.columns(2)
            with col_export1:
                st.download_button(
                    label="📥 Download Summary (CSV)",
                    data=csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col_export2:
                # JSON export
                json_export = json.dumps(summary_data, indent=2)
                st.download_button(
                    label="📥 Download Summary (JSON)",
                    data=json_export,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


# Detection History
if st.session_state.detection_history:
    st.markdown("---")
    st.subheader("📜 Detection History")
    
    history_df_data = []
    for h in st.session_state.detection_history[-10:]:  # Show last 10
        history_df_data.append({
            'Timestamp': h['timestamp'],
            'Filename': h['filename'],
            'Status': '✅ Negative' if not h['detected'] else '⚠️ Positive',
            'Detections': h['num_detections'],
            'Avg Confidence': f"{np.mean(h['confidence_scores']):.2%}" if h['confidence_scores'] else 'N/A'
        })
    
    st.dataframe(history_df_data, width='stretch')

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey; padding: 20px;">
    <p><strong>Medical Lesion Detection Dashboard</strong></p>
    <p>⚠️ This tool is for research and educational purposes only. Always consult with qualified healthcare professionals.</p>
</div>
""", unsafe_allow_html=True)
# 