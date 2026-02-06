from ultralytics import YOLO
import cv2

def run_prediction(model_path, image_path):
    """
    Run YOLOv8 prediction on a single image.
    """
    # Load model
    model = YOLO(model_path)
    
    # Predict
    results = model.predict(image_path)
    
    # Return results
    return results

if __name__ == "__main__":
    # Example usage
    # results = run_prediction('models/yolov8_best.pt', 'data/raw/sample.jpg')
    pass
