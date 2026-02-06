import cv2
import os

def apply_clahe(img):
    """
    Apply CLAHE to a grayscale or BGR image.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def preprocess_image(image_path, size=(640, 640)):
    """
    Read, enhance, and resize image.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_enhanced = apply_clahe(img)
    img_resized = cv2.resize(img_enhanced, size)
    return img_resized

def process_and_save(input_path, output_path, size=(640, 640)):
    """
    Process image and save to disk.
    """
    processed = preprocess_image(input_path, size)
    if processed is not None:
        cv2.imwrite(output_path, processed)
        return True
    return False

if __name__ == "__main__":
    # Example usage
    pass
