import cv2
import numpy as np
import matplotlib.pyplot as plt

def isolate_sclera(image_path, eyebox=None, debug=False):
    """
    Isolate the sclera with high robustness for Jaundice.
    Uses YOLO eyebox for target area and HSV masking for yellow-spectrum tissue.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # 1. Target the Eye-Box area (provided by YOLO)
    if eyebox is not None:
        x1, y1, x2, y2 = eyebox.astype(int)
        # Constrain to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        ROI = img[y1:y2, x1:x2]
        
        # 2. Color Thresholding (Dual: White + Yellow)
        hsv_roi = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
        
        # Range for Yellowish Sclera (Jaundice)
        # Hue 20-45 captures yellow. Sat 20-255 captures actual pigment.
        lower_yellow = np.array([15, 30, 100])
        upper_yellow = np.array([45, 255, 255])
        
        # Range for White Sclera (Normal)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        
        mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
        
        # 3. Iris Exclusion (The Blackout)
        # Black out the center 40% of the ROI where the iris usually resides
        roi_h, roi_w = ROI.shape[:2]
        cx, cy = roi_w // 2, roi_h // 2
        bw, bh = int(roi_w * 0.2), int(roi_h * 0.2) # radius of center blackout
        cv2.circle(combined_mask, (cx, cy), max(bw, bh), 0, -1)
        
        # 4. Clean up noise
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Place ROI mask back into full image mask
        mask[y1:y2, x1:x2] = combined_mask
    else:
        # Fallback to old global logic if no YOLO box
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))

    # 5. Connected Components Filtering (Keep Eye-blob)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    if num_labels > 1:
        # Keep largest blob that isn't the background
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype('uint8')
        
    sclera_isolated = cv2.bitwise_and(img, img, mask=mask)
    return sclera_isolated, mask
    
    if debug:
        # Display intermediate steps
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[1].imshow(iris_mask, cmap='gray')
        axes[1].set_title('Iris Mask (Hough)')
        axes[2].imshow(white_mask, cmap='gray')
        axes[2].set_title('White Mask (HSV Threshold)')
        axes[3].imshow(cv2.cvtColor(sclera_isolated, cv2.COLOR_BGR2RGB))
        axes[3].set_title('Final Isolate Sclera')
        plt.show()

    return sclera_isolated, sclera_mask

if __name__ == "__main__":
    print("Sclera Segmentation Module Loaded.")
    # test_img = "Datasets/jaundice/train/Jaundice/sample.jpg"
    # if os.path.exists(test_img):
    #     isolate_sclera(test_img, debug=True)
