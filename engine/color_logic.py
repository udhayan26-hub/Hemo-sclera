import cv2
import numpy as np

def extract_mean_color(image, mask, target_channel='a'):
    """
    Extracts the mean color from a specified LAB channel within a masked region.
    
    Args:
        image: BGR numpy image array.
        mask: Binary mask indicating the region of interest.
        target_channel: 'a' for Red-Green (Eyelid), 'b' for Yellow-Blue (Sclera).
    """
    if image is None or mask is None:
        return None
        
    # Ensure mask is binary
    if len(mask.shape) >= 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    elif len(mask.shape) == 3 and mask.shape[2] == 4:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    elif len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    
    # Ensure dimensions match
    if binary_mask.shape[:2] != image.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    # Convert image to CIE L*a*b*
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    L, a, b = cv2.split(lab_image)
    
    # Select the target channel
    if target_channel == 'a':
        channel_data = a
    elif target_channel == 'b':
        channel_data = b
    elif target_channel == 'L':
        channel_data = L
    else:
        raise ValueError("target_channel must be 'L', 'a', or 'b'")
        
    # Calculate the mean only within the mask
    # cv2.mean returns a tuple, the first element is the mean of the first channel
    mean_val = cv2.mean(channel_data, mask=binary_mask)[0]
    
    return mean_val

def analyze_anemia_eyelid(image, mask):
    """
    Calculates the mean a* value of the eyelid to correlate with Hemoglobin.
    Lower a* = Paler Eyelid = Lower Hb.
    """
    return extract_mean_color(image, mask, target_channel='a')

def analyze_jaundice_sclera(image, mask):
    """
    Calculates the mean b* value of the sclera to correlate with Bilirubin/Jaundice risk.
    Higher b* = More Yellow = Higher Bilirubin / Jaundice Risk.
    """
    return extract_mean_color(image, mask, target_channel='b')

if __name__ == "__main__":
    print("Colorimetry physics module loaded.")
