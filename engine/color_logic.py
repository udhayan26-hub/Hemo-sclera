"""
Color science engine for Hemo-Sclera.
Calculates physiological signals (CIE L*a*b*) from patient tissue regions.
"""
import cv2
import numpy as np

def analyze_anemia_eyelid(image, mask):
    """
    Extracts the average a* channel value (Green -> Red axis) 
    from the palpebral conjunctiva region.
    """
    if image is None:
        return None
    
    # Convert BGR to CIE L*a*b*
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    a_channel = lab[:, :, 1]
    
    if mask is not None:
        # Ensure mask is 1-channel grayscale and matches image dimensions
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Calculate mean within the mask
        mean_val = cv2.mean(a_channel, mask=mask)[0]
        return mean_val
    
    return float(np.mean(a_channel))

def analyze_jaundice_sclera(image, mask):
    """
    Extracts the average b* channel value (Blue -> Yellow axis)
    from the sclera region.
    """
    if image is None:
        return None
    
    # Convert BGR to CIE L*a*b*
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    b_channel = lab[:, :, 2]
    
    if mask is not None:
        # Ensure mask is 1-channel grayscale and matches image dimensions
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Calculate mean within the mask
        mean_val = cv2.mean(b_channel, mask=mask)[0]
        return mean_val
    
    return float(np.mean(b_channel))

def extract_mean_color(image, mask, target_channel='b'):
    """Deprecated: Kept for legacy compatibility."""
    if target_channel == 'a':
        return analyze_anemia_eyelid(image, mask)
    return analyze_jaundice_sclera(image, mask)

def refine_tissue_mask(cropped_bgr_image):
    """Deprecated: Kept for legacy compatibility."""
    return None, 0.0, 0.0
