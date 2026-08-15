import numpy as np
import pandas as pd
from skimage import filters, morphology, measure, color

def classical_segmentation_and_features(gray_image):
    """
    Applies Otsu thresholding, cleans noise, labels objects, 
    and extracts region properties safely.
    """
    # Ensure image is strictly 2D grayscale
    if len(gray_image.shape) == 3:
        gray_image = color.rgb2gray(gray_image)
    
    # 1. Otsu Thresholding
    thresh = filters.threshold_otsu(gray_image)
    binary = gray_image > thresh
    
    # 2. Morphological cleanup (compatible with all skimage versions)
    clean_mask = morphology.remove_small_objects(binary, min_size=30)
    clean_mask = morphology.remove_small_holes(clean_mask, area_threshold=10)
    
    # 3. Label connected components
    labeled = measure.label(clean_mask)
    
    # 4. Extract region properties (features)
    properties = ['label', 'area', 'eccentricity', 'solidity', 'mean_intensity']
    props_table = measure.regionprops_table(
        labeled, 
        intensity_image=gray_image, 
        properties=properties
    )
    
    df_features = pd.DataFrame(props_table)
    return clean_mask, df_features

def create_text_summary(df_features):
    """Converts the pandas feature table into a numbers-first text summary."""
    n_objects = len(df_features)
    if n_objects == 0:
        return "No objects detected."
    
    mean_area = df_features['area'].mean()
    mean_ecc = df_features['eccentricity'].mean()
    mean_solidity = df_features['solidity'].mean()
    
    summary = (
        f"Detected {n_objects} distinct objects. "
        f"The average area is {mean_area:.2f} pixels. "
        f"The average eccentricity (elongation) is {mean_ecc:.2f}. "
        f"The average solidity is {mean_solidity:.2f}."
    )
    return summary