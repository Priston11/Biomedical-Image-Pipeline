import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_preprocess(image_path, target_size=(256, 256), is_mask=False):
    """
    Loads an image, converts it to grayscale, and resizes it.
    If is_mask is True, returns binary values strictly scaled between 0.0 and 1.0.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image at: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if is_mask:
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_NEAREST)
        # Scale strictly to 0.0 and 1.0 for PyTorch loss functions
        return (resized > 127).astype(np.float32)
    else:
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        return resized

def plot_eda(image_paths, num_samples=3):
    """Displays sample images and their pixel histograms."""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 7))

    for i in range(num_samples):
        path = image_paths[i]
        img = load_and_preprocess(path)

        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"{path.name}")
        axes[0, i].axis("off")

        axes[1, i].hist(img.ravel(), bins=256, range=(0, 256), color="blue", alpha=0.7)
        axes[1, i].set_title("Histogram")

    plt.tight_layout()
    plt.show()