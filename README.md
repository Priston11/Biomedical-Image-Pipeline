# Biomedical Image Analysis & Hybrid AI Pipeline

A modular, end-to-end Python pipeline for biomedical image processing, cell nuclei segmentation, deep learning evaluation, and structured clinical reporting using PyTorch, scikit-image, and local Vision-Language Models via Ollama.

---

## Repository Overview & Modular Architecture

* **`main.py`**: The core execution script that orchestrates the entire end-to-end pipeline from Tasks 1 through 4.
* **`preprocessing.py`**: Handles image loading, normalization, grayscale conversion, and Exploratory Data Analysis (EDA) histogram plotting.
* **`classical_cv.py`**: Implements classical computer vision routines, including Otsu thresholding, morphological filtering, and `scikit-image` region properties measurement.
* **`unet_model.py`**: Defines the lightweight PyTorch U-Net architecture, training loops, binary cross-entropy loss, and evaluation metrics (Dice coefficient and IoU).
* **`llm_handler.py`**: Manages local Ollama model interactions (`llama3.2-vision` and text models) utilizing strict prompt engineering and JSON schema parsing.
* **`dataset.py`**: Manages custom dataset loaders for training, validation, and unseen test splits.

---

## Core Prompt Texts Used in the Pipeline

Below are the exact prompt templates embedded within `llm_handler.py` to prompt the local Ollama LLMs and enforce rigid JSON schemas to prevent hallucinations.

### 1. Multimodal VLM Prompt (Task 1)
*Used with `llama3.2-vision` to objectively extract imaging metadata and guard against clinical hallucination.*

```text
You are an expert biomedical image analyst. Describe this image purely objectively. 
Do NOT attempt to diagnose any condition. If a feature is unclear, you MUST output "uncertain".
Output your response STRICTLY as a JSON object with exactly these keys:
- "modality": (e.g., fluorescence microscopy, ultrasound, MRI, x-ray, or uncertain)
- "tissue_type": (description of the dominant tissue or cells)
- "notable_features": (any distinct shapes, boundaries, or textures)
- "image_quality": (e.g., high, low, noisy, blurry)




2. Numbers-First LLM Summary Prompt (Task 2)
Used with the local text model (llama3.2) to summarize quantifiable geometric and spatial features extracted via classical computer vision.

You are a clinical data reporting assistant. You will be provided with statistical measurements extracted from a biomedical microscopy image. 
Summarize these exact numerical metrics objectively. Do not invent or extrapolate unprovided symptoms.
Provide your response in a clear clinical narrative format followed by a JSON metrics block containing:
- "object_count"
- "mean_area"
- "mean_eccentricity"
- "density_classification"

Pipeline Execution Instructions
Ensure you have your local Ollama server active with llama3.2 and llama3.2-vision pulled, along with your PyTorch environment configured. Run the entire pipeline directly from your terminal:

python3 main.py
