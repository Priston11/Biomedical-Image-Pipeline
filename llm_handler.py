import base64
import json
import cv2
import ollama
import numpy as np

def image_to_base64(image_array):
    """Converts a numpy image array into a base64 string for the Ollama API."""
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    success, buffer = cv2.imencode('.png', image_array)
    if not success:
        raise ValueError("Could not encode image to PNG format.")
    return base64.b64encode(buffer).decode('utf-8')

def get_vlm_description(image_array, prompt, model="llava", force_json=False):
    """
    Sends an image and a prompt to the local Ollama vision model (llava).
    """
    img_b64 = image_to_base64(image_array)
    
    # If force_json is True, tell Ollama to strictly return JSON
    format_arg = 'json' if force_json else ''
    
    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [img_b64]
        }],
        format=format_arg
    )
    
    return response['message']['content']

def get_text_description(prompt, model="llama3.2", force_json=True):
    """
    Sends a text-only prompt to the local Ollama model (llama3.2).
    """
    format_arg = 'json' if force_json else ''
    
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        format=format_arg
    )
    
    return response['message']['content']