#!/usr/bin/env python3
"""
Preprocessing module for texture analysis experiments.
Contains functions for generating synthetic textures and preprocessing images.
"""

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import os
import json
import glob

def load_config(config_path="config/experiment_params/config.json"):
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def set_random_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def generate_texture(width, height, imbalance_factor=0.2):
    """
    Generate a synthetic grayscale texture with controlled dark/light imbalance.
    A rectangular (darker) region is inserted into the random texture.
    
    Args:
        width (int): Width of the texture
        height (int): Height of the texture
        imbalance_factor (float): Factor controlling the darkness of the inserted region
        
    Returns:
        numpy.ndarray: Generated texture as a grayscale image
    """
    base_texture = np.random.rand(height, width) * 255
    cv2.rectangle(base_texture, (width//4, height//4), (width//2, height//2),
                  int(255 * (1 - imbalance_factor)), -1)
    return base_texture.astype(np.uint8)

def extract_multiscale_patches(img, scales=[1.0, 0.5, 0.25]):
    """
    Extract multi-scale patches from an image using different scaling factors.
    
    Args:
        img (numpy.ndarray): Input image
        scales (list): List of scaling factors
        
    Returns:
        list: List of resized images at different scales
    """
    patches = []
    for s in scales:
        resized = cv2.resize(img, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        patches.append(resized)
    return patches

def generate_sine_texture(width, height, frequency):
    """
    Generate a sine texture (sinusoidal pattern) with a specified frequency.
    
    Args:
        width (int): Width of the texture
        height (int): Height of the texture
        frequency (float): Frequency of the sine wave
        
    Returns:
        numpy.ndarray: Generated sine texture as a grayscale image
    """
    x = np.linspace(0, 2 * np.pi * frequency, width)
    sine_row = (np.sin(x) * 0.5 + 0.5) * 255
    texture = np.tile(sine_row, (height, 1))
    return texture.astype(np.uint8)

def prepare_tensor(image):
    """
    Convert image (numpy.ndarray) to tensor, normalize, and add channel and batch dimensions.
    Assumes grayscale image.
    
    Args:
        image (numpy.ndarray): Input grayscale image
        
    Returns:
        torch.Tensor: Normalized tensor with added dimensions
    """
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return tensor

def get_image_transform():
    """
    Get torchvision transform for preprocessing natural images.
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()  # scales to [0, 1]
    ])

def load_natural_images(image_dir):
    """
    Load natural images from a directory.
    
    Args:
        image_dir (str): Directory containing natural images
        
    Returns:
        list: List of image paths
    """
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    if len(image_paths) == 0:
        print(f"No natural images found in '{image_dir}' directory.")
    return image_paths

def create_synthetic_dataset(config):
    """
    Create a synthetic dataset of textures with ground truth values.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        tuple: (textures, ground_truth) where textures is a list of synthetic textures
               and ground_truth is a list of corresponding ground truth values
    """
    batch_size = config.get("batch_size", 10)
    width, height = config.get("texture_size", [128, 128])
    imbalance_factor = config.get("imbalance_factor", 0.3)
    
    textures = []
    ground_truth = []
    
    for _ in range(batch_size):
        tex = generate_texture(width, height, imbalance_factor)
        textures.append(tex)
        gt_val = np.mean(tex) / 255.0
        ground_truth.append(gt_val)
    
    return textures, ground_truth

def create_sine_dataset(config):
    """
    Create a dataset of sine textures with known frequencies.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        tuple: (sine_textures, frequencies) where sine_textures is a list of sine textures
               and frequencies is a list of corresponding frequencies
    """
    width, height = config.get("texture_size", [128, 128])
    frequencies = config.get("frequencies", [3, 5, 7, 10])
    
    sine_textures = [generate_sine_texture(width, height, f) for f in frequencies]
    
    return sine_textures, frequencies

if __name__ == "__main__":
    config = load_config()
    set_random_seed(config.get("random_seed", 42))
    
    textures, ground_truth = create_synthetic_dataset(config)
    print(f"Generated {len(textures)} synthetic textures")
    
    sine_textures, frequencies = create_sine_dataset(config)
    print(f"Generated {len(sine_textures)} sine textures with frequencies: {frequencies}")
    
    image_paths = load_natural_images(config.get("natural_images_dir", "data/natural_images"))
    print(f"Found {len(image_paths)} natural images")
