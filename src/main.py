#!/usr/bin/env python3
"""
Main orchestration script for texture analysis experiments.
Coordinates the entire experimental pipeline from preprocessing through evaluation.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import argparse

from preprocess import (
    load_config, 
    set_random_seed, 
    create_synthetic_dataset, 
    create_sine_dataset, 
    load_natural_images,
    generate_texture
)
from train import (
    get_device, 
    initialize_models, 
    initialize_optimizers, 
    train_model, 
    train_sine_model, 
    save_model
)
from evaluate import (
    experiment1_synthetic_texture_test, 
    experiment2_frequency_visualization, 
    experiment3_natural_image_test
)

def create_output_directories():
    """Create necessary output directories if they don't exist."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/base_method", exist_ok=True)
    os.makedirs("models/new_method", exist_ok=True)
    os.makedirs("data/synthetic_textures", exist_ok=True)
    os.makedirs("data/natural_images", exist_ok=True)

def ensure_natural_images_exist(config):
    """
    Ensure that some natural images exist for Experiment 3.
    If no images are found, create some synthetic ones.
    
    Args:
        config (dict): Configuration parameters
    """
    image_dir = config.get("natural_images_dir", "data/natural_images")
    os.makedirs(image_dir, exist_ok=True)
    
    image_paths = load_natural_images(image_dir)
    
    if len(image_paths) == 0:
        print(f"No natural images found in '{image_dir}'. Creating synthetic ones for testing.")
        
        width, height = config.get("texture_size", [128, 128])
        
        random_texture = generate_texture(width, height, 0.3)
        img_path = os.path.join(image_dir, "random_texture.jpg")
        Image.fromarray(random_texture).save(img_path)
        
        gradient = np.zeros((height, width), dtype=np.uint8)
        for i in range(width):
            gradient[:, i] = int(255 * i / width)
        img_path = os.path.join(image_dir, "gradient.jpg")
        Image.fromarray(gradient).save(img_path)
        
        checkerboard = np.zeros((height, width), dtype=np.uint8)
        square_size = 16
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    checkerboard[i:i+square_size, j:j+square_size] = 255
        img_path = os.path.join(image_dir, "checkerboard.jpg")
        Image.fromarray(checkerboard).save(img_path)
        
        print(f"Created 3 synthetic images in '{image_dir}'.")

def update_status_enum(config_path, status="stopped"):
    """
    Update the status_enum in the config file.
    
    Args:
        config_path (str): Path to the config file
        status (str): New status value
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config["status_enum"] = status
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Updated status_enum to '{status}' in {config_path}")

def run_experiment(config_path="config/experiment_params/config.json", output_dir="logs"):
    """
    Run the complete experimental pipeline.
    
    Args:
        config_path (str): Path to the config file
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Results from all experiments
    """
    start_time = time.time()
    
    create_output_directories()
    
    config = load_config(config_path)
    
    ensure_natural_images_exist(config)
    
    set_random_seed(config.get("random_seed", 42))
    
    device = get_device(config)
    
    base_model, new_model = initialize_models(config, device)
    
    criterion, optimizer_base, optimizer_new = initialize_optimizers(base_model, new_model, config)
    
    print("\nCreating synthetic dataset for Experiment 1...")
    textures, ground_truth = create_synthetic_dataset(config)
    
    for i, tex in enumerate(textures[:3]):  # Save first 3 textures
        img_path = os.path.join("data/synthetic_textures", f"texture_{i}.jpg")
        Image.fromarray(tex).save(img_path)
        print(f"Saved example texture to {img_path}")
    
    print("\nCreating sine dataset for Experiment 2...")
    sine_textures, frequencies = create_sine_dataset(config)
    
    for i, (tex, freq) in enumerate(zip(sine_textures, frequencies)):
        img_path = os.path.join("data/synthetic_textures", f"sine_texture_freq_{freq}.jpg")
        Image.fromarray(tex).save(img_path)
        print(f"Saved sine texture (freq={freq}) to {img_path}")
    
    print("\nTraining Base Method on synthetic textures...")
    train_model(base_model, optimizer_base, criterion, textures, ground_truth, device, config)
    
    save_model(base_model, "models/base_method/base_model.pth")
    
    print("\nTraining New Method on sine textures for Experiment 2...")
    train_sine_model(new_model, optimizer_new, criterion, sine_textures, frequencies, device, config)
    
    save_model(new_model, "models/new_method/new_model.pth")
    
    print("\n" + "="*80)
    exp1_results = experiment1_synthetic_texture_test(base_model, new_model, textures, ground_truth, device, output_dir)
    
    print("\n" + "="*80)
    exp2_results = experiment2_frequency_visualization(new_model, device, config, output_dir)
    
    print("\n" + "="*80)
    exp3_results = experiment3_natural_image_test(base_model, new_model, device, config, output_dir)
    
    results = {
        "experiment1": {
            "mae_base": exp1_results[0] if exp1_results else None,
            "mae_new": exp1_results[1] if exp1_results else None,
            "p_value": exp1_results[2] if exp1_results else None
        },
        "experiment2": {
            "learned_freqs_shape": exp2_results[0].shape if exp2_results else None,
            "fixed_freqs_shape": exp2_results[1].shape if exp2_results else None
        },
        "experiment3": {
            "num_images": len(exp3_results) if exp3_results else 0
        }
    }
    
    print("\n" + "="*80)
    print("Experiment Summary:")
    print(f"Experiment 1 - Synthetic Texture Segregation Test:")
    print(f"  Mean Absolute Error (Base Method): {results['experiment1']['mae_base']:.4f}")
    print(f"  Mean Absolute Error (New Method): {results['experiment1']['mae_new']:.4f}")
    print(f"  P-value: {results['experiment1']['p_value']:.4f}")
    print(f"Experiment 2 - Adaptive Frequency Visualization:")
    print(f"  Learned Frequencies Shape: {results['experiment2']['learned_freqs_shape']}")
    print(f"  Fixed Fourier Frequencies Shape: {results['experiment2']['fixed_freqs_shape']}")
    print(f"Experiment 3 - Natural Image Generalization Test:")
    print(f"  Number of Images Processed: {results['experiment3']['num_images']}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    
    update_status_enum(config_path, "stopped")
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run texture analysis experiments")
    parser.add_argument("--config", type=str, default="config/experiment_params/config.json",
                        help="Path to the configuration file")
    parser.add_argument("--output-dir", type=str, default="logs",
                        help="Directory to save output files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    print("Starting texture analysis experiments...")
    print(f"Using config file: {args.config}")
    print(f"Output directory: {args.output_dir}")
    
    results = run_experiment(args.config, args.output_dir)
    
    print("\nAll experiments completed successfully.")
