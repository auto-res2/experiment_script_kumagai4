#!/usr/bin/env python3
"""
Evaluation module for texture analysis experiments.
Contains functions for evaluating models and visualizing results.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import os
from PIL import Image
import glob
from torchvision import transforms

from preprocess import prepare_tensor, load_config, set_random_seed, load_natural_images, get_image_transform
from train import get_device, BaseMethod, NewMethod

def experiment1_synthetic_texture_test(base_model, new_model, textures, ground_truth, device, output_dir="."):
    """
    Run Experiment 1: Synthetic Texture Segregation Test.
    
    Args:
        base_model (nn.Module): Base Method model
        new_model (nn.Module): New Method model
        textures (list): List of synthetic textures
        ground_truth (list): List of ground truth values
        device (torch.device): Device to use for evaluation
        output_dir (str): Directory to save output files
        
    Returns:
        tuple: (mae_base, mae_new, p_val) Mean absolute errors and p-value
    """
    print("Running Experiment 1: Synthetic Texture Segregation Test")
    
    base_model.eval()
    new_model.eval()
    
    base_outputs = []
    new_outputs = []
    
    for i, (img, gt) in enumerate(zip(textures, ground_truth)):
        tensor_in = prepare_tensor(img).to(device)
        
        with torch.no_grad():
            pred_base = base_model(tensor_in)
            base_outputs.append(pred_base.item())
        
        with torch.no_grad():
            pred_new = new_model(tensor_in)
            new_outputs.append(pred_new.item())
        
        print(f"Texture {i+1}: Ground Truth = {gt:.4f}, Base Prediction = {pred_base.item():.4f}, New Prediction = {pred_new.item():.4f}")
    
    errors_base = np.abs(np.array(base_outputs) - np.array(ground_truth))
    errors_new = np.abs(np.array(new_outputs) - np.array(ground_truth))
    
    mae_base = np.mean(errors_base)
    mae_new = np.mean(errors_new)
    
    print(f"\nMean Absolute Error (Base Method): {mae_base:.4f}")
    print(f"Mean Absolute Error (New Method): {mae_new:.4f}")
    
    t_stat, p_val = ttest_rel(errors_base, errors_new)
    print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(textures)+1), errors_base, 'o-', label="Base Method Error")
    plt.plot(range(1, len(textures)+1), errors_new, 's-', label="New Method Error")
    plt.xlabel("Texture Sample Index")
    plt.ylabel("Absolute Error")
    plt.title("Error Comparison between Base and New Methods")
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "salience_error_comparison.pdf")
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()
    
    print(f"Error comparison plot saved to {pdf_path}")
    
    return mae_base, mae_new, p_val

def experiment2_frequency_visualization(new_model, device, config, output_dir="."):
    """
    Run Experiment 2: Adaptive Frequency Visualization and Analysis.
    
    Args:
        new_model (nn.Module): New Method model
        device (torch.device): Device to use for evaluation
        config (dict): Configuration parameters
        output_dir (str): Directory to save output files
        
    Returns:
        tuple: (learned_freqs, fixed_fourier_freqs) Learned and fixed frequencies
    """
    print("Running Experiment 2: Adaptive Frequency Visualization and Analysis")
    
    learned_freqs = new_model.sinusoid.freqs.detach().cpu().numpy()
    
    plt.figure(figsize=(8, 4))
    sns.histplot(learned_freqs.flatten(), bins=20, color="blue", label="Learned Frequencies", kde=False)
    plt.xlabel("Frequency Value")
    plt.ylabel("Count")
    plt.title("Distribution of Learned Sinusoidal Frequencies")
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "frequency_distribution_newmethod.pdf")
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()
    
    print(f"Frequency distribution plot saved to {pdf_path}")
    
    total_elements = learned_freqs.size
    fixed_fourier_freqs = np.linspace(0, 10, num=total_elements)
    
    plt.figure(figsize=(8, 4))
    sns.histplot(fixed_fourier_freqs, bins=20, color="green", label="Fixed Fourier Frequencies", kde=False)
    plt.xlabel("Frequency Value")
    plt.ylabel("Count")
    plt.title("Distribution of Fixed Fourier Frequencies")
    plt.legend()
    
    pdf_path = os.path.join(output_dir, "frequency_distribution_fixed_fourier.pdf")
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()
    
    print(f"Fixed Fourier frequency distribution plot saved to {pdf_path}")
    
    return learned_freqs, fixed_fourier_freqs

def experiment3_natural_image_test(base_model, new_model, device, config, output_dir="."):
    """
    Run Experiment 3: Natural Image Generalization Test.
    
    Args:
        base_model (nn.Module): Base Method model
        new_model (nn.Module): New Method model
        device (torch.device): Device to use for evaluation
        config (dict): Configuration parameters
        output_dir (str): Directory to save output files
        
    Returns:
        list: Results containing image paths and salience predictions
    """
    print("Running Experiment 3: Natural Image Generalization Test")
    
    preprocess = get_image_transform()
    
    image_dir = config.get("natural_images_dir", "data/natural_images")
    image_paths = load_natural_images(image_dir)
    
    if len(image_paths) == 0:
        print(f"No natural images found in '{image_dir}' directory. Skipping Experiment 3.")
        return []
    
    base_model.eval()
    new_model.eval()
    
    results = []
    
    def get_salience_map(model, image_tensor):
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))  # add batch dimension
        return output.item()
    
    for path in image_paths:
        try:
            image = Image.open(path).convert("L")  # convert to grayscale
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            continue
        
        tensor_img = preprocess(image).to(device)
        
        salience_base = get_salience_map(base_model, tensor_img)
        salience_new = get_salience_map(new_model, tensor_img)
        
        print(f"Image: {os.path.basename(path)} | Base Salience: {salience_base:.4f} | New Salience: {salience_new:.4f}")
        results.append((path, salience_base, salience_new))
    
    if len(results) == 0:
        print("No valid images were processed. Skipping visualization.")
        return results
    
    indices = np.arange(len(results))
    base_preds = [r[1] for r in results]
    new_preds = [r[2] for r in results]
    
    plt.figure(figsize=(10, 5))
    width = 0.35
    plt.bar(indices - width/2, base_preds, width, label="Base Method")
    plt.bar(indices + width/2, new_preds, width, label="New Method")
    plt.xlabel("Image Index")
    plt.ylabel("Predicted Salience")
    plt.title("Comparison of Salience Predictions on Natural Images")
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "natural_image_salience_comparison.pdf")
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()
    
    print(f"Natural image salience comparison plot saved to {pdf_path}")
    print("Experiment 3 completed on natural images.")
    
    return results

def run_all_experiments(config, output_dir="."):
    """
    Run all three experiments.
    
    Args:
        config (dict): Configuration parameters
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Results from all experiments
    """
    from preprocess import create_synthetic_dataset, create_sine_dataset
    from train import initialize_models, initialize_optimizers, train_sine_model
    
    set_random_seed(config.get("random_seed", 42))
    
    device = get_device(config)
    
    base_model, new_model = initialize_models(config, device)
    
    criterion, optimizer_base, optimizer_new = initialize_optimizers(base_model, new_model, config)
    
    textures, ground_truth = create_synthetic_dataset(config)
    
    sine_textures, frequencies = create_sine_dataset(config)
    
    print("Training New Method on sine textures for Experiment 2")
    train_sine_model(new_model, optimizer_new, criterion, sine_textures, frequencies, device, config)
    
    exp1_results = experiment1_synthetic_texture_test(base_model, new_model, textures, ground_truth, device, output_dir)
    
    exp2_results = experiment2_frequency_visualization(new_model, device, config, output_dir)
    
    exp3_results = experiment3_natural_image_test(base_model, new_model, device, config, output_dir)
    
    results = {
        "experiment1": {
            "mae_base": exp1_results[0],
            "mae_new": exp1_results[1],
            "p_value": exp1_results[2]
        },
        "experiment2": {
            "learned_freqs_shape": exp2_results[0].shape,
            "fixed_freqs_shape": exp2_results[1].shape
        },
        "experiment3": {
            "num_images": len(exp3_results)
        }
    }
    
    return results

if __name__ == "__main__":
    config = load_config()
    output_dir = "."
    
    results = run_all_experiments(config, output_dir)
    print("All experiments completed successfully.")
    print(f"Results: {results}")
