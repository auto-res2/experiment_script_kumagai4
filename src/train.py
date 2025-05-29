#!/usr/bin/env python3
"""
Training module for texture analysis experiments.
Contains model definitions and training functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from preprocess import prepare_tensor, set_random_seed, load_config

class BaseMethod(nn.Module):
    """
    Base Method:
    A simple CNN that processes an image and outputs a single scalar (e.g., salience threshold).
    """
    def __init__(self):
        super(BaseMethod, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(8, 1)
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)

class SinusoidalEncoding(nn.Module):
    """
    Adaptive sinusoidal encoding layer.
    Uses learnable frequency and phase parameters.
    """
    def __init__(self, in_channels, embedding_size):
        super(SinusoidalEncoding, self).__init__()
        self.freqs = nn.Parameter(torch.randn(in_channels, embedding_size))
        self.phases = nn.Parameter(torch.randn(in_channels, embedding_size))
        
    def forward(self, x):
        x_expanded = x.unsqueeze(-1)  # shape: batch x channels x H x W x 1
        encoded = torch.sin(x_expanded * self.freqs.view(1, -1, 1, 1, self.freqs.shape[-1]) 
                              + self.phases.view(1, -1, 1, 1, self.phases.shape[-1]))
        encoded = torch.mean(encoded, dim=-1)
        return encoded

class NewMethod(nn.Module):
    """
    New Method:
    Starts with the sinusoidal encoding layer followed by similar CNN layers as BaseMethod.
    """
    def __init__(self, embedding_size=10):
        super(NewMethod, self).__init__()
        self.sinusoid = SinusoidalEncoding(in_channels=1, embedding_size=embedding_size)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # after encoding channels remain the same
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(8, 1)
        
    def forward(self, x):
        x_encoded = self.sinusoid(x)
        out = self.conv(x_encoded)
        out = out.view(out.size(0), -1)
        return self.fc(out)

def get_device(config):
    """
    Get the device to use for training.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        torch.device: Device to use for training
    """
    device_config = config.get("device", "auto")
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_config == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    return device

def initialize_models(config, device):
    """
    Initialize the Base and New Method models.
    
    Args:
        config (dict): Configuration parameters
        device (torch.device): Device to use for training
        
    Returns:
        tuple: (base_model, new_model) initialized models
    """
    embedding_size = config.get("embedding_size", 10)
    
    base_model = BaseMethod().to(device)
    new_model = NewMethod(embedding_size=embedding_size).to(device)
    
    return base_model, new_model

def initialize_optimizers(base_model, new_model, config):
    """
    Initialize optimizers for both models.
    
    Args:
        base_model (nn.Module): Base Method model
        new_model (nn.Module): New Method model
        config (dict): Configuration parameters
        
    Returns:
        tuple: (criterion, optimizer_base, optimizer_new) loss function and optimizers
    """
    learning_rate = config.get("learning_rate", 0.001)
    
    criterion = nn.MSELoss()
    optimizer_base = optim.Adam(base_model.parameters(), lr=learning_rate)
    optimizer_new = optim.Adam(new_model.parameters(), lr=learning_rate)
    
    return criterion, optimizer_base, optimizer_new

def train_model(model, optimizer, criterion, textures, ground_truth, device, config):
    """
    Train a model on synthetic textures.
    
    Args:
        model (nn.Module): Model to train
        optimizer (optim.Optimizer): Optimizer for the model
        criterion (nn.Module): Loss function
        textures (list): List of synthetic textures
        ground_truth (list): List of ground truth values
        device (torch.device): Device to use for training
        config (dict): Configuration parameters
        
    Returns:
        float: Final loss value
    """
    epochs = config.get("epochs", 20)
    
    model.train()
    final_loss = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for img, gt in zip(textures, ground_truth):
            tensor_in = prepare_tensor(img).to(device)
            gt_tensor = torch.tensor([[gt]], dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            pred = model(tensor_in)
            loss = criterion(pred, gt_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(textures)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        final_loss = avg_loss
    
    return final_loss

def train_sine_model(model, optimizer, criterion, sine_textures, frequencies, device, config):
    """
    Train a model on sine textures with known frequencies.
    
    Args:
        model (nn.Module): Model to train
        optimizer (optim.Optimizer): Optimizer for the model
        criterion (nn.Module): Loss function
        sine_textures (list): List of sine textures
        frequencies (list): List of frequencies
        device (torch.device): Device to use for training
        config (dict): Configuration parameters
        
    Returns:
        float: Final loss value
    """
    epochs = config.get("epochs", 20)
    
    model.train()
    final_loss = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for f, texture in zip(frequencies, sine_textures):
            tensor_in = prepare_tensor(texture).to(device)
            gt_val = np.mean(texture) / 255.0 + 0.1 * np.log(f + 1)
            gt_tensor = torch.tensor([[gt_val]], dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            pred = model(tensor_in)
            loss = criterion(pred, gt_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(sine_textures)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        final_loss = avg_loss
    
    return final_loss

def save_model(model, model_path):
    """
    Save a model to disk.
    
    Args:
        model (nn.Module): Model to save
        model_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def load_model(model, model_path, device):
    """
    Load a model from disk.
    
    Args:
        model (nn.Module): Model to load into
        model_path (str): Path to load the model from
        device (torch.device): Device to load the model onto
        
    Returns:
        nn.Module: Loaded model
    """
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found. Using untrained model.")
    
    return model

if __name__ == "__main__":
    from preprocess import create_synthetic_dataset, create_sine_dataset
    
    config = load_config()
    set_random_seed(config.get("random_seed", 42))
    
    device = get_device(config)
    base_model, new_model = initialize_models(config, device)
    criterion, optimizer_base, optimizer_new = initialize_optimizers(base_model, new_model, config)
    
    textures, ground_truth = create_synthetic_dataset(config)
    print(f"Training Base Method on {len(textures)} synthetic textures")
    train_model(base_model, optimizer_base, criterion, textures, ground_truth, device, config)
    
    sine_textures, frequencies = create_sine_dataset(config)
    print(f"Training New Method on {len(sine_textures)} sine textures")
    train_sine_model(new_model, optimizer_new, criterion, sine_textures, frequencies, device, config)
    
    save_model(base_model, "models/base_method/base_model.pth")
    save_model(new_model, "models/new_method/new_model.pth")
