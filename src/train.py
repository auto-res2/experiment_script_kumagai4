import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For saving figures as PDF without display
import matplotlib.pyplot as plt

class SampleTransformer(nn.Module):
    """A sample transformer-like model with configurable shape parameters."""
    
    def __init__(self, num_layers, width, num_heads, vocab_size=1000):
        """Initialize the transformer model.
        
        Args:
            num_layers: Number of transformer encoder layers
            width: Width of the model (embedding dimension)
            num_heads: Number of attention heads per layer
            vocab_size: Size of the vocabulary
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=width, nhead=num_heads) 
            for _ in range(num_layers)
        ])
        self.embedding = nn.Embedding(vocab_size, width)
        self.fc = nn.Linear(width, vocab_size)
    
    def forward(self, x):
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (sequence length, batch size)
            
        Returns:
            Output tensor of shape (sequence length, batch size, vocab_size)
        """
        # Ensure input is on the same device as model parameters
        device = next(self.parameters()).device
        x = x.to(device)
        
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)

def compute_gradient_variance(module):
    """Compute the average variance of gradients for a module.
    
    Args:
        module: PyTorch module to compute gradient variance for
        
    Returns:
        Average variance of gradients across all parameters
    """
    grad_vars = []
    for param in module.parameters():
        if param.grad is not None:
            grad_vars.append(param.grad.data.cpu().numpy().var())
    return np.mean(grad_vars) if grad_vars else 0.0

def train_epoch(model, optimizer, scaler, data_loader, use_dynamic_precision=True):
    """Train the model for one epoch.
    
    Args:
        model: PyTorch model to train
        optimizer: PyTorch optimizer
        scaler: GradScaler for mixed precision training
        data_loader: Data loader yielding (input, target) pairs
        use_dynamic_precision: Whether to use dynamic precision assignment
        
    Returns:
        Average loss over the epoch
    """
    model.train()
    losses = []
    criterion = nn.CrossEntropyLoss()
    GRAD_VARIANCE_THRESHOLD = 1e-4
    
    for batch in data_loader:
        optimizer.zero_grad()
        input_data, target = batch
        
        if use_dynamic_precision:
            device_type = 'cuda' if next(model.parameters()).device.type == 'cuda' else 'cpu'
            with autocast(device_type):
                x = model.embedding(input_data)
            for layer in model.layers:
                with autocast(device_type):
                    x = layer(x)
            with autocast(device_type):
                output = model.fc(x)
        else:
            with autocast('cuda' if next(model.parameters()).device.type == 'cuda' else 'cpu'):
                output = model(input_data)
        
        loss = criterion(output.view(-1, output.shape[-1]), target.view(-1))
        losses.append(loss.item())
        scaler.scale(loss).backward()

        if use_dynamic_precision:
            for idx, layer in enumerate(model.layers):
                grad_var = compute_gradient_variance(layer)
                if grad_var > GRAD_VARIANCE_THRESHOLD:
                    print(f"    [Dynamic] Layer {idx} gradient variance {grad_var:.6f} exceeds threshold.")
        
        scaler.step(optimizer)
        scaler.update()
    
    return np.mean(losses)

def dynamic_mixed_precision_experiment(model, device='cpu'):
    """Run the mixed-precision fine-tuning experiment.
    
    Args:
        model: PyTorch model to use
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (static_loss, dynamic_loss)
    """
    from preprocess import dummy_data_loader
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler('cuda' if device == 'cuda' else 'cpu')
    
    data_loader_static = list(dummy_data_loader(device=device))
    data_loader_dynamic = list(dummy_data_loader(device=device))

    print("  Training with static mixed precision ...")
    loss_static = train_epoch(model, optimizer, scaler, data_loader_static, use_dynamic_precision=False)
    print(f"    Static precision average loss: {loss_static:.4f}")

    print("  Training with dynamic mixed precision ...")
    loss_dynamic = train_epoch(model, optimizer, scaler, data_loader_dynamic, use_dynamic_precision=True)
    print(f"    Dynamic precision average loss: {loss_dynamic:.4f}")
    
    methods = ['Static', 'Dynamic']
    losses = [loss_static, loss_dynamic]
    plt.figure(figsize=(5, 4))
    plt.bar(methods, losses, color=['green', 'orange'])
    plt.ylabel('Average Training Loss')
    plt.title('Static vs. Dynamic Mixed Precision Training')
    plt.savefig("logs/training_loss_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("  Training loss comparison plot saved as 'logs/training_loss_comparison.pdf'.")
    
    return loss_static, loss_dynamic
