import torch
import numpy as np

def dummy_data_loader(num_batches=10, batch_size=32, seq_len=10, vocab_size=1000, device='cpu'):
    """Generate synthetic data for training/evaluation.
    
    Args:
        num_batches: Number of batches to generate
        batch_size: Number of samples per batch
        seq_len: Sequence length for each sample
        vocab_size: Size of the vocabulary
        device: Device to place tensors on ('cpu' or 'cuda')
        
    Yields:
        Tuple of (input_data, target) tensors
    """
    for _ in range(num_batches):
        input_data = torch.randint(0, vocab_size, (seq_len, batch_size)).to(device)
        target = torch.randint(0, vocab_size, (seq_len, batch_size)).to(device)
        yield input_data, target

def create_synthetic_input(seq_len=10, batch_size=32, vocab_size=1000, device='cpu'):
    """Create a single batch of synthetic input data.
    
    Args:
        seq_len: Sequence length for each sample
        batch_size: Number of samples in the batch
        vocab_size: Size of the vocabulary
        device: Device to place tensor on ('cpu' or 'cuda')
        
    Returns:
        Tensor of shape (seq_len, batch_size) with random integers
    """
    return torch.randint(0, vocab_size, (seq_len, batch_size)).to(device)
