import time
import torch
from torch.amp import autocast
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For saving figures as PDF without display
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

def inference_call(model, input_batch):
    """Perform a single inference call with the model.
    
    Args:
        model: PyTorch model to use for inference
        input_batch: Input tensor to the model
        
    Returns:
        Latency in seconds for the inference call
    """
    with torch.no_grad():
        start = time.perf_counter()
        with autocast('cuda' if next(model.parameters()).device.type == 'cuda' else 'cpu'):
            _ = model(input_batch)
        end = time.perf_counter()
    return end - start  # latency in seconds

def simulate_inference(model, batch_size, num_requests, num_workers, device='cpu'):
    """Simulate inference under concurrency conditions.
    
    Args:
        model: PyTorch model to use for inference
        batch_size: Batch size for each inference call
        num_requests: Number of inference requests to simulate
        num_workers: Number of concurrent workers
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        List of latencies for each inference call
    """
    from preprocess import create_synthetic_input
    
    latencies = []
    seq_len = 10
    vocab_size = 1000
    
    input_batch = create_synthetic_input(
        seq_len=seq_len, 
        batch_size=batch_size, 
        vocab_size=vocab_size, 
        device=device
    )
    
    model.to(device)
    model.eval()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(inference_call, model, input_batch) for _ in range(num_requests)]
        for future in as_completed(futures):
            latencies.append(future.result())
    
    return latencies

def inference_efficiency_experiment(model, device='cpu'):
    """Run the inference efficiency experiment under varied concurrency conditions.
    
    Args:
        model: PyTorch model to use for inference
        device: Device to run inference on ('cpu' or 'cuda')
    """
    print("\n[Experiment 3] Inference Efficiency Under Varied Concurrency Conditions")
    
    batch_sizes = [16, 32, 64]
    concurrency_levels = [1, 4, 8]
    results = {}
    
    for bs in batch_sizes:
        for workers in concurrency_levels:
            latencies = simulate_inference(
                model, 
                batch_size=bs, 
                num_requests=20, 
                num_workers=workers, 
                device=device
            )
            avg_latency = np.mean(latencies)
            results[(bs, workers)] = avg_latency
            print(f"  Batch size: {bs}, Workers: {workers}, Average Latency: {avg_latency:.4f} sec")
    
    plt.figure(figsize=(10, 8))
    
    batch_sizes_unique = sorted(list(set([key[0] for key in results.keys()])))
    workers_unique = sorted(list(set([key[1] for key in results.keys()])))
    
    latency_matrix = np.zeros((len(batch_sizes_unique), len(workers_unique)))
    for i, bs in enumerate(batch_sizes_unique):
        for j, workers in enumerate(workers_unique):
            if (bs, workers) in results:
                latency_matrix[i, j] = results[(bs, workers)]
    
    plt.imshow(latency_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Avg. Inference Latency (sec)')
    
    plt.xlabel('Concurrent Workers')
    plt.ylabel('Batch Size')
    plt.xticks(range(len(workers_unique)), workers_unique)
    plt.yticks(range(len(batch_sizes_unique)), batch_sizes_unique)
    
    for i in range(len(batch_sizes_unique)):
        for j in range(len(workers_unique)):
            plt.text(j, i, f"{latency_matrix[i, j]:.4f}", 
                     ha="center", va="center", color="white")
    
    plt.title('Inference Latency vs. Concurrency Conditions')
    plt.savefig("logs/inference_latency_vs_concurrency.pdf", bbox_inches="tight")
    plt.close()
    print("  Inference latency plot saved as 'logs/inference_latency_vs_concurrency.pdf'.")
    
    return results
