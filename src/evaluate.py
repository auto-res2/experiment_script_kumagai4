import time
import torch
from torch.cuda.amp import autocast
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
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    xs = [key[0] for key in results.keys()]
    ys = [key[1] for key in results.keys()]
    zs = [results[key] for key in results.keys()]
    scatter = ax.scatter(xs, ys, zs, c='purple', marker='o')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Concurrent Workers')
    ax.set_zlabel('Avg. Inference Latency (sec)')
    ax.set_title('Inference Latency vs. Concurrency Conditions')
    plt.savefig("logs/inference_latency_vs_concurrency.pdf", bbox_inches="tight")
    plt.close()
    print("  Inference latency plot saved as 'logs/inference_latency_vs_concurrency.pdf'.")
    
    return results
