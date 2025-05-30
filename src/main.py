import os
import time
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # For saving figures as PDF without display
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

from train import SampleTransformer, dynamic_mixed_precision_experiment
from evaluate import inference_efficiency_experiment
from preprocess import create_synthetic_input

def multi_objective_architecture_search():
    """Run the multi-objective architecture search experiment.
    
    Returns:
        The best candidate architecture parameters [num_layers, width, num_heads]
    """
    print("\n[Experiment 1] Multi-Objective Architecture Search")
    
    def evaluate_architecture(individual):
        num_layers, width, num_heads = individual
        # Ensure width is divisible by num_heads
        if width % num_heads != 0:
            # Adjust width to be divisible by num_heads
            width = (width // num_heads) * num_heads
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"    Evaluating model with layers={num_layers}, width={width}, heads={num_heads} on {device}")
        
        dummy_input = create_synthetic_input(device=device)
        input_device = dummy_input.device
        
        # Create model and explicitly move to the same device as input
        model = SampleTransformer(num_layers, width, num_heads)
        model = model.to(input_device)  # Reassign to ensure all submodules are moved
        
        model_device = next(model.parameters()).device
        if str(model_device) != str(input_device):
            print(f"    WARNING: Device mismatch detected! Model on {model_device}, input on {input_device}")
            model = model.to('cpu')
            dummy_input = dummy_input.to('cpu')
            print(f"    Forced both model and input to CPU device for compatibility")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_latency = time.perf_counter() - start_time
        
        training_cost = num_layers * width * num_heads  
        accuracy = 0.8 - (training_cost % 10)*0.005  
        
        return training_cost, inference_latency

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    valid_layers = [4, 6, 8]
    valid_widths = [128, 256, 512]
    valid_heads = [4, 8]  # Removed 16 to ensure width is divisible by num_heads
    
    toolbox.register("num_layers", random.choice, valid_layers)
    toolbox.register("width", random.choice, valid_widths)
    toolbox.register("num_heads", random.choice, valid_heads)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.num_layers, toolbox.width, toolbox.num_heads), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_architecture)
    toolbox.register("mate", tools.cxTwoPoint)
    
    def custom_mutate(individual, indpb):
        if random.random() < indpb:
            individual[0] = random.choice(valid_layers)  # Mutate num_layers
        if random.random() < indpb:
            individual[1] = random.choice(valid_widths)  # Mutate width
        if random.random() < indpb:
            individual[2] = random.choice(valid_heads)   # Mutate num_heads
        return individual,
    
    toolbox.register("mutate", custom_mutate, indpb=0.3)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=20)
    NGEN = 10

    fits = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fits):
        ind.fitness.values = fit
        
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
        population = toolbox.select(population + offspring, k=len(population))
        print(f"  Generation {gen+1} completed.")
    
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    print(f"  Pareto front (number of candidates): {len(pareto_front)}")
    
    valid_front = [ind for ind in pareto_front if hasattr(ind, 'fitness') and hasattr(ind.fitness, 'values') and len(ind.fitness.values) >= 2]
    print(f"  Valid candidates with fitness values: {len(valid_front)}")
    
    for i, ind in enumerate(valid_front):
        print(f"    Candidate {i+1}: layers={ind[0]}, width={ind[1]}, heads={ind[2]} -> cost={ind.fitness.values[0]:.1f}, latency={ind.fitness.values[1]:.4f}")
    
    training_costs = [ind.fitness.values[0] for ind in valid_front]
    latencies = [ind.fitness.values[1] for ind in valid_front]
    
    plt.figure(figsize=(6, 5))
    plt.scatter(training_costs, latencies, c='blue')
    plt.xlabel('Training Cost (proxy)')
    plt.ylabel('Inference Latency (sec)')
    plt.title('Pareto Front: Training Cost vs. Inference Latency')
    plt.savefig("logs/training_cost_vs_latency.pdf", bbox_inches="tight")
    plt.close()
    print("  Pareto front plot saved as 'logs/training_cost_vs_latency.pdf'.")

    best_candidate = valid_front[0] if valid_front else [6, 256, 4]  # Default fallback if no valid candidates
    return best_candidate  # [num_layers, width, num_heads]

def test_all():
    """Run all experiments in sequence for a quick test."""
    print("=" * 80)
    print("STARTING ADAPTIVE ARCHITECTURE + MIXED-PRECISION EXPERIMENTS")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("Default tensor type set to CUDA for consistency")
    else:
        print("Running on CPU")
        torch.set_default_tensor_type('torch.FloatTensor')
    print("=" * 80)
    
    os.makedirs("logs", exist_ok=True)
    print("Created logs directory for experiment outputs")
    
    with open("logs/status.txt", "w") as f:
        f.write("status_enum: running\n")
    print("Status set to 'running'")
    print("=" * 80)
    
    best_candidate = multi_objective_architecture_search()
    
    print("\n" + "=" * 80)
    print("[Experiment 2] Mixed-Precision Fine-Tuning with Dynamic Precision Assignment")
    print("=" * 80)
    print(f"Using best architecture from search: layers={best_candidate[0]}, width={best_candidate[1]}, heads={best_candidate[2]}")
    
    num_layers, width, num_heads = best_candidate
    model = SampleTransformer(num_layers, width, num_heads)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Creating model and moving to {device}")
    model.to(device)  # Ensure model is on the correct device
    dynamic_mixed_precision_experiment(model, device)
    
    print("\n" + "=" * 80)
    print("[Experiment 3] Inference Efficiency Under Varied Concurrency Conditions")
    print("=" * 80)
    print(f"Creating new model with same architecture: layers={best_candidate[0]}, width={best_candidate[1]}, heads={best_candidate[2]}")
    model = SampleTransformer(best_candidate[0], best_candidate[1], best_candidate[2])
    model.to(device)  # Ensure model is on the correct device
    inference_efficiency_experiment(model, device)
    
    with open("logs/status.txt", "w") as f:
        f.write("status_enum: stopped\n")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print("All experiments executed successfully:")
    print("  1. Multi-Objective Architecture Search - Complete")
    print("  2. Mixed-Precision Fine-Tuning - Complete")
    print("  3. Inference Efficiency Analysis - Complete")
    print("\nPDF outputs saved in the logs directory:")
    print("  - logs/training_cost_vs_latency.pdf")
    print("  - logs/training_loss_comparison.pdf")
    print("  - logs/inference_latency_vs_concurrency.pdf")
    print("\nStatus set to 'stopped'")
    print("=" * 80)

if __name__ == '__main__':
    test_all()
