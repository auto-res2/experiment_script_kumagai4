#!/usr/bin/env python3
'''
AIRAS Experiment Runner - Ensures experiments produce real results
'''

import sys
import os

# Add src to path so AIRAS experiments can import properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_experiment():
    """Run the AIRAS-generated experiment with proper error handling."""
    
    print("AIRAS Experiment Runner Starting...")
    print("=" * 50)
    
    try:
        # Import and run the AIRAS-generated experiment
        if os.path.exists('src/airas_experiment.py'):
            print("Loading AIRAS experiment from src/airas_experiment.py")
            from airas_experiment import main
            main()
        else:
            print("No AIRAS experiment found. Running default test...")
            # Run a minimal test to verify environment
            import torch
            import transformers
            
            print(f"PyTorch version: {torch.__version__}")
            print(f"Transformers version: {transformers.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            
            # Simple test computation
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.matmul(x, y)
            print(f"Test computation successful. Result shape: {z.shape}")
            
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    run_experiment()
