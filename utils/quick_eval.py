"""
Evaluation script for VAE models on MNIST. 
Supports single-model evaluation or batch processing of multiple snapshots.
"""

import torch
import sys
import os
import argparse

# 1. Add the parent directory to sys.path to enable absolute imports of models/utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.VAE import VAE
from evaluation import evaluate_vae
from load_data import load_static_mnist

# 2. Configure CLI arguments
parser = argparse.ArgumentParser(description='Evaluate trained VAE models.')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Path to a specific model directory or a root containing multiple snapshots.')
parser.add_argument('--several_models', action='store_true', default=False,
                    help='Flag to indicate model_dir contains multiple snapshot subdirectories.')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Force CPU evaluation even if CUDA is available.')
args_eval = parser.parse_args()

# 3. Register VAE class for safe torch.load (required for weights_only=False)
torch.serialization.add_safe_globals([VAE])

def get_model(model_dir):
    """Loads a VAE model from a saved .model file."""
    model_path = os.path.join(model_dir, 'vae.model')
    if not os.path.exists(model_path):
        print(f"Error: Could not find model file at {model_path}")
        return None
    model = torch.load(model_path, weights_only=False)
    return model

def _evaluate_model(model, model_dir, args_eval, train_loader, test_loader):
    """Executes the standard VAE evaluation pipeline for a single model."""
    model.eval()

    # 4. Move model to GPU if requested and available
    cuda = not args_eval.no_cuda and torch.cuda.is_available()
    if cuda:
        model.cuda()

    # 5. Define and create results directory inside the model's snapshot folder
    out_dir = os.path.join(model_dir, 'eval_results/')
    os.makedirs(out_dir, exist_ok=True)

    # 6. Run quantitative evaluation (ELBO, LL) and generate qualitative visualizations
    print(f"Running evaluation for model in: {model_dir}")
    evaluate_vae(model.args, model, train_loader, test_loader, 0, out_dir, mode="test")
    
    # 7. Explicitly clear VRAM to prevent accumulation during batch processing
    if cuda:
        del model
        torch.cuda.empty_cache()

# --- Main Execution Flow ---

if args_eval.several_models:
    # Get list of all subdirectories containing models
    model_dirs = [os.path.join(args_eval.model_dir, d) for d in os.listdir(args_eval.model_dir) 
                  if os.path.isdir(os.path.join(args_eval.model_dir, d))]
    
    train_loader, val_loader, test_loader = None, None, None

    # Process each model directory sequentially to manage CPU/GPU load
    for m_dir in model_dirs:
        model = get_model(m_dir)
        if model is None:
            continue
            
        # Initialize data loaders exactly once using the first model's configuration
        if train_loader is None:
            print("Initializing Data Loaders (Shared for all models)...")
            train_loader, val_loader, test_loader, _ = load_static_mnist(model.args)
            
        print(f"\n---> Starting Evaluation for: {m_dir}")
        _evaluate_model(model, m_dir, args_eval, train_loader, test_loader)

else:
    # Single model evaluation mode
    model = get_model(args_eval.model_dir)
    if model:
        print("Initializing Data Loaders...")
        train_loader, val_loader, test_loader, _ = load_static_mnist(model.args)
        _evaluate_model(model, args_eval.model_dir, args_eval, train_loader, test_loader)
