"""
Quick evaluation script on static MNIST using predefined and added metrics.
"""

import torch
import sys
import os
import argparse

# 1. Add the parent directory (cwd) to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.VAE import VAE
from evaluation import evaluate_vae

# 2. Setup Argument Parser
parser = argparse.ArgumentParser(description='Evaluate a trained VAE model.')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Path to the snapshot directory containing vae.model')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
args_eval = parser.parse_args()

# Tell PyTorch it's safe to load your VAE class
torch.serialization.add_safe_globals([VAE])

# 3. Construct paths and Load Model
model_path = os.path.join(args_eval.model_dir, 'vae.model')

if not os.path.exists(model_path):
    print(f"Error: Could not find model file at {model_path}")
    sys.exit(1)

model = torch.load(model_path, weights_only=False)
model.eval()

# Update cuda setting based on parser or model's own saved args
cuda = not args_eval.no_cuda and torch.cuda.is_available()
if cuda:
    model.cuda()

# 4. Setup Data Loaders
from load_data import load_static_mnist
train_loader, val_loader, test_loader, _ = load_static_mnist(model.args)

# 5. Create the output directory (inside the snapshot folder to avoid overwrites)
out_dir = os.path.join(args_eval.model_dir, 'eval_results/')
os.makedirs(out_dir, exist_ok=True)

# 6. Evaluate the model (ELBO, log-likelihood, reconstructions, generations, etc.)
evaluate_vae(model.args, model, train_loader, test_loader, 0, out_dir, mode="test")
