import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def torch_load_metrics(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.atleast_1d(data)
    except:
        return None

def get_clean_info(folder_name):
    """Extracts prior and weighting info from the folder name."""
    # Example: 2026-01-18_static_mnist_vae_vampprior_K500_weightedTrue_seed69
    parts = folder_name.split('_')
    
    # Identify Prior
    if 'vampprior' in folder_name:
        prior = "VampPrior"
    elif 'flow' in folder_name:
        prior = "Flow Prior"
    else:
        prior = "Standard Prior"
        
    # Identify Weighting
    weighting = "Unweighted"
    for p in parts:
        if p.startswith("weightedTrue"):
            weighting = "Weighted"
        elif p.startswith("weightedFalse"):
            weighting = "Unweighted"
            
    return prior, weighting

def create_separate_plots(snapshot_path):
    folder_name = os.path.basename(snapshot_path)
    prior, weighting = get_clean_info(folder_name)
    
    # Simple Title Prefix
    title_prefix = f"{prior} ({weighting})"
    
    eval_dir = os.path.join(snapshot_path, 'eval_results')
    os.makedirs(eval_dir, exist_ok=True)
    
    metrics_to_plot = {
        'loss': ('Total Loss (ELBO)', '#1f77b4'),
        're': ('Reconstruction Error', '#2ca02c'),
        'kl': ('KL Divergence', '#9467bd'),
        'log_likelihood': ('Log-Likelihood', '#d62728')
    }

    for suffix, (metric_name, color) in metrics_to_plot.items():
        train = torch_load_metrics(snapshot_path, f'vae.train_{suffix}')
        val = torch_load_metrics(snapshot_path, f'vae.val_{suffix}')
        test = torch_load_metrics(snapshot_path, f'vae.test_{suffix}')

        if train is None and val is None and test is None:
            continue

        plt.figure(figsize=(9, 5))
        
        if train is not None and train.size > 0:
            plt.plot(train, label='Train', color=color, linewidth=2, alpha=0.7)
        
        if val is not None and val.size > 0:
            plt.plot(val, label='Validation', color='orange', linestyle='--', linewidth=2)

        if test is not None and test.size > 0:
            if test.size == 1:
                plt.axhline(y=test[0], color='red', linestyle=':', label=f'Test: {test[0]:.2f}')
            else:
                plt.plot(test, label='Test', color='red', alpha=0.5)

        # Clean, simple title
        plt.title(f"{title_prefix}: {metric_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend(frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        save_path = os.path.join(eval_dir, f"plot_{suffix}.png")
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"    ✅ {title_prefix} -> plot_{suffix}.png")

# Execution loop
base_dir = 'snapshots'
if os.path.exists(base_dir):
    folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    for folder in folders:
        create_separate_plots(os.path.join(base_dir, folder))