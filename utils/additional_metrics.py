"""
Here, we define additional evaluation metrics according to the following sources:

Active Units (AU):
    - Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). 
    Importance Weighted Autoencoders. International Conference on Learning Representations (ICLR).
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_active_units(model, data_loader, out_dir, device, threshold=0.01):
    model.eval()
    all_means = []
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            # TODO: what does my model output?
            # For HVAE, we look at the top-level latent encoder (z2)
            if hasattr(model, 'q_z'):
                mu, _ = model.q_z(x)
            # Fallback for standard VAE
            elif hasattr(model, 'q_z2'):
                mu, _ = model.q_z2(x)
            else:
                # If it's a different naming convention
                continue
            all_means.append(mu)
            
    # Stack all means: [Total_Samples, Latent_Dim]
    all_means = torch.cat(all_means, dim=0)
    
    # The paper's "Cov_x" for a single dimension is simply the Variance
    # across the dataset samples.
    means_variance = torch.var(all_means, dim=0)
    
    # Count how many dimensions exceed the threshold
    active_units = (means_variance > threshold).sum().item()

    # --- Visualization ---
    means_variance = means_variance.cpu().numpy()

    plt.figure(figsize=(10, 4))
    colors = ["#2e6dcc" if v > threshold else '#e74c3c' for v in means_variance]
    plt.bar(range(len(means_variance)), means_variance, color=colors)
    plt.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    plt.yscale('log')
    plt.title(f'Latent Activity (Active: {active_units}/{len(means_variance)})')
    plt.xlabel('Latent Dim')
    plt.ylabel('Variance of Posterior Means')
    
    # Save plot to the snapshot directory if possible
    # We grab the directory from the model's args if saved, or use current
    plot_path = os.path.join(out_dir, 'latent_activity.png')
    plt.savefig(plot_path)
    plt.close()

    # --- Table Logging ---
    log_file = os.path.join(out_dir, 'au_summary_metrics.csv')
    row = {
        'Prior': [model.args.prior],
        'Active_Units': [active_units],
        'Z_Dim': [len(means_variance)],
        'Avg_Var': [np.mean(means_variance)]
    }
    df = pd.DataFrame(row)
    df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    
    return active_units, means_variance
