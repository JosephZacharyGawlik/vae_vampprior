import torch
import torch.nn as nn
import math
from models.FlowPrior import FlowPrior

class VampFlowPrior(nn.Module):
    def __init__(self, encoder, pseudo_inputs, latent_dim, flow_layers, flow_hidden_dim, cuda=False, weighted=False):
        super().__init__()
        self.encoder = encoder  
        self.pseudo_inputs = nn.Parameter(pseudo_inputs)  # shape [C, input_dim]
        self.n_components = pseudo_inputs.size(0)
        self.latent_dim = latent_dim
        self.cuda = cuda
        self.weighted = weighted

        # ONE Single FlowPrior that warps the entire mixture space
        self.flow = FlowPrior(latent_dim, flow_hidden_dim, flow_layers)

        # Learnable mixture weights for the pseudo-input components
        if self.weighted:
            self.logits = nn.Parameter(torch.zeros(self.n_components))
        else:
            self.logits = None

    def log_prob(self, z):
        """
        Calculates log p(z) using the change of variables formula:
        log p(z) = log p(z0) + log |det J_inv|
        """
        MB = z.size(0)
        device = z.device

        # 1. Map z (complex) -> z0 (base space) using the inverse flow
        z0, log_det_jacobian = self.flow.inverse(z)

        # 2. Get pseudo-input parameters from the encoder
        q_mean, q_logvar = self.encoder(self.pseudo_inputs)
        
        # Numerical Stability: Clamp log-variance to prevent nans
        q_logvar = torch.clamp(q_logvar, min=-7.0, max=2.0)

        # 3. Compute log p(z0) under the VampPrior mixture
        # Broadcast z0: [MB, 1, L] and components: [1, C, L]
        z0_expand = z0.unsqueeze(1)
        means = q_mean.unsqueeze(0)
        logvars = q_logvar.unsqueeze(0)

        diff = (z0_expand - means) ** 2
        var = torch.exp(logvars) + 1e-6
        # log_p_z0_per_component shape: [MB, C]
        log_p_z0_components = -0.5 * (diff / var + logvars + math.log(2 * math.pi)).sum(-1)

        # 4. Mixture weights logic
        if self.weighted:
            log_weights = torch.log_softmax(self.logits, dim=0)
        else:
            log_weights = torch.full((self.n_components,), -math.log(self.n_components), device=device)

        # 5. LogSumExp over components: [MB, C] -> [MB]
        weighted_log_probs = log_p_z0_components + log_weights.unsqueeze(0)
        log_p_z0 = torch.logsumexp(weighted_log_probs, dim=1)

        # 6. Apply Change of Variables: log p(z) = log p(z0) + log_det
        return log_p_z0 + log_det_jacobian

    def sample(self, n_samples):
        """
        Generative path: Sample z0 from VampPrior mixture, then warp z0 -> z via forward flow.
        """
        device = self.pseudo_inputs.device
        
        # 1. Select which pseudo-input components to sample from
        if self.weighted:
            weights = torch.softmax(self.logits, dim=0)
        else:
            weights = torch.ones(self.n_components, device=device) / self.n_components
        
        # Choose n_samples indices from the C available components
        indices = torch.multinomial(weights, n_samples, replacement=True)
        
        # 2. Get the encoder distribution for those specific pseudo-inputs
        q_mean, q_logvar = self.encoder(self.pseudo_inputs)
        q_logvar = torch.clamp(q_logvar, min=-7.0, max=2.0)
        
        batch_mu = q_mean[indices]
        batch_logvar = q_logvar[indices]
        
        # 3. Sample z0 from the base mixture
        std = torch.exp(0.5 * batch_logvar)
        eps = torch.randn(n_samples, self.latent_dim, device=device)
        z0 = batch_mu + eps * std
        
        # 4. Warp z0 to the final latent space z using the FORWARD flow
        z, _ = self.flow.forward(z0)
        
        return z