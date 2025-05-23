import torch
import numpy as np


class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """
    
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # Mimicking how compvis repo creates schedule
        self.betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        )
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the nosie predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        # Ensure correct broadcasting for all time-dependent coefficients
        def expand_coeff(coeff, t, shape):
            # coeff: (num_timesteps,)
            # t: (B,)
            # returns (B, 1, 1, 1, ...)
            out = coeff.to(xt.device)[t]
            for _ in range(len(shape) - 1):
                out = out.unsqueeze(-1)
            return out

        B = xt.shape[0]
        shape = xt.shape

        sqrt_one_minus_alpha_cum_prod = expand_coeff(self.sqrt_one_minus_alpha_cum_prod, t, shape)
        alpha_cum_prod = expand_coeff(self.alpha_cum_prod, t, shape)
        betas = expand_coeff(self.betas, t, shape)
        alphas = expand_coeff(self.alphas, t, shape)

        x0 = ((xt - (sqrt_one_minus_alpha_cum_prod * noise_pred)) / torch.sqrt(alpha_cum_prod))
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - (betas * noise_pred) / sqrt_one_minus_alpha_cum_prod
        mean = mean / torch.sqrt(alphas)

        # t can be a tensor, so check for t == 0 for each batch element
        if (t == 0).all():
            return mean, x0
        else:
            # For each batch, compute variance and sigma
            # For t > 0, t-1 is valid; for t==0, variance is not used
            t_minus_1 = torch.clamp(t - 1, min=0)
            alpha_cum_prod_tm1 = expand_coeff(self.alpha_cum_prod, t_minus_1, shape)
            variance = (1 - alpha_cum_prod_tm1) / (1.0 - alpha_cum_prod)
            variance = variance * betas
            sigma = variance ** 0.5
            z = torch.randn_like(xt)
            return mean + sigma * z, x0
