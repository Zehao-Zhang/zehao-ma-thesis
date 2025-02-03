import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import Trainer
import math

def calculate_epsilon(num_updates, q, sigma, delta, alphas=[10, 100]):
    """
    Calculate the epsilon value for the given parameters.
    
    Parameters:
    - num_updates: Total number of updates
    - q: Sampling probability (batch_size / N)
    - sigma: Noise multiplier
    - delta: Privacy parameter delta
    - alphas: List of alpha values for RDP (Rényi Differential Privacy)
    
    Returns:
    - A dictionary containing epsilon values for each alpha
    """
    epsilons = {}
    for alpha in alphas:
        # RDP per update step
        rdp_per_step = (alpha * q**2) / (2 * sigma**2)
        # Total RDP accumulated over all updates
        total_rdp = num_updates * rdp_per_step
        # Convert to epsilon
        epsilon = (total_rdp - math.log(delta) / (alpha - 1)) / alpha
        epsilons[alpha] = epsilon
    return epsilons


def _seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


class dp_tabulaTrainer(Trainer):
    """ dp_tabula Trainer

    Adds DP-SGD functionality by clipping gradients and adding noise in each training step.
    """
    def __init__(self, *args, use_dp=False, clip_coeff=1.0, sigma=1.0, micro_batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dp = use_dp          # Flag to enable or disable differential privacy
        self.clip_coeff = clip_coeff  # Coefficient for gradient clipping
        self.sigma = sigma            # Standard deviation of noise for differential privacy
        self.micro_batch_size = micro_batch_size  # Micro-batch size for DP-SGD
        self.total_updates = 0

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_collator = self.data_collator
        train_dataset = self.train_dataset
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
        )


    def training_step(self, model, inputs, delta=1e-5):
        """
        Perform a training step with gradient accumulation and only add noise at the end of each full batch.
        """
        model.train()
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Clear gradients
        model.zero_grad()
        cumulative_loss = 0.0

        # Initialize cumulative gradients for each parameter
        cumulative_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

        # Loop through each micro-batch
        for i in range(0, self.args.per_device_train_batch_size, self.micro_batch_size):
            end_idx = min(i + self.micro_batch_size, self.args.per_device_train_batch_size)
            if end_idx <= i:
                continue

            # Get micro-batch data
            micro_batch = {k: v[i:end_idx] for k, v in inputs.items()}

            # Forward pass and loss calculation
            outputs = model(**micro_batch)
            micro_loss = outputs.loss / (self.args.per_device_train_batch_size / self.micro_batch_size)
            cumulative_loss += micro_loss.item()

            # Backward pass without clearing gradients
            micro_loss.backward()

            # Gradient clipping and accumulate gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_coeff)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    cumulative_grads[name] += param.grad

            model.zero_grad()  # Reset gradients for the next micro-batch

        # Only add noise after accumulating gradients over all micro-batches
        if self.use_dp:
            for name, param in model.named_parameters():
                if cumulative_grads[name] is not None:
                    noise = torch.randn_like(cumulative_grads[name]) * (self.sigma * self.clip_coeff)
                    param.grad = (cumulative_grads[name] + noise) / (self.args.per_device_train_batch_size / self.micro_batch_size)
                else:
                    param.grad = cumulative_grads[name]

        return torch.tensor(cumulative_loss, device=device, requires_grad=True)





