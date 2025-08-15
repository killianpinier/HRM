import torch
from hrm_act import HRM, HRM_Config
from act_loss import ACTHeadLoss

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

def generate_full_sudoku():
    """Generates a full valid 9x9 Sudoku grid using backtracking."""
    grid = np.zeros((9, 9), dtype=int)

    def is_valid(num, row, col):
        # Check row, column, and 3x3 box
        if num in grid[row, :]:
            return False
        if num in grid[:, col]:
            return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[start_row:start_row+3, start_col:start_col+3]:
            return False
        return True

    def solve(cell=0):
        if cell == 81:
            return True
        row, col = divmod(cell, 9)
        if grid[row, col] != 0:
            return solve(cell + 1)
        nums = list(range(1, 10))
        random.shuffle(nums)
        for num in nums:
            if is_valid(num, row, col):
                grid[row, col] = num
                if solve(cell + 1):
                    return True
                grid[row, col] = 0
        return False

    solve()
    return grid

def make_puzzle(full_grid, n_remove=40):
    """Remove n_remove numbers to create a puzzle."""
    puzzle = full_grid.copy()
    indices = list(range(81))
    random.shuffle(indices)
    for idx in indices[:n_remove]:
        row, col = divmod(idx, 9)
        puzzle[row, col] = 0
    return puzzle

def print_sudoku(grid):
    """
    Pretty-print a 9x9 Sudoku grid.
    Zeros are displayed as dots for empty cells.
    """
    for i, row in enumerate(grid):
        row_str = ""
        for j, val in enumerate(row):
            if val == 0:
                row_str += ". "
            else:
                row_str += f"{val} "
            # Add vertical separators for 3x3 blocks
            if (j + 1) % 3 == 0 and j < 8:
                row_str += "| "
        print(row_str)
        # Add horizontal separators for 3x3 blocks
        if (i + 1) % 3 == 0 and i < 8:
            print("- " * 11)

class SudokuDataset(Dataset):
    def __init__(self, n_samples=1000, n_remove=40):
        puzzles, solutions = [], []
        for _ in range(n_samples):
            full = generate_full_sudoku()          # (9, 9) numpy int array, values 1..9
            puzzle = make_puzzle(full, n_remove)   # (9, 9) numpy int array, zeros for blanks
            puzzles.append(puzzle)
            solutions.append(full)

        puzzles = np.array(puzzles, dtype=np.int64)    # (N, 9, 9)
        solutions = np.array(solutions, dtype=np.int64)

        # Flatten to (N, 81)
        puzzles_1d = puzzles.reshape(len(puzzles), 81)
        solutions_1d = solutions.reshape(len(solutions), 81)

        # Labels: map 1..9 -> 0..8, and ignore blanks (where puzzle==0) with -100
        labels_1d = solutions_1d - 1                   # 0..8
        labels_1d = np.where(puzzles_1d == 0, -100, labels_1d)

        self.inputs = torch.from_numpy(puzzles_1d)     # (N, 81), ints 0..9
        self.labels = torch.from_numpy(labels_1d)      # (N, 81), ints -100 or 0..8

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],   # (81,)
            "labels": self.labels[idx],   # (81,)
        }

device="mps"

# Assume SudokuDataset is defined as before
dataset = SudokuDataset(n_samples=1008)
dataset.inputs = dataset.inputs.to(device)
dataset.labels = dataset.labels.to(device)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from hrm_act import HRM, HRM_Config
from act_loss import ACTHeadLoss
import os
import logging
from uuid import uuid4

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_hrm(model: HRM, dataloader: DataLoader, config: dict, device: str = 'cuda'):
    """
    Training loop for HRM on Sudoku dataset.
    
    Args:
        model: Initialized HRM model
        dataloader: DataLoader with batches containing 'inputs' and 'labels'
        config: Dict with training hyperparameters (epochs, lr, weight_decay, etc.)
        device: Device to train on ('cuda' or 'cpu')
    """
    # Move model to device
    model = model.to(device)
    model.train()
    
    # Initialize loss and optimizer
    loss_fn = ACTHeadLoss(model)
    optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=config.get('weight_decay', 1.0))
    
    # Training config
    epochs = config.get('epochs', 20000)
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    log_interval = config.get('log_interval', 100)
    grad_accum_steps = config.get('grad_accum_steps', 1)
    checkpoint_interval = config.get('checkpoint_interval', 1000)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize metrics tracking
    total_steps = 0
    running_loss = 0.0
    running_metrics = {'count': 0, 'accuracy': 0.0, 'exact_accuracy': 0.0, 'q_halt_accuracy': 0.0, 
                      'steps': 0.0, 'lm_loss': 0.0, 'q_halt_loss': 0.0, 'q_continue_loss': 0.0}
    
    for epoch in range(epochs):
        # Initialize carry for the epoch (reset per epoch for simplicity)
        batch = next(iter(dataloader))  # Get sample batch for shape
        
        
        with torch.device(device):
            carry = loss_fn.initial_carry(batch)
        # carry = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in carry.__dict__.items()}
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            print(batch_idx)
            
            # Forward pass
            new_carry, loss, metrics, outputs, all_halted = loss_fn(carry, batch, return_keys=['logits', 'q_halt_logits', 'q_continue_logits'])
            
            # Backward pass with gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update carry for next iteration
            carry = new_carry
            
            # Aggregate metrics
            total_steps += 1
            running_loss += loss.item() * grad_accum_steps
            for k, v in metrics.items():
                running_metrics[k] += v.item()
            
            # Log metrics periodically
            if total_steps % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_metrics = {k: v / log_interval for k, v in running_metrics.items()}
                logger.info(f"Epoch {epoch+1}/{epochs}, Step {total_steps}, Loss: {avg_loss:.4f}, "
                           f"Acc: {avg_metrics['accuracy']:.4f}, Exact Acc: {avg_metrics['exact_accuracy']:.4f}, "
                           f"Q-Halt Acc: {avg_metrics['q_halt_accuracy']:.4f}, Avg Steps: {avg_metrics['steps']:.2f}")
                running_loss = 0.0
                running_metrics = {k: 0.0 for k in running_metrics}
            
            # Save checkpoint
            if total_steps % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'hrm_checkpoint_{uuid4().hex}.pth')
                torch.save({
                    'epoch': epoch,
                    'step': total_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Stop if all samples halted (optional, for debugging)
            if all_halted and config.get('early_stop_on_halt', False):
                logger.info(f"All samples halted at step {total_steps}")
                break
        
        # Reset carry at end of epoch to avoid state leakage
        carry = loss_fn.initial_carry(batch)
        carry = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in carry.__dict__.items()}
    
    logger.info("Training completed!")
    return model

if __name__ == '__main__':
    # Example configuration for Sudoku
    hrm_config = HRM_Config(
        H_cycles=2,
        L_cycles=5,
        H_layers=4,
        L_layers=4,
        seq_len=81,  # 9x9 Sudoku grid
        vocab_size=10,  # 0-9 digits
        hidden_size=512,
        num_heads=8,
        pos_encodings='rope',
        halt_max_steps=20,
        halt_exploration_prob=0.2,
        rms_norm_eps=1e-5,
        rope_theta=10000.0
    )
    
    # Initialize model
    model = HRM(hrm_config, device)
    
    # Example DataLoader (user must provide)
    # dataloader = DataLoader(sudoku_dataset, batch_size=32, shuffle=True)
    
    # Training config
    train_config = {
        'epochs': 1000,
        'lr': 1e-4,
        'weight_decay': 1.0,
        'log_interval': 100,
        'grad_accum_steps': 1,
        'checkpoint_interval': 1000,
        'checkpoint_dir': './checkpoints',
        'early_stop_on_halt': False
    }
    
    # Run training (uncomment when dataloader is ready)
    train_hrm(model, dataloader, train_config, device='mps')