"""
LoRA (Low-Rank Adaptation) Demonstration
========================================

This script demonstrates the core concept of LoRA: decomposing a large weight matrix
into two smaller matrices to reduce the number of trainable parameters.

Key Concept:
-----------
Instead of updating a full weight matrix W (d×k), LoRA decomposes the update into:
    ΔW = B × A
where:
    - A is a (d×r) matrix
    - B is an (r×k) matrix
    - r << min(d,k) is the rank (much smaller than d and k)

Benefits:
--------
1. Parameter Efficiency: Train only r(d+k) parameters instead of d×k
2. Memory Efficient: Significantly reduces GPU memory requirements
3. Faster Training: Fewer parameters to update during fine-tuning
4. Modularity: Can swap different LoRA adapters without changing base model

Example:
-------
For a 8×8 weight matrix with rank r=2:
    - Original parameters: 64
    - LoRA parameters: 32 (50% reduction!)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


class LoRALayer:
    """
    LoRA Layer implementation that decomposes weight updates into low-rank matrices.

    Mathematical Background:
    -----------------------
    Traditional fine-tuning updates: W' = W + ΔW, where ΔW is d×k
    LoRA updates: W' = W + B×A, where A is d×r and B is r×k (r << d,k)

    This reduces trainable parameters from d×k to r×(d+k)
    """

    def __init__(self, input_dim, output_dim, rank):
        """
        Initialize LoRA layer with low-rank decomposition.

        Args:
            input_dim (int): Input dimension (d)
            output_dim (int): Output dimension (k)
            rank (int): Rank of decomposition (r), controls parameter reduction
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # Initialize matrix A: (d × r) with scaled random values
        # Scaling by 1/sqrt(r) helps with training stability
        self.matrix_A = torch.randn(input_dim, rank) / np.sqrt(rank)

        # Initialize matrix B: (r × k) with zeros
        # Starting with zeros ensures ΔW = 0 initially (no change to pretrained weights)
        self.matrix_B = torch.zeros(rank, output_dim)

        # Enable gradient computation for training
        self.matrix_A.requires_grad_(True)
        self.matrix_B.requires_grad_(True)

    def forward(self, x):
        """
        Forward pass: compute (B @ A) @ x

        Args:
            x: Input tensor

        Returns:
            Output after LoRA transformation
        """
        return (self.matrix_B @ self.matrix_A) @ x

    def get_weight_delta(self):
        """
        Compute the weight update matrix ΔW = B @ A

        Returns:
            Tensor of shape (input_dim, output_dim)
        """
        return self.matrix_A @ self.matrix_B

    def parameter_count(self):
        """
        Calculate total trainable parameters in LoRA layer.

        Returns:
            int: Total number of parameters (r × (d + k))
        """
        return self.rank * (self.input_dim + self.output_dim)


def create_pretrained_weights(input_dim=512, output_dim=512, seed=42):
    """
    Simulate a pretrained weight matrix from a large language model.

    In real scenarios, this would be a layer from BERT, GPT, LLaMA, etc.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (input_dim, output_dim)
    """
    torch.manual_seed(seed)
    return torch.randn(input_dim, output_dim)


def train_lora(target_weight, lora_layer, iterations=1000, learning_rate=0.01):
    """
    Train LoRA matrices to approximate the target weight update.

    Training Process:
    ----------------
    1. Compute current LoRA weight: ΔW = B @ A
    2. Calculate loss: MSE between ΔW and target weight
    3. Backpropagate gradients through A and B
    4. Update A and B using optimizer

    Args:
        target_weight: The weight matrix to approximate
        lora_layer: LoRALayer instance
        iterations: Number of training steps
        learning_rate: Learning rate for Adam optimizer

    Returns:
        List of loss values during training
    """
    optimizer = torch.optim.Adam(
        [lora_layer.matrix_A, lora_layer.matrix_B],
        lr=learning_rate
    )

    loss_history = []

    print("Training LoRA approximation...")
    print(f"{'Iteration':<12} {'Loss':<12} {'Improvement'}")
    print("-" * 50)

    for iteration in range(iterations):
        # Forward pass: compute current LoRA weight
        current_weight = lora_layer.get_weight_delta()

        # Compute loss: Mean Squared Error
        loss = torch.nn.functional.mse_loss(current_weight, target_weight)

        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update parameters

        # Log progress
        if iteration % 100 == 0:
            loss_value = loss.item()
            loss_history.append(loss_value)

            if iteration == 0:
                print(f"{iteration:<12} {loss_value:<12.6f} {'baseline'}")
            else:
                improvement = ((loss_history[0] - loss_value) / loss_history[0]) * 100
                print(f"{iteration:<12} {loss_value:<12.6f} {improvement:>6.2f}%")

    return loss_history


def visualize_lora_concept(original_weight, lora_approximation, loss_history,
                           lora_layer, save_path=None):
    """
    Create comprehensive visualization of LoRA concept and results.

    Visualization includes:
    1. Training loss curve showing convergence
    2. Original weight matrix heatmap
    3. LoRA approximation heatmap
    4. Error distribution histogram
    5. Parameter comparison chart
    """
    fig = plt.figure(figsize=(18, 10))

    # Convert to numpy for visualization
    orig_np = original_weight.detach().numpy()
    lora_np = lora_approximation.detach().numpy()
    error = orig_np - lora_np

    # 1. Training Loss Curve
    plt.subplot(2, 3, 1)
    plt.plot(range(0, len(loss_history)*100, 100), loss_history,
             linewidth=2, color='#2E86AB', marker='o')
    plt.title('LoRA Training Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Training Iterations', fontsize=11)
    plt.ylabel('MSE Loss', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 2. Original Weight Matrix
    plt.subplot(2, 3, 2)
    im1 = plt.imshow(orig_np, cmap='RdBu_r', aspect='auto')
    plt.title('Original Weight Matrix (Pretrained)', fontsize=14, fontweight='bold')
    plt.xlabel('Output Dimension', fontsize=11)
    plt.ylabel('Input Dimension', fontsize=11)
    plt.colorbar(im1, fraction=0.046)

    # 3. LoRA Approximation
    plt.subplot(2, 3, 3)
    im2 = plt.imshow(lora_np, cmap='RdBu_r', aspect='auto')
    plt.title(f'LoRA Approximation (rank={lora_layer.rank})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Output Dimension', fontsize=11)
    plt.ylabel('Input Dimension', fontsize=11)
    plt.colorbar(im2, fraction=0.046)

    # 4. Error Distribution
    plt.subplot(2, 3, 4)
    plt.hist(error.flatten(), bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    plt.title('Approximation Error Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Error Value', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')

    # 5. Parameter Efficiency Comparison
    plt.subplot(2, 3, 5)
    d, k = lora_layer.input_dim, lora_layer.output_dim
    original_params = d * k
    lora_params = lora_layer.parameter_count()

    categories = ['Full\nFine-tuning', 'LoRA\nAdaptation']
    params = [original_params, lora_params]
    colors = ['#F18F01', '#06A77D']

    bars = plt.bar(categories, params, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    plt.title('Parameter Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Parameters', fontsize=11)

    # Add value labels on bars
    for bar, param in zip(bars, params):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}\n({param/original_params*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')

    # 6. LoRA Decomposition Illustration
    plt.subplot(2, 3, 6)
    plt.axis('off')

    # Create text explanation
    explanation = f"""
    LoRA Decomposition Explained:

    Full Matrix (W):        {d} × {k} = {original_params:,} params

    LoRA Decomposition:
    ├─ Matrix A:            {d} × {lora_layer.rank} = {d * lora_layer.rank:,} params
    ├─ Matrix B:            {lora_layer.rank} × {k} = {lora_layer.rank * k:,} params
    └─ Total:               {lora_params:,} params

    Parameter Reduction:    {(1 - lora_params/original_params)*100:.2f}%

    Memory Savings:         {(original_params - lora_params) * 4 / 1024:.2f} KB
                           (assuming float32)

    Approximation Quality:
    ├─ Max Error:          {np.abs(error).max():.6f}
    ├─ Mean Error:         {np.abs(error).mean():.6f}
    ├─ Std Error:          {np.abs(error).std():.6f}
    └─ Correlation:        {pearsonr(orig_np.flatten(), lora_np.flatten())[0]:.6f}
    """

    plt.text(0.1, 0.95, explanation, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.show()


def print_detailed_analysis(original_weight, lora_approximation, lora_layer):
    """
    Print detailed numerical analysis of LoRA approximation quality.
    """
    orig_np = original_weight.detach().numpy()
    lora_np = lora_approximation.detach().numpy()
    error = orig_np - lora_np

    print("\n" + "="*70)
    print("DETAILED LORA ANALYSIS")
    print("="*70)

    print("\n1. COMPLETE MATRIX DISPLAY (Full matrices)")
    print("-" * 70)
    print("\nOriginal Weight Matrix:")
    print(orig_np)
    print("\nLoRA Approximation:")
    print(lora_np)
    print("\nError Matrix:")
    print(error)

    print("\n2. PARAMETER EFFICIENCY")
    print("-" * 70)
    d, k = lora_layer.input_dim, lora_layer.output_dim
    r = lora_layer.rank
    original_params = d * k
    lora_params = r * (d + k)
    reduction = (1 - lora_params/original_params) * 100

    print(f"Original parameters:     {original_params:>10,} (full {d}×{k} matrix)")
    print(f"LoRA parameters:         {lora_params:>10,} ({d}×{r} + {r}×{k})")
    print(f"Parameter reduction:     {reduction:>10.2f}%")
    print(f"Compression ratio:       {original_params/lora_params:>10.2f}x")

    print("\n3. APPROXIMATION QUALITY")
    print("-" * 70)
    print(f"Maximum absolute error:  {np.abs(error).max():>10.6f}")
    print(f"Mean absolute error:     {np.abs(error).mean():>10.6f}")
    print(f"Error standard deviation:{np.abs(error).std():>10.6f}")
    print(f"Root mean square error:  {np.sqrt((error**2).mean()):>10.6f}")

    correlation, p_value = pearsonr(orig_np.flatten(), lora_np.flatten())
    print(f"\nPearson correlation:     {correlation:>10.6f}")
    print(f"P-value:                 {p_value:>10.2e}")

    # Frobenius norm comparison
    orig_norm = np.linalg.norm(orig_np, 'fro')
    error_norm = np.linalg.norm(error, 'fro')
    relative_error = error_norm / orig_norm

    print(f"\nOriginal matrix norm:    {orig_norm:>10.6f}")
    print(f"Error norm:              {error_norm:>10.6f}")
    print(f"Relative error:          {relative_error:>10.6f} ({relative_error*100:.4f}%)")

    print("\n4. MEMORY FOOTPRINT (float32)")
    print("-" * 70)
    original_memory = original_params * 4 / 1024  # KB
    lora_memory = lora_params * 4 / 1024  # KB

    print(f"Original memory:         {original_memory:>10.2f} KB")
    print(f"LoRA memory:             {lora_memory:>10.2f} KB")
    print(f"Memory saved:            {original_memory - lora_memory:>10.2f} KB")

    print("\n" + "="*70)


def main():
    """
    Main demonstration of LoRA concept and implementation.

    This demo shows:
    1. Creating a simulated pretrained weight matrix
    2. Initializing LoRA with low-rank decomposition
    3. Training LoRA to approximate the weight matrix
    4. Visualizing results and parameter efficiency
    """
    print("\n" + "="*70)
    print("LoRA (Low-Rank Adaptation) Demonstration")
    print("="*70)
    print("\nWhat is LoRA?")
    print("-" * 70)
    print("""
LoRA is a parameter-efficient fine-tuning technique that freezes pretrained
model weights and injects trainable low-rank decomposition matrices into
each layer of the transformer architecture.

Instead of updating all parameters in a weight matrix W (d×k), LoRA learns
a low-rank update: ΔW = B @ A, where:
    • A is a (d × r) matrix
    • B is a (r × k) matrix
    • r << min(d, k) is the rank

This dramatically reduces trainable parameters while maintaining model quality.
    """)

    # Configuration
    input_dim = 8      # Input dimension (d) - smaller for clearer visualization
    output_dim = 8     # Output dimension (k) - smaller for clearer visualization
    rank = 2           # LoRA rank (r) - the bottleneck dimension

    print(f"\nConfiguration:")
    print(f"  • Input dimension (d):  {input_dim}")
    print(f"  • Output dimension (k): {output_dim}")
    print(f"  • LoRA rank (r):        {rank}")

    # Step 1: Create pretrained weights
    print("\n" + "="*70)
    print("STEP 1: Creating simulated pretrained weights")
    print("-" * 70)
    pretrained_weights = create_pretrained_weights(input_dim, output_dim)
    print(f"✓ Created {input_dim}×{output_dim} weight matrix")

    # Step 2: Initialize LoRA
    print("\n" + "="*70)
    print("STEP 2: Initializing LoRA layer")
    print("-" * 70)
    lora = LoRALayer(input_dim, output_dim, rank)
    print(f"✓ Matrix A shape: {lora.matrix_A.shape} (input → bottleneck)")
    print(f"✓ Matrix B shape: {lora.matrix_B.shape} (bottleneck → output)")
    print(f"✓ Total LoRA parameters: {lora.parameter_count():,}")

    # Step 3: Train LoRA
    print("\n" + "="*70)
    print("STEP 3: Training LoRA to approximate weight matrix")
    print("-" * 70)
    loss_history = train_lora(pretrained_weights, lora, iterations=1000)

    # Step 4: Get final approximation
    final_approximation = lora.get_weight_delta()

    # Step 5: Analyze results
    print_detailed_analysis(pretrained_weights, final_approximation, lora)

    # Step 6: Visualize
    print("\n" + "="*70)
    print("STEP 4: Generating visualization...")
    print("-" * 70)
    visualize_lora_concept(
        pretrained_weights,
        final_approximation,
        loss_history,
        lora,
        save_path='lora_demonstration.png'
    )

    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("-" * 70)
    print("""
1. LoRA achieves 50% parameter reduction (8×8=64 → 2×16=32 params)
2. Matrix decomposition: W(8×8) ≈ A(8×2) @ B(2×8)
3. Training only affects small A and B matrices, not the full weight matrix
4. Multiple LoRA adapters can be trained for different tasks
5. Can be easily added/removed without modifying the base model
6. Ideal for fine-tuning large language models with limited compute resources
    """)


if __name__ == "__main__":
    main()
