"""
QLoRA (Quantized Low-Rank Adaptation) Demonstration
===================================================

This script demonstrates QLoRA: combining LoRA with quantization to further
reduce memory usage while fine-tuning large language models.

Key Concept:
-----------
QLoRA = LoRA + 4-bit Quantization of base model weights

Instead of storing pretrained weights in float32 (32 bits), QLoRA:
1. Quantizes base model weights to 4-bit integers (8x memory reduction)
2. Adds LoRA adapters in float32 for training
3. Uses "double quantization" and "paged optimizers" for efficiency

Benefits Over LoRA:
------------------
1. 8x less memory for base weights (4-bit vs 32-bit)
2. Can fine-tune 65B models on single 48GB GPU
3. Minimal accuracy loss despite aggressive quantization
4. Same parameter efficiency as LoRA

Example:
-------
For a 8×8 weight matrix with rank r=2:
    - Original (float32):     64 params × 32 bits = 2,048 bits
    - LoRA (float32):         32 params × 32 bits = 1,024 bits
    - QLoRA (base 4-bit):     64 params × 4 bits + 32 params × 32 bits = 1,280 bits
    - Memory saving vs Original: 37.5% reduction
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def quantize_to_4bit(weight, scale=None):
    """
    Simulate 4-bit quantization of weight matrix.

    Process:
    -------
    1. Find min/max values in weight matrix
    2. Map values to 16 levels (4-bit = 2^4 = 16 possible values)
    3. Store quantization scale for dequantization

    Args:
        weight: Float32 weight tensor
        scale: Optional pre-computed scale factor

    Returns:
        quantized: 4-bit quantized values (stored as int8)
        scale: Scale factor for dequantization
    """
    # Calculate min and max for quantization range
    min_val = weight.min()
    max_val = weight.max()

    # Calculate scale factor (map to 4-bit range: 0-15)
    if scale is None:
        scale = (max_val - min_val) / 15.0

    # Quantize: map float values to 0-15 range
    quantized = torch.clamp(
        torch.round((weight - min_val) / scale),
        0, 15
    ).to(torch.int8)

    return quantized, scale, min_val


def dequantize_from_4bit(quantized, scale, min_val):
    """
    Dequantize 4-bit values back to float32.

    Args:
        quantized: 4-bit quantized values (as int8)
        scale: Scale factor from quantization
        min_val: Minimum value from original weight

    Returns:
        Dequantized float32 tensor
    """
    return quantized.float() * scale + min_val


class QLoRALayer:
    """
    QLoRA Layer: Combines 4-bit quantized base weights with float32 LoRA adapters.

    Architecture:
    ------------
    Base model:  W_base (quantized to 4-bit, frozen)
    LoRA adapt:  ΔW = A @ B (float32, trainable)
    Forward:     output = (W_base_dequantized + A @ B) @ input

    Memory Efficiency:
    -----------------
    - Base weights: 4 bits per parameter (vs 32 bits)
    - LoRA weights: 32 bits per parameter (only r×(d+k) params)
    - Total memory << traditional fine-tuning
    """

    def __init__(self, base_weight, rank):
        """
        Initialize QLoRA layer with quantized base and LoRA adapters.

        Args:
            base_weight: Pretrained weight matrix (will be quantized)
            rank: LoRA rank for low-rank decomposition
        """
        self.input_dim, self.output_dim = base_weight.shape
        self.rank = rank

        # Quantize base weights to 4-bit (frozen, not trainable)
        self.base_quantized, self.scale, self.min_val = quantize_to_4bit(base_weight)

        # Initialize LoRA adapters (trainable)
        self.lora_A = torch.randn(self.input_dim, rank) / np.sqrt(rank)
        self.lora_B = torch.zeros(rank, self.output_dim)

        # Enable gradients only for LoRA parameters
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def get_full_weight(self):
        """
        Reconstruct full weight: W = W_base_dequantized + LoRA_delta

        Returns:
            Combined weight matrix (dequantized base + LoRA update)
        """
        # Dequantize base weights
        base_dequantized = dequantize_from_4bit(
            self.base_quantized,
            self.scale,
            self.min_val
        )

        # Add LoRA update
        lora_delta = self.lora_A @ self.lora_B

        return base_dequantized + lora_delta

    def forward(self, x):
        """
        Forward pass through QLoRA layer.

        Args:
            x: Input tensor

        Returns:
            Output after QLoRA transformation
        """
        full_weight = self.get_full_weight()
        return full_weight @ x

    def memory_usage(self):
        """
        Calculate memory usage for different components.

        Returns:
            Dictionary with memory breakdown
        """
        base_4bit = self.input_dim * self.output_dim * 4  # 4 bits per param
        lora_32bit = self.rank * (self.input_dim + self.output_dim) * 32  # 32 bits
        total_bits = base_4bit + lora_32bit

        return {
            'base_quantized_bits': base_4bit,
            'lora_bits': lora_32bit,
            'total_bits': total_bits,
            'base_quantized_kb': base_4bit / 8 / 1024,
            'lora_kb': lora_32bit / 8 / 1024,
            'total_kb': total_bits / 8 / 1024
        }


def create_pretrained_weights(input_dim=10, output_dim=10, seed=42):
    """
    Create simulated pretrained weights.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        seed: Random seed

    Returns:
        Pretrained weight tensor
    """
    torch.manual_seed(seed)
    return torch.randn(input_dim, output_dim)


def train_qlora(target_weight, qlora_layer, iterations=1000, learning_rate=0.01):
    """
    Train QLoRA adapters to match target weight.

    Training Process:
    ----------------
    1. Dequantize base weights (4-bit → float32)
    2. Add LoRA update: W_full = W_base + A @ B
    3. Compute loss vs target
    4. Backprop through LoRA only (base is frozen)
    5. Update A and B matrices

    Args:
        target_weight: Target weight to approximate
        qlora_layer: QLoRALayer instance
        iterations: Training iterations
        learning_rate: Learning rate

    Returns:
        Loss history during training
    """
    optimizer = torch.optim.Adam(
        [qlora_layer.lora_A, qlora_layer.lora_B],
        lr=learning_rate
    )

    loss_history = []

    print("Training QLoRA approximation...")
    print(f"{'Iteration':<12} {'Loss':<12} {'Improvement'}")
    print("-" * 50)

    for iteration in range(iterations):
        # Get full weight (dequantized base + LoRA)
        full_weight = qlora_layer.get_full_weight()

        # Compute loss
        loss = torch.nn.functional.mse_loss(full_weight, target_weight)

        # Optimize only LoRA parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


def visualize_qlora(original_weight, qlora_layer, loss_history, save_path=None):
    """
    Create comprehensive visualization of QLoRA concept.

    Visualization includes:
    1. Training loss curve
    2. Original weight matrix
    3. Quantized base weights (dequantized for viewing)
    4. QLoRA full reconstruction
    5. Quantization error analysis
    6. Memory comparison chart
    """
    fig = plt.figure(figsize=(20, 12))

    # Get different weight representations
    orig_np = original_weight.detach().numpy()
    base_dequantized = dequantize_from_4bit(
        qlora_layer.base_quantized,
        qlora_layer.scale,
        qlora_layer.min_val
    ).numpy()
    qlora_full = qlora_layer.get_full_weight().detach().numpy()

    quantization_error = orig_np - base_dequantized
    reconstruction_error = orig_np - qlora_full

    # 1. Training Loss Curve
    plt.subplot(3, 3, 1)
    plt.plot(range(0, len(loss_history)*100, 100), loss_history,
             linewidth=2, color='#2E86AB', marker='o')
    plt.title('QLoRA Training Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Training Iterations', fontsize=11)
    plt.ylabel('MSE Loss', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 2. Original Weight Matrix
    plt.subplot(3, 3, 2)
    im1 = plt.imshow(orig_np, cmap='RdBu_r', aspect='auto')
    plt.title('Original Weight (Float32)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, fraction=0.046)

    # 3. Quantized Base (4-bit, dequantized for display)
    plt.subplot(3, 3, 3)
    im2 = plt.imshow(base_dequantized, cmap='RdBu_r', aspect='auto')
    plt.title('4-bit Quantized Base (Dequantized)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, fraction=0.046)

    # 4. QLoRA Full Reconstruction
    plt.subplot(3, 3, 4)
    im3 = plt.imshow(qlora_full, cmap='RdBu_r', aspect='auto')
    plt.title('QLoRA Reconstruction (Base + LoRA)', fontsize=14, fontweight='bold')
    plt.colorbar(im3, fraction=0.046)

    # 5. Quantization Error
    plt.subplot(3, 3, 5)
    plt.hist(quantization_error.flatten(), bins=30, color='#F18F01',
             alpha=0.7, edgecolor='black', label='Quantization Error')
    plt.title('4-bit Quantization Error', fontsize=14, fontweight='bold')
    plt.xlabel('Error Value', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()

    # 6. Reconstruction Error
    plt.subplot(3, 3, 6)
    plt.hist(reconstruction_error.flatten(), bins=30, color='#06A77D',
             alpha=0.7, edgecolor='black', label='QLoRA Error')
    plt.title('QLoRA Reconstruction Error', fontsize=14, fontweight='bold')
    plt.xlabel('Error Value', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()

    # 7. Memory Usage Comparison
    plt.subplot(3, 3, 7)
    d, k = qlora_layer.input_dim, qlora_layer.output_dim
    r = qlora_layer.rank

    # Calculate memory in bits
    full_32bit = d * k * 32
    lora_32bit = (d * k + r * (d + k)) * 32  # base + lora
    qlora_bits = qlora_layer.memory_usage()['total_bits']

    methods = ['Full\nFloat32', 'LoRA\n(Float32)', 'QLoRA\n(4-bit+Float32)']
    memories = [full_32bit, lora_32bit, qlora_bits]
    colors = ['#F18F01', '#A23B72', '#06A77D']

    bars = plt.bar(methods, memories, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    plt.title('Memory Usage Comparison (bits)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Bits', fontsize=11)

    for bar, mem in zip(bars, memories):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(mem)}\n({mem/full_32bit*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # 8. Quantized Values Heatmap (actual 4-bit integers)
    plt.subplot(3, 3, 8)
    im4 = plt.imshow(qlora_layer.base_quantized.numpy(), cmap='viridis',
                     aspect='auto', vmin=0, vmax=15)
    plt.title('4-bit Quantized Values (0-15)', fontsize=14, fontweight='bold')
    plt.colorbar(im4, fraction=0.046, ticks=range(0, 16, 2))

    # 9. QLoRA Architecture Explanation
    plt.subplot(3, 3, 9)
    plt.axis('off')

    mem_info = qlora_layer.memory_usage()
    explanation = f"""
    QLoRA Architecture:

    Base Model (Frozen):
    ├─ Quantization:       4-bit (16 levels)
    ├─ Shape:              {d} × {k}
    ├─ Memory:             {mem_info['base_quantized_kb']:.4f} KB
    └─ Parameters:         {d*k} (frozen)

    LoRA Adapters (Trainable):
    ├─ Matrix A:           {d} × {r}
    ├─ Matrix B:           {r} × {k}
    ├─ Memory:             {mem_info['lora_kb']:.4f} KB
    └─ Parameters:         {r*(d+k)} (trainable)

    Total Memory:          {mem_info['total_kb']:.4f} KB

    Memory Reduction:
    ├─ vs Full (32-bit):   {(1 - qlora_bits/full_32bit)*100:.1f}%
    └─ vs LoRA (32-bit):   {(1 - qlora_bits/lora_32bit)*100:.1f}%

    Quality Metrics:
    ├─ Quant Error:        {np.abs(quantization_error).mean():.6f}
    ├─ QLoRA Error:        {np.abs(reconstruction_error).mean():.6f}
    └─ Correlation:        {pearsonr(orig_np.flatten(), qlora_full.flatten())[0]:.6f}
    """

    plt.text(0.05, 0.95, explanation, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.show()


def print_detailed_analysis(original_weight, qlora_layer):
    """
    Print detailed analysis of QLoRA quantization and reconstruction.
    """
    orig_np = original_weight.detach().numpy()
    base_dequant = dequantize_from_4bit(
        qlora_layer.base_quantized,
        qlora_layer.scale,
        qlora_layer.min_val
    ).numpy()
    qlora_full = qlora_layer.get_full_weight().detach().numpy()

    quant_error = orig_np - base_dequant
    recon_error = orig_np - qlora_full

    print("\n" + "="*70)
    print("DETAILED QLORA ANALYSIS")
    print("="*70)

    print("\n1. MATRIX COMPARISON (Full matrices)")
    print("-" * 70)
    print("\nOriginal Weight (Float32):")
    print(orig_np)
    print("\n4-bit Quantized Base (Dequantized):")
    print(base_dequant)
    print("\nQLoRA Reconstruction (Base + LoRA):")
    print(qlora_full)

    print("\n2. QUANTIZATION ANALYSIS")
    print("-" * 70)
    print("\n4-bit Quantized Values (0-15 integers):")
    print(qlora_layer.base_quantized.numpy())
    print(f"\nQuantization scale:      {qlora_layer.scale:.6f}")
    print(f"Minimum value:           {qlora_layer.min_val:.6f}")
    print(f"Dynamic range:           {qlora_layer.scale * 15:.6f}")

    print("\n3. MEMORY EFFICIENCY")
    print("-" * 70)
    d, k, r = qlora_layer.input_dim, qlora_layer.output_dim, qlora_layer.rank

    full_params = d * k
    lora_params = r * (d + k)

    full_mem_bits = full_params * 32
    lora_mem_bits = (full_params + lora_params) * 32
    qlora_mem = qlora_layer.memory_usage()

    print(f"Full fine-tuning:        {full_mem_bits:>10} bits ({full_mem_bits/8/1024:.4f} KB)")
    print(f"LoRA (32-bit):           {lora_mem_bits:>10} bits ({lora_mem_bits/8/1024:.4f} KB)")
    print(f"QLoRA (4-bit + 32-bit):  {qlora_mem['total_bits']:>10.0f} bits ({qlora_mem['total_kb']:.4f} KB)")
    print(f"\nMemory reduction vs Full: {(1 - qlora_mem['total_bits']/full_mem_bits)*100:>6.2f}%")
    print(f"Memory reduction vs LoRA: {(1 - qlora_mem['total_bits']/lora_mem_bits)*100:>6.2f}%")

    print("\n4. APPROXIMATION QUALITY")
    print("-" * 70)
    print("Quantization Error (4-bit only):")
    print(f"  Max error:             {np.abs(quant_error).max():>10.6f}")
    print(f"  Mean error:            {np.abs(quant_error).mean():>10.6f}")
    print(f"  Std error:             {np.abs(quant_error).std():>10.6f}")

    print("\nQLoRA Reconstruction Error (4-bit + LoRA):")
    print(f"  Max error:             {np.abs(recon_error).max():>10.6f}")
    print(f"  Mean error:            {np.abs(recon_error).mean():>10.6f}")
    print(f"  Std error:             {np.abs(recon_error).std():>10.6f}")

    corr_quant = pearsonr(orig_np.flatten(), base_dequant.flatten())[0]
    corr_qlora = pearsonr(orig_np.flatten(), qlora_full.flatten())[0]

    print(f"\nCorrelation (quantized): {corr_quant:>10.6f}")
    print(f"Correlation (QLoRA):     {corr_qlora:>10.6f}")

    print("\n" + "="*70)


def main():
    """
    Main QLoRA demonstration.

    Shows:
    1. 4-bit quantization of pretrained weights
    2. LoRA adapter initialization
    3. Training LoRA to compensate for quantization error
    4. Memory efficiency comparison
    5. Quality analysis
    """
    print("\n" + "="*70)
    print("QLoRA (Quantized Low-Rank Adaptation) Demonstration")
    print("="*70)
    print("\nWhat is QLoRA?")
    print("-" * 70)
    print("""
QLoRA combines two powerful techniques:
1. 4-bit quantization of pretrained model weights (8x memory reduction)
2. LoRA adapters for parameter-efficient fine-tuning

Key Innovation:
- Base model weights are quantized to 4-bit and frozen
- Only small LoRA adapters (A and B matrices) are trained in float32
- This allows fine-tuning 65B parameter models on a single consumer GPU!

The magic: Despite aggressive 4-bit quantization, QLoRA maintains accuracy
comparable to full 16-bit fine-tuning by learning LoRA adapters that
compensate for quantization errors.
    """)

    # Configuration
    input_dim = 8
    output_dim = 8
    rank = 2

    print(f"\nConfiguration:")
    print(f"  • Matrix size:          {input_dim} × {output_dim}")
    print(f"  • LoRA rank:            {rank}")
    print(f"  • Base quantization:    4-bit (16 levels)")
    print(f"  • LoRA precision:       32-bit float")

    # Create pretrained weights
    print("\n" + "="*70)
    print("STEP 1: Creating pretrained weights")
    print("-" * 70)
    pretrained_weights = create_pretrained_weights(input_dim, output_dim)
    print(f"✓ Created {input_dim}×{output_dim} weight matrix")

    # Initialize QLoRA layer
    print("\n" + "="*70)
    print("STEP 2: Initializing QLoRA layer")
    print("-" * 70)
    qlora = QLoRALayer(pretrained_weights, rank)
    print(f"✓ Quantized base to 4-bit: {qlora.base_quantized.shape}")
    print(f"✓ LoRA Matrix A: {qlora.lora_A.shape}")
    print(f"✓ LoRA Matrix B: {qlora.lora_B.shape}")

    mem_info = qlora.memory_usage()
    print(f"\nMemory breakdown:")
    print(f"  • Base (4-bit):         {mem_info['base_quantized_kb']:.4f} KB")
    print(f"  • LoRA (32-bit):        {mem_info['lora_kb']:.4f} KB")
    print(f"  • Total:                {mem_info['total_kb']:.4f} KB")

    # Train QLoRA
    print("\n" + "="*70)
    print("STEP 3: Training QLoRA adapters")
    print("-" * 70)
    loss_history = train_qlora(pretrained_weights, qlora, iterations=1000)

    # Analyze results
    print_detailed_analysis(pretrained_weights, qlora)

    # Visualize
    print("\n" + "="*70)
    print("STEP 4: Generating visualization...")
    print("-" * 70)
    visualize_qlora(
        pretrained_weights,
        qlora,
        loss_history,
        save_path='qlora_demonstration.png'
    )

    print("\n" + "="*70)
    print("QLoRA Demonstration Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("-" * 70)
    print("""
1. QLoRA = 4-bit base (64 params) + Float32 LoRA (32 params)
2. Memory: 8×8=64 → 256 bits (4-bit) + 1024 bits (LoRA) = 1280 bits total
3. Memory reduction: 37.5% vs full float32 (2048 bits)
4. LoRA adapters compensate for quantization errors during training
5. Enables fine-tuning massive models (65B+) on consumer hardware
6. Maintains quality close to full-precision fine-tuning
7. Real-world use: Fine-tune LLaMA, Mistral, etc. on single GPU
    """)


if __name__ == "__main__":
    main()
