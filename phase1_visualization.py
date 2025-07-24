"""
Phase 1 visualization demo
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_model_comparison():
    """Show basic model architecture differences"""
    
    # Simulate model statistics
    models = ['Jetson\n(32-bit)', 'Akida\n(4-bit)']
    parameters = [1.2, 1.2]  # Same model, different precision
    memory_mb = [4.8, 0.6]   # 4-bit uses less memory
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Parameters comparison
    ax1.bar(models, parameters, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Parameters (M)')
    ax1.set_title('Model Parameters')
    
    # Memory comparison  
    ax2.bar(models, memory_mb, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Memory Usage')
    
    plt.tight_layout()
    plt.savefig('results/phase1_model_comparison.png', dpi=150)
    plt.show()
    print("✅ Model comparison plot saved")

def plot_aggregation_concept():
    """Visualize mixed-precision aggregation concept"""
    
    # Simulate weight values
    rounds = np.arange(1, 6)
    jetson_weights = np.array([0.5, 0.52, 0.48, 0.51, 0.49])
    akida_weights = np.array([0.4, 0.45, 0.42, 0.44, 0.43])  # Quantized, different values
    aggregated = (jetson_weights + akida_weights) / 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, jetson_weights, 'b-o', label='Jetson (32-bit)', linewidth=2)
    plt.plot(rounds, akida_weights, 'r-s', label='Akida (4-bit)', linewidth=2)
    plt.plot(rounds, aggregated, 'g-^', label='Aggregated', linewidth=2)
    
    plt.xlabel('FL Round')
    plt.ylabel('Weight Value')
    plt.title('Mixed-Precision Weight Aggregation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('results/phase1_aggregation_concept.png', dpi=150)
    plt.show()
    print("✅ Aggregation concept plot saved")

if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    print("📊 Generating Phase 1 visualizations...")
    plot_model_comparison()
    plot_aggregation_concept()
    print("✅ Phase 1 visualizations complete!")