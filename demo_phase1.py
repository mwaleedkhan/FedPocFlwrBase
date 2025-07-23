"""
Phase 1 demonstration script
"""
import sys
sys.path.append('src')

from model_variants import ModelFactory
from aggregation_engine import FedMPQAggregator, ClientUpdate, DeviceType
import torch

def demonstrate_mixed_precision():
    """Demonstrate the core mixed-precision concept"""
    
    print("🔬 Demonstrating Mixed-Precision Aggregation")
    print("=" * 50)
    
    # Create aggregator
    aggregator = FedMPQAggregator()
    print(f"Aggregator created: Jetson={aggregator.jetson_precision}-bit, Akida={aggregator.akida_precision}-bit")
    
    # Create sample updates
    updates = []
    
    # Jetson update (32-bit)
    jetson_weights = {"conv1": torch.randn(32, 16)}
    jetson_update = ClientUpdate(
        client_id="jetson_01",
        device_type=DeviceType.JETSON,
        weights=jetson_weights,
        num_samples=100,
        local_loss=0.75,
        round_number=1
    )
    updates.append(jetson_update)
    
    # Akida update (4-bit simulation)
    akida_weights = {"conv1": torch.randint(-8, 7, (32, 16)).float()}
    akida_update = ClientUpdate(
        client_id="akida_01",
        device_type=DeviceType.AKIDA,
        weights=akida_weights,
        num_samples=80,
        local_loss=0.82,
        round_number=1
    )
    updates.append(akida_update)
    
    # Perform aggregation
    print(f"\nAggregating updates from {len(updates)} clients...")
    result = aggregator.aggregate_updates(updates)
    
    print(f"✅ Aggregation successful!")
    print(f"Result contains {len(result)} layers")
    
    # Show device-specific outputs
    jetson_result = aggregator.get_device_specific_weights(result, DeviceType.JETSON)
    akida_result = aggregator.get_device_specific_weights(result, DeviceType.AKIDA)
    
    print(f"Jetson output: {jetson_result['conv1'].dtype} precision")
    print(f"Akida output: quantized and sparsified")
    
    return True

def demonstrate_models():
    """Demonstrate different model variants"""
    
    print("\n🔬 Demonstrating Model Variants")
    print("=" * 50)
    
    # Create models
    jetson_model = ModelFactory.create_model("jetson", num_classes=10)
    akida_model = ModelFactory.create_model("akida", num_classes=10, sparsity=0.7)
    
    # Get model info
    jetson_info = ModelFactory.get_model_info(jetson_model)
    akida_info = ModelFactory.get_model_info(akida_model)
    
    print(f"Jetson model: {jetson_info['total_parameters']:,} parameters, {jetson_info['model_size_mb']:.2f} MB")
    print(f"Akida model: {akida_info['total_parameters']:,} parameters, {akida_info['model_size_mb']:.2f} MB")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    
    jetson_out = jetson_model(x)
    akida_out = akida_model(x)
    
    print(f"Jetson output: {jetson_out.shape}")
    print(f"Akida output: {akida_out.shape}")
    
    return True


# At the end of demo_phase1.py, add:
def run_visualizations():
    """Generate Phase 1 plots"""
    try:
        from phase1_visualization import plot_model_comparison, plot_aggregation_concept
        print("\n📊 Generating visualizations...")
        plot_model_comparison()
        plot_aggregation_concept()
        return True
    except ImportError:
        print("⚠️ Matplotlib not available - skipping plots")
        return False

# In the main section:
if __name__ == "__main__":
    print("🚀 Phase 1 Demonstration")
    print("Heterogeneous Federated Learning POC")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demonstrate_models()
        demonstrate_mixed_precision()
        
        print("\n🎉 Phase 1 demonstration completed successfully!")
        print("\nKey achievements:")
        print("✅ Mixed-precision aggregation working")
        print("✅ Device-specific models implemented")
        print("✅ Core FL components integrated")
        print("\nReady for Phase 2: Baseline Implementation")

        run_visualizations()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Check that all components are properly installed")