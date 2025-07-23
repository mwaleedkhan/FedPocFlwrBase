import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

class DepthwiseSeparableConv(nn.Module):
    """Core building block of MobileNet"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, stride=stride, padding=1, 
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class JetsonMobileNet(nn.Module):
    """Full-precision MobileNet V1 for Jetson devices"""
    
    def __init__(self, num_classes: int = 10, width_multiplier: float = 1.0):
        super().__init__()
        
        # Calculate channel numbers based on width multiplier
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        # First layer
        input_channel = make_divisible(32 * width_multiplier)
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        
        # Depthwise separable layers
        # Format: [channels, stride]
        self.layer_configs = [
            [64, 1],   # 32x32 -> 32x32
            [128, 2],  # 32x32 -> 16x16
            [128, 1],  # 16x16 -> 16x16
            [256, 2],  # 16x16 -> 8x8
            [256, 1],  # 8x8 -> 8x8
            [512, 2],  # 8x8 -> 4x4
            [512, 1],  # 4x4 -> 4x4 (repeat 5 times)
            [512, 1],
            [512, 1],
            [512, 1],
            [512, 1],
            [1024, 2], # 4x4 -> 2x2
            [1024, 1], # 2x2 -> 2x2
        ]
        
        # Build layers
        self.layers = nn.ModuleList()
        for out_channels, stride in self.layer_configs:
            out_channels = make_divisible(out_channels * width_multiplier)
            self.layers.append(DepthwiseSeparableConv(input_channel, out_channels, stride))
            input_channel = out_channels
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(input_channel, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Track device type for federated learning
        self.device_type = "jetson"
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

class QuantizedDepthwiseSeparableConv(nn.Module):
    """4-bit quantized version for Akida simulation"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Use quantization-aware training modules
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, stride=stride, padding=1, 
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Quantization parameters
        self.weight_bits = 4
        self.activation_bits = 8  # Higher precision for activations
        
    def quantize_weights(self, weights):
        """Quantize weights to 4-bit"""
        max_val = torch.max(torch.abs(weights))
        scale = max_val / (2**(self.weight_bits-1) - 1)
        quantized = torch.round(weights / scale).clamp(-(2**(self.weight_bits-1)), 2**(self.weight_bits-1) - 1)
        return quantized * scale
    
    def forward(self, x):
        # Quantize weights during forward pass
        with torch.no_grad():
            self.depthwise.weight.data = self.quantize_weights(self.depthwise.weight.data)
            self.pointwise.weight.data = self.quantize_weights(self.pointwise.weight.data)
        
        # Forward pass with quantized weights
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        
        # Simulate neuromorphic sparsity
        sparsity_mask = (torch.rand_like(x) > 0.3).float()  # 70% sparsity
        x = x * sparsity_mask
        
        return x

class AkidaMobileNet(nn.Module):
    """Quantized/Sparse MobileNet V1 for Akida neuromorphic simulation"""
    
    def __init__(self, num_classes: int = 10, width_multiplier: float = 1.0, sparsity: float = 0.7):
        super().__init__()
        
        self.sparsity = sparsity
        
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        # First layer (quantized)
        input_channel = make_divisible(32 * width_multiplier)
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        
        # Same layer config as Jetson version
        self.layer_configs = [
            [64, 1], [128, 2], [128, 1], [256, 2], [256, 1], [512, 2],
            [512, 1], [512, 1], [512, 1], [512, 1], [512, 1], [1024, 2], [1024, 1]
        ]
        
        # Build quantized layers
        self.layers = nn.ModuleList()
        for out_channels, stride in self.layer_configs:
            out_channels = make_divisible(out_channels * width_multiplier)
            self.layers.append(QuantizedDepthwiseSeparableConv(input_channel, out_channels, stride))
            input_channel = out_channels
        
        # Classifier with on-chip learning simulation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(input_channel, num_classes)
        
        # Neuromorphic-specific parameters
        self.spike_threshold = 1.0
        self.membrane_potential = 0.0
        self.device_type = "akida"
        
        self._initialize_weights()
        
    def forward(self, x):
        # First layer with quantization
        x = self.conv1(x)
        x = self.quantize_activations(x)
        x = F.relu(self.bn1(x))
        
        # Process through quantized layers
        for layer in self.layers:
            x = layer(x)
            x = self.apply_sparsity(x)
        
        # Final classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def quantize_activations(self, x, bits=8):
        """Quantize activations to simulate neuromorphic processing"""
        max_val = torch.max(x)
        min_val = torch.min(x)
        scale = (max_val - min_val) / (2**bits - 1)
        quantized = torch.round((x - min_val) / scale) * scale + min_val
        return quantized
    
    def apply_sparsity(self, x):
        """Apply sparsity pattern typical of neuromorphic chips"""
        if self.training:
            # During training, apply stochastic sparsity
            mask = (torch.rand_like(x) > self.sparsity).float()
        else:
            # During inference, use fixed sparsity based on magnitude
            threshold = torch.quantile(torch.abs(x), self.sparsity)
            mask = (torch.abs(x) > threshold).float()
        
        return x * mask
    
    def get_sparse_weight_deltas(self, old_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract sparse weight deltas typical of neuromorphic learning"""
        deltas = {}
        
        for name, param in self.named_parameters():
            if name in old_weights:
                # Calculate delta
                delta = param.data - old_weights[name]
                
                # Apply sparsity to delta (neuromorphic chips update sparsely)
                sparse_mask = (torch.rand_like(delta) > 0.8).float()  # Very sparse updates
                sparse_delta = delta * sparse_mask
                
                deltas[name] = sparse_delta
        
        return deltas
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize with smaller values for quantization stability
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.5  # Reduce initial magnitude
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

class ModelFactory:
    """Factory for creating appropriate models for each device type"""
    
    @staticmethod
    def create_model(device_type: str, num_classes: int = 10, **kwargs) -> nn.Module:
        """Create model based on device type"""
        
        if device_type.lower() == "jetson":
            return JetsonMobileNet(num_classes=num_classes, **kwargs)
        elif device_type.lower() == "akida":
            return AkidaMobileNet(num_classes=num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown device type: {device_type}")
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict:
        """Get model information for debugging"""
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        param_size = total_params * 4  # 4 bytes per float32
        if hasattr(model, 'device_type') and model.device_type == "akida":
            param_size = total_params * 0.5  # 4-bit quantization
        
        return {
            'device_type': getattr(model, 'device_type', 'unknown'),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': param_size / (1024 * 1024),
            'architecture': type(model).__name__
        }

# Example usage and testing
def test_models():
    """Test both model variants"""
    
    # Create models
    jetson_model = ModelFactory.create_model("jetson", num_classes=10)
    akida_model = ModelFactory.create_model("akida", num_classes=10, sparsity=0.7)
    
    # Test with dummy input (CIFAR-10 size)
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch size 4
    
    print("=== Model Comparison ===")
    
    # Jetson model
    jetson_output = jetson_model(dummy_input)
    jetson_info = ModelFactory.get_model_info(jetson_model)
    
    print(f"Jetson Model:")
    print(f"  Output shape: {jetson_output.shape}")
    print(f"  Parameters: {jetson_info['total_parameters']:,}")
    print(f"  Model size: {jetson_info['model_size_mb']:.2f} MB")
    print(f"  Device type: {jetson_info['device_type']}")
    
    # Akida model  
    akida_output = akida_model(dummy_input)
    akida_info = ModelFactory.get_model_info(akida_model)
    
    print(f"\nAkida Model:")
    print(f"  Output shape: {akida_output.shape}")
    print(f"  Parameters: {akida_info['total_parameters']:,}")
    print(f"  Model size: {akida_info['model_size_mb']:.2f} MB")
    print(f"  Device type: {akida_info['device_type']}")
    print(f"  Sparsity: {akida_model.sparsity:.1%}")
    
    # Test compatibility for federated learning
    print(f"\n=== FL Compatibility Test ===")
    
    # Check if models have same architecture (for weight sharing)
    jetson_layers = [name for name, _ in jetson_model.named_parameters()]
    akida_layers = [name for name, _ in akida_model.named_parameters()]
    
    compatible_layers = set(jetson_layers) & set(akida_layers)
    print(f"Compatible layers: {len(compatible_layers)}/{len(jetson_layers)}")
    
    if len(compatible_layers) == len(jetson_layers):
        print("✅ Models are fully compatible for federated learning")
    else:
        print("❌ Models have architecture differences")
        missing = set(jetson_layers) - set(akida_layers)
        print(f"Missing in Akida: {missing}")
    
    return jetson_model, akida_model

def demonstrate_weight_extraction():
    """Show how to extract weights for federated learning"""
    
    print("\n=== Weight Extraction Demo ===")
    
    akida_model = ModelFactory.create_model("akida", num_classes=10)
    
    # Get initial weights
    initial_weights = {name: param.clone() for name, param in akida_model.named_parameters()}
    
    # Simulate some training (dummy)
    dummy_input = torch.randn(2, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (2,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(akida_model.parameters(), lr=0.01)
    
    # One training step
    output = akida_model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()
    
    # Extract sparse deltas (neuromorphic style)
    sparse_deltas = akida_model.get_sparse_weight_deltas(initial_weights)
    
    print(f"Extracted {len(sparse_deltas)} weight delta tensors")
    
    for name, delta in list(sparse_deltas.items())[:3]:  # Show first 3
        sparsity_level = (delta == 0).float().mean().item()
        print(f"  {name}: shape={delta.shape}, sparsity={sparsity_level:.1%}")
    
    return sparse_deltas

if __name__ == "__main__":
    # Run tests
    jetson_model, akida_model = test_models()
    sparse_deltas = demonstrate_weight_extraction()
    
    print("\n✅ Model variants created successfully!")
    print("Ready for federated learning experiments.")