import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class DeviceType(Enum):
    JETSON = "jetson"
    AKIDA = "akida"

@dataclass
class ClientUpdate:
    """Represents a model update from a client device"""
    client_id: str
    device_type: DeviceType
    weights: Dict[str, torch.Tensor]
    num_samples: int
    local_loss: float
    round_number: int
    metadata: Optional[Dict] = None

class QuantizationStrategy:
    """Handles quantization/dequantization for different bit widths"""
    
    @staticmethod
    def quantize_weights(weights: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """Quantize weights to specified bit width"""
        if bits == 32:
            return weights  # No quantization needed
        
        # Symmetric quantization
        max_val = torch.max(torch.abs(weights))
        scale = max_val / (2**(bits-1) - 1)
        
        quantized = torch.round(weights / scale).clamp(-(2**(bits-1)), 2**(bits-1) - 1)
        return quantized * scale
    
    @staticmethod
    def dequantize_to_float32(quantized_weights: torch.Tensor) -> torch.Tensor:
        """Convert quantized weights to float32 for aggregation"""
        return quantized_weights.float()
    
    @staticmethod
    def apply_sparsity_mask(weights: torch.Tensor, sparsity: float = 0.7) -> torch.Tensor:
        """Apply sparsity pattern typical of neuromorphic chips"""
        mask = (torch.rand_like(weights) > sparsity).float()
        return weights * mask

class FedMPQAggregator:
    """Mixed-Precision Quantized Federated Aggregation"""
    
    def __init__(self, 
                 jetson_precision: int = 32,
                 akida_precision: int = 4,
                 error_compensation: bool = True):
        self.jetson_precision = jetson_precision
        self.akida_precision = akida_precision
        self.error_compensation = error_compensation
        self.quantizer = QuantizationStrategy()
        
        # Error compensation buffers
        self.jetson_error_buffer = {}
        self.akida_error_buffer = {}
        
        # Aggregation statistics
        self.round_stats = []
    
    def aggregate_updates(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """
        Core FedMPQ aggregation algorithm
        
        Steps:
        1. Separate updates by device type
        2. Dequantize Akida updates to float32
        3. Apply error compensation if enabled
        4. Weighted average all updates
        5. Store error for next round
        """
        
        if not updates:
            raise ValueError("No updates provided for aggregation")
        
        # Separate updates by device type
        jetson_updates = [u for u in updates if u.device_type == DeviceType.JETSON]
        akida_updates = [u for u in updates if u.device_type == DeviceType.AKIDA]
        
        print(f"Aggregating {len(jetson_updates)} Jetson + {len(akida_updates)} Akida updates")
        
        # Get model structure from first update
        model_keys = list(updates[0].weights.keys())
        aggregated_weights = {}
        
        for key in model_keys:
            aggregated_weights[key] = self._aggregate_layer_weights(
                key, jetson_updates, akida_updates
            )
        
        # Record statistics
        self._record_round_stats(updates, aggregated_weights)
        
        return aggregated_weights
    
    def _aggregate_layer_weights(self, 
                                layer_key: str,
                                jetson_updates: List[ClientUpdate],
                                akida_updates: List[ClientUpdate]) -> torch.Tensor:
        """Aggregate weights for a specific layer"""
        
        all_weights = []
        all_sample_counts = []
        
        # Process Jetson updates (already float32)
        for update in jetson_updates:
            weights = update.weights[layer_key]
            
            # Apply error compensation
            if self.error_compensation and layer_key in self.jetson_error_buffer:
                weights = weights + self.jetson_error_buffer[layer_key]
            
            all_weights.append(weights)
            all_sample_counts.append(update.num_samples)
        
        # Process Akida updates (dequantize first)
        for update in akida_updates:
            quantized_weights = update.weights[layer_key]
            
            # Dequantize to float32
            weights = self.quantizer.dequantize_to_float32(quantized_weights)
            
            # Apply error compensation
            if self.error_compensation and layer_key in self.akida_error_buffer:
                weights = weights + self.akida_error_buffer[layer_key]
            
            all_weights.append(weights)
            all_sample_counts.append(update.num_samples)
        
        # Weighted average
        total_samples = sum(all_sample_counts)
        aggregated = torch.zeros_like(all_weights[0])
        
        for weights, sample_count in zip(all_weights, all_sample_counts):
            weight_factor = sample_count / total_samples
            aggregated += weights * weight_factor
        
        # Store quantization error for next round
        if self.error_compensation:
            self._update_error_buffers(layer_key, aggregated, jetson_updates, akida_updates)
        
        return aggregated
    
    def _update_error_buffers(self, 
                             layer_key: str,
                             aggregated_weights: torch.Tensor,
                             jetson_updates: List[ClientUpdate],
                             akida_updates: List[ClientUpdate]):
        """Update error compensation buffers"""
        
        # Jetson error (minimal, mostly for consistency)
        jetson_quantized = self.quantizer.quantize_weights(aggregated_weights, self.jetson_precision)
        self.jetson_error_buffer[layer_key] = aggregated_weights - jetson_quantized
        
        # Akida error (significant due to 4-bit quantization)
        akida_quantized = self.quantizer.quantize_weights(aggregated_weights, self.akida_precision)
        self.akida_error_buffer[layer_key] = aggregated_weights - akida_quantized
    
    def _record_round_stats(self, updates: List[ClientUpdate], aggregated_weights: Dict[str, torch.Tensor]):
        """Record statistics for this aggregation round"""
        
        round_stat = {
            'round': updates[0].round_number,
            'total_clients': len(updates),
            'jetson_clients': len([u for u in updates if u.device_type == DeviceType.JETSON]),
            'akida_clients': len([u for u in updates if u.device_type == DeviceType.AKIDA]),
            'avg_loss': np.mean([u.local_loss for u in updates]),
            'total_samples': sum(u.num_samples for u in updates),
            'weight_diversity': self._calculate_weight_diversity(updates),
        }
        
        self.round_stats.append(round_stat)
    
    def _calculate_weight_diversity(self, updates: List[ClientUpdate]) -> float:
        """Calculate diversity in weight updates (higher = more heterogeneous)"""
        if len(updates) < 2:
            return 0.0
        
        # Compare first layer weights as a proxy for overall diversity
        first_key = list(updates[0].weights.keys())[0]
        weights_list = []
        
        for update in updates:
            weights = update.weights[first_key].flatten()
            if update.device_type == DeviceType.AKIDA:
                weights = self.quantizer.dequantize_to_float32(weights)
            weights_list.append(weights)
        
        # Calculate pairwise cosine distances
        diversities = []
        for i in range(len(weights_list)):
            for j in range(i+1, len(weights_list)):
                cos_sim = F.cosine_similarity(weights_list[i], weights_list[j], dim=0)
                diversity = 1 - cos_sim.item()  # Distance = 1 - similarity
                diversities.append(diversity)
        
        return np.mean(diversities)
    
    def get_device_specific_weights(self, 
                                  aggregated_weights: Dict[str, torch.Tensor],
                                  device_type: DeviceType) -> Dict[str, torch.Tensor]:
        """Convert aggregated weights to device-specific format"""
        
        if device_type == DeviceType.JETSON:
            # Return float32 weights
            return {k: v.float() for k, v in aggregated_weights.items()}
        
        elif device_type == DeviceType.AKIDA:
            # Quantize to 4-bit and apply sparsity
            quantized_weights = {}
            for k, v in aggregated_weights.items():
                quantized = self.quantizer.quantize_weights(v, self.akida_precision)
                sparse = self.quantizer.apply_sparsity_mask(quantized, sparsity=0.7)
                quantized_weights[k] = sparse
            return quantized_weights
        
        else:
            raise ValueError(f"Unknown device type: {device_type}")

# Example usage for AI agent testing
def example_aggregation():
    """Example of how the FedMPQ aggregator works"""
    
    # Create mock updates
    aggregator = FedMPQAggregator()
    
    # Simulate model weights (simplified)
    layer_shape = (64, 32)  # Conv layer example
    
    jetson_update = ClientUpdate(
        client_id="jetson_01",
        device_type=DeviceType.JETSON,
        weights={"conv1.weight": torch.randn(layer_shape)},
        num_samples=100,
        local_loss=0.85,
        round_number=1
    )
    
    akida_update = ClientUpdate(
        client_id="akida_01", 
        device_type=DeviceType.AKIDA,
        weights={"conv1.weight": torch.randint(-8, 7, layer_shape).float()},  # 4-bit quantized
        num_samples=80,
        local_loss=0.92,
        round_number=1
    )
    
    # Aggregate
    updates = [jetson_update, akida_update]
    aggregated = aggregator.aggregate_updates(updates)
    
    # Get device-specific weights
    jetson_weights = aggregator.get_device_specific_weights(aggregated, DeviceType.JETSON)
    akida_weights = aggregator.get_device_specific_weights(aggregated, DeviceType.AKIDA)
    
    print("Aggregation successful!")
    print(f"Round stats: {aggregator.round_stats[-1]}")
    
    return aggregated, jetson_weights, akida_weights

if __name__ == "__main__":
    example_aggregation()