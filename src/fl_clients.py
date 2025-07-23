import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import time
import random
from collections import OrderedDict

# Import our models
from mobilenet_variants import ModelFactory, JetsonMobileNet, AkidaMobileNet

class BaseFlowerClient(fl.client.NumPyClient):
    """Base client implementation with common functionality"""
    
    def __init__(
        self,
        client_id: str,
        device_type: str,
        data_shard_id: int,
        model_params: Dict,
        training_params: Dict,
        dropout_config: Optional[Dict] = None
    ):
        self.client_id = client_id
        self.device_type = device_type
        self.data_shard_id = data_shard_id
        self.dropout_config = dropout_config or {}
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() and device_type == "jetson" else "cpu")
        
        # Create model
        self.model = ModelFactory.create_model(device_type, **model_params)
        self.model.to(self.device)
        
        # Setup training
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer(training_params)
        self.training_params = training_params
        
        # Load data
        self.train_loader, self.test_loader = self._load_data()
        
        # Client state tracking
        self.current_round = 0
        self.is_dropped = False
        self.drop_start_round = None
        self.training_history = []
        self.local_model_state = None  # For dropout scenarios
        
        # Logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{client_id}")
        self.logger.info(f"Initialized {device_type} client {client_id} with {len(self.train_loader.dataset)} training samples")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as numpy arrays"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model and return updated parameters"""
        
        round_num = config.get("round", 0)
        self.current_round = round_num
        
        # Handle dropout scenarios (Research Questions Q1-Q4)
        if self._should_drop_this_round(round_num):
            return self._handle_dropout_scenario(parameters, config)
        
        # If rejoining after dropout
        if self.is_dropped:
            self._handle_rejoin_scenario(parameters, config)
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Perform local training
        train_metrics = self._train_local_model(config)
        
        # Get updated parameters
        updated_params = self.get_parameters(config)
        
        # Record training history
        self.training_history.append({
            "round": round_num,
            "train_loss": train_metrics["train_loss"],
            "train_accuracy": train_metrics.get("train_accuracy", 0.0),
            "num_samples": len(self.train_loader.dataset),
            "training_time": train_metrics.get("training_time", 0.0)
        })
        
        return updated_params, len(self.train_loader.dataset), train_metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model and return loss, number of samples, and metrics"""
        
        # Skip evaluation if dropped
        if self.is_dropped:
            return float('inf'), 0, {"accuracy": 0.0, "status": "dropped"}
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate model
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Device-specific forward pass
                output = self._device_specific_forward(data)
                
                # Calculate loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "test_loss": avg_loss,
            "device_type": self.device_type,
            "client_id": self.client_id
        }
        
        return avg_loss, total, metrics
    
    def _create_optimizer(self, training_params: Dict):
        """Create optimizer based on device type and parameters"""
        lr = training_params.get("learning_rate", 0.01)
        weight_decay = training_params.get("weight_decay", 1e-4)
        
        if self.device_type == "akida":
            # Neuromorphic devices might need different optimization
            return optim.Adam(self.model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
        else:
            # Standard SGD for Jetson
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    def _load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load CIFAR-10 data shard for this client"""
        
        # CIFAR-10 transforms
        if self.device_type == "akida":
            # Simpler transforms for neuromorphic processing
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            # Standard data augmentation for Jetson
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Load full datasets
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        # Create data shards (Non-IID distribution)
        train_indices = self._create_non_iid_shard(trainset, self.data_shard_id)
        test_indices = self._create_test_shard(testset, self.data_shard_id)
        
        # Create data loaders
        batch_size = 32 if self.device_type == "jetson" else 16  # Smaller batches for Akida
        
        train_subset = Subset(trainset, train_indices)
        test_subset = Subset(testset, test_indices)
        
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_subset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, test_loader
    
    def _create_non_iid_shard(self, dataset, shard_id: int, alpha: float = 0.5):
        """Create non-IID data shard using Dirichlet distribution"""
        
        num_classes = 10
        num_clients = 4  # Assume 4 clients total
        
        # Get labels
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        # Create Dirichlet distribution for non-IID data
        label_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)
        
        # Calculate number of samples per class for this client
        samples_per_class = (label_distribution[shard_id] * len(dataset) / num_clients).astype(int)
        
        # Select indices for each class
        selected_indices = []
        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            np.random.shuffle(class_indices)
            
            num_samples = min(samples_per_class[class_id], len(class_indices))
            start_idx = (shard_id * num_samples) % len(class_indices)
            end_idx = start_idx + num_samples
            
            if end_idx <= len(class_indices):
                selected_indices.extend(class_indices[start_idx:end_idx])
            else:
                # Wrap around if needed
                selected_indices.extend(class_indices[start_idx:])
                selected_indices.extend(class_indices[:end_idx - len(class_indices)])
        
        return selected_indices
    
    def _create_test_shard(self, dataset, shard_id: int):
        """Create test data shard (smaller, balanced)"""
        
        num_clients = 4
        samples_per_client = len(dataset) // num_clients
        
        start_idx = shard_id * samples_per_client
        end_idx = start_idx + samples_per_client
        
        return list(range(start_idx, min(end_idx, len(dataset))))
    
    def _device_specific_forward(self, data):
        """Perform device-specific forward pass"""
        if self.device_type == "akida":
            # Add some artificial delays to simulate neuromorphic processing
            time.sleep(0.001)  # 1ms delay
        
        return self.model(data)
    
    def _train_local_model(self, config: Dict) -> Dict:
        """Perform local training"""
        
        epochs = config.get("local_epochs", 1)
        start_time = time.time()
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Device-specific forward pass
                output = self._device_specific_forward(data)
                
                loss = self.criterion(output, target)
                loss.backward()
                
                # Device-specific optimization step
                if self.device_type == "akida":
                    # Simulate spike-based learning with gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        training_time = time.time() - start_time
        avg_loss = total_loss / (len(self.train_loader) * epochs)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "training_time": training_time,
            "device_type": self.device_type,
            "local_epochs": epochs
        }
    
    # Dropout/Rejoin Scenario Handling (Research Questions Q1-Q4)
    
    def _should_drop_this_round(self, round_num: int) -> bool:
        """Determine if client should drop out this round"""
        
        if not self.dropout_config:
            return False
        
        drop_probability = self.dropout_config.get("drop_probability", 0.0)
        drop_rounds = self.dropout_config.get("drop_rounds", [])
        drop_duration = self.dropout_config.get("drop_duration", 1)
        
        # Specific round dropouts
        if round_num in drop_rounds:
            self.is_dropped = True
            self.drop_start_round = round_num
            self.logger.info(f"Client {self.client_id} dropping out at round {round_num}")
            return True
        
        # Random dropouts
        if random.random() < drop_probability and not self.is_dropped:
            self.is_dropped = True
            self.drop_start_round = round_num
            self.logger.info(f"Client {self.client_id} randomly dropping out at round {round_num}")
            return True
        
        # Check if still in dropout period
        if self.is_dropped:
            rounds_dropped = round_num - self.drop_start_round
            if rounds_dropped < drop_duration:
                return True
            else:
                # Time to rejoin
                self.is_dropped = False
                self.logger.info(f"Client {self.client_id} rejoining at round {round_num}")
                return False
        
        return False
    
    def _handle_dropout_scenario(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Handle client dropout scenario (Q2: idle vs continue training)"""
        
        continue_training = self.dropout_config.get("continue_training_when_dropped", False)
        
        if continue_training:
            # Q2: Continue local training while dropped
            self.logger.info(f"Client {self.client_id} continues training while dropped")
            
            # Use last known model state or current parameters
            if self.local_model_state is not None:
                self.set_parameters(self.local_model_state)
            else:
                self.set_parameters(parameters)
            
            # Train locally
            train_metrics = self._train_local_model(config)
            updated_params = self.get_parameters(config)
            
            # Store local state for potential rejoin
            self.local_model_state = updated_params
            
            # Return dummy data (won't be aggregated)
            metrics = {
                "train_loss": train_metrics["train_loss"],
                "status": "dropped_but_training",
                "device_type": self.device_type
            }
            
        else:
            # Q2: Remain idle while dropped
            self.logger.info(f"Client {self.client_id} remaining idle while dropped")
            
            metrics = {
                "train_loss": float('inf'),
                "status": "dropped_idle",
                "device_type": self.device_type
            }
            updated_params = parameters  # No change
        
        return updated_params, 0, metrics  # 0 samples to exclude from aggregation
    
    def _handle_rejoin_scenario(self, parameters: List[np.ndarray], config: Dict):
        """Handle client rejoin scenario (Q3: reset vs buffered updates)"""
        
        rejoin_strategy = self.dropout_config.get("rejoin_strategy", "reset")
        
        if rejoin_strategy == "reset":
            # Q3: Reset to global model state
            self.logger.info(f"Client {self.client_id} resetting to global model on rejoin")
            self.set_parameters(parameters)
            self.local_model_state = None
            
        elif rejoin_strategy == "continue_local":
            # Q3: Continue from local state
            if self.local_model_state is not None:
                self.logger.info(f"Client {self.client_id} continuing from local state on rejoin")
                self.set_parameters(self.local_model_state)
            else:
                self.logger.warning(f"Client {self.client_id} has no local state, resetting to global")
                self.set_parameters(parameters)
            
        elif rejoin_strategy == "weighted_merge":
            # Q3: Merge local and global states
            if self.local_model_state is not None:
                self.logger.info(f"Client {self.client_id} merging local and global states on rejoin")
                
                # Simple weighted average (could be more sophisticated)
                global_state = parameters
                local_state = self.local_model_state
                merge_weight = 0.5  # Could be adaptive
                
                merged_params = []
                for global_param, local_param in zip(global_state, local_state):
                    merged = merge_weight * global_param + (1 - merge_weight) * local_param
                    merged_params.append(merged)
                
                self.set_parameters(merged_params)
            else:
                self.set_parameters(parameters)
        
        self.is_dropped = False
        self.drop_start_round = None

class JetsonFlowerClient(BaseFlowerClient):
    """Jetson GPU client with full-precision training"""
    
    def __init__(self, client_id: str, data_shard_id: int, **kwargs):
        model_params = kwargs.pop("model_params", {"num_classes": 10, "width_multiplier": 1.0})
        training_params = kwargs.pop("training_params", {
            "learning_rate": 0.01,
            "weight_decay": 1e-4
        })
        
        super().__init__(
            client_id=client_id,
            device_type="jetson",
            data_shard_id=data_shard_id,
            model_params=model_params,
            training_params=training_params,
            **kwargs
        )
    
    def _device_specific_forward(self, data):
        """Jetson-specific forward pass with full precision"""
        # Enable mixed precision training if available
        if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self.model(data)
        else:
            return self.model(data)

class AkidaFlowerClient(BaseFlowerClient):
    """Akida neuromorphic client with quantized/sparse training"""
    
    def __init__(self, client_id: str, data_shard_id: int, **kwargs):
        model_params = kwargs.pop("model_params", {
            "num_classes": 10, 
            "width_multiplier": 1.0,
            "sparsity": 0.7
        })
        training_params = kwargs.pop("training_params", {
            "learning_rate": 0.001,  # Lower learning rate for quantized training
            "weight_decay": 1e-5
        })
        
        super().__init__(
            client_id=client_id,
            device_type="akida",
            data_shard_id=data_shard_id,
            model_params=model_params,
            training_params=training_params,
            **kwargs
        )
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return quantized parameters"""
        params = []
        for val in self.model.state_dict().values():
            # Quantize to 4-bit before sending
            quantized = self._quantize_4bit(val.cpu())
            params.append(quantized.numpy())
        return params
    
    def _quantize_4bit(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to 4-bit"""
        max_val = torch.max(torch.abs(tensor))
        scale = max_val / 7  # 4-bit signed: -8 to 7
        quantized = torch.round(tensor / scale).clamp(-8, 7)
        return quantized * scale
    
    def _device_specific_forward(self, data):
        """Akida-specific forward pass with neuromorphic simulation"""
        # Simulate neuromorphic processing delays and spike generation
        time.sleep(0.002)  # 2ms processing delay
        
        output = self.model(data)
        
        # Simulate spike-based output quantization
        output = self._apply_spike_quantization(output)
        
        return output
    
    def _apply_spike_quantization(self, output: torch.Tensor) -> torch.Tensor:
        """Apply spike-based quantization to outputs"""
        # Simple threshold-based spiking simulation
        threshold = 0.5
        spike_output = (output > threshold).float() * output
        return spike_output

# Client Factory
class ClientFactory:
    """Factory for creating appropriate client instances"""
    
    @staticmethod
    def create_client(
        client_id: str,
        device_type: str,
        data_shard_id: int,
        dropout_config: Optional[Dict] = None,
        **kwargs
    ) -> BaseFlowerClient:
        """Create client based on device type"""
        
        if device_type.lower() == "jetson":
            return JetsonFlowerClient(
                client_id=client_id,
                data_shard_id=data_shard_id,
                dropout_config=dropout_config,
                **kwargs
            )
        elif device_type.lower() == "akida":
            return AkidaFlowerClient(
                client_id=client_id,
                data_shard_id=data_shard_id,
                dropout_config=dropout_config,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown device type: {device_type}")

# Example client creation and testing
def test_clients():
    """Test client creation and basic functionality"""
    
    print("=== Testing Client Creation ===")
    
    # Create test clients
    jetson_client = ClientFactory.create_client(
        client_id="jetson_01",
        device_type="jetson",
        data_shard_id=0,
        dropout_config={
            "drop_probability": 0.1,
            "drop_duration": 2,
            "continue_training_when_dropped": True,
            "rejoin_strategy": "reset"
        }
    )
    
    akida_client = ClientFactory.create_client(
        client_id="akida_01", 
        device_type="akida",
        data_shard_id=1,
        dropout_config={
            "drop_rounds": [5, 10],
            "drop_duration": 3,
            "continue_training_when_dropped": False,
            "rejoin_strategy": "weighted_merge"
        }
    )
    
    # Test parameter extraction
    jetson_params = jetson_client.get_parameters({})
    akida_params = akida_client.get_parameters({})
    
    print(f"Jetson client: {len(jetson_params)} parameter arrays")
    print(f"Akida client: {len(akida_params)} parameter arrays")
    
    # Test training step
    config = {"round": 1, "local_epochs": 1}
    
    print("\n=== Testing Training Step ===")
    
    # Jetson training
    jetson_result = jetson_client.fit(jetson_params, config)
    print(f"Jetson training result: {len(jetson_result[0])} params, {jetson_result[1]} samples")
    print(f"Jetson metrics: {jetson_result[2]}")
    
    # Akida training  
    akida_result = akida_client.fit(akida_params, config)
    print(f"Akida training result: {len(akida_result[0])} params, {akida_result[1]} samples")
    print(f"Akida metrics: {akida_result[2]}")
    
    # Test evaluation
    print("\n=== Testing Evaluation ===")
    
    jetson_eval = jetson_client.evaluate(jetson_params, config)
    akida_eval = akida_client.evaluate(akida_params, config)
    
    print(f"Jetson evaluation: loss={jetson_eval[0]:.4f}, samples={jetson_eval[1]}")
    print(f"Akida evaluation: loss={akida_eval[0]:.4f}, samples={akida_eval[1]}")
    
    return jetson_client, akida_client

if __name__ == "__main__":
    # Run client tests
    jetson_client, akida_client = test_clients()
    print("\n✅ Flower clients created and tested successfully!")
    print("Ready for federated learning experiments.")