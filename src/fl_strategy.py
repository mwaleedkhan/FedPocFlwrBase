import flwr as fl
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from flwr.common import (
    FitRes, FitIns, EvaluateRes, EvaluateIns, 
    Parameters, Scalar, Config, parameters_to_ndarrays, ndarrays_to_parameters
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from dataclasses import dataclass
import logging
import json

# Import our custom aggregator
from fedmpq_algorithm import FedMPQAggregator, ClientUpdate, DeviceType

@dataclass
class ClientCapability:
    """Information about client device capabilities"""
    client_id: str
    device_type: str  # "jetson" or "akida"
    precision_bits: int
    memory_mb: float
    compute_flops: float
    max_batch_size: int
    supports_quantization: bool

class FedMPQFlowerStrategy(Strategy):
    """
    Custom Flower strategy implementing Mixed-Precision Quantized Federated Learning
    
    This strategy handles:
    - Heterogeneous device types (Jetson GPU vs Akida neuromorphic)
    - Mixed-precision aggregation (32-bit + 4-bit)
    - Client capability negotiation
    - Dropout/rejoin scenarios
    """
    
    def __init__(
        self,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[callable] = None,
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        # FedMPQ specific parameters
        jetson_precision: int = 32,
        akida_precision: int = 4,
        error_compensation: bool = True,
        adaptive_aggregation: bool = True,
        dropout_tolerance: float = 0.3,  # Allow 30% client dropout
    ):
        super().__init__()
        
        # Standard Flower parameters
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        
        # FedMPQ aggregator
        self.aggregator = FedMPQAggregator(
            jetson_precision=jetson_precision,
            akida_precision=akida_precision,
            error_compensation=error_compensation
        )
        
        # Strategy parameters
        self.adaptive_aggregation = adaptive_aggregation
        self.dropout_tolerance = dropout_tolerance
        
        # Client management
        self.client_capabilities: Dict[str, ClientCapability] = {}
        self.client_history: Dict[str, List[Dict]] = {}
        self.dropped_clients: Dict[str, Dict] = {}  # Track dropped clients
        self.global_model_parameters = initial_parameters
        
        # Round tracking
        self.current_round = 0
        self.experiment_log = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def initialize_parameters(
        self, client_manager: Any
    ) -> Optional[Parameters]:
        """Initialize global model parameters"""
        return self.global_model_parameters
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: Any
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for training round"""
        
        self.current_round = server_round
        self.logger.info(f"Configuring fit for round {server_round}")
        
        # Get available clients
        available_clients = client_manager.all()
        
        # Handle client dropouts
        self._handle_client_dropouts(available_clients)
        
        # Select clients for this round
        selected_clients = self._select_clients_for_round(available_clients)
        
        # Create fit configuration
        config = {"round": server_round}
        if self.on_fit_config_fn:
            config.update(self.on_fit_config_fn(server_round))
        
        # Create fit instructions for each client
        fit_instructions = []
        for client in selected_clients:
            # Get client-specific parameters
            client_params = self._get_client_parameters(client, parameters)
            fit_ins = FitIns(parameters=client_params, config=config)
            fit_instructions.append((client, fit_ins))
        
        self.logger.info(f"Selected {len(selected_clients)} clients for training")
        return fit_instructions
    
    def aggregate_fit(
        self, 
        server_round: int, 
        results: List[Tuple[ClientProxy, FitRes]], 
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results using FedMPQ"""
        
        self.logger.info(f"Aggregating {len(results)} client updates for round {server_round}")
        
        if not results:
            return None, {}
        
        # Handle failures
        if failures and not self.accept_failures:
            return None, {}
        
        # Convert Flower results to our format
        client_updates = []
        for client_proxy, fit_res in results:
            try:
                update = self._convert_flower_result_to_update(
                    client_proxy, fit_res, server_round
                )
                client_updates.append(update)
            except Exception as e:
                self.logger.warning(f"Failed to convert update from {client_proxy.cid}: {e}")
                if not self.accept_failures:
                    return None, {}
        
        if not client_updates:
            return None, {}
        
        # Perform FedMPQ aggregation
        try:
            aggregated_weights = self.aggregator.aggregate_updates(client_updates)
            
            # Convert back to Flower parameters format
            aggregated_params = self._convert_weights_to_parameters(aggregated_weights)
            
            # Update global model
            self.global_model_parameters = aggregated_params
            
            # Calculate metrics
            metrics = self._calculate_round_metrics(client_updates, server_round)
            
            # Log experiment data
            self._log_round_data(server_round, client_updates, metrics)
            
            return aggregated_params, metrics
            
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            return None, {}
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: Any
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure clients for evaluation"""
        
        if self.evaluate_fn is None:
            # Client-side evaluation
            available_clients = client_manager.all()
            selected_clients = self._select_evaluation_clients(available_clients)
            
            config = {"round": server_round}
            if self.on_evaluate_config_fn:
                config.update(self.on_evaluate_config_fn(server_round))
            
            evaluate_instructions = []
            for client in selected_clients:
                client_params = self._get_client_parameters(client, parameters)
                eval_ins = EvaluateIns(parameters=client_params, config=config)
                evaluate_instructions.append((client, eval_ins))
            
            return evaluate_instructions
        
        return []  # Server-side evaluation
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""
        
        if not results:
            return None, {}
        
        # Server-side evaluation
        if self.evaluate_fn is not None:
            parameters_ndarrays = parameters_to_ndarrays(self.global_model_parameters)
            eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
            if eval_res is not None:
                loss, metrics = eval_res
                return loss, metrics
        
        # Client-side evaluation aggregation
        total_examples = 0
        total_loss = 0.0
        device_metrics = {"jetson": [], "akida": []}
        
        for client_proxy, eval_res in results:
            # Aggregate loss
            total_examples += eval_res.num_examples
            total_loss += eval_res.loss * eval_res.num_examples
            
            # Track per-device metrics
            client_id = client_proxy.cid
            if client_id in self.client_capabilities:
                device_type = self.client_capabilities[client_id].device_type
                device_metrics[device_type].append({
                    "loss": eval_res.loss,
                    "accuracy": eval_res.metrics.get("accuracy", 0.0),
                    "num_examples": eval_res.num_examples
                })
        
        # Calculate aggregated metrics
        aggregated_loss = total_loss / total_examples if total_examples > 0 else 0.0
        
        # Per-device performance analysis
        metrics = {
            "aggregated_loss": aggregated_loss,
            "total_examples": total_examples,
        }
        
        for device_type, device_results in device_metrics.items():
            if device_results:
                avg_loss = np.mean([r["loss"] for r in device_results])
                avg_acc = np.mean([r["accuracy"] for r in device_results])
                total_examples = sum([r["num_examples"] for r in device_results])
                
                metrics[f"{device_type}_avg_loss"] = avg_loss
                metrics[f"{device_type}_avg_accuracy"] = avg_acc
                metrics[f"{device_type}_total_examples"] = total_examples
        
        return aggregated_loss, metrics
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side model evaluation"""
        
        if self.evaluate_fn is None:
            return None
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        
        if eval_res is not None:
            loss, metrics = eval_res
            return loss, metrics
        
        return None
    
    def _handle_client_dropouts(self, available_clients: List[ClientProxy]):
        """Handle client dropout and rejoin scenarios"""
        
        current_client_ids = {client.cid for client in available_clients}
        previous_client_ids = set(self.client_capabilities.keys())
        
        # Detect new dropouts
        dropped_this_round = previous_client_ids - current_client_ids
        for client_id in dropped_this_round:
            if client_id not in self.dropped_clients:
                self.dropped_clients[client_id] = {
                    "dropped_at_round": self.current_round,
                    "last_state": self.client_history.get(client_id, [])[-1] if client_id in self.client_history else None
                }
                self.logger.info(f"Client {client_id} dropped at round {self.current_round}")
        
        # Detect rejoining clients
        rejoined_this_round = current_client_ids - previous_client_ids
        for client_id in rejoined_this_round:
            if client_id in self.dropped_clients:
                dropout_info = self.dropped_clients[client_id]
                rounds_absent = self.current_round - dropout_info["dropped_at_round"]
                self.logger.info(f"Client {client_id} rejoined after {rounds_absent} rounds")
                
                # Research Question Q3: Handle rejoin strategy
                # For now, client gets current global model (reset strategy)
                # Alternative: could implement buffered update replay
                
                del self.dropped_clients[client_id]
    
    def _select_clients_for_round(self, available_clients: List[ClientProxy]) -> List[ClientProxy]:
        """Select clients for training round with device type balancing"""
        
        if len(available_clients) < self.min_available_clients:
            return available_clients
        
        # Separate by device type
        jetson_clients = []
        akida_clients = []
        unknown_clients = []
        
        for client in available_clients:
            if client.cid in self.client_capabilities:
                device_type = self.client_capabilities[client.cid].device_type
                if device_type == "jetson":
                    jetson_clients.append(client)
                elif device_type == "akida":
                    akida_clients.append(client)
                else:
                    unknown_clients.append(client)
            else:
                unknown_clients.append(client)
        
        # Balance selection across device types
        selected = []
        
        # Ensure at least one of each type if available
        if jetson_clients:
            selected.extend(jetson_clients[:max(1, len(jetson_clients) // 2)])
        if akida_clients:
            selected.extend(akida_clients[:max(1, len(akida_clients) // 2)])
        
        # Add remaining clients up to min_fit_clients
        remaining_clients = [c for c in available_clients if c not in selected]
        needed = max(0, self.min_fit_clients - len(selected))
        selected.extend(remaining_clients[:needed])
        
        return selected[:self.min_fit_clients]
    
    def _get_client_parameters(self, client: ClientProxy, global_parameters: Parameters) -> Parameters:
        """Get device-specific parameters for a client"""
        
        if client.cid not in self.client_capabilities:
            # Unknown client, send global parameters as-is
            return global_parameters
        
        device_type = self.client_capabilities[client.cid].device_type
        
        # Convert to weights
        global_weights_arrays = parameters_to_ndarrays(global_parameters)
        global_weights = {
            f"layer_{i}": torch.from_numpy(arr) 
            for i, arr in enumerate(global_weights_arrays)
        }
        
        # Get device-specific weights
        if device_type == "jetson":
            device_weights = self.aggregator.get_device_specific_weights(
                global_weights, DeviceType.JETSON
            )
        elif device_type == "akida":
            device_weights = self.aggregator.get_device_specific_weights(
                global_weights, DeviceType.AKIDA
            )
        else:
            device_weights = global_weights
        
        # Convert back to parameters
        device_arrays = [weights.numpy() for weights in device_weights.values()]
        return ndarrays_to_parameters(device_arrays)
    
    def _convert_flower_result_to_update(
        self, client_proxy: ClientProxy, fit_res: FitRes, server_round: int
    ) -> ClientUpdate:
        """Convert Flower FitRes to our ClientUpdate format"""
        
        # Get device type
        device_type = DeviceType.JETSON  # Default
        if client_proxy.cid in self.client_capabilities:
            device_type_str = self.client_capabilities[client_proxy.cid].device_type
            device_type = DeviceType.JETSON if device_type_str == "jetson" else DeviceType.AKIDA
        
        # Convert parameters to weights dict
        weights_arrays = parameters_to_ndarrays(fit_res.parameters)
        weights_dict = {
            f"layer_{i}": torch.from_numpy(arr)
            for i, arr in enumerate(weights_arrays)
        }
        
        # Extract metrics
        local_loss = fit_res.metrics.get("train_loss", 0.0)
        
        return ClientUpdate(
            client_id=client_proxy.cid,
            device_type=device_type,
            weights=weights_dict,
            num_samples=fit_res.num_examples,
            local_loss=local_loss,
            round_number=server_round,
            metadata=fit_res.metrics
        )
    
    def _convert_weights_to_parameters(self, weights_dict: Dict[str, torch.Tensor]) -> Parameters:
        """Convert weights dict back to Flower Parameters"""
        arrays = [weights.numpy() for weights in weights_dict.values()]
        return ndarrays_to_parameters(arrays)
    
    def _calculate_round_metrics(self, updates: List[ClientUpdate], round_num: int) -> Dict[str, Scalar]:
        """Calculate metrics for this round"""
        
        # Get latest aggregator stats
        if self.aggregator.round_stats:
            latest_stats = self.aggregator.round_stats[-1]
        else:
            latest_stats = {}
        
        # Device type distribution
        jetson_count = len([u for u in updates if u.device_type == DeviceType.JETSON])
        akida_count = len([u for u in updates if u.device_type == DeviceType.AKIDA])
        
        # Loss statistics
        losses = [u.local_loss for u in updates]
        
        metrics = {
            "round": round_num,
            "participating_clients": len(updates),
            "jetson_clients": jetson_count,
            "akida_clients": akida_count,
            "avg_client_loss": np.mean(losses),
            "std_client_loss": np.std(losses),
            "min_client_loss": np.min(losses),
            "max_client_loss": np.max(losses),
            "total_samples": sum(u.num_samples for u in updates),
        }
        
        # Add aggregator-specific metrics
        metrics.update(latest_stats)
        
        return metrics
    
    def _log_round_data(self, round_num: int, updates: List[ClientUpdate], metrics: Dict):
        """Log data for experiment analysis"""
        
        round_data = {
            "round": round_num,
            "timestamp": np.datetime64('now').astype(str),
            "metrics": metrics,
            "client_updates": [
                {
                    "client_id": u.client_id,
                    "device_type": u.device_type.value,
                    "num_samples": u.num_samples,
                    "local_loss": u.local_loss,
                    "metadata": u.metadata
                }
                for u in updates
            ],
            "dropped_clients": list(self.dropped_clients.keys()),
            "active_clients": [u.client_id for u in updates]
        }
        
        self.experiment_log.append(round_data)
        
        # Log to file periodically
        if round_num % 10 == 0:
            self._save_experiment_log()
    
    def _save_experiment_log(self):
        """Save experiment log to file"""
        try:
            with open(f"fedmpq_experiment_log_round_{self.current_round}.json", "w") as f:
                json.dump(self.experiment_log, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save experiment log: {e}")
    
    def register_client_capability(self, client_id: str, capability: ClientCapability):
        """Register client capabilities for device-specific handling"""
        self.client_capabilities[client_id] = capability
        self.logger.info(f"Registered {capability.device_type} client: {client_id}")

# Helper function to create strategy
def create_fedmpq_strategy(
    min_clients: int = 2,
    evaluate_fn: Optional[callable] = None,
    jetson_precision: int = 32,
    akida_precision: int = 4,
    **kwargs
) -> FedMPQFlowerStrategy:
    """Create FedMPQ strategy with sensible defaults"""
    
    return FedMPQFlowerStrategy(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_fn=evaluate_fn,
        jetson_precision=jetson_precision,
        akida_precision=akida_precision,
        accept_failures=True,
        **kwargs
    )

if __name__ == "__main__":
    # Example strategy creation
    strategy = create_fedmpq_strategy(min_clients=2)
    print("✅ FedMPQ Flower strategy created successfully!")
    print(f"Jetson precision: {strategy.aggregator.jetson_precision}-bit")
    print(f"Akida precision: {strategy.aggregator.akida_precision}-bit")
    print(f"Error compensation: {strategy.aggregator.error_compensation}")