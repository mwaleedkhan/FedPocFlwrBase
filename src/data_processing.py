"""
Data Processing Utilities for Federated Learning POC

Handles:
- CIFAR-10 data loading and preprocessing
- Non-IID data distribution across clients
- Data shard creation and validation
- Statistical analysis of data heterogeneity
"""

import os
import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FederatedDataset(Dataset):
    """Custom dataset for federated learning with client-specific data shards"""
    
    def __init__(self, base_dataset, indices: List[int], client_id: str, device_type: str):
        self.base_dataset = base_dataset
        self.indices = indices
        self.client_id = client_id
        self.device_type = device_type
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in this shard"""
        labels = [self.base_dataset[idx][1] for idx in self.indices]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

def create_non_iid_shards(
    dataset, 
    num_clients: int, 
    alpha: float = 0.5,
    seed: int = 42
) -> List[List[int]]:
    """
    Create non-IID data shards using Dirichlet distribution
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of federated clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed
        
    Returns:
        List of index lists, one per client
    """
    
    np.random.seed(seed)
    
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = {}
    for class_id in range(num_classes):
        class_indices[class_id] = np.where(labels == class_id)[0]
    
    # Generate Dirichlet distribution for each client
    client_class_distributions = np.random.dirichlet([alpha] * num_classes, num_clients)
    
    # Allocate data to clients
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        # Get indices for this class
        class_idx = class_indices[class_id]
        np.random.shuffle(class_idx)
        
        # Calculate how many samples each client gets from this class
        class_distribution = client_class_distributions[:, class_id]
        class_distribution = class_distribution / class_distribution.sum()
        
        samples_per_client = (class_distribution * len(class_idx)).astype(int)
        
        # Ensure all samples are allocated
        remaining = len(class_idx) - samples_per_client.sum()
        for i in range(remaining):
            samples_per_client[i % num_clients] += 1
        
        # Distribute samples
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + samples_per_client[client_id]
            client_indices[client_id].extend(class_idx[start_idx:end_idx])
            start_idx = end_idx
    
    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices

def create_device_specific_transforms(device_type: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create device-specific data transforms"""
    
    if device_type == "jetson":
        # Full data augmentation for Jetson (more compute)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    elif device_type == "akida":
        # Minimal augmentation for neuromorphic processing
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    else:
        # Default transform
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    # Test transform is the same for all devices
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return train_transform, test_transform

def setup_data_shards(
    num_clients: int = 4,
    dataset: str = "cifar10",
    non_iid_alpha: float = 0.5,
    data_dir: str = "./data",
    save_shards: bool = True
) -> Dict[str, any]:
    """
    Setup federated data shards for all clients
    
    Returns:
        Dictionary containing dataset info and client assignments
    """
    
    logger.info(f"Setting up {dataset} data shards for {num_clients} clients")
    
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Load dataset
    if dataset.lower() == "cifar10":
        # Download CIFAR-10
        base_train_transform = transforms.ToTensor()  # Minimal for indexing
        
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=base_train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=base_train_transform
        )
        
        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Create non-IID training shards
    train_client_indices = create_non_iid_shards(
        trainset, num_clients, alpha=non_iid_alpha
    )
    
    # Create balanced test shards (for evaluation)
    test_client_indices = []
    test_per_client = len(testset) // num_clients
    
    for client_id in range(num_clients):
        start_idx = client_id * test_per_client
        end_idx = start_idx + test_per_client
        if client_id == num_clients - 1:  # Last client gets remaining samples
            end_idx = len(testset)
        test_client_indices.append(list(range(start_idx, end_idx)))
    
    # Assign device types (alternating Jetson and Akida)
    device_assignments = {}
    client_configs = {}
    
    for client_id in range(num_clients):
        if client_id < num_clients // 2:
            device_type = "jetson"
            client_name = f"jetson_{(client_id % (num_clients // 2)) + 1:02d}"
        else:
            device_type = "akida"
            client_name = f"akida_{(client_id % (num_clients // 2)) + 1:02d}"
        
        device_assignments[client_id] = device_type
        client_configs[client_name] = {
            "client_id": client_id,
            "device_type": device_type,
            "train_indices": train_client_indices[client_id],
            "test_indices": test_client_indices[client_id],
            "num_train_samples": len(train_client_indices[client_id]),
            "num_test_samples": len(test_client_indices[client_id])
        }
    
    # Create data summary
    data_summary = {
        "dataset": dataset,
        "num_classes": num_classes,
        "class_names": class_names,
        "total_train_samples": len(trainset),
        "total_test_samples": len(testset),
        "num_clients": num_clients,
        "non_iid_alpha": non_iid_alpha,
        "client_configs": client_configs,
        "device_assignments": device_assignments
    }
    
    # Save data shards if requested
    if save_shards:
        shards_file = data_path / f"data_shards_{dataset}_{num_clients}clients_alpha{non_iid_alpha}.pkl"
        
        with open(shards_file, 'wb') as f:
            pickle.dump({
                "train_indices": train_client_indices,
                "test_indices": test_client_indices,
                "data_summary": data_summary
            }, f)
        
        logger.info(f"Data shards saved to {shards_file}")
    
    # Generate data distribution analysis
    analyze_data_distribution(trainset, train_client_indices, device_assignments, data_dir)
    
    logger.info(f"✅ Data setup complete: {num_clients} clients, α={non_iid_alpha}")
    
    return data_summary

def load_data_shards(shards_file: str) -> Dict:
    """Load pre-created data shards"""
    
    with open(shards_file, 'rb') as f:
        shard_data = pickle.load(f)
    
    return shard_data

def get_client_dataloader(
    client_name: str,
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 32,
    is_train: bool = True,
    shards_file: Optional[str] = None
) -> DataLoader:
    """
    Get DataLoader for a specific client
    
    Args:
        client_name: Name of the client (e.g., "jetson_01", "akida_01")
        dataset: Dataset name
        data_dir: Data directory
        batch_size: Batch size
        is_train: Whether to get training or test data
        shards_file: Path to pre-created shards file
        
    Returns:
        DataLoader for the client
    """
    
    # Load shards data
    if shards_file and os.path.exists(shards_file):
        shard_data = load_data_shards(shards_file)
        data_summary = shard_data["data_summary"]
        train_indices = shard_data["train_indices"]
        test_indices = shard_data["test_indices"]
    else:
        # Create new shards (fallback)
        data_summary = setup_data_shards(data_dir=data_dir, dataset=dataset)
        # This would need to be re-implemented to return indices as well
        raise ValueError("No existing shards file found. Run setup_data_shards first.")
    
    # Get client configuration
    if client_name not in data_summary["client_configs"]:
        raise ValueError(f"Client {client_name} not found in data configuration")
    
    client_config = data_summary["client_configs"][client_name]
    device_type = client_config["device_type"]
    
    # Get appropriate transforms
    train_transform, test_transform = create_device_specific_transforms(device_type)
    
    # Load base dataset
    if dataset.lower() == "cifar10":
        if is_train:
            base_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=False, transform=train_transform
            )
            indices = client_config["train_indices"]
        else:
            base_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=False, transform=test_transform
            )
            indices = client_config["test_indices"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Create federated dataset
    federated_dataset = FederatedDataset(
        base_dataset=base_dataset,
        indices=indices,
        client_id=client_name,
        device_type=device_type
    )
    
    # Adjust batch size for device type
    if device_type == "akida":
        batch_size = min(batch_size, 16)  # Smaller batches for neuromorphic
    
    # Create DataLoader
    dataloader = DataLoader(
        federated_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader

def analyze_data_distribution(
    dataset, 
    client_indices: List[List[int]], 
    device_assignments: Dict[int, str],
    output_dir: str
):
    """Analyze and visualize data distribution across clients"""
    
    logger.info("Analyzing data distribution across clients")
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    num_clients = len(client_indices)
    
    # Calculate distribution matrix
    distribution_matrix = np.zeros((num_clients, num_classes))
    
    for client_id, indices in enumerate(client_indices):
        client_labels = labels[indices]
        for class_id in range(num_classes):
            distribution_matrix[client_id, class_id] = np.sum(client_labels == class_id)
    
    # Normalize to get proportions
    proportion_matrix = distribution_matrix / distribution_matrix.sum(axis=1, keepdims=True)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Raw counts
    plt.subplot(2, 2, 1)
    sns.heatmap(distribution_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Sample Counts per Client-Class')
    plt.xlabel('Class')
    plt.ylabel('Client')
    
    # Subplot 2: Proportions
    plt.subplot(2, 2, 2)
    sns.heatmap(proportion_matrix, annot=True, fmt='.2f', cmap='RdYlBu')
    plt.title('Class Proportions per Client')
    plt.xlabel('Class')
    plt.ylabel('Client')
    
    # Subplot 3: Samples per client
    plt.subplot(2, 2, 3)
    samples_per_client = distribution_matrix.sum(axis=1)
    device_colors = ['blue' if device_assignments[i] == 'jetson' else 'red' 
                    for i in range(num_clients)]
    
    bars = plt.bar(range(num_clients), samples_per_client, color=device_colors, alpha=0.7)
    plt.title('Total Samples per Client')
    plt.xlabel('Client')
    plt.ylabel('Number of Samples')
    
    # Add legend for device types
    jetson_patch = plt.Rectangle((0,0),1,1, fc='blue', alpha=0.7)
    akida_patch = plt.Rectangle((0,0),1,1, fc='red', alpha=0.7)
    plt.legend([jetson_patch, akida_patch], ['Jetson', 'Akida'])
    
    # Subplot 4: Class distribution diversity
    plt.subplot(2, 2, 4)
    # Calculate entropy for each client (measure of diversity)
    entropies = []
    for client_id in range(num_clients):
        props = proportion_matrix[client_id]
        # Add small epsilon to avoid log(0)
        props = props + 1e-10
        entropy = -np.sum(props * np.log(props))
        entropies.append(entropy)
    
    bars = plt.bar(range(num_clients), entropies, color=device_colors, alpha=0.7)
    plt.title('Data Diversity per Client (Entropy)')
    plt.xlabel('Client')
    plt.ylabel('Entropy')
    plt.axhline(y=np.log(num_classes), color='black', linestyle='--', 
                label=f'Max Entropy ({np.log(num_classes):.2f})')
    plt.legend()
    
    plt.tight_layout()
    
    # Save visualization
    output_path = Path(output_dir) / "data_distribution_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save distribution statistics
    stats = {
        "distribution_matrix": distribution_matrix.tolist(),
        "proportion_matrix": proportion_matrix.tolist(),
        "samples_per_client": samples_per_client.tolist(),
        "entropies": entropies,
        "device_assignments": device_assignments,
        "mean_entropy": np.mean(entropies),
        "std_entropy": np.std(entropies)
    }
    
    stats_file = Path(output_dir) / "data_distribution_stats.pkl"
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    
    logger.info(f"Data distribution analysis saved to {output_dir}")
    
    return stats

def validate_data_distribution(data_summary: Dict) -> bool:
    """Validate that data distribution is reasonable"""
    
    logger.info("Validating data distribution")
    
    client_configs = data_summary["client_configs"]
    
    # Check that all clients have data
    for client_name, config in client_configs.items():
        if config["num_train_samples"] == 0:
            logger.error(f"Client {client_name} has no training data")
            return False
        
        if config["num_test_samples"] == 0:
            logger.error(f"Client {client_name} has no test data")
            return False
    
    # Check total sample count
    total_train = sum(config["num_train_samples"] for config in client_configs.values())
    total_test = sum(config["num_test_samples"] for config in client_configs.values())
    
    expected_train = data_summary["total_train_samples"]
    expected_test = data_summary["total_test_samples"]
    
    if total_train != expected_train:
        logger.error(f"Train sample mismatch: {total_train} != {expected_train}")
        return False
    
    if total_test != expected_test:
        logger.error(f"Test sample mismatch: {total_test} != {expected_test}")
        return False
    
    # Check device type balance
    device_counts = {}
    for config in client_configs.values():
        device_type = config["device_type"]
        device_counts[device_type] = device_counts.get(device_type, 0) + 1
    
    if len(device_counts) < 2:
        logger.warning("Only one device type found - this is homogeneous FL")
    
    logger.info("✅ Data distribution validation passed")
    logger.info(f"Device distribution: {device_counts}")
    
    return True

def create_centralized_dataset(
    dataset: str = "cifar10",
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """Create centralized dataset (for baseline comparison)"""
    
    # Standard transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if dataset.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data setup
    print("🧪 Testing data setup...")
    
    data_summary = setup_data_shards(
        num_clients=4,
        dataset="cifar10",
        non_iid_alpha=0.5,
        data_dir="./test_data"
    )
    
    print(f"✅ Data setup complete for {data_summary['num_clients']} clients")
    
    # Test client dataloader
    print("\n🧪 Testing client dataloader...")
    
    # This would need the shards file path
    # dataloader = get_client_dataloader("jetson_01", batch_size=16, is_train=True)
    # print(f"✅ Client dataloader created with {len(dataloader)} batches")
    
    # Validate distribution
    print("\n🧪 Validating data distribution...")
    is_valid = validate_data_distribution(data_summary)
    print(f"✅ Validation {'passed' if is_valid else 'failed'}")
    
    print("\n🎉 Data utilities test completed!")