"""
Configuration Management and Utilities for FL POC

Handles:
- YAML configuration loading and validation
- Environment variable management
- Logging setup
- Dependency checking
- Directory creation and management
"""

import os
import sys
import yaml
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pkg_resources
import torch

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_environment()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with defaults"""
        
        # Default configuration
        default_config = {
            "experiment": {
                "name": "heterogeneous_fl_poc",
                "description": "Mixed-precision federated learning with Jetson and Akida",
                "output_dir": "./results",
                "random_seed": 42
            },
            "data": {
                "dataset": "cifar10",
                "data_dir": "./data",
                "num_clients": 4,
                "non_iid_alpha": 0.5,
                "batch_size_jetson": 32,
                "batch_size_akida": 16
            },
            "model": {
                "num_classes": 10,
                "width_multiplier": 1.0,
                "jetson_precision": 32,
                "akida_precision": 4,
                "akida_sparsity": 0.7
            },
            "training": {
                "num_rounds": 50,
                "local_epochs": 1,
                "jetson_lr": 0.01,
                "akida_lr": 0.001,
                "weight_decay": 1e-4,
                "min_clients_per_round": 2
            },
            "aggregation": {
                "strategy": "fedmpq",
                "error_compensation": True,
                "adaptive_aggregation": True,
                "dropout_tolerance": 0.3
            },
            "research_questions": {
                "enabled": ["Q1", "Q2", "Q3", "Q4"],
                "Q1": {
                    "rejoin_strategies": ["reset", "continue_local", "weighted_merge"],
                    "dropout_duration": 5
                },
                "Q2": {
                    "compare_strategies": ["continue_training", "remain_idle"]
                },
                "Q3": {
                    "state_strategies": ["reset", "continue_local", "weighted_merge"]
                },
                "Q4": {
                    "dropout_rounds": [5, 15, 30]
                }
            },
            "execution": {
                "parallel": False,
                "max_parallel": 2,
                "timeout_minutes": 60,
                "retry_failed": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "fl_poc.log",
                "console": True
            },
            "network": {
                "flower_server_port": 8000,
                "zeromq_base_port": 6000,
                "dashboard_port": 3000,
                "timeout_seconds": 30
            },
            "docker": {
                "enabled": False,
                "network_name": "fl_network",
                "compose_file": "docker-compose.yml"
            }
        }
        
        # Load user configuration if exists
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                
                # Merge configurations (user overrides default)
                config = self._deep_merge(default_config, user_config)
                
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration")
                config = default_config
        else:
            # Create default config file
            self._save_config(default_config)
            config = default_config
        
        return config
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _save_config(self, config: Dict):
        """Save configuration to YAML file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save config to {self.config_path}: {e}")
    
    def _setup_environment(self):
        """Setup environment variables from config"""
        
        # Set random seeds
        seed = self.config.get("experiment", {}).get("random_seed", 42)
        torch.manual_seed(seed)
        
        # Set environment variables
        env_vars = {
            "PYTHONPATH": str(Path.cwd()),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "FL_CONFIG_PATH": str(self.config_path),
            "FL_OUTPUT_DIR": self.config.get("experiment", {}).get("output_dir", "./results")
        }
        
        for key, value in env_vars.items():
            if value:
                os.environ[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'data.batch_size')"""
        
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self):
        """Save current configuration to file"""
        self._save_config(self.config)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        
        issues = []
        
        # Check required fields
        required_fields = [
            "experiment.name",
            "data.dataset",
            "data.num_clients",
            "model.num_classes",
            "training.num_rounds"
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                issues.append(f"Missing required field: {field}")
        
        # Validate values
        if self.get("data.num_clients", 0) < 2:
            issues.append("data.num_clients must be at least 2")
        
        if self.get("training.num_rounds", 0) < 1:
            issues.append("training.num_rounds must be at least 1")
        
        if self.get("data.non_iid_alpha", 0) <= 0:
            issues.append("data.non_iid_alpha must be positive")
        
        # Check device configuration
        num_clients = self.get("data.num_clients", 4)
        if num_clients % 2 != 0:
            issues.append("num_clients should be even for balanced Jetson/Akida split")
        
        return issues

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    
    # Get logging configuration
    log_level = config.get("level", "INFO")
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("file", "fl_poc.log")
    console_logging = config.get("console", True)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create logs directory
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create logger for our package
    logger = logging.getLogger("fl_poc")
    logger.info("Logging setup complete")
    
    return logger

def check_dependencies() -> bool:
    """Check if all required dependencies are installed"""
    
    required_packages = [
        "torch",
        "torchvision", 
        "flwr",
        "numpy",
        "pandas",
        "plotly",
        "zmq",
        "yaml",
        "matplotlib",
        "seaborn",
        "tqdm",
        "sklearn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "yaml":
                importlib.import_module("yaml")
            elif package == "zmq":
                importlib.import_module("zmq")
            elif package == "sklearn":
                importlib.import_module("sklearn")
            else:
                importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    # Check PyTorch version
    try:
        torch_version = torch.__version__
        major, minor = map(int, torch_version.split('.')[:2])
        if major < 2:
            print(f"⚠️  PyTorch version {torch_version} is older than recommended (2.0+)")
        else:
            print(f"✅ PyTorch version {torch_version}")
    except:
        print("⚠️  Could not verify PyTorch version")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("ℹ️  CUDA not available - using CPU only")
    
    print("✅ All dependencies satisfied")
    return True

def create_directories(directories: List[str], base_path: str = ".") -> bool:
    """Create necessary directories"""
    
    base = Path(base_path)
    
    try:
        for directory in directories:
            dir_path = base / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Created directories: {', '.join(directories)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return False

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    
    import platform
    import psutil
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        })
    
    return info

def print_system_info():
    """Print system information"""
    
    info = get_system_info()
    
    print("\n" + "="*50)
    print("🖥️  SYSTEM INFORMATION")
    print("="*50)
    
    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['torch_version']}")
    print(f"CPU cores: {info['cpu_count']}")
    print(f"Memory: {info['memory_gb']} GB total, {info['available_memory_gb']} GB available")
    
    if info['cuda_available']:
        print(f"CUDA: {info['cuda_version']}")
        print(f"GPUs: {info['gpu_count']} ({', '.join(info['gpu_names'])})")
    else:
        print("CUDA: Not available")
    
    print("="*50)

def create_experiment_config(
    experiment_name: str,
    research_questions: List[str],
    num_clients: int = 4,
    num_rounds: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """Create experiment-specific configuration"""
    
    config = {
        "experiment": {
            "name": experiment_name,
            "description": f"FL experiment for {', '.join(research_questions)}",
            "timestamp": str(pd.Timestamp.now()),
            "research_questions": research_questions
        },
        "data": {
            "num_clients": num_clients
        },
        "training": {
            "num_rounds": num_rounds
        }
    }
    
    # Add any additional parameters
    for key, value in kwargs.items():
        if '.' in key:
            # Handle nested keys
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            config[key] = value
    
    return config

# Utility functions for environment management
def set_environment_variables(config: Dict[str, Any]):
    """Set environment variables based on configuration"""
    
    env_mapping = {
        "FL_OUTPUT_DIR": config.get("experiment", {}).get("output_dir", "./results"),
        "FL_DATA_DIR": config.get("data", {}).get("data_dir", "./data"),
        "FL_NUM_CLIENTS": str(config.get("data", {}).get("num_clients", 4)),
        "FL_NUM_ROUNDS": str(config.get("training", {}).get("num_rounds", 50)),
        "FL_FLOWER_PORT": str(config.get("network", {}).get("flower_server_port", 8000)),
        "FL_ZEROMQ_BASE_PORT": str(config.get("network", {}).get("zeromq_base_port", 6000)),
        "FL_DASHBOARD_PORT": str(config.get("network", {}).get("dashboard_port", 3000))
    }
    
    for key, value in env_mapping.items():
        os.environ[key] = value

def validate_gpu_setup() -> Dict[str, Any]:
    """Validate GPU setup for Jetson simulation"""
    
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "memory_info": [],
        "recommendations": []
    }
    
    if torch.cuda.is_available():
        gpu_info["device_count"] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            gpu_info["devices"].append(device_name)
            
            # Get memory info
            memory_total = torch.cuda.get_device_properties(i).total_memory
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            
            gpu_info["memory_info"].append({
                "device": i,
                "total_gb": memory_total / (1024**3),
                "reserved_gb": memory_reserved / (1024**3),
                "allocated_gb": memory_allocated / (1024**3),
                "available_gb": (memory_total - memory_reserved) / (1024**3)
            })
        
        # Generate recommendations
        if gpu_info["device_count"] >= 2:
            gpu_info["recommendations"].append("Multiple GPUs detected - can simulate multiple Jetson clients")
        elif gpu_info["device_count"] == 1:
            gpu_info["recommendations"].append("Single GPU detected - will simulate all Jetson clients on same device")
        
        total_memory = sum(info["total_gb"] for info in gpu_info["memory_info"])
        if total_memory < 4:
            gpu_info["recommendations"].append("Low GPU memory - consider reducing batch sizes")
        
    else:
        gpu_info["recommendations"].append("No CUDA GPUs detected - Jetson clients will run on CPU")
        gpu_info["recommendations"].append("Performance will be slower but experiments will still work")
    
    return gpu_info

def create_run_scripts(config: Dict[str, Any], output_dir: str = "."):
    """Create convenience run scripts"""
    
    output_path = Path(output_dir)
    
    # Main run script
    run_script = f"""#!/bin/bash
# Heterogeneous Federated Learning POC - Main Runner
# Generated automatically

set -e

echo "🚀 Starting Heterogeneous FL POC"
echo "================================"

# Check dependencies
echo "📦 Checking dependencies..."
python -c "from src.utils import check_dependencies; check_dependencies()"

# Setup data
echo "📊 Setting up data shards..."
python -c "from src.data_utils import setup_data_shards; setup_data_shards()"

# Run experiments based on mode
MODE=${{1:-full}}

case $MODE in
    "setup")
        echo "🔧 Running setup only..."
        python main.py --mode setup
        ;;
    "baseline")
        echo "📊 Running baseline experiments..."
        python main.py --mode baseline
        ;;
    "research")
        echo "🔬 Running research question experiments..."
        python main.py --mode research
        ;;
    "analysis")
        echo "📈 Running analysis and dashboard generation..."
        python main.py --mode analysis
        ;;
    "full")
        echo "🚀 Running full experimental pipeline..."
        python main.py --mode full
        ;;
    *)
        echo "Usage: $0 [setup|baseline|research|analysis|full]"
        exit 1
        ;;
esac

echo "✅ Experiment completed!"
echo "📁 Results available in: {config.get('experiment', {}).get('output_dir', './results')}"
echo "🌐 Dashboard: http://localhost:{config.get('network', {}).get('dashboard_port', 3000)}"
"""
    
    with open(output_path / "run_experiment.sh", 'w') as f:
        f.write(run_script)
    
    # Make executable
    os.chmod(output_path / "run_experiment.sh", 0o755)
    
    # Docker run script
    docker_script = f"""#!/bin/bash
# Docker-based FL POC Runner

set -e

echo "🐳 Starting FL POC with Docker"
echo "=============================="

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting FL services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 30

# Run experiments
echo "🔬 Running experiments..."
docker-compose exec fl-orchestrator python main.py --mode full

# Show logs
echo "📋 Recent logs:"
docker-compose logs --tail=50 fl-orchestrator

echo "✅ Docker experiment completed!"
echo "🌐 Dashboard: http://localhost:{config.get('network', {}).get('dashboard_port', 3000)}"
echo "📁 Results mounted to: ./results"

# Optional: Keep services running
read -p "Keep services running? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "🛑 Stopping services..."
    docker-compose down
fi
"""
    
    with open(output_path / "run_docker.sh", 'w') as f:
        f.write(docker_script)
    
    os.chmod(output_path / "run_docker.sh", 0o755)
    
    # Python quick start script
    python_script = f"""#!/usr/bin/env python3
\"\"\"
Quick Start Script for FL POC
\"\"\"

import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from main import FederatedLearningPOC

def quick_start():
    \"\"\"Quick start with default settings\"\"\"
    
    print("🚀 Quick Start - Heterogeneous FL POC")
    print("=====================================")
    
    # Initialize POC
    poc = FederatedLearningPOC()
    
    # Print system info
    from src.utils import print_system_info
    print_system_info()
    
    # Run validation
    print("\\n🔍 Validating environment...")
    if not poc.validate_environment():
        print("❌ Environment validation failed")
        return False
    
    # Ask user what to run
    print("\\n🎯 What would you like to run?")
    print("1. Full experimental pipeline (recommended)")
    print("2. Just baseline experiments")
    print("3. Just research question experiments") 
    print("4. Setup and validation only")
    
    choice = input("Enter choice (1-4): ").strip()
    
    try:
        if choice == "1":
            results = poc.run_full_pipeline()
            print("\\n🎉 Full pipeline completed!")
            
        elif choice == "2":
            poc.run_baseline_experiments()
            print("\\n📊 Baseline experiments completed!")
            
        elif choice == "3":
            poc.run_research_experiments()
            print("\\n🔬 Research experiments completed!")
            
        elif choice == "4":
            poc.run_setup_phase()
            print("\\n🔧 Setup completed!")
            
        else:
            print("Invalid choice")
            return False
        
        print(f"\\n📁 Check results in: {poc.config.get('experiment.output_dir', './results')}")
        return True
        
    except Exception as e:
        print(f"\\n❌ Error: {{e}}")
        return False

if __name__ == "__main__":
    success = quick_start()
    sys.exit(0 if success else 1)
"""
    
    with open(output_path / "quick_start.py", 'w') as f:
        f.write(python_script)
    
    os.chmod(output_path / "quick_start.py", 0o755)
    
    print(f"✅ Created run scripts in {output_path}")
    print("Available scripts:")
    print("  - run_experiment.sh: Main experiment runner")
    print("  - run_docker.sh: Docker-based runner")  
    print("  - quick_start.py: Interactive Python runner")

# Testing and validation utilities
def run_component_tests() -> Dict[str, bool]:
    """Run tests for individual components"""
    
    test_results = {}
    
    print("🧪 Running component tests...")
    
    # Test model creation
    try:
        from src.mobilenet_variants import ModelFactory
        
        jetson_model = ModelFactory.create_model("jetson")
        akida_model = ModelFactory.create_model("akida")
        
        test_results["model_creation"] = True
        print("✅ Model creation test passed")
        
    except Exception as e:
        test_results["model_creation"] = False
        print(f"❌ Model creation test failed: {e}")
    
    # Test aggregation
    try:
        from src.fedmpq_algorithm import FedMPQAggregator
        
        aggregator = FedMPQAggregator()
        test_results["aggregation"] = True
        print("✅ Aggregation test passed")
        
    except Exception as e:
        test_results["aggregation"] = False
        print(f"❌ Aggregation test failed: {e}")
    
    # Test data loading
    try:
        from src.data_utils import setup_data_shards
        
        # Quick test with minimal data
        data_summary = setup_data_shards(
            num_clients=2, 
            data_dir="./test_data",
            save_shards=False
        )
        
        test_results["data_loading"] = True
        print("✅ Data loading test passed")
        
    except Exception as e:
        test_results["data_loading"] = False
        print(f"❌ Data loading test failed: {e}")
    
    # Test configuration
    try:
        config_manager = ConfigManager()
        issues = config_manager.validate()
        
        if not issues:
            test_results["configuration"] = True
            print("✅ Configuration test passed")
        else:
            test_results["configuration"] = False
            print(f"❌ Configuration test failed: {issues}")
            
    except Exception as e:
        test_results["configuration"] = False
        print(f"❌ Configuration test failed: {e}")
    
    # Summary
    passed = sum(test_results.values())
    total = len(test_results)
    
    print(f"\\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All component tests passed!")
    else:
        print("⚠️  Some tests failed - check error messages above")
    
    return test_results

# Example configuration creation
def create_sample_configs():
    """Create sample configuration files"""
    
    # Basic config
    basic_config = {
        "experiment": {
            "name": "basic_fl_test",
            "description": "Basic FL test with minimal settings"
        },
        "data": {
            "num_clients": 4,
            "non_iid_alpha": 0.5
        },
        "training": {
            "num_rounds": 20,
            "local_epochs": 1
        },
        "research_questions": {
            "enabled": ["Q1"]
        }
    }
    
    # Advanced config
    advanced_config = {
        "experiment": {
            "name": "advanced_fl_experiment",
            "description": "Advanced FL experiment with all features"
        },
        "data": {
            "num_clients": 8,
            "non_iid_alpha": 0.3
        },
        "training": {
            "num_rounds": 100,
            "local_epochs": 2
        },
        "research_questions": {
            "enabled": ["Q1", "Q2", "Q3", "Q4"]
        },
        "execution": {
            "parallel": True,
            "max_parallel": 4
        }
    }
    
    # Save configs
    configs = {
        "config_basic.yaml": basic_config,
        "config_advanced.yaml": advanced_config
    }
    
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✅ Created sample configuration files:")
    for filename in configs.keys():
        print(f"  - {filename}")

if __name__ == "__main__":
    # Demo the configuration system
    print("🧪 Testing Configuration Manager")
    print("================================")
    
    # Test config loading
    config_manager = ConfigManager("test_config.yaml")
    
    # Print some config values
    print(f"Experiment name: {config_manager.get('experiment.name')}")
    print(f"Number of clients: {config_manager.get('data.num_clients')}")
    print(f"Number of rounds: {config_manager.get('training.num_rounds')}")
    
    # Validate configuration
    issues = config_manager.validate()
    if issues:
        print(f"Configuration issues: {issues}")
    else:
        print("✅ Configuration is valid")
    
    # Test system info
    print("\\n🖥️  System Information:")
    print_system_info()
    
    # Test dependencies
    print("\\n📦 Dependency Check:")
    check_dependencies()
    
    # Run component tests
    print("\\n🧪 Component Tests:")
    run_component_tests()
    
    print("\\n🎉 Configuration manager test completed!")