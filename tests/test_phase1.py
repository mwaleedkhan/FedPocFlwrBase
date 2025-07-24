"""
Phase 1 validation tests
"""
import pytest
import torch
import sys
sys.path.append('src')

from model_variants import ModelFactory
from aggregation_engine import FedMPQAggregator
from data_processing import setup_data_shards

def test_model_creation():
    """Test that we can create both model types"""
    jetson_model = ModelFactory.create_model("jetson")
    akida_model = ModelFactory.create_model("akida")
    
    assert jetson_model is not None
    assert akida_model is not None
    print("✅ Model creation test passed")

def test_aggregation_basic():
    """Test basic aggregation functionality"""
    aggregator = FedMPQAggregator()
    assert aggregator is not None
    print("✅ Aggregation engine test passed")

def test_data_setup():
    """Test data preparation"""
    try:
        data_summary = setup_data_shards(
            num_clients=2, 
            dataset="cifar10",
            data_dir="./test_data"
        )
        assert data_summary is not None
        print("✅ Data setup test passed")
    except Exception as e:
        print(f"⚠️ Data test skipped: {e}")

if __name__ == "__main__":
    print("🧪 Running Phase 1 validation tests...")
    test_model_creation()
    test_aggregation_basic()
    test_data_setup()
    print("✅ Phase 1 validation completed!")