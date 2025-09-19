"""
Configuration loader for SpecECD project
"""
import toml
import os
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml"""
    config_path = Path(__file__).parent.parent / "config.toml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return toml.load(f)
    else:
        # Default configuration if file doesn't exist
        return {
            "models": {
                "edge_model": "meta-llama/Llama-3.2-1B-Instruct",
                "cloud_model": "meta-llama/Llama-3.1-8B-Instruct",
                "max_edge_tokens": 5,
                "max_cloud_tokens": 10
            },
            "devices": {
                "edge_device": "cpu",
                "cloud_device": "auto",
                "gpu": {
                    "enabled": False,
                    "device_id": 0
                },
                "npu": {
                    "enabled": False,
                    "fallback_to_cpu": True
                }
            },
            "network": {
                "default_host": "localhost", 
                "default_port": 8765
            },
            "performance": {
                "temperature": 0.7,
                "repetition_penalty": 1.1
            }
        }

def get_edge_model_config() -> Dict[str, Any]:
    """Get edge model configuration with device selection"""
    config = load_config()
    
    # Read requested device and low-level backend settings
    edge_device = config.get("devices", {}).get("edge_device", "cpu")
    gpu_config = config.get("devices", {}).get("gpu", {})
    npu_config = config.get("devices", {}).get("npu", {})

    # Return raw configuration values so callers can decide how to apply them
    return {
        "model_name": config.get("models", {}).get("edge_model", "meta-llama/Llama-3.2-1B-Instruct"),
        "device": edge_device,
        "max_tokens": config.get("models", {}).get("max_edge_tokens", 5),
        "temperature": config.get("performance", {}).get("temperature", 0.7),
        "repetition_penalty": config.get("performance", {}).get("repetition_penalty", 1.1),
        "gpu_enabled": bool(gpu_config.get("enabled", False)),
        "gpu_device_id": int(gpu_config.get("device_id", 0)),
        "npu_enabled": bool(npu_config.get("enabled", False)),
        "npu_fallback": bool(npu_config.get("fallback_to_cpu", True))
    }

def get_cloud_model_config() -> Dict[str, Any]:
    """Get cloud model configuration"""
    config = load_config()
    return {
        "model_name": config["models"]["cloud_model"], 
        "device": config["devices"]["cloud_device"],
        "max_tokens": config["models"]["max_cloud_tokens"],
        "temperature": config["performance"]["temperature"],
        "repetition_penalty": config["performance"]["repetition_penalty"]
    }

def get_network_config() -> Dict[str, Any]:
    """Get network configuration"""
    config = load_config()
    return {
        "host": config["network"]["default_host"],
        "port": config["network"]["default_port"],
        "ping_interval": config["network"].get("ping_interval", 300),
        "ping_timeout": config["network"].get("ping_timeout", 300)
    }

def get_performance_config() -> Dict[str, Any]:
    """Get performance test configuration"""
    config = load_config()
    return {
        "warmup_iterations": config["performance"].get("warmup_iterations", 2),
        "test_iterations": config["performance"].get("test_iterations", 2),
        "max_tokens_per_test": config["performance"].get("max_tokens_per_test", 50),
        "temperature": config["performance"].get("temperature", 0.7),
        "repetition_penalty": config["performance"].get("repetition_penalty", 1.1)
    }

def get_deferral_config() -> Dict[str, Any]:
    """Get speculative cascade deferral strategy configuration"""
    config = load_config()
    # Provide safe defaults if section missing
    def_cfg = config.get("deferral", {})
    return {
        "strategy": def_cfg.get("strategy", "never"),
        "batch_threshold": float(def_cfg.get("batch_threshold", 0.0)),  # Only used by adaptive strategies
        "cumulative_threshold": float(def_cfg.get("cumulative_threshold", 0.0)),
        "min_batches_before_consider": int(def_cfg.get("min_batches_before_consider", 1)),
        "require_consecutive_failures": bool(def_cfg.get("require_consecutive_failures", True)),
        "scope": def_cfg.get("scope", "request")
    }

def get_fast_model_config() -> Dict[str, Any]:
    """Get fast model configuration for testing"""
    config = load_config()
    fast_config = config.get("models", {}).get("fast", {})
    if not fast_config:
        # Return None if no fast config exists
        return None
    
    return {
        "cloud_model": fast_config.get("cloud_model"),
        "expected_inference_time": fast_config.get("expected_inference_time")
    }

def set_environment_variables():
    """Set environment variables from configuration"""
    config = load_config()
    
    # Set TORCH environment variables if specified
    torch_cuda_dsa = config.get("TORCH_USE_CUDA_DSA")
    if torch_cuda_dsa is not None:
        os.environ["TORCH_USE_CUDA_DSA"] = str(torch_cuda_dsa)
    
    cuda_launch_blocking = config.get("CUDA_LAUNCH_BLOCKING")
    if cuda_launch_blocking is not None:
        os.environ["CUDA_LAUNCH_BLOCKING"] = str(cuda_launch_blocking)
