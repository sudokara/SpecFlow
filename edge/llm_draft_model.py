"""
LLM-based draft model implementation for speculative decoding
Runs a small language model (<3B parameters) to generate draft tokens
Supports CPU, GPU, and OpenVINO NPU acceleration
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Optional, Tuple, Dict, Any
import time
import logging
from common.protocol import PerformanceMetrics, CreateTimestamp
from edge.draft_model_interface import DraftModelInterface

# Try to import OpenVINO NPU support
try:
    from edge.openvino_model import OpenVINONPUModel, IsNPUAvailable
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    OpenVINONPUModel = None
    IsNPUAvailable = lambda: False

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class LLMDraftModel(DraftModelInterface):
    """LLM-based draft model running on edge device with CPU, GPU, or NPU support"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", device: str = "cpu", gpu_device_id: int = 0, config=None):
        """
        Initialize LLM draft model with device selection
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ("cpu", "gpu", or "npu")
            gpu_device_id: GPU index to use when device is "gpu"
        """
        super().__init__(model_name, device, gpu_device_id=gpu_device_id)
        self.config = config if config is not None else {}
        self.m_tokenizer = None
        self.m_model = None
        self.m_generation_config = None
        self.m_npu_model = None
        self.m_cuda_device = None
        self.m_gpu_device_id = int(gpu_device_id)
        
        # Validate device selection
        if self.m_device == "gpu":
            if not torch.cuda.is_available():
                g_logger.error("GPU device requested but CUDA not available")
                g_logger.info("Falling back to CPU device")
                self.m_device = "cpu"
            else:
                # Set CUDA device using configured GPU index
                self.m_cuda_device = f"cuda:{self.m_gpu_device_id}"
                try:
                    g_logger.info(f"GPU available: {torch.cuda.get_device_name(self.m_gpu_device_id)}")
                    g_logger.info(f"GPU memory: {torch.cuda.get_device_properties(self.m_gpu_device_id).total_memory / 1e9:.1f} GB")
                except Exception:
                    # Fallback to device 0 logging if index lookup fails
                    g_logger.info(f"GPU available (index {self.m_gpu_device_id})")
        
        elif self.m_device == "npu":
            if not OPENVINO_AVAILABLE:
                g_logger.error("NPU device requested but OpenVINO not available")
                g_logger.info("Install OpenVINO with: pip install openvino openvino-dev optimum[openvino]")
                raise RuntimeError("OpenVINO NPU support not available")
            
            if not IsNPUAvailable():
                g_logger.error("NPU device requested but no NPU hardware detected")
                g_logger.info("Falling back to CPU device")
                self.m_device = "cpu"
        
        if self.m_device not in ["cpu", "gpu", "npu"]:
            g_logger.warning(f"Unknown device '{device}', falling back to CPU")
            self.m_device = "cpu"
        
    def LoadModel(self) -> bool:
        """Load the draft model and tokenizer based on device selection"""
        try:
            if self.m_device == "npu":
                return self._LoadNPUModel()
            elif self.m_device == "gpu":
                return self._LoadGPUModel()
            else:
                return self._LoadCPUModel()
                
        except Exception as e:
            g_logger.error(f"Failed to load LLM edge model: {e}")
            return False
    
    def _LoadNPUModel(self) -> bool:
        """Load model for NPU inference using OpenVINO"""
        g_logger.info("Loading LLM model for NPU inference...")
        
        # Create and load NPU model (OpenVINO wrapper handles model selection)
        self.m_npu_model = OpenVINONPUModel(self.m_model_name)
        
        if not self.m_npu_model.LoadModel():
            g_logger.error("Failed to load NPU model")
            g_logger.info("NPU loading failed. Common causes:")
            g_logger.info("1. NPU driver/hardware compatibility (update Intel drivers)")
            g_logger.info("2. OpenVINO version compatibility")
            g_logger.info("3. Model format issues (now using pre-converted model)")
            g_logger.info("Recommended fallbacks:")
            g_logger.info("  python run_tests.py --device cpu")
            g_logger.info("  python run_tests.py --device gpu")
            return False
        
        # Get tokenizer from NPU model
        self.m_tokenizer = self.m_npu_model.m_tokenizer
        
        g_logger.info("NPU LLM model loaded successfully")
        return True
    
    def _LoadGPUModel(self) -> bool:
        """Load model for GPU inference using PyTorch CUDA"""
        g_logger.info("Loading LLM model for GPU inference...")
        
        # Load tokenizer
        self.m_tokenizer = AutoTokenizer.from_pretrained(
            self.m_model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.m_tokenizer.pad_token is None:
            self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
        
        # Load model with GPU optimizations
        g_logger.info(f"Loading model to {self.m_cuda_device}...")
        self.m_model = AutoModelForCausalLM.from_pretrained(
            self.m_model_name,
            torch_dtype=torch.float16,  # Use float16 for GPU memory efficiency
            # device_map="auto",  # Automatically distribute across available GPUs
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.m_cuda_device)
        
        # Configure generation parameters optimized for GPU
        self.m_generation_config = GenerationConfig(
            max_new_tokens=self.config.get("max_tokens", 5),  # Generate few tokens as draft
            temperature=self.config.get("temperature", 0.7),
            do_sample=self.config.get("do_sample", True),
            pad_token_id=self.m_tokenizer.pad_token_id,
            eos_token_id=self.m_tokenizer.eos_token_id,
            repetition_penalty=self.config.get("repetition_penalty", 1.1),
            use_cache=True  # Enable KV cache for faster generation
        )
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            cached = torch.cuda.memory_reserved(0) / 1e9
            g_logger.info(f"GPU memory allocated: {allocated:.2f} GB")
            g_logger.info(f"GPU memory cached: {cached:.2f} GB")
        
        g_logger.info("GPU LLM model loaded successfully")
        return True
    
    def _LoadCPUModel(self) -> bool:
        """Load model for CPU inference using PyTorch"""
        g_logger.info("Loading LLM model for CPU inference...")
        
        # Load tokenizer
        self.m_tokenizer = AutoTokenizer.from_pretrained(
            self.m_model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.m_tokenizer.pad_token is None:
            self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
        
        # Load model with CPU optimizations
        self.m_model = AutoModelForCausalLM.from_pretrained(
            self.m_model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # Load on CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to("cpu")
        
        # Configure generation parameters
        self.m_generation_config = GenerationConfig(
            max_new_tokens=self.config.get("max_tokens", 5),  # Generate few tokens as draft
            temperature=self.config.get("temperature", 0.7),
            do_sample=self.config.get("do_sample", True),
            pad_token_id=self.m_tokenizer.pad_token_id,
            eos_token_id=self.m_tokenizer.eos_token_id,
            repetition_penalty=self.config.get("repetition_penalty", 1.1)
        )
        
        g_logger.info("CPU LLM model loaded successfully")
        return True
    
    def GenerateDraftTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        Generate draft tokens WITH their probabilities
        Routes to appropriate backend (CPU PyTorch, GPU PyTorch, or NPU OpenVINO)
        
        Returns:
            (draft_tokens, draft_probabilities, inference_time)
        """
        if self.m_device == "npu" and self.m_npu_model:
            return self.m_npu_model.GenerateDraftTokensWithProbabilities(
                prompt, num_draft_tokens, temperature
            )
        elif self.m_device == "gpu":
            return self._GenerateGPUTokensWithProbabilities(
                prompt, num_draft_tokens, temperature
            )
        else:
            return self._GenerateCPUTokensWithProbabilities(
                prompt, num_draft_tokens, temperature
            )
    
    def _GenerateCPUTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        Generate draft tokens with probabilities using CPU PyTorch
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("CPU LLM model not loaded")
            return [], [], 0.0
        
        start_time = CreateTimestamp()
        
        try:
            # Encode initial prompt
            input_ids = self.m_tokenizer.encode(prompt, return_tensors="pt").to("cpu")
            
            draft_tokens = []
            draft_probs = []
            
            with torch.no_grad():
                current_input = input_ids
                
                # Generate tokens one by one, tracking probabilities
                for i in range(num_draft_tokens):
                    # Get logits for next token
                    outputs = self.m_model(current_input)
                    logits = outputs.logits[0, -1, :]  # Last position logits
                    
                    # Apply temperature
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    # Convert to probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Sample token
                    token_id = torch.multinomial(probs, 1).item()
                    token_prob = probs[token_id].item()
                    
                    # Decode token
                    token = self.m_tokenizer.decode([token_id])
                    
                    # Store token and its probability
                    draft_tokens.append(token)
                    draft_probs.append(token_prob)
                    
                    # Update input for next iteration (autoregressive)
                    current_input = torch.cat([current_input, torch.tensor([[token_id]]).to("cpu")], dim=1)
                    
                    g_logger.debug(f"Generated token {i+1}: '{token}' (p={token_prob:.3f})")
            
            inference_time = CreateTimestamp() - start_time
            
            g_logger.info(f"CPU LLM generated {len(draft_tokens)} draft tokens with probabilities in {inference_time:.3f}s")
            g_logger.debug(f"Draft sequence: {''.join(draft_tokens)}")
            g_logger.debug(f"Average probability: {sum(draft_probs)/len(draft_probs):.3f}")
            
            return draft_tokens, draft_probs, inference_time
            
        except Exception as e:
            g_logger.error(f"CPU LLM draft generation with probabilities failed: {e}")
            return [], [], CreateTimestamp() - start_time
    
    def _GenerateGPUTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        Generate draft tokens with probabilities using GPU PyTorch
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("GPU LLM model not loaded")
            return [], [], 0.0
        
        start_time = CreateTimestamp()
        
        try:
            # Encode initial prompt
            input_ids = self.m_tokenizer.encode(prompt, return_tensors="pt").to(self.m_cuda_device)
            
            draft_tokens = []
            draft_probs = []
            
            with torch.no_grad():
                current_input = input_ids
                
                # Generate tokens one by one, tracking probabilities
                for i in range(num_draft_tokens):
                    # Get logits for next token
                    outputs = self.m_model(current_input)
                    logits = outputs.logits[0, -1, :]  # Last position logits
                    
                    # Apply temperature
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    # Convert to probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Sample token
                    token_id = torch.multinomial(probs, 1).item()
                    token_prob = probs[token_id].item()
                    
                    # Decode token
                    token = self.m_tokenizer.decode([token_id])
                    
                    # Store token and its probability
                    draft_tokens.append(token)
                    draft_probs.append(token_prob)
                    
                    # Update input for next iteration (autoregressive)
                    current_input = torch.cat([current_input, torch.tensor([[token_id]]).to(self.m_cuda_device)], dim=1)
                    
                    g_logger.debug(f"GPU generated token {i+1}: '{token}' (p={token_prob:.3f})")
            
            inference_time = CreateTimestamp() - start_time
            
            g_logger.info(f"GPU LLM generated {len(draft_tokens)} draft tokens with probabilities in {inference_time:.3f}s")
            g_logger.debug(f"Draft sequence: {''.join(draft_tokens)}")
            g_logger.debug(f"Average probability: {sum(draft_probs)/len(draft_probs):.3f}")
            
            return draft_tokens, draft_probs, inference_time
            
        except Exception as e:
            g_logger.error(f"GPU LLM draft generation with probabilities failed: {e}")
            return [], [], CreateTimestamp() - start_time
    
    def GenerateDraftTokens(self, prompt: str, num_draft_tokens: int = 5) -> Tuple[List[str], float]:
        """
        Legacy interface for backward compatibility
        Note: This doesn't return probabilities, so verification will be suboptimal
        """
        g_logger.warning("Using legacy LLM interface - probabilities not tracked, verification will be suboptimal")
        
        draft_tokens, draft_probs, inference_time = self.GenerateDraftTokensWithProbabilities(
            prompt, num_draft_tokens
        )
        
        return draft_tokens, inference_time
    
    def GetModelInfo(self) -> Dict[str, Any]:
        """Get information about the LLM model"""
        info = {
            "model_name": self.m_model_name,
            "device": self.m_device,
            "model_type": "llm",
            "loaded": self.IsLoaded()
        }
        
        if self.m_device == "gpu" and self.m_cuda_device:
            info["cuda_device"] = self.m_cuda_device
            info["gpu_device_id"] = self.m_gpu_device_id
            
            if torch.cuda.is_available():
                try:
                    info["gpu_name"] = torch.cuda.get_device_name(self.m_gpu_device_id)
                    info["gpu_memory_allocated"] = torch.cuda.memory_allocated(self.m_gpu_device_id) / 1e9
                    info["gpu_memory_cached"] = torch.cuda.memory_reserved(self.m_gpu_device_id) / 1e9
                except Exception:
                    pass
        
        return info
    
    def IsLoaded(self) -> bool:
        """Check if the LLM model is loaded and ready"""
        if self.m_device == "npu":
            return self.m_npu_model is not None and self.m_npu_model.IsLoaded()
        else:
            return self.m_model is not None and self.m_tokenizer is not None
    
    def Cleanup(self):
        """Clean up LLM model resources"""
        if self.m_device == "gpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.m_npu_model:
            self.m_npu_model.Cleanup()
        
        # Clear references
        self.m_model = None
        self.m_tokenizer = None
        self.m_generation_config = None
        self.m_npu_model = None
        
        g_logger.info("LLM draft model resources cleaned up")

# Alias for backward compatibility
EdgeDraftModel = LLMDraftModel