"""
Abstract interface for draft models in speculative decoding
Defines the common contract that all draft model implementations must follow
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class DraftModelInterface(ABC):
    """
    Abstract base class for all draft model implementations
    
    This interface ensures that all draft models (LLM-based, EAGLE, etc.) 
    provide the same core functionality for speculative decoding.
    """
    
    def __init__(self, model_name: str, device: str = "cpu", **kwargs):
        """
        Initialize the draft model
        
        Args:
            model_name: Name/path of the model to use
            device: Device to run on ("cpu", "gpu", "npu", etc.)
            **kwargs: Additional device-specific or model-specific parameters
        """
        self.m_model_name = model_name
        self.m_device = device.lower()
        self.m_kwargs = kwargs
        
        g_logger.info(f"Initializing {self.__class__.__name__}: {model_name}")
        g_logger.info(f"Target device: {self.m_device}")
    
    @abstractmethod
    def LoadModel(self) -> bool:
        """
        Load the draft model and any required components
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def GenerateDraftTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        Generate draft tokens WITH their probabilities for verification
        
        This is the primary method used for speculative decoding.
        
        Args:
            prompt: Input text to continue
            num_draft_tokens: Number of draft tokens to generate
            temperature: Sampling temperature for generation
        
        Returns:
            Tuple containing:
            - draft_tokens: List of generated token strings
            - draft_probabilities: List of probability scores for each token
            - inference_time: Time taken for generation in seconds
        """
        pass
    
    @abstractmethod
    def GenerateDraftTokens(self, prompt: str, num_draft_tokens: int = 5) -> Tuple[List[str], float]:
        """
        Legacy interface for backward compatibility
        
        Note: This doesn't return probabilities, so verification will be suboptimal.
        New implementations should prefer GenerateDraftTokensWithProbabilities.
        
        Args:
            prompt: Input text to continue  
            num_draft_tokens: Number of draft tokens to generate
            
        Returns:
            Tuple containing:
            - draft_tokens: List of generated token strings
            - inference_time: Time taken for generation in seconds
        """
        pass
    
    @abstractmethod
    def GetModelInfo(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary containing model metadata such as:
            - model_name: Name of the model
            - device: Device being used
            - model_type: Type of draft model (e.g., "llm", "eagle")
            - parameters: Number of parameters if applicable
            - memory_usage: Memory consumption if available
        """
        pass
    
    def GetDevice(self) -> str:
        """Get the device this model is running on"""
        return self.m_device
    
    def GetModelName(self) -> str:
        """Get the model name/path"""
        return self.m_model_name
    
    def IsLoaded(self) -> bool:
        """
        Check if the model is loaded and ready for inference
        
        Default implementation returns True, subclasses can override
        """
        return True
    
    def Cleanup(self):
        """
        Clean up resources (optional)
        
        Called when the model is no longer needed
        """
        pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.Cleanup()