"""
Factory for creating different types of draft models
Handles the selection and instantiation of LLM or EAGLE draft models
"""
from typing import Dict, Any
import logging
from edge.draft_model_interface import DraftModelInterface
from edge.llm_draft_model import LLMDraftModel  
from edge.eagle_draft_model import EagleDraftModel

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class DraftModelFactory:
    """Factory class for creating draft models based on configuration"""
    
    @staticmethod
    def create_draft_model(config: Dict[str, Any]) -> DraftModelInterface:
        """
        Create a draft model based on configuration
        
        Args:
            config: Configuration dictionary from get_edge_model_config()
            
        Returns:
            DraftModelInterface: Initialized draft model instance
            
        Raises:
            ValueError: If draft model type is not supported
            RuntimeError: If model creation fails
        """
        draft_type = config.get("draft_type", "llm").lower()
        model_name = config.get("model_name", "meta-llama/Llama-3.2-1B-Instruct")
        device = config.get("device", "cpu")
        
        g_logger.info(f"Creating {draft_type.upper()} draft model: {model_name}")
        g_logger.info(f"Target device: {device}")
        
        try:
            if draft_type == "llm":
                return DraftModelFactory._create_llm_model(config)
            elif draft_type == "eagle":
                return DraftModelFactory._create_eagle_model(config)
            else:
                raise ValueError(f"Unsupported draft model type: {draft_type}")
                
        except Exception as e:
            g_logger.error(f"Failed to create {draft_type} draft model: {e}")
            raise RuntimeError(f"Draft model creation failed: {e}")
    
    @staticmethod
    def _create_llm_model(config: Dict[str, Any]) -> LLMDraftModel:
        """Create LLM-based draft model"""
        return LLMDraftModel(
            model_name=config.get("model_name", "meta-llama/Llama-3.2-1B-Instruct"),
            device=config.get("device", "cpu"),
            gpu_device_id=config.get("gpu_device_id", 0),
            config=config
        )
    
    @staticmethod
    def _create_eagle_model(config: Dict[str, Any]) -> EagleDraftModel:
        """Create EAGLE-based draft model"""
        return EagleDraftModel(
            model_name=config.get("model_name", "meta-llama/Llama-3.2-1B-Instruct"),
            device=config.get("device", "cpu"),
            num_eagle_heads=config.get("eagle_num_heads", 3),
            tree_depth=config.get("eagle_tree_depth", 2),
            base_model_layers=config.get("eagle_base_model_layers", 16),
            eagle_model_path=config.get("eagle_model_path", ""),
            gpu_device_id=config.get("gpu_device_id", 0)  # Pass through for GPU support
        )
    
    @staticmethod
    def get_supported_types() -> list:
        """Get list of supported draft model types"""
        return ["llm", "eagle"]
    
    @staticmethod
    def get_model_info(draft_type: str) -> Dict[str, Any]:
        """
        Get information about a specific draft model type
        
        Args:
            draft_type: Type of draft model ("llm" or "eagle")
            
        Returns:
            Dictionary with model type information
        """
        if draft_type.lower() == "llm":
            return {
                "type": "llm",
                "name": "LLM Draft Model",
                "description": "Traditional autoregressive language model for draft generation",
                "advantages": ["High quality drafts", "Well-tested", "Compatible with all models"],
                "disadvantages": ["Slower generation", "Higher computational cost"],
                "best_for": ["High accuracy requirements", "Small models", "Research"]
            }
        elif draft_type.lower() == "eagle":
            return {
                "type": "eagle", 
                "name": "EAGLE Draft Model",
                "description": "Tree-based speculative decoding with lightweight prediction heads",
                "advantages": ["Faster generation", "Lower computational cost", "Tree-based candidates"],
                "disadvantages": ["More complex setup", "Requires training", "Model-specific"],
                "best_for": ["Speed optimization", "Production deployment", "Large-scale inference"]
            }
        else:
            return {"error": f"Unknown draft model type: {draft_type}"}

# Convenience function for backward compatibility
def create_edge_draft_model(config: Dict[str, Any]) -> DraftModelInterface:
    """
    Convenience function to create draft model from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DraftModelInterface: Created draft model
    """
    return DraftModelFactory.create_draft_model(config)