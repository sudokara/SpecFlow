"""
EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) draft model implementation
A tree-based speculative decoding approach that generates multiple draft candidates
using auto-regression heads trained on target model hidden states
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, list_repo_files
from typing import List, Tuple, Dict, Any, Optional
import time
import logging
import numpy as np
import os
from common.protocol import CreateTimestamp
from edge.draft_model_interface import DraftModelInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class EagleHead(nn.Module):
    """
    EAGLE auto-regression head for draft token prediction
    Lightweight neural network that predicts next tokens from hidden states
    """
    
    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Multi-layer perceptron for token prediction
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(hidden_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Final projection to vocabulary
        layers.append(nn.Linear(hidden_size, vocab_size))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, hidden_states):
        """
        Forward pass through EAGLE head
        
        Args:
            hidden_states: Hidden representations from base model [batch, seq_len, hidden_size]
            
        Returns:
            logits: Token prediction logits [batch, seq_len, vocab_size]
        """
        return self.layers(hidden_states)

class EagleDraftModel(DraftModelInterface):
    """
    EAGLE-based draft model for speculative decoding
    
    Uses lightweight auto-regression heads to predict draft tokens from
    base model hidden states, enabling faster draft generation than
    full auto-regressive inference.
    """
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
        device: str = "cpu",
        num_eagle_heads: int = 3,
        tree_depth: int = 2,
        base_model_layers: int = 16,  # Use first N layers of base model
        eagle_model_path: str = "",  # Path to pre-trained EAGLE model
        **kwargs
    ):
        """
        Initialize EAGLE draft model
        
        Args:
            model_name: Base model name to use for hidden state extraction
            device: Device to run on
            num_eagle_heads: Number of EAGLE prediction heads
            tree_depth: Depth of draft token tree to generate
            base_model_layers: Number of base model layers to use for features
            eagle_model_path: Path or HuggingFace repo ID for pre-trained EAGLE model
        """
        super().__init__(model_name, device, **kwargs)
        
        self.m_num_eagle_heads = num_eagle_heads
        self.m_tree_depth = tree_depth
        self.m_base_model_layers = base_model_layers
        self.m_eagle_model_path = eagle_model_path
        
        # Model components
        self.m_tokenizer = None
        self.m_base_model = None  # Truncated base model for feature extraction
        self.m_eagle_heads = None  # EAGLE prediction heads
        self.m_hidden_size = None
        self.m_vocab_size = None
        
        # Tree-based generation state
        self.m_draft_tree = None
        
        g_logger.info(f"EAGLE configuration:")
        g_logger.info(f"  Number of heads: {num_eagle_heads}")
        g_logger.info(f"  Tree depth: {tree_depth}")
        g_logger.info(f"  Base model layers: {base_model_layers}")
    
    def LoadModel(self) -> bool:
        """Load base model and initialize EAGLE heads"""
        try:
            g_logger.info("Loading EAGLE draft model...")
            
            # Load tokenizer
            self.m_tokenizer = AutoTokenizer.from_pretrained(
                self.m_model_name,
                trust_remote_code=True
            )
            
            if self.m_tokenizer.pad_token is None:
                self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
            
            # Load base model for feature extraction
            g_logger.info("Loading base model for hidden state extraction...")
            self.m_base_model = AutoModelForCausalLM.from_pretrained(
                self.m_model_name,
                torch_dtype=torch.float32 if self.m_device == "cpu" else torch.float16,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.m_device)
            
            # Get model dimensions
            self.m_hidden_size = self.m_base_model.config.hidden_size
            self.m_vocab_size = self.m_base_model.config.vocab_size
            
            g_logger.info(f"Model dimensions: hidden={self.m_hidden_size}, vocab={self.m_vocab_size}")
            
            # Initialize EAGLE heads
            self._InitializeEagleHeads()
            
            # Set model to eval mode
            self.m_base_model.eval()
            for head in self.m_eagle_heads:
                head.eval()
            
            g_logger.info("EAGLE draft model loaded successfully")
            return True
            
        except Exception as e:
            g_logger.error(f"Failed to load EAGLE draft model: {e}")
            return False
    
    def _InitializeEagleHeads(self):
        """Initialize EAGLE prediction heads - either from scratch or pre-trained"""
        
        if self.m_eagle_model_path:
            # Load pre-trained EAGLE model
            g_logger.info(f"Loading pre-trained EAGLE model from: {self.m_eagle_model_path}")
            self._LoadPretrainedEagleModel()
        else:
            # Initialize from scratch
            g_logger.info(f"Initializing {self.m_num_eagle_heads} EAGLE heads from scratch...")
            self._InitializeEagleHeadsFromScratch()
    
    def _LoadPretrainedEagleModel(self):
        """Load pre-trained EAGLE model from HuggingFace or local path"""
        try:
            eagle_path = self.m_eagle_model_path
            
            # Check if it's a HuggingFace repo or local path
            if "/" in eagle_path and not os.path.exists(eagle_path):
                # Assume it's a HuggingFace repo ID
                g_logger.info(f"Downloading EAGLE model from HuggingFace: {eagle_path}")
                
                try:
                    # Try to download the eagle heads file
                    eagle_heads_path = hf_hub_download(
                        repo_id=eagle_path,
                        filename="eagle_heads.pt",
                        local_dir=None
                    )
                    g_logger.info(f"Downloaded EAGLE heads to: {eagle_heads_path}")
                    
                    # Load the pre-trained heads
                    eagle_state = torch.load(eagle_heads_path, map_location=self.m_device)
                    
                except Exception as e:
                    g_logger.warning(f"Could not download eagle_heads.pt: {e}")
                    # Try alternative filename
                    try:
                        eagle_heads_path = hf_hub_download(
                            repo_id=eagle_path,
                            filename="pytorch_model.bin",
                            local_dir=None
                        )
                        eagle_state = torch.load(eagle_heads_path, map_location=self.m_device)
                    except Exception as e2:
                        g_logger.error(f"Could not load pre-trained EAGLE model: {e2}")
                        g_logger.info("Falling back to initialization from scratch")
                        self._InitializeEagleHeadsFromScratch()
                        return
                        
            elif os.path.exists(eagle_path):
                # Local path
                g_logger.info(f"Loading EAGLE model from local path: {eagle_path}")
                
                # Try different possible filenames
                possible_files = ["eagle_heads.pt", "pytorch_model.bin", "model.pt"]
                eagle_state = None
                
                for filename in possible_files:
                    filepath = os.path.join(eagle_path, filename) if os.path.isdir(eagle_path) else eagle_path
                    if os.path.exists(filepath):
                        try:
                            eagle_state = torch.load(filepath, map_location=self.m_device)
                            g_logger.info(f"Loaded EAGLE state from: {filepath}")
                            break
                        except Exception as e:
                            g_logger.warning(f"Could not load {filepath}: {e}")
                            continue
                
                if eagle_state is None:
                    g_logger.error(f"Could not find valid EAGLE model file in: {eagle_path}")
                    self._InitializeEagleHeadsFromScratch()
                    return
            else:
                g_logger.error(f"EAGLE model path not found: {eagle_path}")
                self._InitializeEagleHeadsFromScratch()
                return
            
            # Initialize the heads first
            self._InitializeEagleHeadsFromScratch()
            
            # Load the pre-trained weights
            if isinstance(eagle_state, dict):
                # Handle different formats of saved state
                if "eagle_heads" in eagle_state:
                    heads_state = eagle_state["eagle_heads"]
                elif "model_state_dict" in eagle_state:
                    heads_state = eagle_state["model_state_dict"]
                else:
                    heads_state = eagle_state
                
                # Load state into each head
                try:
                    if isinstance(heads_state, list) and len(heads_state) == len(self.m_eagle_heads):
                        # Individual head states
                        for i, head_state in enumerate(heads_state):
                            self.m_eagle_heads[i].load_state_dict(head_state)
                    else:
                        # Try to load as combined state dict
                        for i, head in enumerate(self.m_eagle_heads):
                            head_key_prefix = f"eagle_heads.{i}."
                            head_state = {
                                k[len(head_key_prefix):]: v 
                                for k, v in heads_state.items() 
                                if k.startswith(head_key_prefix)
                            }
                            if head_state:
                                head.load_state_dict(head_state)
                            else:
                                g_logger.warning(f"No state found for head {i}")
                    
                    g_logger.info("Successfully loaded pre-trained EAGLE heads")
                    
                except Exception as e:
                    g_logger.error(f"Failed to load EAGLE head weights: {e}")
                    g_logger.info("Using randomly initialized heads")
                    
            else:
                g_logger.error("Invalid EAGLE state format")
                g_logger.info("Using randomly initialized heads")
                
        except ImportError:
            g_logger.error("huggingface_hub not available. Install it to use pre-trained EAGLE models.")
            g_logger.info("Falling back to initialization from scratch")
            self._InitializeEagleHeadsFromScratch()
        except Exception as e:
            g_logger.error(f"Error loading pre-trained EAGLE model: {e}")
            g_logger.info("Falling back to initialization from scratch")
            self._InitializeEagleHeadsFromScratch()
    
    def _InitializeEagleHeadsFromScratch(self):
        """Initialize EAGLE heads with random weights"""
        g_logger.info(f"Initializing {self.m_num_eagle_heads} EAGLE heads from scratch...")
        
        self.m_eagle_heads = nn.ModuleList([
            EagleHead(self.m_hidden_size, self.m_vocab_size)
            for _ in range(self.m_num_eagle_heads)
        ]).to(self.m_device)
        
        # Initialize with small random weights
        for head in self.m_eagle_heads:
            for param in head.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:
                    nn.init.zeros_(param)
        
        g_logger.info("EAGLE heads initialized from scratch")
    
    def _ExtractHiddenStates(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract hidden states from base model using partial forward pass
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            hidden_states: Hidden representations [batch_size, seq_len, hidden_size]
        """
        with torch.no_grad():
            # Run partial forward pass through base model
            outputs = self.m_base_model(
                input_ids, 
                output_hidden_states=True,
                use_cache=False
            )
            
            # Use hidden states from specified layer
            # Use average of multiple middle layers for robustness
            layer_indices = [
                max(0, len(outputs.hidden_states) // 2 - 1),
                len(outputs.hidden_states) // 2,
                min(len(outputs.hidden_states) - 1, len(outputs.hidden_states) // 2 + 1)
            ]
            
            hidden_states = torch.stack([
                outputs.hidden_states[i] for i in layer_indices
            ]).mean(dim=0)  # Average across selected layers
            
            return hidden_states
    
    def GenerateDraftTokensWithProbabilities(
        self, 
        prompt: str, 
        num_draft_tokens: int = 5,
        temperature: float = 0.8
    ) -> Tuple[List[str], List[float], float]:
        """
        Generate draft tokens using EAGLE tree-based approach
        
        Returns:
            (draft_tokens, draft_probabilities, inference_time)
        """
        if not self.IsLoaded():
            g_logger.error("EAGLE model not loaded")
            return [], [], 0.0
        
        start_time = CreateTimestamp()
        
        try:
            # Encode prompt
            input_ids = self.m_tokenizer.encode(prompt, return_tensors="pt").to(self.m_device)
            
            # Extract hidden states from base model
            hidden_states = self._ExtractHiddenStates(input_ids)
            
            # Generate draft tree using EAGLE heads
            draft_tokens, draft_probs = self._GenerateDraftTree(
                hidden_states, num_draft_tokens, temperature
            )
            
            inference_time = CreateTimestamp() - start_time
            
            g_logger.info(f"EAGLE generated {len(draft_tokens)} draft tokens in {inference_time:.3f}s")
            g_logger.debug(f"Draft sequence: {''.join(draft_tokens)}")
            if draft_probs:
                g_logger.debug(f"Average probability: {sum(draft_probs)/len(draft_probs):.3f}")
            
            return draft_tokens, draft_probs, inference_time
            
        except Exception as e:
            g_logger.error(f"EAGLE draft generation failed: {e}")
            return [], [], CreateTimestamp() - start_time
    
    def _GenerateDraftTree(
        self, 
        hidden_states: torch.Tensor, 
        num_tokens: int, 
        temperature: float
    ) -> Tuple[List[str], List[float]]:
        """
        Generate draft tokens using tree-based EAGLE approach
        
        Args:
            hidden_states: Base model hidden states [batch, seq_len, hidden_size]
            num_tokens: Number of draft tokens to generate
            temperature: Sampling temperature
            
        Returns:
            (draft_tokens, draft_probabilities)
        """
        draft_tokens = []
        draft_probs = []
        
        # Use the last hidden state as starting point
        current_hidden = hidden_states[:, -1:, :]  # [batch, 1, hidden_size]
        
        # Generate tokens sequentially using EAGLE heads
        for i in range(num_tokens):
            # Ensemble prediction from multiple EAGLE heads
            all_logits = []
            
            for head_idx, head in enumerate(self.m_eagle_heads):
                # Get logits from this EAGLE head
                logits = head(current_hidden)  # [batch, 1, vocab_size]
                logits = logits[:, -1, :]  # Take last position [batch, vocab_size]
                all_logits.append(logits)
            
            # Ensemble: average logits from all heads
            ensemble_logits = torch.stack(all_logits).mean(dim=0)  # [batch, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                ensemble_logits = ensemble_logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(ensemble_logits, dim=-1)
            
            # Sample token
            token_id = torch.multinomial(probs, 1).item()
            token_prob = probs[0, token_id].item()
            
            # Decode token
            token = self.m_tokenizer.decode([token_id])
            
            draft_tokens.append(token)
            draft_probs.append(token_prob)
            
            # Update hidden state for next iteration
            # Simple approach: use token embedding + positional encoding
            token_embedding = self._GetTokenEmbedding(token_id)
            current_hidden = token_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            
            g_logger.debug(f"EAGLE token {i+1}: '{token}' (p={token_prob:.3f}, ensemble)")
        
        return draft_tokens, draft_probs
    
    def _GetTokenEmbedding(self, token_id: int) -> torch.Tensor:
        """
        Get token embedding for hidden state update
        
        Args:
            token_id: Token ID to get embedding for
            
        Returns:
            Token embedding tensor [hidden_size]
        """
        # Simple approach: use base model embeddings
        with torch.no_grad():
            token_tensor = torch.tensor([token_id]).to(self.m_device)
            embedding = self.m_base_model.get_input_embeddings()(token_tensor)
            return embedding[0]  # Return single embedding
    
    def GenerateDraftTokens(self, prompt: str, num_draft_tokens: int = 5) -> Tuple[List[str], float]:
        """Legacy interface for backward compatibility"""
        g_logger.warning("Using legacy EAGLE interface - some optimization may be lost")
        
        draft_tokens, draft_probs, inference_time = self.GenerateDraftTokensWithProbabilities(
            prompt, num_draft_tokens
        )
        
        return draft_tokens, inference_time
    
    def GetModelInfo(self) -> Dict[str, Any]:
        """Get information about the EAGLE model"""
        info = {
            "model_name": self.m_model_name,
            "device": self.m_device,
            "model_type": "eagle",
            "loaded": self.IsLoaded(),
            "num_eagle_heads": self.m_num_eagle_heads,
            "tree_depth": self.m_tree_depth,
            "base_model_layers": self.m_base_model_layers
        }
        
        if self.m_hidden_size:
            info["hidden_size"] = self.m_hidden_size
        if self.m_vocab_size:
            info["vocab_size"] = self.m_vocab_size
        
        return info
    
    def IsLoaded(self) -> bool:
        """Check if EAGLE model is loaded and ready"""
        return (
            self.m_tokenizer is not None and 
            self.m_base_model is not None and 
            self.m_eagle_heads is not None
        )
    
    def Cleanup(self):
        """Clean up EAGLE model resources"""
        if self.m_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear references
        self.m_tokenizer = None
        self.m_base_model = None
        self.m_eagle_heads = None
        
        g_logger.info("EAGLE draft model resources cleaned up")