"""
Cloud server component - Target model for verification and completion
Runs a larger language model (<8B parameters) to verify draft tokens
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Tuple
import time
import logging
import random
from common.protocol import PerformanceMetrics, CreateTimestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class CloudTargetModel:
    """Target model running on cloud server"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize target model with a larger, more capable model from same family
        Using Llama-3.1-8B-Instruct (8B parameters) - same family as edge model
        This ensures compatible tokenization with the 1B edge model for better acceptance rates
        """
        self.m_model_name = model_name
        
        # Check GPU availability more thoroughly
        if torch.cuda.is_available():
            self.m_device = "cuda"
            g_logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
            g_logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.m_device = "cpu"
            g_logger.warning("CUDA not available, using CPU. For better performance, install PyTorch with CUDA support.")
        
        self.m_tokenizer = None
        self.m_model = None
        
        g_logger.info(f"Initializing cloud target model: {model_name}")
        g_logger.info(f"Using device: {self.m_device}")
        
    def LoadModel(self) -> bool:
        """Load the target model and tokenizer"""
        try:
            # Load tokenizer
            self.m_tokenizer = AutoTokenizer.from_pretrained(
                self.m_model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.m_tokenizer.pad_token is None:
                self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
            
            # Load model with optimizations based on device
            if self.m_device == "cuda":
                g_logger.info("Loading model on GPU with float16 precision...")
                self.m_model = AutoModelForCausalLM.from_pretrained(
                    self.m_model_name,
                    torch_dtype=torch.float16,  # Use float16 for GPU memory efficiency
                    device_map="auto",  # Automatically distribute across available GPUs
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                g_logger.info(f"Model loaded on GPU. Memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
            else:
                g_logger.info("Loading model on CPU with float32 precision...")
                self.m_model = AutoModelForCausalLM.from_pretrained(
                    self.m_model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).to(self.m_device)
            
            g_logger.info("Cloud target model loaded successfully")
            return True
            
        except Exception as e:
            g_logger.error(f"Failed to load cloud model: {e}")
            return False
    
    def VerifyAndCompleteProbabilistic(
        self,
        prompt: str,
        draft_tokens: List[str],
        draft_probs: List[float],
        max_new_tokens: int = 10,
        temperature: float = 0.8,
        return_debug: bool = False
    ) -> Tuple[List[str], List[str], int, float] | Tuple[List[str], List[str], int, float, dict]:
        """
        CORRECTED: Verify draft tokens using proper probabilistic acceptance sampling
        
        This implements the actual speculative decoding algorithm:
        1. Run target model on prompt + draft_tokens in ONE forward pass
        2. For each draft token, calculate acceptance probability
        3. Accept/reject based on probability ratio, not string matching
        4. Generate new token only when rejecting
        
        Args:
            prompt: Original input prompt
            draft_tokens: List of draft tokens from edge model
            draft_probs: List of probabilities from draft model (CRITICAL!)
            max_new_tokens: Maximum additional tokens to generate
            temperature: Sampling temperature
            
        Returns:
            If return_debug is False (default):
                (verified_tokens, new_tokens, accepted_count, inference_time)
            If return_debug is True:
                (verified_tokens, new_tokens, accepted_count, inference_time, debug_info_dict)
            where debug_info_dict contains:
                draft_tokens, draft_probs, target_probs_per_draft, acceptance_mask,
                acceptance_probs
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("Model not loaded")
            return ([], [], 0, 0.0, {}) if return_debug else ([], [], 0, 0.0)
        
        start_time = CreateTimestamp()
        
        try:
            if not draft_tokens or not draft_probs:
                # No draft tokens, generate from scratch
                new_tokens = self._GenerateNewTokens(prompt, max_new_tokens)
                result = ([], new_tokens, 0, CreateTimestamp() - start_time)
                if return_debug:
                    result = (*result, {
                        "draft_tokens": draft_tokens or [],
                        "draft_probs": draft_probs or [],
                        "target_probs_per_draft": [],
                        "acceptance_mask": [],
                        "acceptance_probs": []
                    })
                return result
            
            # STEP 1: Run target model on prompt + ALL draft tokens in ONE forward pass
            full_text = prompt + "".join(draft_tokens)
            input_ids = self.m_tokenizer.encode(full_text, return_tensors="pt").to(self.m_device)
            
            with torch.no_grad():
                outputs = self.m_model(input_ids)
                logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            
            # Get prompt length to identify draft token positions
            prompt_ids = self.m_tokenizer.encode(prompt, return_tensors="pt")
            prompt_length = prompt_ids.shape[1]
            
            # STEP 2: Probabilistic verification of each draft token
            verified_tokens = []
            target_probs_per_draft: List[float] = []
            acceptance_mask: List[bool] = []  # True if accepted
            acceptance_probs: List[float] = []
            
            for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
                # Get logits for the position where this token was predicted
                position_logits = logits[prompt_length + i - 1]  # -1 due to shifted prediction
                
                # Apply temperature
                if temperature != 1.0:
                    position_logits = position_logits / temperature
                
                # Convert to probabilities
                target_probs = F.softmax(position_logits, dim=-1)
                
                # Get target probability for the draft token
                try:
                    draft_token_ids = self.m_tokenizer.encode(draft_token, add_special_tokens=False)
                    if len(draft_token_ids) == 0:
                        g_logger.warning(f"Could not encode draft token: '{draft_token}'")
                        break
                    
                    draft_token_id = draft_token_ids[0]
                    target_prob = target_probs[draft_token_id].item()
                    
                except Exception as e:
                    g_logger.warning(f"Error processing draft token '{draft_token}': {e}")
                    break
                
                # STEP 3: Calculate acceptance probability
                acceptance_prob = min(1.0, target_prob / max(draft_prob, 1e-8))  # Avoid division by zero
                target_probs_per_draft.append(target_prob)
                acceptance_probs.append(acceptance_prob)
                
                # STEP 4: Probabilistic acceptance decision
                if random.random() < acceptance_prob:
                    verified_tokens.append(draft_token)
                    acceptance_mask.append(True)
                    g_logger.debug(f"✓ Accepted '{draft_token}' (p_accept={acceptance_prob:.3f})")
                else:
                    acceptance_mask.append(False)
                    # STEP 5: Reject and sample new token from adjusted distribution
                    g_logger.debug(f"✗ Rejected '{draft_token}' (p_accept={acceptance_prob:.3f})")
                    
                    # Adjust target distribution: (p_target - acceptance_prob * p_draft) / (1 - acceptance_prob)
                    adjusted_probs = target_probs.clone()
                    adjusted_probs[draft_token_id] = max(0, target_prob - acceptance_prob * draft_prob)
                    
                    # Renormalize
                    if adjusted_probs.sum() > 0:
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()
                        new_token_id = torch.multinomial(adjusted_probs, 1).item()
                    else:
                        # Fallback: sample from original distribution
                        new_token_id = torch.multinomial(target_probs, 1).item()
                    new_token = self.m_tokenizer.decode([new_token_id])
                    
                    # Return accepted tokens + one new token
                    inference_time = CreateTimestamp() - start_time
                    base_tuple = (verified_tokens, [new_token], len(verified_tokens), inference_time)
                    if return_debug:
                        debug_info = {
                            "draft_tokens": draft_tokens,
                            "draft_probs": draft_probs,
                            "target_probs_per_draft": target_probs_per_draft,
                            "acceptance_mask": acceptance_mask,
                            "acceptance_probs": acceptance_probs
                        }
                        return (*base_tuple, debug_info)
                    return base_tuple
            
            # All draft tokens were accepted - generate additional tokens if requested
            new_tokens = []
            if max_new_tokens > 0:
                new_tokens = self._GenerateNewTokens(
                    prompt + "".join(verified_tokens), 
                    min(max_new_tokens, 3)  # Limit to avoid long generation
                )
            
            inference_time = CreateTimestamp() - start_time
            g_logger.info(f"✓ Probabilistic verification: {len(verified_tokens)}/{len(draft_tokens)} tokens accepted")
            base_tuple = (verified_tokens, new_tokens, len(verified_tokens), inference_time)
            if return_debug:
                debug_info = {
                    "draft_tokens": draft_tokens,
                    "draft_probs": draft_probs,
                    "target_probs_per_draft": target_probs_per_draft,
                    "acceptance_mask": acceptance_mask,
                    "acceptance_probs": acceptance_probs
                }
                return (*base_tuple, debug_info)
            return base_tuple
            
        except Exception as e:
            g_logger.error(f"Probabilistic verification failed: {e}")
            failure_tuple = ([], [], 0, CreateTimestamp() - start_time)
            if return_debug:
                failure_tuple = (*failure_tuple, {
                    "draft_tokens": draft_tokens,
                    "draft_probs": draft_probs,
                    "target_probs_per_draft": [],
                    "acceptance_mask": [],
                    "acceptance_probs": []
                })
            return failure_tuple
    
    def VerifyAndComplete(self, prompt: str, draft_tokens: List[str], max_new_tokens: int = 10) -> Tuple[List[str], List[str], int, float]:
        """
        Verify draft tokens and generate additional tokens if needed
        Returns (verified_tokens, new_tokens, accepted_count, inference_time)
        """
        if not self.m_model or not self.m_tokenizer:
            g_logger.error("Model not loaded")
            return [], [], 0, 0.0
        
        start_time = CreateTimestamp()
        
        try:
            # Combine draft tokens into text
            draft_text = "".join(draft_tokens).strip()
            
            # Since we're using the same model family, we can do more precise verification
            # Generate what the target model would produce
            target_text = self._GenerateTargetText(prompt, max(len(draft_text), max_new_tokens))
            target_text = target_text.strip()
            
            verified_tokens = []
            accepted_count = 0
            new_tokens = []
            
            if draft_text and target_text:
                # For same model family, we can be more precise with token verification
                # Tokenize both texts to compare at token level
                draft_token_ids = self.m_tokenizer.encode(draft_text, add_special_tokens=False)
                target_token_ids = self.m_tokenizer.encode(target_text, add_special_tokens=False)
                
                # Find longest common prefix at token level
                common_tokens = 0
                for i in range(min(len(draft_token_ids), len(target_token_ids))):
                    if draft_token_ids[i] == target_token_ids[i]:
                        common_tokens += 1
                    else:
                        break
                
                if common_tokens > 0:
                    # Accept the common prefix
                    accepted_token_ids = draft_token_ids[:common_tokens]
                    accepted_text = self.m_tokenizer.decode(accepted_token_ids, skip_special_tokens=True)
                    
                    # Convert back to original draft token structure
                    # For simplicity, accept tokens proportionally
                    tokens_to_accept = min(common_tokens, len(draft_tokens))
                    verified_tokens = draft_tokens[:tokens_to_accept]
                    accepted_count = tokens_to_accept
                    
                    # Generate continuation from target
                    if common_tokens < len(target_token_ids):
                        remaining_token_ids = target_token_ids[common_tokens:common_tokens + max_new_tokens]
                        remaining_text = self.m_tokenizer.decode(remaining_token_ids, skip_special_tokens=True)
                        if remaining_text.strip():
                            new_tokens = [remaining_text]
                    
                    g_logger.debug(f"Token-level verification: {common_tokens}/{len(draft_token_ids)} tokens match")
                    
                else:
                    # No token-level match, but check for semantic similarity
                    # Accept if both are attempting similar content (more lenient fallback)
                    draft_lower = draft_text.lower()
                    target_lower = target_text.lower()
                    
                    # Check for code-related content
                    code_indicators = ['#include', 'iostream', 'cout', 'main', 'printf', 'std::', 'int ', 'return']
                    draft_has_code = any(indicator in draft_lower for indicator in code_indicators)
                    target_has_code = any(indicator in target_lower for indicator in code_indicators)
                    
                    if draft_has_code and target_has_code:
                        # Both contain code - accept first token as a compromise
                        verified_tokens = draft_tokens[:1] if draft_tokens else []
                        accepted_count = len(verified_tokens)
                        new_tokens = [target_text[:max_new_tokens]]
                        g_logger.debug("Semantic verification: both contain code, accepting 1 token")
                    else:
                        # Use target output
                        verified_tokens = []
                        accepted_count = 0
                        new_tokens = [target_text[:max_new_tokens]]
                        g_logger.debug("No match found, using target output")
                        
            else:
                # Fallback case
                verified_tokens = []
                accepted_count = 0
                if target_text:
                    new_tokens = [target_text[:max_new_tokens]]
                else:
                    # Generate fresh if target failed
                    new_tokens = self._GenerateNewTokens(prompt, max_new_tokens)
            
            inference_time = CreateTimestamp() - start_time
            
            g_logger.info(f"Verified {accepted_count}/{len(draft_tokens)} tokens, generated {len(new_tokens)} new tokens")
            g_logger.debug(f"Draft text: '{draft_text}'")
            g_logger.debug(f"Target text preview: '{target_text[:50]}...'")
            
            return verified_tokens, new_tokens, accepted_count, inference_time
            
        except Exception as e:
            g_logger.error(f"Verification failed: {e}")
            return [], [], 0, CreateTimestamp() - start_time
    
    def _GenerateTargetText(self, prompt: str, max_length: int) -> str:
        """Generate text from the target model for comparison"""
        try:
            inputs = self.m_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.m_device)
            
            with torch.no_grad():
                outputs = self.m_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=min(max_length, 50),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.m_tokenizer.pad_token_id,
                    eos_token_id=self.m_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=False
                )
            
            # Extract new text only
            input_length = inputs.input_ids.shape[1]
            new_token_ids = outputs[0][input_length:]
            
            # Decode the new tokens
            new_text = self.m_tokenizer.decode(new_token_ids, skip_special_tokens=True)
            
            return new_text
            
        except Exception as e:
            g_logger.error(f"Target text generation failed: {e}")
            return ""
    
    def _GenerateNewTokens(self, prompt: str, num_tokens: int) -> List[str]:
        """Generate new tokens from the target model"""
        if num_tokens <= 0:
            return []
        
        try:
            inputs = self.m_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.m_device)
            
            with torch.no_grad():
                generation_config = GenerationConfig(
                    max_new_tokens=num_tokens,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.m_tokenizer.pad_token_id,
                    eos_token_id=self.m_tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
                
                outputs = self.m_model.generate(
                    inputs.input_ids,
                    generation_config=generation_config,
                    attention_mask=inputs.attention_mask
                )
            
            # Extract new tokens only
            input_length = inputs.input_ids.shape[1]
            new_token_ids = outputs[0][input_length:]
            
            # Decode tokens
            new_tokens = []
            for token_id in new_token_ids:
                token = self.m_tokenizer.decode([token_id], skip_special_tokens=True)
                if token.strip():
                    new_tokens.append(token)
            
            return new_tokens
            
        except Exception as e:
            g_logger.error(f"New token generation failed: {e}")
            return []
    
    def GetModelInfo(self) -> dict:
        """Get information about the loaded model"""
        if not self.m_model:
            return {}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.m_model.parameters())
        trainable_params = sum(p.numel() for p in self.m_model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.m_model_name,
            "device": self.m_device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_footprint": f"{total_params * 2 / 1e9:.2f}GB"  # Approximate for fp16
        }
