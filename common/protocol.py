"""
Common utilities and data structures for SpecECD implementation
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import time
import json

class MessageType(Enum):
    DRAFT_REQUEST = "draft_request"
    DRAFT_RESPONSE = "draft_response"
    VERIFY_REQUEST = "verify_request"
    VERIFY_RESPONSE = "verify_response"
    BASELINE_REQUEST = "baseline_request"  # New: For cloud-only inference
    BASELINE_RESPONSE = "baseline_response"  # New: For cloud-only response

@dataclass
class SpeculativeRequest:
    """Request sent from edge to cloud for verification"""
    prompt: str
    draft_tokens: List[str]
    draft_probabilities: Optional[List[float]] = None  # CRITICAL: Add probabilities for proper verification
    max_new_tokens: int = 50
    temperature: float = 0.7
    request_id: str = ""
    timestamp: float = 0.0

@dataclass
class SpeculativeResponse:
    """Response sent from cloud to edge"""
    verified_tokens: List[str]
    new_tokens: List[str]
    accepted_count: int
    total_draft_count: int
    early_exit: bool = False
    request_id: str = ""
    timestamp: float = 0.0
    defer_future_drafts: bool = False
    deferral_reason: str = ""

@dataclass
class BaselineRequest:
    """Request sent for cloud-only baseline inference"""
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    request_id: str = ""
    timestamp: float = 0.0

@dataclass
class BaselineResponse:
    """Response sent for cloud-only baseline inference"""
    generated_text: str
    tokens_generated: int
    inference_time: float
    request_id: str = ""
    timestamp: float = 0.0

@dataclass
class PerformanceMetrics:
    """Performance tracking for the speculative decoding system"""
    end_to_end_latency: float = 0.0
    network_latency: float = 0.0
    edge_inference_time: float = 0.0
    cloud_inference_time: float = 0.0
    token_acceptance_rate: float = 0.0
    total_tokens_generated: int = 0
    speedup_ratio: float = 0.0
    # Deferral / cascade metrics
    deferred: bool = False
    deferral_reason: str = ""
    speculative_phase_tokens: int = 0  # Tokens (accepted + new) generated before deferral
    baseline_phase_tokens: int = 0     # Tokens generated after deferral (cloud-only)
    batches_before_deferral: int = 0
    
def CalculateAcceptanceRate(verified_count: int, total_draft_count: int) -> float:
    """Calculate token acceptance rate"""
    if total_draft_count == 0:
        return 0.0
    return verified_count / total_draft_count

def CalculateSpeedup(spec_time: float, baseline_time: float) -> float:
    """Calculate speedup ratio compared to baseline"""
    if spec_time == 0:
        return 0.0
    return baseline_time / spec_time

def CreateTimestamp() -> float:
    """Create current timestamp"""
    return time.time()

def SerializeMessage(message: Any) -> str:
    """Serialize message to JSON string"""
    if hasattr(message, '__dict__'):
        return json.dumps(message.__dict__)
    return json.dumps(message)

def DeserializeMessage(message_str: str, message_type: type) -> Any:
    """Deserialize JSON string to message object"""
    data = json.loads(message_str)
    
    # Handle optional fields that might not be present in older messages
    if message_type == SpeculativeRequest:
        if 'draft_probabilities' not in data:
            data['draft_probabilities'] = None
    
    return message_type(**data)
