"""
Edge client for communicating with cloud server
Handles network communication and request/response processing
"""
import asyncio
import websockets
import json
from typing import Optional, List
import logging
from common.protocol import (
    SpeculativeRequest, SpeculativeResponse, PerformanceMetrics,
    SerializeMessage, DeserializeMessage, CreateTimestamp,
    BaselineRequest, BaselineResponse
)
from common.config import get_edge_model_config
from edge.draft_model import EdgeDraftModel

# Configure logging
logging.basicConfig(level=logging.INFO)
g_logger = logging.getLogger(__name__)

class EdgeClient:
    """Client running on edge device to communicate with cloud"""
    
    def __init__(self, cloud_host: str = "localhost", cloud_port: int = 8765, device: str = None):
        self.m_cloud_host = cloud_host
        self.m_cloud_port = cloud_port
        
        # Get device configuration
        edge_config = get_edge_model_config()
        self.m_device = device if device is not None else edge_config["device"]
        self.m_gpu_device_id = edge_config.get("gpu_device_id", 0)
        
        # Initialize draft model with configured device and GPU index
        self.m_draft_model = EdgeDraftModel(
            model_name=edge_config["model_name"],
            device=self.m_device,
            gpu_device_id=self.m_gpu_device_id
        )
        
        self.m_websocket = None
        self.m_performance_metrics = PerformanceMetrics()
        self.m_deferred = False  # Set True when cloud instructs to stop drafting
        self.m_deferral_reason = ""
        
        g_logger.info(f"EdgeClient initialized with device: {self.m_device}")
        
    async def Initialize(self) -> bool:
        """Initialize the edge client and load draft model"""
        g_logger.info("Initializing edge client...")
        
        # Load draft model
        if not self.m_draft_model.LoadModel():
            g_logger.error("Failed to load draft model")
            return False
            
        g_logger.info("Edge client initialized successfully")
        return True
    
    async def ConnectToCloud(self) -> bool:
        """Establish WebSocket connection to cloud server"""
        try:
            uri = f"ws://{self.m_cloud_host}:{self.m_cloud_port}"
            g_logger.info(f"Connecting to cloud server at {uri}")
            
            # Increase timeout for slow model inference
            self.m_websocket = await websockets.connect(
                uri,
                ping_interval=300,  # 5 minutes ping interval
                ping_timeout=300,   # 5 minutes ping timeout
                close_timeout=60    # 1 minute close timeout
            )
            g_logger.info("Connected to cloud server")
            return True
            
        except Exception as e:
            g_logger.error(f"Failed to connect to cloud: {e}")
            return False
    
    async def GenerateWithSpeculation(self, prompt: str, max_tokens: int = 50) -> tuple[str, PerformanceMetrics]:
        """
        Generate text using speculative decoding with cloud verification
        Returns (generated_text, performance_metrics)
        """
        if not self.m_websocket:
            g_logger.error("Not connected to cloud server")
            return "", self.m_performance_metrics
        
        start_time = CreateTimestamp()
        generated_text = ""
        total_accepted = 0
        total_drafted = 0
        
        try:
            current_prompt = prompt
            remaining_tokens = max_tokens
            
            while remaining_tokens > 0:
                # If deferral has been triggered, switch to baseline (cloud-only) mode
                if self.m_deferred:
                    g_logger.info(f"Baseline mode engaged (deferral). Remaining tokens: {remaining_tokens}. Reason: {self.m_deferral_reason}")
                    baseline_request = BaselineRequest(
                        prompt=current_prompt,
                        max_new_tokens=remaining_tokens,
                        temperature=self.m_performance_metrics.token_acceptance_rate or 0.7,  # reuse metric placeholder or default
                        request_id=f"baseline_{CreateTimestamp()}",
                        timestamp=CreateTimestamp()
                    )
                    network_start = CreateTimestamp()
                    try:
                        await self.m_websocket.send(SerializeMessage(baseline_request))
                        response_data = await asyncio.wait_for(
                            self.m_websocket.recv(),
                            timeout=300.0
                        )
                        network_time = CreateTimestamp() - network_start
                        self.m_performance_metrics.network_latency += network_time
                    except asyncio.TimeoutError:
                        g_logger.error("Cloud server response timeout (baseline mode)")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        g_logger.error("Connection to cloud server lost (baseline mode)")
                        break

                    baseline_response = DeserializeMessage(response_data, BaselineResponse)
                    generated_text += baseline_response.generated_text
                    current_prompt += baseline_response.generated_text
                    # Update baseline phase tokens (difference after baseline completion)
                    self.m_performance_metrics.baseline_phase_tokens = (
                        len(generated_text.split()) - self.m_performance_metrics.speculative_phase_tokens
                    )
                    # Treat words as tokens for counting; more accurate token accounting could be added later
                    remaining_tokens = 0  # Assume baseline satisfies remainder
                    g_logger.info(
                        f"Baseline completion tokens_generated={baseline_response.tokens_generated} inference_time={baseline_response.inference_time:.3f}s"
                    )
                    break  # Exit main loop after baseline completion

                # Generate draft tokens on edge with probabilities
                draft_start = CreateTimestamp()
                # Use the new method that returns probabilities
                draft_tokens, draft_probs, edge_inference_time = self.m_draft_model.GenerateDraftTokensWithProbabilities(
                    current_prompt, 
                    min(5, remaining_tokens)
                )
                
                if not draft_tokens:
                    g_logger.warning("No draft tokens generated, stopping")
                    break
                
                # Create verification request with probabilities
                request = SpeculativeRequest(
                    prompt=current_prompt,
                    draft_tokens=draft_tokens,
                    draft_probabilities=draft_probs,  # Include draft probabilities
                    max_new_tokens=remaining_tokens,
                    request_id=f"req_{CreateTimestamp()}",
                    timestamp=CreateTimestamp()
                )
                
                # Send to cloud for verification
                network_start = CreateTimestamp()
                try:
                    await self.m_websocket.send(SerializeMessage(request))
                    
                    # Receive response from cloud with timeout
                    response_data = await asyncio.wait_for(
                        self.m_websocket.recv(),
                        timeout=300.0  # 5 minute timeout for slow inference
                    )
                    network_time = CreateTimestamp() - network_start
                    
                except asyncio.TimeoutError:
                    g_logger.error("Cloud server response timeout")
                    break
                except websockets.exceptions.ConnectionClosed:
                    g_logger.error("Connection to cloud server lost")
                    break
                
                # Parse response
                response = DeserializeMessage(response_data, SpeculativeResponse)
                
                # Process verified tokens
                verified_text = "".join(response.verified_tokens)
                new_text = "".join(response.new_tokens)
                
                # Update generated text and prompt
                accepted_text = verified_text + new_text
                generated_text += accepted_text
                current_prompt += accepted_text
                
                # Update metrics
                total_accepted += response.accepted_count
                total_drafted += response.total_draft_count
                remaining_tokens -= len(response.verified_tokens) + len(response.new_tokens)
                
                # Update performance metrics
                self.m_performance_metrics.network_latency += network_time
                self.m_performance_metrics.edge_inference_time += edge_inference_time
                
                g_logger.info(f"Accepted: {response.accepted_count}/{response.total_draft_count} draft tokens")
                # Check for deferral instruction
                if getattr(response, 'defer_future_drafts', False):
                    self.m_deferred = True
                    self.m_deferral_reason = getattr(response, 'deferral_reason', '')
                    g_logger.info(
                        f"Deferral instruction received from cloud. Reason: {self.m_deferral_reason or 'n/a'}. Switching to baseline mode."
                    )
                    # Continue loop; baseline path will execute at top next iteration
                    continue
                
                # Check for early exit
                if response.early_exit or remaining_tokens <= 0:
                    break
            
            # Calculate final metrics
            total_time = CreateTimestamp() - start_time
            self.m_performance_metrics.end_to_end_latency = total_time
            self.m_performance_metrics.token_acceptance_rate = (
                total_accepted / total_drafted if total_drafted > 0 else 0.0
            )
            self.m_performance_metrics.total_tokens_generated = len(generated_text.split())
            
            g_logger.info(f"Generation completed in {total_time:.3f}s")
            g_logger.info(f"Token acceptance rate: {self.m_performance_metrics.token_acceptance_rate:.2%}")
            
            return generated_text, self.m_performance_metrics
            
        except Exception as e:
            g_logger.error(f"Speculative generation failed: {e}")
            return generated_text, self.m_performance_metrics
    
    async def Disconnect(self):
        """Close connection to cloud server"""
        if self.m_websocket:
            await self.m_websocket.close()
            g_logger.info("Disconnected from cloud server")
    
    def GetModelInfo(self) -> dict:
        """Get information about the edge model"""
        return self.m_draft_model.GetModelInfo()
