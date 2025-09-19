# SpecECD: Speculative Edge-Cloud Decoding

Implementation of speculative decoding with corrected probabilistic verification based on "Fast Inference from Transformers via Speculative Decoding".

## Overview

This project implements a distributed inference system where:
- **Edge Device**: Runs a small draft model (Llama-3.2-1B-Instruct) to generate speculative tokens with probabilities
  - **CPU Support**: PyTorch-based inference for broad compatibility
  - **GPU Support**: CUDA acceleration for faster edge inference
  - 4. **RoPE Scaling Configuration Error** (`rope_scaling must be a dictionary with two fields`):
   - **Problem**: Older OpenVINO versions had issues with Llama 3.2's extended RoPE scaling format
   - **Solution**: Upgrade to OpenVINO 2025.3.0 which includes improved RoPE support:
     ```bash
     # Upgrade to latest OpenVINO with RoPE fixes
     pip uninstall openvino openvino-dev optimum -y
     pip install "openvino==2025.3.0" "openvino-dev==2025.3.0" "optimum[openvino]>=1.22.0"
     ```
   - **If issue persists**: Use fallback devices while we investigate:
     ```bash
     # Use CPU mode (always works)
     python run_tests.py --device cpu
     
     # Use GPU mode if NVIDIA GPU available
     python run_tests.py --device gpu
     
     # Check what devices work on your system
     python run_edge.py --list-devices
     ```
   - **Note**: Latest OpenVINO 2025.3.0 should resolve most RoPE compatibility issuesport**: OpenVINO NPU acceleration for Intel Arc GPUs and dedicated NPU hardware
- **Cloud Server**: Runs a larger target model (Llama-3.1-8B-Instruct) for probabilistic verification and completion
- **Communication**: WebSocket-based protocol for real-time coordination

## Architecture

```
┌─────────────────┐    WebSocket     ┌─────────────────┐
│   Edge Client   │◄────────────────►│  Cloud Server   │
│  (Draft Model)  │    Verification  │ (Target Model)  │
│     1B params   │     Requests     │    8B params    │
└─────────────────┘                  └─────────────────┘
```

## Model Selection

### Edge (Draft Model)
- **Model**: meta-llama/Llama-3.2-1B-Instruct
- **Parameters**: 1.2B
- **Features**: Probability tracking for each generated token
- **Acceleration Options**:
  - **CPU**: PyTorch inference (default, compatible with all systems)
  - **GPU**: CUDA acceleration (NVIDIA GPUs, faster inference)
  - **NPU**: OpenVINO NPU acceleration (Intel Arc GPUs, dedicated NPU hardware)
    - Automatically uses pre-converted model: `srang992/Llama-3.2-1B-Instruct-ov-INT8`
    - INT8 quantization for better NPU performance
    - Bypasses RoPE scaling conversion issues

### Cloud (Target Model)  
- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Parameters**: 8B
- **Features**: Probabilistic verification using single forward pass

## Setup Instructions

### Cloud Server Setup (CUDA GPU Only)

The cloud server only needs CUDA GPU support for running the larger target model.

```bash
# 1. Create environment (recommended)
conda create -n specd-cloud python=3.11
conda activate specd-cloud

# OR with venv
python -m venv specd_cloud_env
# Windows
specd_cloud_env\Scripts\activate
# Linux/Mac
source specd_cloud_env/bin/activate

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install cloud dependencies
pip install -r requirements-cloud.txt
```

### Edge Client Setup (CPU, GPU, NPU Support)

The edge client supports multiple acceleration options. Choose the appropriate setup for your hardware:

#### Option A: CPU-Only Edge Inference
Perfect for development, testing, or systems without GPU/NPU acceleration.

```bash
# 1. Create environment
conda create -n specd-edge-cpu python=3.11
conda activate specd-edge-cpu

# OR with venv
python -m venv specd_edge_cpu_env
# Windows: specd_edge_cpu_env\Scripts\activate
# Linux/Mac: source specd_edge_cpu_env/bin/activate

# 2. Install CPU-only dependencies
pip install torch torchvision torchaudio  # CPU version
pip install -r requirements-edge-cpu.txt
```

#### Option B: CUDA GPU Edge Inference  
For NVIDIA GPUs - provides significant speedup over CPU.

```bash
# 1. Create environment
conda create -n specd-edge-gpu python=3.11
conda activate specd-edge-gpu

# OR with venv
python -m venv specd_edge_gpu_env
# Windows: specd_edge_gpu_env\Scripts\activate
# Linux/Mac: source specd_edge_gpu_env/bin/activate

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install GPU-specific dependencies
pip install -r requirements-edge-gpu.txt
```

#### Option C: OpenVINO NPU Edge Inference
For Intel Arc GPUs and NPU-enabled CPUs - optimized for AI inference.

```bash
# 1. Create environment
conda create -n specd-edge-npu python=3.11
conda activate specd-edge-npu

# OR with venv
python -m venv specd_edge_npu_env
# Windows: specd_edge_npu_env\Scripts\activate
# Linux/Mac: source specd_edge_npu_env/bin/activate

# 2. Install NPU-specific dependencies
pip install torch torchvision torchaudio  # CPU version works with NPU
pip install -r requirements-edge-npu.txt
```

### Development Setup (Both Cloud and Edge)

For development or testing both components:

```bash
# 1. Create environment
conda create -n specd-dev python=3.11
conda activate specd-dev

# 2. Install PyTorch with CUDA (for cloud compatibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install both cloud and edge dependencies
pip install -r requirements-cloud.txt
pip install -r requirements-edge.txt
```

**Verify Installation:**
```bash
# Verify core functionality (all setups)
python -c "import torch, transformers, websockets; print('Core imports successful')"

# Verify GPU support (cloud and edge GPU setups)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

# Verify GPU monitoring (edge GPU setup)
python -c "import pynvml; print('NVIDIA-ML available')"

# Verify NPU support (edge NPU setup)  
python -c "import openvino; print('OpenVINO version:', openvino.__version__)"
python -c "import optimum.intel; print('Optimum Intel available')"
```

### GPU Setup (Optional)

CUDA GPU provides significant acceleration for edge inference:

#### 1. Hardware Requirements
- **NVIDIA GPU**: GeForce RTX, Quadro, or Tesla series with CUDA support
- **CUDA Drivers**: Latest NVIDIA drivers (minimum 11.8)
- **Memory**: At least 4GB VRAM for 1B model

#### 2. Check GPU Availability
```bash
# List available edge devices (includes GPU info)
python run_edge.py --list-devices

# Test GPU functionality
python test_gpu.py
```

#### 3. Enable GPU in Configuration
Edit `config.toml`:
```toml
[devices]
edge_device = "gpu"  # Force GPU usage

# Or enable GPU with settings
[devices.gpu]
enabled = true       # Try GPU, fallback to CPU if unavailable
device_id = 0        # Use first GPU
```

### OpenVINO NPU Setup (Optional)

OpenVINO NPU provides hardware acceleration for Intel Arc GPUs and dedicated NPU chips:

#### 1. Check NPU Availability
```bash
# List available edge devices
python run_edge.py --list-devices

# Test NPU functionality
python test_npu.py
```

#### 2. Enable NPU in Configuration
Edit `config.toml`:
```toml
[devices]
edge_device = "npu"  # Force NPU usage

# Or enable NPU with fallback
[devices.npu]
enabled = true       # Try NPU, fallback to CPU if unavailable
```

#### 3. Hardware Requirements
- **Intel Arc GPUs**: Intel Arc A-series with NPU support
- **Intel CPUs**: 12th gen Core or newer with built-in NPU
- **Dedicated NPU**: Intel Movidius or similar accelerators
- **Drivers**: Latest Intel graphics drivers and OpenVINO runtime

### Network Configuration
1. Ensure both machines are on the same local network
2. Configure firewall to allow traffic on port 8765
3. Update IP addresses in the scripts

## Usage

### 1. Start Cloud Server
```bash
python run_cloud.py
```

### 2. Run Edge Client
```bash
# Connect to localhost (testing) - uses device from config
python run_edge.py

# Connect with specific device
python run_edge.py --device cpu   # Force CPU inference
python run_edge.py --device gpu   # Force GPU acceleration
python run_edge.py --device npu   # Force NPU acceleration

# Connect to remote server with GPU
python run_edge.py --host 192.168.1.100 --device gpu

# Check available devices
python run_edge.py --list-devices
```

### 3. Run Performance Tests
```bash
# Test with default 5 prompts, 2 iterations each (uses config.toml device)
python run_tests.py

# Test with custom number of prompts
python run_tests.py --num-prompts 3 --iterations 1

# Test with specific device
python run_tests.py --device cpu     # Force CPU testing
python run_tests.py --device gpu     # Force GPU testing  
python run_tests.py --device npu     # Force NPU testing

# Test with remote server and specific device
python run_tests.py --host 192.168.1.100 --port 8765 --device gpu

# Performance comparison across devices
python run_tests.py --device cpu --num-prompts 3 --iterations 2
python run_tests.py --device gpu --num-prompts 3 --iterations 2
python run_tests.py --device npu --num-prompts 3 --iterations 2
```

## Test Options

### Command Line Arguments
- `--num-prompts N`: Number of test prompts to use (1-10, default: 5)
- `--iterations N`: Number of iterations per prompt (default: 2)
- `--host HOST`: Cloud server host (default: localhost)
- `--port PORT`: Cloud server port (default: 8765)
- `--device DEVICE`: Edge device to use (cpu, gpu, npu). If not specified, uses config.toml setting

### Device Testing Examples
```bash
# Compare CPU vs GPU performance
python run_tests.py --device cpu --num-prompts 5 --iterations 2
python run_tests.py --device gpu --num-prompts 5 --iterations 2

# Test NPU acceleration
python run_tests.py --device npu --num-prompts 3 --iterations 1

# Remote server with GPU edge
python run_tests.py --host 192.168.1.100 --device gpu
```

### Test Prompts
The system tests diverse prompts including:
- AI and technology topics
- Scientific discoveries
- Programming tasks
- General knowledge topics

### Performance Testing Benefits
- **Device Comparison**: Test CPU vs GPU vs NPU performance side-by-side
- **Hardware Validation**: Verify acceleration is working correctly
- **Optimization Guidance**: Identify best device for your workload
- **Acceptance Rate Analysis**: Compare token acceptance across devices
- **Latency Breakdown**: Detailed timing analysis per device type

## File Structure

```
SpecECD/
├── requirements.txt              # All dependencies (for development)
├── requirements-cloud.txt        # Cloud server dependencies (CUDA GPU only)
├── requirements-edge.txt         # Edge client dependencies (all options)
├── requirements-edge-cpu.txt     # Edge client dependencies (CPU only)
├── requirements-edge-gpu.txt     # Edge client dependencies (CUDA GPU)
├── requirements-edge-npu.txt     # Edge client dependencies (OpenVINO NPU)
├── config.toml                   # Configuration file with device options
├── run_cloud.py              # Cloud server entry point
├── run_edge.py               # Edge client entry point  
├── run_tests.py              # Performance test entry point
├── test_npu.py               # NPU functionality tests
├── test_gpu.py               # GPU functionality tests
├── common/
│   ├── config.py             # Configuration loader with GPU/NPU support
│   └── protocol.py           # Shared data structures and utilities
├── edge/
│   ├── draft_model.py        # Edge draft model with CPU/GPU/NPU support
│   ├── openvino_model.py     # OpenVINO NPU model wrapper
│   └── client.py             # Edge client with WebSocket communication
├── cloud/
│   ├── target_model.py       # Cloud target model with probabilistic verification
│   └── server.py             # Cloud server with WebSocket handling
└── tests/
    └── performance_test.py   # Performance evaluation suite
```

## Corrected Algorithm

This implementation includes the **corrected speculative decoding algorithm**:

### Draft Model (Edge)
- Generates tokens using autoregressive sampling
- **Tracks probability for each generated token**
- Sends both tokens and probabilities to cloud

### Target Model (Cloud)
- **Single forward pass** on prompt + draft tokens
- For each draft token: `p_accept = min(1.0, p_target / p_draft)`
- **Probabilistic acceptance** using random sampling
- Generates new token only when rejecting

### Key Improvements
- ✅ **Probabilistic verification** (not string matching)
- ✅ **Single forward pass** (not double inference)  
- ✅ **Proper probability tracking** at each step
- ✅ **Corrected acceptance sampling** algorithm

## Performance Metrics

The system measures and reports:

- **End-to-end latency**: Total generation time
- **Token acceptance rate**: Critical metric for algorithm effectiveness
- **Network latency**: Communication overhead
- **Edge inference time**: Draft model processing (CPU, GPU, or NPU)
- **Component breakdown**: Detailed timing analysis

### Expected Results
| Configuration | Speedup | Acceptance Rate | Notes |
|---------------|---------|-----------------|-------|
| **Cloud Only** | 1.0x | - | Baseline performance |
| **Edge CPU + Cloud** | 2-3x | 60-80% | Standard speculative decoding |
| **Edge GPU + Cloud** | 3-4x | 60-80% | GPU acceleration benefit |
| **Edge NPU + Cloud** | 3-5x | 60-80% | NPU acceleration benefit |

### Hardware Acceleration Benefits
- **CPU**: Broad compatibility, no special hardware required
- **GPU**: High throughput, significant speedup over CPU (2-5x faster)
- **NPU**: Efficient low-power acceleration with strong speculative throughput

## Speculative Cascade Deferral (Adaptive Drafting)

The system supports a *cascade deferral* mechanism: the cloud can instruct the edge to stop
producing draft tokens when speculation becomes inefficient (e.g., low acceptance rates or
probability divergence). After deferral, generation continues in **baseline (cloud-only)** mode.

### Why Deferral?
Speculative decoding offers speedups only if a meaningful fraction of draft tokens are accepted.
When acceptance collapses (model mismatch, drift, domain shift), continuing to draft wastes edge
compute and increases latency. Deferral lets the system adapt dynamically.

### How It Works
1. Edge sends: prompt + draft tokens + draft token probabilities.
2. Cloud verifies probabilistically (single forward pass) and gathers per-token stats.
3. A pluggable deferral strategy evaluates acceptance / probability signals.
4. If deferral is triggered, cloud sets `defer_future_drafts=true` in `SpeculativeResponse`.
5. Edge switches to baseline mode (single `BaselineRequest` for the remaining tokens).

### Configuration (`config.toml`)
```toml
[deferral]
# Strategy options: "never", "low_acceptance", "prob_divergence"
strategy = "low_acceptance"

# Low-acceptance strategy thresholds
batch_threshold = 0.30                # Per-batch acceptance requirement
cumulative_threshold = 0.45            # Overall (since start) acceptance requirement
min_batches_before_consider = 2        # Don't evaluate until N speculative batches processed
require_consecutive_failures = true    # Require two sequential failing batches to defer

# Probability divergence (KL heuristic)
kl_divergence_threshold = 5.0          # Only used by prob_divergence strategy

scope = "request"                      # Reserved for future finer control
```

### Strategy Details
| Strategy | Name(s) | Condition | Use Case |
|----------|---------|-----------|----------|
| Never | `never`, `none`, `off` | Always continue drafting | Baseline / debugging |
| Low Acceptance | `low_acceptance` | Batch & cumulative acceptance below thresholds (optionally consecutive) | General adaptive control |
| Probability Divergence | `prob_divergence`, `kl` | Average per-token KL divergence above threshold | Draft model drift / mismatch |

### Runtime Signals (Cloud → Edge)
`SpeculativeResponse` fields:
```jsonc
{
   "verified_tokens": ["..."],
   "new_tokens": ["..."],
   "accepted_count": 3,
   "total_draft_count": 5,
   "defer_future_drafts": true,
   "deferral_reason": "low_acceptance batch=0.20 < 0.30 and cumulative=0.40 < 0.45"
}
```

### Metrics Extensions
`PerformanceMetrics` now includes:
- `deferred` (bool)
- `deferral_reason`
- `speculative_phase_tokens`
- `baseline_phase_tokens`

### Forcing a Quick Deferral (Testing)
Set very high thresholds so nearly any realistic batch fails:
```toml
[deferral]
strategy = "low_acceptance"
batch_threshold = 0.95
cumulative_threshold = 0.95
min_batches_before_consider = 1
require_consecutive_failures = false
```
Run a short generation; you should see a log message:
```
Deferring further drafts ... reason=low_acceptance batch=... cumulative=...
```

### Disabling Deferral
```toml
[deferral]
strategy = "never"
```

## Deferral Smoke Test
See `tests/deferral_test.py` for an automated heuristic test harness that
simulates a low-acceptance setting and asserts that deferral is triggered.
- **NPU**: Lower power consumption, optimized for AI inference

## Performance Assessment

The test report includes automatic assessment:
- **EXCELLENT**: >50% acceptance rate (algorithm working correctly)
- **GOOD**: >30% acceptance rate (significant improvement)  
- **MODERATE**: >10% acceptance rate (some improvement)
- **POOR**: <10% acceptance rate (needs further tuning)

## Key Features

1. **Corrected Speculative Decoding**: Implements proper probabilistic verification
2. **Probability Tracking**: Draft model tracks and sends token probabilities
3. **Single Forward Pass**: Target model processes all draft tokens at once
4. **Multi-Device Support**: CPU (PyTorch), GPU (CUDA), and NPU (OpenVINO) acceleration
5. **Flexible Configuration**: Easy device switching via config or command line
6. **Automatic Fallback**: GPU/NPU fall back to CPU if hardware unavailable
7. **Flexible Testing**: Configurable number of prompts and iterations
8. **Real-time Communication**: WebSocket-based low-latency protocol
9. **Performance Monitoring**: Comprehensive metrics and assessment

## Troubleshooting

### Common Issues
1. **Connection Failed**: Check firewall settings and IP addresses
2. **Model Loading Error**: Ensure sufficient RAM/VRAM available
3. **Low Acceptance Rate**: May indicate model mismatch or algorithm issues
4. **CUDA Issues**: Verify PyTorch CUDA installation
5. **GPU Not Available**: Run `python test_gpu.py` to check GPU status
6. **NPU Not Available**: Run `python test_npu.py` to check NPU status

### Dependency Conflicts
**Problem**: pip dependency resolver shows version conflicts (numpy, networkx, pillow)

**Common Error Messages**:
```
contourpy 1.2.0 requires numpy<2.0,>=1.20, but you have numpy 2.1.2
openvino-dev requires networkx<=3.1.0, but you have networkx 3.3
streamlit requires pillow<11,>=7.1.0, but you have pillow 11.0.0
```

**Solutions**:

1. **Use Device-Specific Requirements (Recommended)**:
   ```bash
   # For cloud only (no OpenVINO conflicts)
   pip install -r requirements-cloud.txt
   
   # For edge CPU (minimal dependencies)
   pip install -r requirements-edge-cpu.txt
   
   # For edge GPU (no OpenVINO conflicts)
   pip install -r requirements-edge-gpu.txt
   
   # For edge NPU (has pre-resolved compatibility constraints)
   pip install -r requirements-edge-npu.txt
   ```

2. **Clean Environment**:
   ```bash
   # Create device-specific environment
   conda create -n specd-edge-npu python=3.11  # or cpu/gpu
   conda activate specd-edge-npu
   
   # Install device-specific requirements
   pip install -r requirements-edge-npu.txt
   ```

3. **Force Compatible Versions** (only needed for NPU setup):
   ```bash
   # Only needed if using the combined requirements-edge.txt
   pip install "numpy>=1.24.0,<2.0.0" "networkx<=3.1.0" "pillow<11.0.0" --force-reinstall
   pip install -r requirements-edge-npu.txt --force-reinstall
   ```

**Verification**:
```bash
# Check for conflicts
pip check

# Verify device-specific imports
python -c "import torch, transformers; print('Core imports successful')"

# For NPU setup only
python -c "import openvino, numpy, networkx; print('NPU imports successful')"
```

### GPU Issues
1. **GPU Not Detected**:
   - Check hardware compatibility (NVIDIA GPU with CUDA support)
   - Update NVIDIA drivers
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

2. **PyTorch CUDA Installation Issues**:
   - **Problem**: "CUDA not available" warning even after installing CUDA PyTorch
   - **Solution**: Uninstall and reinstall PyTorch with CUDA:
     ```bash
     pip uninstall torch torchvision torchaudio
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - **Verify**: `python -c "import torch; print('CUDA version:', torch.version.cuda)"`

3. **CUDA Out of Memory**:
   - Close other GPU applications to free memory
   - Try CPU fallback: `--device cpu`
   - Monitor GPU memory usage: `nvidia-smi`

4. **Performance Lower Than Expected**:
   - Check if GPU is actually being used in model info output
   - Monitor GPU utilization: `nvidia-smi -l 1`
   - Compare with CPU baseline: `python run_edge.py --device cpu`

### OpenVINO NPU Issues
1. **NPU Not Detected**: 
   - Check hardware compatibility (Intel Arc GPU or NPU-enabled CPU)
   - Update Intel graphics drivers
   - Verify OpenVINO installation: `python -c "import openvino; print('OpenVINO OK')"`

2. **OpenVINO API Compatibility Error** (`module 'openvino' has no attribute 'Node'`):
   - **Problem**: Version mismatch between OpenVINO and optimum[openvino]
   - **Solution**: Complete uninstall and reinstall with latest versions:
     ```bash
     # Step 1: Complete uninstall
     pip uninstall openvino openvino-dev optimum -y
     pip uninstall transformers huggingface-hub -y
     
     # Step 2: Clear pip cache
     pip cache purge
     
     # Step 3: Reinstall with latest versions (includes Llama 3.2 RoPE support)
     pip install "openvino==2025.3.0" "openvino-dev==2024.6.0" "optimum[openvino]>=1.25.2"
     
     # Step 4: Verify installation
     python -c "import openvino; print('OpenVINO version:', openvino.__version__)"
     python -c "from optimum.intel import OVModelForCausalLM; print('Optimum Intel OK')"
     ```
   - **Alternative**: Use the NPU requirements file:
     ```bash
     pip uninstall openvino openvino-dev optimum transformers huggingface-hub -y
     pip install -r requirements-edge-npu.txt
     ```

3. **Model Conversion Failed**:
   - Ensure sufficient disk space for IR model files
   - Check internet connection for model download
   - Try CPU fallback: `--device cpu`

4. **RoPE Scaling Configuration Error** (`rope_scaling must be a dictionary with two fields`):
   - **Problem**: Original Llama 3.2 models use extended RoPE scaling format not supported by OpenVINO
   - **Solution**: For NPU inference, we automatically use a pre-converted OpenVINO model:
     - **NPU Model**: `srang992/Llama-3.2-1B-Instruct-ov-INT8`
     - **Benefits**: No conversion needed, INT8 quantization, better NPU performance
     - **Automatic**: System detects Llama 3.2 and switches to pre-converted model for NPU only
     - **CPU/GPU**: Continue using original model without issues
   - **If issue still persists**: Use fallback devices:
     ```bash
     python run_tests.py --device cpu   # Fallback to CPU
     python run_tests.py --device gpu   # Use GPU if available
     ```
   - **Note**: Pre-converted model only used for NPU - CPU/GPU use original model

5. **Performance Lower Than Expected**:
   - Check if NPU is actually being used in model info output
   - Verify no other processes are using NPU
   - Compare with CPU baseline: `python run_edge.py --device cpu`

### Performance Tips
1. Use GPU acceleration when available on cloud server
2. Monitor acceptance rates - should be >30% for effective speedup
3. **NPU**: Use for edge inference, especially on battery-powered devices
4. Adjust temperature and sampling parameters if needed
5. Ensure models are properly loaded before testing

## Research Applications

This implementation demonstrates:
- Correct speculative decoding algorithm implementation
- Edge-cloud distributed inference patterns
- Multi-device acceleration (CPU, NPU) for edge AI
- Probabilistic token verification methods
- Performance analysis and benchmarking across hardware platforms
- Model combination effectiveness evaluation
- OpenVINO integration for AI acceleration
