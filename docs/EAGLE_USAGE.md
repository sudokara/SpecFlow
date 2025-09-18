"""
EAGLE Draft Model Usage Guide
============================

This document explains how to use EAGLE (Extrapolation Algorithm for Greater Language-model 
Efficiency) draft models as an alternative to traditional LLM-based draft generation.

## Overview

EAGLE is a tree-based speculative decoding approach that uses lightweight prediction heads 
to generate multiple draft candidates simultaneously, potentially offering faster inference 
than traditional autoregressive generation.

## Configuration

### Enable EAGLE in config.toml

```toml
[models.draft]
type = "eagle"  # Switch from "llm" to "eagle"

[models.draft.eagle]
num_heads = 3            # Number of EAGLE prediction heads
tree_depth = 2           # Depth of draft token tree  
base_model_layers = 16   # Number of base model layers for features
temperature = 0.8        # Sampling temperature for EAGLE heads

# Optional: Use pre-trained EAGLE model
eagle_model_path = ""    # Leave empty to train from scratch
# eagle_model_path = "yuhuili/EAGLE-llama2-chat-7B"  # HuggingFace repo
# eagle_model_path = "./local/eagle-model"           # Local path
```

### Pre-trained EAGLE Models

EAGLE models can be used in two modes:

#### 1. Training from Scratch (Default)
```toml
[models.draft.eagle]
eagle_model_path = ""    # Empty = train from scratch
num_heads = 3            # Will create 3 randomly initialized heads
```

#### 2. Using Pre-trained Models
```toml
[models.draft.eagle]
# From HuggingFace Hub
eagle_model_path = "yuhuili/EAGLE-llama2-chat-7B"

# From local path
# eagle_model_path = "./models/my-eagle-model"
```

#### Available Pre-trained Models

Popular EAGLE models on HuggingFace:
- `yuhuili/EAGLE-llama2-chat-7B` - For Llama 2 Chat 7B
- `yuhuili/EAGLE-Vicuna-7B-v1.3` - For Vicuna 7B v1.3
- `yuhuili/EAGLE-llama2-chat-13B` - For Llama 2 Chat 13B

**Important**: Ensure the EAGLE model matches your base model architecture!

#### Model Compatibility

| Base Model | Compatible EAGLE Models |
|------------|-------------------------|
| Llama 2 7B Chat | `yuhuili/EAGLE-llama2-chat-7B` |
| Vicuna 7B v1.3 | `yuhuili/EAGLE-Vicuna-7B-v1.3` |
| Llama 2 13B Chat | `yuhuili/EAGLE-llama2-chat-13B` |

### Requirements for Pre-trained Models

Add to your requirements:
```bash
pip install huggingface_hub>=0.16.0
```

Or install from requirements:
```bash
pip install -r requirements-edge.txt
```

### LLM vs EAGLE Comparison

| Aspect | LLM Draft | EAGLE Draft |
|--------|-----------|-------------|
| **Generation Method** | Autoregressive | Tree-based with prediction heads |
| **Speed** | Slower | Potentially faster |
| **Memory Usage** | Higher (full model) | Lower (lightweight heads) |
| **Quality** | High (proven) | Variable (experimental) |
| **Setup Complexity** | Simple | More complex |
| **Training Required** | No | Yes (for heads) |

## Implementation Details

### EAGLE Architecture

1. **Base Model**: Uses the first N layers of the edge model for feature extraction
2. **Prediction Heads**: Lightweight MLPs that predict next tokens from hidden states
3. **Ensemble**: Averages predictions from multiple heads for robustness
4. **Tree Generation**: Generates draft tokens in a tree structure for parallel verification

### Key Components

- `EagleHead`: Individual prediction head (MLP)
- `EagleDraftModel`: Main EAGLE implementation
- `DraftModelFactory`: Creates EAGLE or LLM models based on config

## Performance Considerations

### When to Use EAGLE

✅ **Good for:**
- Production environments requiring speed
- Large-scale inference with many requests
- When computational efficiency is critical
- Scenarios with repetitive or predictable text patterns

❌ **Avoid when:**
- High accuracy is critical
- Working with very diverse or creative text
- Limited development time for tuning
- First-time implementation (start with LLM)

### Memory Requirements

EAGLE typically uses less memory than full LLM generation:
- Base model: Partial forward pass only
- EAGLE heads: Small MLPs (~1-10MB each)
- Total: Usually 20-50% less than full autoregressive

## Tuning Parameters

### num_heads (1-10)
- More heads = better ensemble but higher compute
- Start with 3, increase if quality is insufficient
- Diminishing returns after 5-7 heads

### tree_depth (1-3)
- Deeper trees = more draft candidates
- Higher depth increases complexity exponentially
- Usually 2 is optimal

### base_model_layers (8-32)
- More layers = better features but slower extraction
- Use ~50% of total model layers as starting point
- Tune based on speed/quality tradeoff

### temperature (0.1-1.5)
- Lower = more conservative, higher acceptance rates
- Higher = more diverse, potentially lower acceptance
- Start with 0.8, adjust based on acceptance rates

## Monitoring and Debugging

### Key Metrics

```python
# Get model info
model_info = draft_model.GetModelInfo()
print(f"Model type: {model_info['model_type']}")
print(f"EAGLE heads: {model_info['num_eagle_heads']}")

# Monitor acceptance rates
# Higher acceptance rates indicate better EAGLE tuning
```

### Common Issues

1. **Low Acceptance Rates (<30%)**
   - Reduce temperature
   - Increase num_heads
   - Check base_model_layers configuration
   - **Try a pre-trained model** if using random initialization

2. **Slow Performance**
   - Reduce num_heads
   - Reduce base_model_layers
   - Check GPU memory usage

3. **Poor Quality Drafts**
   - Increase temperature slightly
   - Increase base_model_layers
   - **Switch to pre-trained model** for better quality
   - Consider switching back to LLM

4. **Pre-trained Model Loading Issues**
   - Check internet connection for HuggingFace downloads
   - Verify model compatibility with base model
   - Check logs for specific error messages
   - Ensure `huggingface_hub` is installed

5. **Model Compatibility Errors**
   - Verify EAGLE model matches base model architecture
   - Check hidden dimensions and vocabulary sizes
   - Use models trained on the same base architecture

## Migration Guide

### From LLM to EAGLE

1. **Backup current configuration**
2. **Install dependencies**:
   ```bash
   pip install huggingface_hub>=0.16.0
   ```
3. **Change config.toml**:
   ```toml
   [models.draft]
   type = "eagle"  # Was "llm"
   
   [models.draft.eagle]
   # For pre-trained models (recommended):
   eagle_model_path = "yuhuili/EAGLE-llama2-chat-7B"  # Match your base model
   
   # Or train from scratch:
   # eagle_model_path = ""
   num_heads = 3
   tree_depth = 2
   base_model_layers = 16
   temperature = 0.8
   ```
4. **Test with small workload**
5. **Monitor acceptance rates**
6. **Tune parameters if needed**

### Using Pre-trained Models

#### Step 1: Choose Compatible Model
```bash
# Check your base model
grep "edge_model" config.toml
# Output: edge_model = "meta-llama/Llama-2-7b-chat-hf"

# Select matching EAGLE model:
# meta-llama/Llama-2-7b-chat-hf → yuhuili/EAGLE-llama2-chat-7B
```

#### Step 2: Update Configuration
```toml
[models.draft.eagle]
eagle_model_path = "yuhuili/EAGLE-llama2-chat-7B"
```

#### Step 3: Run and Monitor
The system will automatically:
1. Download the EAGLE model from HuggingFace
2. Load pre-trained prediction heads
3. Use them for draft generation

Check logs for successful loading:
```
INFO - Loading pre-trained EAGLE model from: yuhuili/EAGLE-llama2-chat-7B
INFO - Downloaded EAGLE heads to: /cache/models/...
INFO - Successfully loaded pre-trained EAGLE heads
```

### Rollback Plan

If EAGLE doesn't work well:
```toml
[models.draft]
type = "llm"  # Switch back to LLM
```

## Future Enhancements

- [ ] Pre-trained EAGLE heads for popular models
- [ ] Dynamic head selection based on input
- [ ] Adaptive temperature tuning
- [ ] Integration with model quantization
- [ ] Multi-GPU EAGLE head distribution

## References

- EAGLE Paper: [Original EAGLE research]
- Speculative Decoding: [Background on speculative decoding]
- Model Architecture: See `edge/eagle_draft_model.py` for implementation details