# Pre-trained EAGLE Models Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install huggingface_hub>=0.16.0
# Or use the full requirements
pip install -r requirements-edge.txt
```

### 2. Configure for Pre-trained EAGLE Model
Edit `config.toml`:
```toml
[models.draft]
type = "eagle"

[models.draft.eagle]
# Use a pre-trained model (recommended)
eagle_model_path = "yuhuili/EAGLE-llama2-chat-7B"

# Other EAGLE settings
num_heads = 3
tree_depth = 2
base_model_layers = 16
temperature = 0.8
```

### 3. Ensure Base Model Compatibility
Make sure your base model matches the EAGLE model:

| Base Model | Compatible EAGLE Model |
|------------|------------------------|
| `meta-llama/Llama-2-7b-chat-hf` | `yuhuili/EAGLE-llama2-chat-7B` |
| `lmsys/vicuna-7b-v1.3` | `yuhuili/EAGLE-Vicuna-7B-v1.3` |
| `meta-llama/Llama-2-13b-chat-hf` | `yuhuili/EAGLE-llama2-chat-13B` |

### 4. Run the Edge Client
```bash
python run_edge.py
```

## What Happens Behind the Scenes

1. **Model Detection**: The system detects you've specified a pre-trained model path
2. **Download**: If it's a HuggingFace repo, it downloads the EAGLE heads automatically
3. **Loading**: Pre-trained weights are loaded into the EAGLE prediction heads
4. **Fallback**: If loading fails, it falls back to random initialization

## Verification

Check the logs for successful loading:
```
INFO - Loading pre-trained EAGLE model from: yuhuili/EAGLE-llama2-chat-7B
INFO - Downloaded EAGLE heads to: /home/user/.cache/huggingface/...
INFO - Successfully loaded pre-trained EAGLE heads
```

## Troubleshooting

### Model Not Found
```
ERROR - Could not download eagle_heads.pt: HTTP 404
```
**Solution**: Check the model repository exists and contains EAGLE weights

### Compatibility Issues
```
ERROR - Failed to load EAGLE head weights: size mismatch
```
**Solution**: Ensure base model and EAGLE model are compatible (same architecture)

### Network Issues
```
ERROR - huggingface_hub not available
```
**Solution**: Install huggingface_hub: `pip install huggingface_hub>=0.16.0`

### Fallback to Random Initialization
```
INFO - Falling back to initialization from scratch
```
This is normal when pre-trained models can't be loaded. The system continues with random weights.

## Local Models

You can also use local EAGLE models:
```toml
[models.draft.eagle]
eagle_model_path = "./my-local-eagle-model"
```

The system will look for these files:
- `eagle_heads.pt`
- `pytorch_model.bin` 
- `model.pt`

## Performance Expectations

Pre-trained EAGLE models typically provide:
- **Better acceptance rates** (40-70% vs 20-40% random)
- **Faster convergence** (no training needed)
- **More stable performance** across different inputs

## Creating Your Own EAGLE Models

To train custom EAGLE models for your base model:
1. Collect target model hidden states on your dataset
2. Train EAGLE heads to predict next tokens from these states
3. Save as `eagle_heads.pt` with the expected format
4. Use local path in configuration

Format for saved EAGLE heads:
```python
{
    "eagle_heads": [
        head_0_state_dict,
        head_1_state_dict,
        head_2_state_dict
    ]
}
```