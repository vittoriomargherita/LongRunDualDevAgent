# Planner Model Recommendations for RTX 3080 10GB VRAM

## Overview
The RTX 3080 with 10GB VRAM can efficiently run quantized language models for the Planner role. This document provides recommendations based on model size, quantization level, and expected performance.

## Recommended Models (Ranked by Quality/Performance)

### 1. **Qwen2.5-14B-Instruct** (Q4_K_M) - ⭐ BEST BALANCE
- **VRAM Usage**: ~8-9GB
- **Quality**: Excellent reasoning, strong code understanding
- **Speed**: ~15-25 tokens/s
- **Download**: `ollama pull qwen2.5:14b-instruct-q4_K_M`
- **Why**: Best balance between model size and reasoning capability. Qwen2.5-14B has shown excellent performance on coding tasks and architectural planning.

### 2. **Llama 3.1 8B Instruct** (Q5_K_M) - ⭐ RECOMMENDED
- **VRAM Usage**: ~5-6GB
- **Quality**: Very good reasoning, fast inference
- **Speed**: ~30-40 tokens/s
- **Download**: `ollama pull llama3.1:8b-instruct-q5_K_M`
- **Why**: Excellent performance-to-size ratio. Llama 3.1 8B is specifically optimized for instruction following and reasoning tasks.

### 3. **Qwen2.5-7B-Instruct** (Q5_K_M) - ⭐ FASTEST
- **VRAM Usage**: ~4-5GB
- **Quality**: Good reasoning, very fast
- **Speed**: ~40-50 tokens/s
- **Download**: `ollama pull qwen2.5:7b-instruct-q5_K_M`
- **Why**: Fastest option while maintaining good quality. Good for rapid iteration.

### 4. **Mistral 7B Instruct** (Q6_K) - ⭐ HIGH QUALITY
- **VRAM Usage**: ~5-6GB
- **Quality**: Excellent reasoning, high precision
- **Speed**: ~25-35 tokens/s
- **Download**: `ollama pull mistral:7b-instruct-q6_K`
- **Why**: Higher quantization (Q6_K) provides better quality, Mistral 7B is known for strong reasoning.

### 5. **DeepSeek Coder 6.7B** (Q5_K_M) - ⭐ CODE-SPECIFIC
- **VRAM Usage**: ~4-5GB
- **Quality**: Excellent for code understanding
- **Speed**: ~35-45 tokens/s
- **Download**: `ollama pull deepseek-coder:6.7b-q5_K_M`
- **Why**: Specifically trained on code, excellent for understanding code structure and dependencies.

## Quantization Levels Explained

- **Q4_K_M**: Good balance, ~4.5 bits per weight, recommended for 14B models
- **Q5_K_M**: Better quality, ~5 bits per weight, recommended for 7-8B models
- **Q6_K**: Highest quality, ~6 bits per weight, use when VRAM allows

## Configuration

Update `config.json`:

```json
{
  "planner": {
    "model": "qwen2.5:14b-instruct-q4_K_M",
    "base_url": "http://localhost:11434",
    "temperature": 0.7
  },
  "executor": {
    "model": "qwen2.5:14b-instruct-q4_K_M",
    "base_url": "http://localhost:11434",
    "temperature": 0.3
  }
}
```

## Performance Comparison

| Model | Size | VRAM | Tokens/s | Quality | Best For |
|-------|------|------|----------|---------|----------|
| Qwen2.5-14B Q4 | 14B | 8-9GB | 15-25 | ⭐⭐⭐⭐⭐ | Complex planning |
| Llama 3.1 8B Q5 | 8B | 5-6GB | 30-40 | ⭐⭐⭐⭐ | Balanced |
| Qwen2.5-7B Q5 | 7B | 4-5GB | 40-50 | ⭐⭐⭐⭐ | Fast iteration |
| Mistral 7B Q6 | 7B | 5-6GB | 25-35 | ⭐⭐⭐⭐⭐ | High precision |
| DeepSeek 6.7B Q5 | 6.7B | 4-5GB | 35-45 | ⭐⭐⭐⭐ | Code-specific |

## Installation

1. Install Ollama: https://ollama.ai
2. Pull the model: `ollama pull <model-name>`
3. Update `config.json` with the model name
4. Test: `ollama run <model-name> "Plan a feature for a PHP booking system"`

## Recommendations by Use Case

### For Maximum Quality (Complex Projects)
- **Primary**: Qwen2.5-14B-Instruct (Q4_K_M)
- **Fallback**: Mistral 7B Instruct (Q6_K)

### For Speed (Rapid Development)
- **Primary**: Qwen2.5-7B-Instruct (Q5_K_M)
- **Fallback**: DeepSeek Coder 6.7B (Q5_K_M)

### For Balanced Performance
- **Primary**: Llama 3.1 8B Instruct (Q5_K_M)
- **Fallback**: Qwen2.5-7B-Instruct (Q5_K_M)

## Notes

- All models listed fit comfortably in 10GB VRAM
- You can run both Planner and Executor on the same GPU if using smaller models (7B)
- For 14B models, consider using CPU for Executor or a separate GPU
- Monitor VRAM usage: `nvidia-smi` or `watch -n 1 nvidia-smi`

## Testing Your Setup

Run this test to verify your model works:

```bash
python3 -c "
from code_agent import CodeAgent
agent = CodeAgent()
# Test planner
response = agent.planner_client.chat_completion([
    {'role': 'user', 'content': 'Plan a feature: User authentication'}
])
print('Planner test:', 'OK' if response else 'FAILED')
"
```

