# Gemma3 Emoji Fine-Tuning

A demonstration project showing how to fine-tune Google's Gemma3 language model to convert text emotions into emoji sequences using LoRA.

## Purpose

This project is a technical demonstration of fine-tuning techniques, specifically LoRA (Low-Rank Adaptation) on the Gemma3-270M model. Fine-tuning was performed in Google Colab using a Tesla T4 GPU.

## Features

- LoRA fine-tuning with Unsloth optimizations
- 4-bit quantization for memory efficiency
- GGUF export and Ollama integration
- 186 emotion-to-emoji training examples

## 🏗️ Architecture

```
Input Text (Emotion) → Gemma3-270M + LoRA → Emoji Sequence
```

- **Base Model**: `unsloth/gemma-3-270m-it` (270M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit loading for memory efficiency
- **LoRA Rank**: 64 with alpha 128
- **Max Sequence Length**: 2048 tokens
- **Training Dataset**: 186 emotion-to-emoji examples

## 📋 Prerequisites

### Hardware Requirements
- **CUDA-compatible GPU** (Tesla T4 tested and recommended)
- **Minimum 16GB GPU memory** for optimal performance
- **CPU training possible** but significantly slower
- **Storage**: 2GB free space

### Software Requirements
- Python 3.8+
- CUDA Toolkit 11.0+ (for GPU acceleration)
- Git

## Setup

### Dependencies
```bash
pip install unsloth trl peft accelerate bitsandbytes torch datasets
```

### Training (Google Colab)
```bash
jupyter notebook Fine_Tuning.ipynb
```
Execute cells sequentially to fine-tune the model.

### Testing
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "I'm so happy right now!"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=64, temperature=0.7)
response = tokenizer.batch_decode(outputs)[0]
print(response)
```

### Export (Optional)
```python
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
```

### Ollama Deployment (Optional)
```bash
ollama create emoji-model -f Modelfile
ollama run emoji-model
```

## Files

```
gemma3-finetune/
├── README.md                    # This file
├── CLAUDE.md                    # Development guide
├── Fine_Tuning.ipynb           # Training notebook (used in Google Colab)
├── fine_tuning.py              # Python script version
├── json_extraction_dataset_186.json  # Training dataset (186 examples)
├── Modelfile                   # Ollama configuration
└── outputs/                    # Model checkpoints
```

## Training Details

### Model Configuration
- **Base**: `unsloth/gemma-3-270m-it`
- **LoRA Rank**: 64, Alpha: 128
- **Sequence Length**: 2048 tokens
- **Training Time**: ~15 minutes (Tesla T4)

### Hyperparameters
- Batch size: 2 (gradient accumulation: 4)
- Learning rate: 2e-4
- Epochs: 3
- Optimizer: AdamW 8-bit

### Dataset Format
```json
{
  "messages": [
    {"role": "user", "content": "I'm so happy right now!"},
    {"role": "assistant", "content": "😊😃🎉"}
  ]
}
```

## Example Output

```
Input: "I'm so happy right now!"
Output: 😊😃🎉
```

## Resources

- [Gemma3 Documentation](https://huggingface.co/google/gemma-3-270m-it)
- [Unsloth](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
