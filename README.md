# Gemma-3-1B-IT QNM 4-bit (Edge LLM Inference Runtime)

## Research Context
This repository contains the official runtime implementation for the research study: **"Precision Transition Dynamics: Standardizing BF16 to U8 Containers in Sub-Block Quantization for Edge LLM Inference"** by the **POKDARTIK AI Engineering Team**. 

The framework investigates the dynamics of converting high-precision `bfloat16` weights into standardized unsigned 8-bit (`U8`) sub-block quantization containers (`signed-grid` / Q4M), enabling highly efficient local model execution and optimized memory-bandwidth utilization on edge hardware.

The core model and quantized tensor weights are hosted on Hugging Face: [WiTi4Bit/gemma-3-1b-it-qnm-4bit](https://huggingface.co/WiTi4Bit/gemma-3-1b-it-qnm-4bit).

---

## Key Features

* **Edge CPU Int4 Optimization**: Utilizes PyTorch native CPU int4 packaging and matrix multiplication kernels (`int4pack` and `int4mm`) for accelerated low-bit inference.
* **Sub-Block Quantization Container Handling**: Seamlessly reconstructs and maps custom `U8` sub-block quantized weights (`signed-grid`) alongside native `bfloat16` fallback components (biases, normalization layers, and embeddings).
* **Alternative Attention Dynamics (DR9 & NoExp)**: Implements experimental Digital Root 9 (`dr9`) attention weighting combined with direct repetition penalty handling to stabilize generation coherence on edge devices.
* **Interactive Multi-Turn Chat Runtime**: Fully compatible with Gemma 3 chat templates for conversational inference workflows.

---

## System Requirements & Dependencies

Ensure you have Python 3.10+ and the required packages installed:

```bash
pip install torch transformers safetensors
```

---

## Model Download

Clone the model repository using Git LFS from Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/WiTi4Bit/gemma-3-1b-it-qnm-4bit
```

---

## Usage Guide

The `chat_runtime_public.py` script supports both interactive sessions and single-shot generation via command-line arguments.

### 1. Interactive Chat Mode (Default)
```bash
python3 chat_runtime_public.py ./gemma-3-1b-it-qnm-4bit
```

### 2. Single-Shot Prompt Mode
```bash
python3 chat_runtime_public.py ./gemma-3-1b-it-qnm-4bit --prompt "Jelaskan konsep dasar kuantisasi sub-block secara singkat."
```

### 3. Command-Line Arguments
* `--attention`: Select attention implementation (`dr9` [default], `noexp`, or standard `softmax`).
* `--threads`: Specify the number of CPU execution threads (default: `8`).
* `--temperature`: Adjust response creativity (default: `0.7`).
* `--system`: Set a custom system prompt (default in Indonesian language).

Example with custom thread allocation:
```bash
python3 chat_runtime_public.py ./gemma-3-1b-it-qnm-4bit --threads 4 --temperature 0.5
```

Inside the interactive chat interface:
* Type `/reset` or `/new` to clear conversation history.
* Type `keluar`, `exit`, or `/quit` to close the runtime.

---

## License & Terms of Use

Copyright (c) 2026 POKDARTIK AI Engineering Team. All rights reserved.

This core software is licensed strictly for **NON-COMMERCIAL**, educational, research, and scientific peer-review purposes. Users are permitted to download, run, and share this code provided that proper attribution and credit are given to POKDARTIK. Commercial use is strictly prohibited.

> *Third-Party Dependency Note:* The Gemma 3 model (`Gemma3-1B-IT`) is governed by Google's Gemma Terms of Use.
