# vLLM Setup for CausalArmor Proxy

CausalArmor uses a **proxy model** (Gemma 3 12B) served by vLLM to compute Leave-One-Out (LOO) log-probability scores. This guide explains how to set it up on a DGX or any machine with an NVIDIA GPU.

## Prerequisites

- NVIDIA GPU with at least 24 GB VRAM (e.g., A100, H100, RTX 4090)
- Docker with NVIDIA runtime (`nvidia-container-toolkit`)
- A HuggingFace account with access to `google/gemma-3-12b-it`

## 1. Accept the Gemma License

Go to [huggingface.co/google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) and accept the license agreement. Then create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
export HF_TOKEN=hf_...
```

## 2. Start vLLM

```bash
docker run --runtime nvidia --gpus '"device=0"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model google/gemma-3-12b-it \
  --dtype auto --max-model-len 8192
```

The first launch downloads the model weights (~24 GB). Subsequent starts use the cached weights.

### Multi-GPU

To use multiple GPUs (e.g., for larger models or higher throughput):

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model google/gemma-3-12b-it \
  --dtype auto --max-model-len 8192 \
  --tensor-parallel-size 2
```

## 3. Health Check

```bash
# Check the server is running
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models

# Test a completion (the endpoint CausalArmor uses)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-12b-it",
    "prompt": "The capital of France is",
    "max_tokens": 1,
    "logprobs": 1
  }'
```

## 4. Configure the Demo

Set the environment variables in your `.env` file or shell:

```bash
export VLLM_BASE_URL=http://localhost:8000
export VLLM_MODEL=google/gemma-3-12b-it
```

If vLLM is running on a different machine (e.g., a DGX in your cluster):

```bash
export VLLM_BASE_URL=http://dgx-hostname:8000
```

## Troubleshooting

### Out of Memory

Reduce `--max-model-len` or use `--dtype float16`:

```bash
--max-model-len 4096 --dtype float16
```

### Model Download Fails

Verify your HuggingFace token has read access and you've accepted the Gemma license:

```bash
huggingface-cli whoami  # Should show your username
huggingface-cli download google/gemma-3-12b-it --dry-run
```

### Connection Refused

Ensure the container port is mapped correctly and no firewall is blocking port 8000. Check Docker logs:

```bash
docker logs <container-id>
```

### Slow First Request

The first request after startup may take a few seconds while vLLM compiles CUDA kernels. Subsequent requests are fast.
