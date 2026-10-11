# Numi-OT.01

**Original sparse Mixture-of-Experts language model.**

Built from scratch. No weights or architecture copied from any existing model.

## Design inspirations

| Component        | Inspiration          |
|------------------|----------------------|
| Attention + RoPE | Llama 3              |
| RMSNorm          | Llama / Mistral      |
| SwiGLU experts   | Llama / Mixtral      |
| Top-k routing    | Mixtral              |
| Load balancing   | Switch Transformer   |

## Architecture (default config)

- **24 layers**
- Hidden size: 2048
- Attention heads: 16 (GQA with 4 KV heads)
- Experts: 8 (top-2 routing)
- Context: 8192 tokens
- Vocab: 128256

## Status

Development — architecture complete, training pipeline next.
