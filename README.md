<!-- markdownlint-disable MD036 MD029 -->

# VIT.TRITON

Complete implementation of ViT (Vision Transformer) model using Triton kernels. The weights are imported from HF (HuggingFace). We have also compared the implementation against the official HF implementation. The results are in the `benchmarks` folder. You can use this ViT implementation as an education resource or in your pipeline. This implementation is completely valid and functional. It only supports forward passes for now.

There are some accompanying posts under the [posts](./posts/) folder to help you get started with GPU programming.

This repo can help you:

1. Learn how to write Triton kernels
2. How to implement an architecture using PyTorch but by calling custom triton kernels
3. Load weights to your own implementation from a different repository like HuggingFace

## Repo structure

```plaintext
- vit/
    - kernels/              # All triton kernels reside here
    - load_weights.py       # Functions for loading weights from HF
    - utils.py              # Utils
    - vit.py                # Architecture written in torch, but calling triton kernels
- benchmarks/               # Benchmark results
- posts/                    # Blog posts related to GPU programming 
- examples/                 # Small examples used in posts
```

## Setup

If you'd like to test this implementation on your machine, all you need to do is,

```python
git clone https://github.com/cmeraki/vit.triton.git
cd vit.triton
python -m venv .venv
source ~/.venv/bin/activate
pip install -r requirements.txt                         # Requriements are suited for NVIDIA GPU and linux setup
python -m vit.vit                                       # This will run the benchmarking on both HF implementation of ViT and the custom implementation
```

## Contact

In case you have any questions or suggestions, feel free to raise an issue!
