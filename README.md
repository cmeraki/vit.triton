# VIT.TRITON

This repository implements VIT from scratch using kernels written in Triton

## Learning GPU programming through an example

LLMs are biiig, and use a lot of compute. Hence, there is some alpha in learning how to run them as fast as possible. If you can reduce latency or increase throughput, that opens up a lot of doors for LLM applications.

To learn how to run these big models as fast as possible, we have to understand how they are executed on the hardware. So this blog will help you learn just that. This blog will walk you through a basic layout of GPUs, a mental model of how the code is executed on them, and help you learn how to write your own optimized code to run on GPUs.
Apart from that, there is just a deep satisfaction in knowing how things work on the hardware. 

### Hardware

To learn how to run programs on GPUs efficiently, we have to first understand what is so special about GPUs that make them extremely efficient for LLMs. This boils down to the having a understanding of the GPU hardware.


#### Why is GPU faster for AI?


#### GPU hardware layout

Broadly, the GPU hardware is divided into 3 peices:

1. 



### Hardware to software mapping

### A simple example in CUDA

### A simple example in Triton

### How you can rewrite the complete architecture using optimized kernel
