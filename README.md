# VIT.TRITON

This repository implements VIT from scratch using kernels written in Triton

## Learning GPU programming through an example

LLMs are biiig, and use a lot of compute. Hence, there is some alpha in running them as fast as possible, because that is what the real bottle neck is. If you can reduce latency or increase throughput, that opens up a lot of doors for LLM applications.

To learn how to run these big models as fast as possible, we have to understand how they are executed on the hardware. So this blog will help you learn just that. This blog will walk you through a basic layout of GPUs, a mental model of how the code is executed on them, and help you learn how to write your own optimized code to run on GPUs.
Apart from that, there is just a deep satisfaction in knowing how things work on the hardware. 

### Hardware

To learn how to run programs on GPUs efficiently, we have to first understand what is so special about GPUs that make them extremely efficient for LLMs. This boils down to the having a understanding of the GPU hardware.


#### Why is GPU faster for AI?


#### GPU hardware layout

Broadly, the GPU hardware is divided into 3 peices:

1. Streaming Multiprocessors
2. Blocks
3. CUDA cores


**Streaming Multiprocessors**

The whole GPU is divided into smaller boxes. Each of this box has a group of processing units (which are called CUDA cores, and we'll come up to them later). Every box has small amount of memory associated with it which can not be accesses by other box. This memory can be shared amongst all the CUDA cores inside a box but not by any other CUDA core outside this box.

This is called the shared memory and this is extremely fast. What do I mean by fast? The memory transfer spped. It's ~19TB/s on H100.

But this is also small in terms of capacity. So it's essential to use this memory judiciously.

Why are GPUs divided like this? It's to enable smaller groups of processing units to share memory amongst them. With every next generation, usually the SMs are increased and CUDA cores per SMs are also increased.

**CUDA Cores**

Here is where the magic actually happens. These are the units inside SMs (which are themselves inside a GPU) that actually does computation. And the amount of operations these bad boys can do per second is what gives rise to FLOP numbers.
For example, H100 has 500 CUDA cores that can each do 20 computations per second. In total, theoretically if all the cores are processing something at all times, we can achieve 1000 TFLOPs.



### Hardware to software mapping

### A simple example in CUDA

### A simple example in Triton

### How you can rewrite the complete architecture using optimized kernel
