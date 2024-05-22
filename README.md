<!-- markdownlint-disable MD036 -->

# VIT.TRITON

This repository implements VIT from scratch using kernels written in Triton

## Learning GPU programming

LLMs are biiig, and can use a lot of compute. Hence, there is some alpha in running them as fast as possible, because that is what the real bottle neck is. If you can reduce latency or increase throughput, that opens up a lot of doors for LLM applications.

To learn how to run these big models as fast as possible, we have to understand how they are executed on the hardware. This blog will help you learn just that. This blog will walk you through a basic layout of GPUs hardware, a mental model of how the programming model of CUDA looks like and help you learn how to write your own optimized code to run on GPUs.

Apart from that, there is just a deep satisfaction in knowing how things work on the hardware.

### Hardware

To learn how to run programs on GPUs efficiently, we have to first understand what is so special about GPUs that make them extremely efficient for LLMs. This boils down to the having a understanding of the GPU hardware. GPUs are optimized for throughput whereas CPUs are optimized for latency.

#### Why is GPU faster for AI?

But why are GPUs faster than CPUs? The answer that may come to your head is because GPU can process data parallely and it operates in [SIMD](add url) fashion. CPUs are mostly designed to work with sequential tasks. 
But even then, what makes GPU process data parallely? Majorly 2 things:

1. CPUs have a lot of space dedicated to cache and registers on the chip. GPUs make a design choice that they reduce the size of the cache and increase the number of processing cores. This way they can fit more cores in the same area. Cores are essentially the processing units that processes every unit of data.
2. CPUs have a lot of functionalities in their cores. These functionalities help them operate in a variety of different tasks and hence CPUs are very robust. GPUs reduce these special functionalities which helps it to make cores smaller. If the cores are smaller, you can fit more cores in the same area.

(Insert an image for comparison)

Figure 1: The figure above helps explain both the points. You can see how the cache and control logic sizes are extremely reduced. The core size is also reduced. Both of these trade offs helps us to fit more cores in the GPU.

You can probably see from the above two points that its all a game of cores. You may think that the more number of cores you can fit in the chip, the better the performance. It's only partially true though!

#### GPU hardware layout

To understand the layout of the GPU hardware, you really need to understand how cores are organized on the GPU (and why are they organized that way). Let's go through some basic concepts:

**CUDA Cores**

Here is where the magic actually happens. These are the processing units and come in different flavours, eg: Tensor Cores, Single precision cores, Double precision cores, and all of these cores handle different kinds of operations. The GPU decides where to send the operation based on the data type and instruction. The amount of operations these bad boys can do per second is what gives rise to FLOP numbers. Each of these different flavours have different performance numbers in terms of FLOPs because all of them do different kinds of operations.

For example, H100 has 16986 FP32 CUDA cores that can each do 2 floating point operations per cycle. The clock speed of the GPU is 1593 MHz. In total, theoretically if all the cores are processing data at all times, it can achieve

$$ 1.593 * 10^9 * 2 * 16.986 * 10^3 ~ 54.1 * 10^12 FLOPS$$ or 54 teraFLOPs

This is close to what is shown on the official specs of [H100](https://www.nvidia.com/en-us/data-center/h100/). (There is a difference in the teraFLOPs number on the website which I am not able to figure out why? If you know, please drop me a mail!)

**Streaming Multiprocessors**

All the cores in a GPU are organized into groups. Each of this group is called a streaming multiporcess (SM). Every SM has some memory associated with it. This memory can be shared amongst all the cores inside an SM but not by any other core outside this SM. This memory is called as shared memory and this is extremely fast. What do I mean by fast? The data transfer speed. But this is also small in terms of capacity. So it's essential to use this memory judiciously.

Why are GPUs divided like this? It's to enable smaller groups of cores to share memory amongst them and work together. With every next generation, usually the SMs and cores per SMs are increased.

Let's take some real numbers to understand the capacity. An H100 SXM GPU contains:

1. 132 streaming multiprocessors (SM)
2. Each SM has 128 FP32 CUDA cores (and a total of 16896 CUDA cores)
3. Each SM having 227 KB of shared memory
4. And this memory has a bandwidth of 33 TB/s

SMs are also grouped into TPCs (Texture/Processor Cluster). For reference, the above hardware has 2 SMs per single TPC. But for our purposes, it doesn't have a lot of meaning.

**Memory**

There are three kinds of memory on the GPU

1. HBM/Global memory - You can think of it as the equivalent of PC's RAM. This is the slowest and largest memory available on the GPU.
   1. For refernce, H100 SXM has 80GB of HBM with 3 TB/s of bandwidth (i.e. it can transfer 3 TB per second to and from HBM)
   2. This is where the model is loaded when we do `model.to(device='cuda:0')`
2. L2 Cache - Faster than HBM but limited in size. This is shared among all the SMs.
   1. For reference, H100 SXM has 50 MB (lol, in comparison to HBM) of L2 cache with 12 TB/s of bandwidth.
3. Shared memory - Fastest and smalles memory available on the GPU. Every SM has it's own shared memory and all the cores executing instructions in an SM have access to it.

### Back to LLMs

Okay, so now that you have a good mental model of how the GPU hardware looks like, let me cite a working examples to drive home a point - For LLMs, you should probably not worry about TFLOPs.

Why?

Take an example that H100 SXM GPU can do 67 teraFlops (FP32) of computation. The memory bandwidth of the HBM is 3 TB/s. That means GPU can transfer about 3 TB of data to the compute layer per second. Considering FP32 (4 bytes), we can transfer about 750 billion numbers to the compute layer in one second. But the compute layer can perform 67 trillion operations per second. Just to break even with the computation speed, we would either:

1. Need to transfer ~90x the data (67 trillion/750 billion) from the memory to the computer layer per second
2. Or perform, ~90 operations on every datapoint each second

So, the main bottleneck in making LLMs faster is memory bandwidth. You can read more in depth in an article by Horace He [here](https://horace.io/brrr_intro.html).

Another practical example is stated in the article: [How is Llama.cpp possible?](https://finbarr.ca/how-is-llama-cpp-possible/)

### Hardware to software mapping

Okay, let's move on to software. How can you translate your learning of GPU hardware to software?

Let's start with CUDA. CUDA is a programming extension to C/C++ that helps us write programs that run on GPUs. It helps us 

### A simple example in CUDA

### A simple example in Triton

### How you can rewrite the complete architecture using optimized kernel
