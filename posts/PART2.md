<!-- markdownlint-disable MD036 MD029 -->
# Understanding the GPU programming model

Given that you have read Part 1 of the series, you should have a basic understanding of the GPU hardware. Let's now understand the software that is used to run programs on the GPUs.

## Hardware to software mapping and programming model of the GPU

2 things to keep in mind before we start:

1. The physical concepts of hardware do not necessarily translate one-to-one to logical concepts in software.
2. In GPU programming, we write a function that needs to be executed on the GPU. This is called a kernel. We can write multiple kernels in a single program and "launch" them from the CPU.

Okay, with that out of the mind, let's understand a few important concepts of GPU programming -

### Threads

Each kernel is executed by a thread in the GPU. And every thread executes the same kernel (assuming you have a single kernel in the program). This makes it necessary for us to write our kernel such that a single function can operate on all the data points. When we launch a kernel, what we are actually starting are GPU threads that will execute instructions written inside the kernel. We can start a lot of threads at once and these are the true powerhouse of the GPU.

All threads have some small memory associated with it which is called local memory. Apart from that, threads can also access shared memory, L2 cache, and global memory.

Physically, threads are assigned to cores. Cores execute software threads.

### Blocks

Threads are logically organized into blocks. Every block has a pre-defined number of threads assigned to it. *Just for logical purposes*, we can arrange threads inside a block in either a 1D, 2D, or 3D array layout. You can think of blocks as an array of threads. It's important to understand that this 1D, 2D, or 3D arrangement is purely logical and for the developer's convenience only. This arrangement is provided so it's easier to visualize our input and output data. If we imagine that we want to operate on a 100x100 matrix, then we can just launch a kernel with a block size of 100 by 100 threads. That will start a total of $10^4$ (100x100) threads which we can use to map to the matrix. We can write our kernel such that every single thread operates on every single element of the matrix.

In the physical world, every block is assigned an SM. Throughout its execution, the block will only be executed on the same SM. Since every block is assigned an SM, it also has access to the SM's shared memory (which we learned in the first part). All the threads that are part of a single block can access and share this memory.

### Grids

Similar to how threads are organized in blocks, blocks are themselves organized into a grid. That helps us to launch multiple blocks at one time. As we discussed earlier, a single GPU has multiple SMs, we can launch multiple blocks at once so that all of our SMs and cores are utilized. Let's assume that our program executes 25 blocks and our GPU has 10 SMs. Then the program will execute 10 blocks in the first wave, 10 blocks in the second wave, and 5 blocks in the third wave. The first two waves will have 100% optimization but the last wave will have 50% utilization.

Blocks inside a grid can be organized in the same way that threads are organized inside a block. A grid can have a 1D, 2D, or 3D array layout of the blocks. The arrangement of blocks and threads is just logical. A single program only executes a single grid at a time.

During execution, we start a total of `blocks per thread (b) * number of blocks (num)` physical threads. Each physical thread is numbered from `0` to `(b*num)-1`. So, how do you map your 2D or 3D structure of logical thread blocks to the physical thread? By unrolling.

A 2D array layout can be to 1D. If it's row-major ordering, then a 2D matrix after unrolling will look like this:

![matrix unrolling](image-1.png)

Figure 1: Element `A[2][3]` in the 2D matrix will be `A[5]` in the flattened 1D array. This is how you can think of mapping 2D blocks of thread to the 1D thread array.

## A simple example in CUDA

Now that you have hopefully understood what threads, blocks, and grids are, let's start with CUDA. CUDA is a programming extension of C/C++ that helps us write heterogeneous programs. These programs allow us to define and launch kernels from the CPU. CUDA is very powerful and offers a lot of ways to optimize your kernel. It's just a bit ... too expressive.

Let's slowly work through an example to understand how it works. Let's implement a very naive implementation of matrix multiplication. We will be using some CUDA function calls, they should be self-explanatory, but in case they are not, just google the syntax. This is a relatively simple kernel, so should be easy to follow along.

1. Allocate the memory for the data (both input and output) on the CPU memory (also called as host). We will allocate memory for our input (X), weight matrix (W), and output (O). Assuming B as the batch size, N as the number of rows or sequence length in transformers, D_in as the number of columns or embedding dimension, and D_out as the hidden dimension.

```C
float *X = (float*)malloc(B*N*D_in*sizeof(float));      // Input data
float *W = (float*)malloc(D_in*D_out*sizeof(float));    // Weights
float *O = (float*)malloc(B*N*D_out*sizeof(float));     // Output data
```

2. Allocate the memory for the data on the GPU (also called as device)

```C
float *d_X, *d_W, *d_O;

cudaMalloc((void**) &d_X, B*N*D_in*sizeof(float));      //cudaMalloc is a CUDA function and allocates memory on the GPU memory
cudaMalloc((void**) &d_W, D_in*D_out*sizeof(float));
cudaMalloc((void**) &d_O, B*N*D_out*sizeof(float));
```

3. Copy the relevant data from the CPU memory to the GPU memory. Let's assume that we have loaded `X` and `W` with relevant data. Now we transfer the data to the GPU.

```C
cudaMemcpy(d_X, X, B*N*D_in*sizeof(float), cudaMemcpyHostToDevice);     // cudaMemcpy is again a CUDA function
cudaMemcpy(d_W, W, D_in*D_out*sizeof(float), cudaMemcpyHostToDevice);

```

4. Launch the kernel. Assuming that our kernel is called `matMul`, `grid` defines how the blocks are arranged and `blocks` define how threads are arranged in each block. For this example, the `grids` will be a 1D array equal to the batch size. `blocks` will have the same layout as the output dimension of our output matrix (`N*D_out`). This means that every block will process a single batch of data and every thread will process a single cell of our output matrix.

```C
matMul<<<grid, blocks>>>(
    d_X,
    d_W,
    d_O,
    B,
    N,
    D_in,
    D_out
);
```

In total we have launched: `B*N*D_out` threads.

5. Copy the relevant data (usually only the output) from the GPU memory to the CPU memory. Once the kernel execution is completed, we need to copy the output from the GPU memory back to our CPU memory so that we can use it for any downstream processing.

```C
cudaMemcpy(O, d_O, B*N*D_out*sizeof(float), cudaMemcpyDeviceToHost);
```

These 5 steps are followed in almost all GPU programs. Let's now dive deep into the actual kernel:

```C
__global__ void matMul(
    float* X,
    float* W,
    float* OO,
    int B,
    int N,
    int D_in,
    int D_out
) {
    /*
    This kernel takes a batch of data: (B x N x Din)
    and a weight matrix: (Din X Dout)
    and produces: (B x N x Dout)
    */

    int batch = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    int out_offset = N*D_out*batch + row*D_out + col;

    if ((batch < B) && (col < D_out) && (row < N)) {
        float sum = 0.0f;
        for (int i = 0; i < D_in; i++) {
            sum += X[N * D_in * batch + row * D_in + i] * W[i * D_out + col];
        }
        OO[out_offset] = sum;
    }
}
```

Remember that physically there is no 2D or 3D arrangement of threads. That construct is just provided by CUDA to help us map the problems appropriately. Physically it's just a single 1D array of threads. Since we have started `B*N*D_out` threads, it maps exactly with the 1D layout of our output matrix.

To figure out which data a particular thread should process, the kernel just needs to figure out which thread is it executing. Depending on the batch, row, and column, each thread will load different parts of the input and weight matrix. These are called offsets and we have calculated three offsets in our code:

1. `batch_offset`: Figure out which batch this kernel is processing
   1. `blockDim.x` gives us the size of the block in the x direction (number of columns in the grid)
   2. `blockIdx.x` gives us the index of the batch that the particular thread is executing
2. `elem_offset`: Figure out within a batch, which cell is the kernel processing
3. `out_offset`: Just a summation of `batch_offset` and `elem_offset`

Hopefully, this diagram will make it more clear.

![threads-blocks](image-2.png)

Figure 2: Grids/Blocks/Threads layout
Source: Borrowed from [this](https://siboehm.com/articles/22/CUDA-MMM) excellent blog.

After calculating these offsets, we are loading the corresponding row from `X` and the corresponding column from `W` and doing a single vector multiplication in a for loop. The important part here is to understand the offset calculation and how we can make use of the block and grid layout to make our lives easier!

The complete code is present [here](https://github.com/cmeraki/vit.triton/blob/main/examples/matmul_batch.cu). You would need to install `nvcc` (the compiler for CUDA programs), have an NVIDIA GPU to run the program, and have the CUDA drivers, and CUDA toolkit installed.

## A simple example in Triton

CUDA is amazing and lets us do a lot of optimizations. But it is quite verbose. Plus, if you are coming from the ML/DS land, you are probably more familiar with Python. Open AI released a package called [Triton](https://triton-lang.org/) that provides a Python environment to write kernels and compile them for any GPU. By using Triton, you can write very performant kernels in Python directly.

But instead of working with individual threads, Triton works with blocks. Instead of each kernel being assigned a thread, in Triton each kernel is assigned a block. Triton abstracts out the thread computation completely so that you can focus on slightly higher-order computation.

In our example of matrix multiplication, instead of computing a single element of the output in our kernel, Triton can help us compute values for small "blocks" of the output matrix.

![alt text](<image-4.png>)

Figure 3: (Left) CUDA execution model vs (Right) Triton execution model
Source: [Triton documentation](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)

Let's reimplement the matrix multiplication example using Triton.

## How you can rewrite the complete architecture using optimized kernel

Congrats on making this far away. Now that you understand the basics of GPU hardware and its programming model, you can go ahead and implement any network from scratch, this time not relying on PyTroch for operations but writing your kernels in CUDA or Triton.

What would you require for that? If you want to implement a transformer encoder network, you would need to implement all the basic layers and operations in Triton or Kernel.

1. Matrix multiplication
2. Layernorm
3. Softmax
4. Addition
5. Concatenation

You can then wrap these kernels in the PyTorch module and load weights from HF to compare your implementation with other PyTorch/TF native implementations. If this sounds interesting, this is exactly what we did too. We implemented most of the operations used in Vision Transformer (including patching and addition operations) in Triton and used HF weights to run a forward pass. You can look at the code [here](https://github.com/cmeraki/vit.triton) and maybe implement your favorite model too using custom kernels!
