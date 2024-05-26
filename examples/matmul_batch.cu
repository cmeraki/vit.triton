// nvcc -o matmul_batch.o matmul_batch.cu && ./matmul_batch.o

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

void display_3dmat(char* name, float* M, int B, int N, int D) {
    /*
    Helper function to print a 3D matrix with size BxNxD
    */
    printf("\nMatrix: %s\n", name);

    for (int i=0; i<B; i++) {
        for (int j=0; j<N; j++) {
            for (int k=0; k<D; k++) {
                // skip batches, skip rows, skip columns
                int offset = i*N*D + j*D + k;
                printf("%0.1f ", M[offset]);
            }
            printf("\n");
        }
        printf("--\n");
    }
}

void display_2dmat(char* name, float* M, int N, int D) {
    /*
    Helper function to print a 2D matrix with size NxD
    */
    printf("\nMatrix: %s\n", name);

    for (int i=0; i<N; i++) {
        for (int j=0; j<D; j++) {
            int offset = i*D + j;
            printf("%0.1f ", M[offset]);
        }
        printf("\n");
    }
}

int main() {
    int B = 2;      // Batch size
    int N = 5;      // Sequence length
    int D_in = 4;   // Input dimension
    int D_out = 8;  // Output dimension

    /*
    For the current implementation, N*D_out needs to be < 1024
    */

    srand(42);

    // Step 1: CPU memory (Host) allocation
    float *X = (float*)malloc(B*N*D_in*sizeof(float));      // Input data
    float *W = (float*)malloc(D_in*D_out*sizeof(float));    // Weights
    float *O = (float*)malloc(B*N*D_out*sizeof(float));     // Output data

    // Filling the matrix with random data
    for (int i=0; i<B*N*D_in; i++) {
        X[i] = rand() % 10;
    }

    for (int i=0; i<D_in*D_out; i++) {
        W[i] = rand() % 5;
    }

    /*
    Step 2: GPU (Device) memory allocation
    Prepended with `_d` and serve the same purpose as 
    the above ones (X, W, O)
    */
    float *d_X, *d_W, *d_O;

    cudaMalloc((void**) &d_X, B*N*D_in*sizeof(float));
    cudaMalloc((void**) &d_W, D_in*D_out*sizeof(float));
    cudaMalloc((void**) &d_O, B*N*D_out*sizeof(float));

    // Step 3: Copy the relevant data from the CPU memory to the GPU memory
    cudaMemcpy(d_X, X, B*N*D_in*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, D_in*D_out*sizeof(float), cudaMemcpyHostToDevice);

    // We would launch B blocks, each block processing a single batch
    dim3 grid(B);
    /*
    We would arrange the threads inside a block in the same dimension as our output
    i.e N*D_out, so that logically each thread corresponds to a single element in the
    output matrix. Hence, each thread is responsible for computing a single element of the output.
    */
    dim3 blocks(D_out, N);

    // Step 4: Launch the kernel
    matMul<<<grid, blocks>>>(
        d_X,
        d_W,
        d_O,
        B,
        N,
        D_in,
        D_out
    );

    cudaDeviceSynchronize();

    // Step 5: Copy the results back to 1D array
    cudaMemcpy(O, d_O, B*N*D_out*sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    // Let's visualize the matrices to make sure we are getting the correct results
    display_3dmat("X", X, B, N, D_in);
    display_2dmat("W", W, D_in, D_out);
    display_3dmat("O", O, B, N, D_out);

    // Cleanup
    free(X); free(W); free(O);
    cudaFree(d_X); cudaFree(d_W); cudaFree(d_O);

    return 0;
}