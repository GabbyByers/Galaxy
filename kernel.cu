#include "kernel.cuh"

__global__ void kernel(int* a, int* b, int* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void go(int* a, int* b, int* c, int arraySize) {
    cudaSetDevice(0);

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;
    cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_c, arraySize * sizeof(int));

    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    kernel <<<1, arraySize>>> (dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}