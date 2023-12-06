#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#define NNZ 7
#define N 4
#define BLOCK_SIZE 4

__global__ void spmv(float* data, int* col_index, int* row_ptr, float* x, float* y) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N) {
		float dot = 0;
		int row_start = row_ptr[row];
		int row_end = row_ptr[row + 1];
		for (int elem = row_start; elem < row_end; elem++) {
			dot += data[elem] * x[col_index[elem]];
		}
		y[row] = dot;
	}
}

int main() {
	float data[NNZ] = { 3.0, 1.0, 2.0, 4.0, 1.0, 1.0, 1.0 };
	int col_index[NNZ] = { 0,2,1,2,3,0,3 };
	int row_pointers[N + 1] = { 0,2,2,5,7 };

	float x[N] = { 1.0, 2.0, 3.0, 4.0 };

	float* d_y;
	cudaMalloc((void**)&d_y, N * sizeof(float));

	float* d_data, * d_x;
	int* d_col_index, * d_row_ptr;
	cudaMalloc((void**)&d_data, NNZ * sizeof(float));
	cudaMalloc((void**)&d_col_index, NNZ * sizeof(int));
	cudaMalloc((void**)&d_row_ptr, (N+1) * sizeof(int));
	cudaMalloc((void**)&d_x, N * sizeof(float));

	cudaMemcpy(d_data, data, NNZ * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_index, col_index, NNZ * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row_ptr, row_pointers, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

	int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	spmv << <num_blocks, BLOCK_SIZE >> > (d_data, d_col_index, d_row_ptr, d_x, d_y);

	float y[N];
	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_data);
	cudaFree(d_col_index);
	cudaFree(d_row_ptr);
	cudaFree(d_x);
	cudaFree(d_y);

	for (int i = 0; i < N; i++)
		printf("y[%d] = %f\n", i, y[i]);

	return 0;
}