#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void out(int* a, int*b, int wa) {
	int ridA = threadIdx.x;
	int ha = blockDim.x;
	
	int p;
	if (ridA > 0 && ridA < ha - 1) {
		for (int k = 1; k < wa - 1; k++) {
			p = a[ridA * wa + k];
			int rem, bin = 0, place = 1;

			while (p != 0) {
				rem = p % 2;
				p = p / 2;
				bin += rem * place;
				place = place * 10;
			}

			b[ridA * wa + k] = bin;
		}
	}
}

int main() {
	int* a, * b, ha, wa;
	int* da, * db;

	printf("Enter the no.of rows and columns: ");
	scanf("%d %d", &ha, &wa);

	int size = sizeof(int) * ha * wa;

	a = (int*)malloc(size);
	b = (int*)malloc(size);

	printf("Enter the matrix: ");
	for (int i = 0; i < ha; i++)
		for (int j = 0; j < wa; j++)
			scanf("%d", &a[i * wa + j]);

	cudaMalloc((void**)&da, size);
	cudaMalloc((void**)&db, size);

	cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

	out << <1, ha >> > (da, db, wa);

	cudaMemcpy(b, db, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < ha; i++) {
		for (int j = 0; j < wa; j++)
			printf("%d\t", b[i * wa + j]);
		printf("\n");
	}
	cudaFree(da);
	cudaFree(db);

	return 0;
}