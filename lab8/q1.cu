#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void out(int* a, int* b, int wa) {
	int ridA = threadIdx.x;
	int ha = blockDim.x;
	
	int p = 1;
	if (ridA < ha) {
		for (int k = 0; k < wa; k++) {
			for (int z = 0; z <= ridA; z++) {
				p = p * a[ridA * wa + k];
			}
			b[ridA * wa + k] = p;
			p = 1;
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
		for(int j = 0; j < wa; j++)
			scanf("%d", &a[i*wa + j]);

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