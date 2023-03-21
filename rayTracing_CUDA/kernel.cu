#include "Kernel.h"
#include <iostream>

static __global__ void sum(int* d_a, int* d_b, const int N) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < N) {
		d_a[index] += d_b[index];
	}
}

Kernel::Kernel(): N(1024), TPB(32){
	srand(time(NULL));

	t1 = new int[N];
	t2 = new int[N];

	for (int i = 0; i < N; i++) {
		t1[i] = rand() % 10 + 1;
		t2[i] = rand() % 10 + 1;
	}


	// Printing
	std::cout << "T1: ";
	for (int i = 0; i < TPB; i++) {
		std::cout.width(3);
		std::cout << t1[i];
	}
	std::cout << '\n';
	std::cout << "T2: ";
	for (int i = 0; i < TPB; i++) {
		std::cout.width(3);
		std::cout << t2[i];
	}

	std::cout << '\n';
	std::cout << '\n';
}

void Kernel::run_kernel() {
	int* d_a, * d_b;
	cudaMalloc(&d_a, N * sizeof(*t1));
	cudaMalloc(&d_b, N * sizeof(*t1));

	cudaMemcpy(d_a, t1, N * sizeof(*t1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, t2, N * sizeof(*t2), cudaMemcpyHostToDevice);

	sum << < (N + TPB - 1) / TPB, TPB >> > (d_a, d_b, N);

	cudaMemcpy(t1, d_a, N * sizeof(*t1), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);

	for (int i = 0; i < TPB; i++) {
		std::cout.width(3);
		std::cout << t1[i];
	}
	std::cout << '\n';
}

string Kernel::getText()
{
	string out;
	
	for (int i = 0; i < N; i++) {
		out += t1[i];
	}
	return out;
	/*char *buff = new char[out.length()];

	for (int i = 0; i < N; i++) {
		buff[i] = out[i];
	}
	return buff;*/
}

Kernel::~Kernel() {
	delete[] t1;
	delete[] t2;
}