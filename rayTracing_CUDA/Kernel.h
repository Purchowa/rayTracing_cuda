#pragma once
#include <cuda_runtime.h>
#include <random>
#include <ctime>
#include <string>

using std::string;

class Kernel {
public:
	Kernel::Kernel();
	Kernel::~Kernel();
	void run_kernel();

	string getText();

private:
	const int N;
	const int TPB;
	int* t1, * t2;

};
