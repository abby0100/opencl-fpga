#include <iostream>
#include <stdio.h>
#include <stdlib.h>
//#include "CL/cl.h"
#include "CL/opencl.h"

using namespace std;

#define checkError(status, errInfo)	\
	if(status != CL_SUCCESS) {		\
		cout << errInfo << endl;	\
		exit(1);					\
	}

void clinfo() {}
void callback(const char* errInfo, const void*, size_t, void*) {
	cout << "Context callback error:\t" << errInfo << endl;
}
void loadProgramSource(const char* files[], size_t number, char** buffers, size_t* sizes) {
	for(size_t i=0; i<number; ++i) {
		FILE* file = fopen(files[i], "r");
		if(!file) {
			cerr << "Failed to open OpenCL kernel file:\t" << files[i] << endl;
			exit(1);
		}

		fseek(file, 0, SEEK_END);
		sizes[i] = ftell(file);
		rewind(file);

		buffers[i] = new char[sizes[i] + 1];
		fread(buffers[i], sizeof(char), sizes[i], file);
		buffers[i][sizes[i]] = 0;
		fclose(file);
	}
}

void matmul() {
	cout << "[" << __func__ << ":" << __LINE__ << "]\t" << endl;
	cl_int status = CL_SUCCESS;
	int length = 128;
	char clInfo[length];
	size_t infoSize = 0;

	// platform
	cl_uint numPlatforms;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkError(status, "Failed to get Platforms number");
	cl_platform_id platforms[numPlatforms];
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	checkError(status, "Failed to create Platforms");
	for(size_t i=0; i<numPlatforms; ++i) {
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, length, clInfo, &infoSize);
		cout << "Platform " << i << " info:\t" << clInfo << endl;
	}

	// device
	cl_uint numDevices;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);	// 0: GPU
	//status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);	// 1: FPGA
	checkError(status, "Failed to get Devices number");
	cl_device_id devices[numDevices];
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
	//status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
	checkError(status, "Failed to create devices");
	for(size_t i=0; i<numDevices; ++i) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, length, clInfo, &infoSize);
		cout << "Device " << i << " info:\t" << clInfo << endl;
	}

	// context
	cl_context context = clCreateContext(NULL, numDevices, devices, &callback, NULL, &status);
	checkError(status, "Failed to create Context");

	// queue
	cl_command_queue queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command Queue");

	// program
	// way 1
	int numCL = 1;
	char* kernelbuffs[numCL];
	size_t kernelsizes[numCL];
	const char* kernelfiles[numCL] = {"matmul.cl"};
	loadProgramSource(kernelfiles, numCL, kernelbuffs, kernelsizes);
	cout << "Kernel source length:\t" << kernelsizes[0] << endl;

	cl_program program = clCreateProgramWithSource(context, numCL, (const char**)kernelbuffs, kernelsizes, &status);
	checkError(status, "Failed to create Program");

	// build
	const char options[] = "-cl-finite-math-only -cl-no-signed-zeros";
	status = clBuildProgram(program, 1, devices, options, NULL, NULL);
	if(status != CL_SUCCESS) {
		size_t logsize = 0;
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
		char programLog[logsize + 1];
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, logsize+1, programLog, NULL);
		cerr << "\n=== Error ===\n" << programLog << "\n=============" << endl;
		exit(1);
	}

	// kernel
	cl_kernel kernel = clCreateKernel(program, "matmul", &status);
	checkError(status, "Failed to create Kernel");

	// buffer
	// args

	// NDRange
	int dims = 2;
	size_t gsizes[] = {4,4};
	size_t lsizes[] = {2,2};
	status = clEnqueueNDRangeKernel(queue, kernel, dims, NULL, gsizes, lsizes, 0, NULL, NULL);
	checkError(status, "Failed to enqueue NDRange Kernel");

	// finish
	status = clFinish(queue);
	checkError(status, "Failed to wait Queue finish");

	//
}

int main(int argc, char** argv) {
	cout << "[" << __func__ << ":" << __LINE__ << "]\t" << endl;

	clinfo();

	matmul();
	return 0;
}
