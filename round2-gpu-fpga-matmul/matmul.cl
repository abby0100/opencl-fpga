#ifndef OPENCL
#define OPENCL
#endif

__kernel void matmul() {
	size_t gid = get_global_id(0);
	size_t gid1 = get_global_id(1);

	printf("<matmul> gid(%d,%d)\n", gid,gid1);
}
