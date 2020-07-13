// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

// AOC kernel demonstrating device-side printf call

__kernel void hello_world(int thread_id_from_which_to_print_message, __global const float* restrict da, __global const float* restrict db, __global float* restrict dc, int colA, int colB) {
  // Get index of the work item
  //unsigned thread_id = get_global_id(0);
  //if(thread_id == thread_id_from_which_to_print_message) {
  //  printf("Thread #%u, idx #%d: Hello from Altera's OpenCL Compiler!\n", thread_id, gid + gid1);
  //}

  unsigned gid = get_global_id(0);
  unsigned gid1 = get_global_id(1);
  unsigned lid = get_local_id(0);
  unsigned lid1 = get_local_id(1);

  unsigned gpid = get_group_id(0);
  unsigned gpid1 = get_group_id(1);
  unsigned gsize = get_global_size(0);
  unsigned gsize1 = get_global_size(1);
  unsigned lsize = get_local_size(0);
  unsigned lsize1 = get_local_size(1);

  unsigned thread_id = gid + gid1 * gsize;
  //printf("[+] Thread #%u,\tgid(%d,%d),\tgpid(%d,%d),\tgps(%d,%d)\n", thread_id, gid,gid1, gpid,gpid1, gps,gps1);
  //printf("[+] Thread #%u,\tgid(%d,%d),\tlid(%d,%d),\tgpid(%d,%d),\tgsize(%d,%d),\tlsize(%d,%d)\n", thread_id, gid,gid1, lid,lid1, gpid,gpid1, gsize,gsize1, lsize,lsize1);

  //printf("[Thread #%u]\tgid(%d,%d),\tlid(%d,%d),\tgpid(%d,%d),\tgsize(%d,%d),\tlsize(%d,%d),\tdata(%f,%f)\n", thread_id, gid,gid1, lid,lid1, gpid,gpid1, gsize,gsize1, lsize,lsize1, da[thread_id], db[thread_id]);
  //printf("[Thread #%u]\tgid(%d,%d),\tlid(%d,%d),\tgpid(%d,%d),\tgsize(%d,%d),\tlsize(%d,%d),\tdata(%f,%f)\n", thread_id, gid,gid1, lid,lid1, gpid,gpid1, gsize,gsize1, lsize,lsize1, da[gid*colA], db[gid1]);
  printf("[Thread #%u]\tgid(%d,%d),\tlid(%d,%d),\tgpid(%d,%d),\tgsize(%d,%d),\tlsize(%d,%d),\tdata(%f,%f)\n", thread_id, gid,gid1, lid,lid1, gpid,gpid1, gsize,gsize1, lsize,lsize1, da[gid1*colA], db[gid]);

  // matmul
  float sum = 0.0f;
  for(size_t k=0; k<colA; ++k) {
    //sum += da[gid*colA + k] * db[k*colB + gid1];
    sum += da[gid1*colA + k] * db[k*colB + gid];
  }
  //dc[gid*colB + gid1] = sum;
  dc[gid1*colB + gid] = sum;

  return;
}

