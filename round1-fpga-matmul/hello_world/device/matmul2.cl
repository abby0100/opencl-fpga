#ifndef OPENCL
#define OPENCL
#endif

__kernel void hello_world(int thread_id_from_which_to_print_message, __global const float* restrict da, __global const float* restrict db, __global float* restrict dc, int colA, int colB) {

  const size_t TILE_DIM = 2;

  //unsigned gid = get_global_id(0);
  //unsigned gid1 = get_global_id(1);
  unsigned gpid = get_group_id(0);
  unsigned gpid1 = get_group_id(1);
  unsigned lid = get_local_id(0);
  unsigned lid1 = get_local_id(1);
  unsigned lsize = get_local_size(0);
  unsigned lsize1 = get_local_size(1);
  unsigned gsize = get_global_size(0);
  unsigned gsize1 = get_global_size(1);
  unsigned thread_id = (gpid*lsize + lid) + ((gpid1*lsize1 + lid1) * gsize);
  unsigned thread_id_rev = (gpid1*lsize1 + lid1) + ((gpid*lsize + lid) * gsize1);

  const int lrow = lid;
  const int lcol = lid1;
  const int gprow = gpid * lsize;
  const int gpcol = gpid1 * lsize1;

  //printf("[Thread #%u]\tgid(%d,%d),\tlid(%d,%d),\tgpid(%d,%d),\tgsize(%d,%d),\tlsize(%d,%d),\tdata(%f,%f)\n", thread_id, gid,gid1, lid,lid1, gpid,gpid1, gsize,gsize1, lsize,lsize1, da[gid1*colA], db[gid]);
  //printf("[Thread #%u]\tgpid(%d,%d),\tlid(%d,%d),\tdata(%f,%f)\n", thread_id, gpid,gpid1, lid,lid1, da[thread_id], db[thread_id_rev]);
  //printf("[Thread #%u]\tgpid(%d,%d),\tlid(%d,%d),\tdata(%f,%f)\n", thread_id, gpid,gpid1, lid,lid1, da[gprow*colA], db[gpcol]);

  printf("[Thread #%u]\tgpid(%d,%d),\tlid(%d,%d),\tdata(%f,%f %f,%f)\n", thread_id, gpid,gpid1, lid,lid1, da[lrow*colA],da[lrow*colA+1], db[lcol],db[colB + lcol]);
  //printf("[Thread #%u]\tgpid(%d,%d),\tlid(%d,%d),\tdata(%f,%f %f,%f)\n", thread_id, gpid,gpid1, lid,lid1, da[gprow*colA],da[gprow*colA+1], db[gpcol],db[colB + gpcol]);

  __local float blocka[TILE_DIM][TILE_DIM];
  __local float blockb[TILE_DIM][TILE_DIM];
  size_t nblock = colA/TILE_DIM;

  // matmul
  float sum = 0.0f;
  for(size_t n=0; n<nblock; ++n) {
    blocka[lrow][lcol] = da[(gprow + lrow)*colA + TILE_DIM * n + lcol];
    blockb[lcol][lrow] = db[(TILE_DIM * n + lrow)*colB + gpcol + lcol];
    //printf("[Thread #%u] barrier local memory\n", thread_id);
    barrier(CLK_LOCAL_MEM_FENCE);

    for(size_t k=0; k<TILE_DIM; ++k) {
      //printf("[Thread #%u] add(%f,%f)\n", thread_id, blocka[lrow][k], blockb[lcol][k]);
      sum += blocka[lrow][k] * blockb[lcol][k];
    }
    //printf("[Thread #%u] barrier sum\n", thread_id);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  dc[(gprow + lrow)*colB + gpcol + lcol] = sum;
  return;
}

