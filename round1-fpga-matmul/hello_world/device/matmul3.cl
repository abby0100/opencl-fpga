#ifndef OPENCL
#define OPENCL
#endif

__kernel void hello_world(int thread_id_from_which_to_print_message, __global const float* restrict da, __global const float* restrict db, __global float* restrict dc, int colA, int colB) {

  unsigned gid = get_global_id(0);
  unsigned gid1 = get_global_id(1);
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

  const int SIMD = 4;
  const int TILE_DIM = 4;
  const int lrow = lid;
  const int lcol = lid1;
  const int gprow = gpid * lsize;
  const int gpcol = gpid1 * lsize1;

  printf("[Thread #%u]\tgpid(%d,%d),\tlid(%d,%d),\tdata(%f,%f %f,%f)\n", thread_id, gpid,gpid1, lid,lid1, da[lrow*colA],da[lrow*colA+1], db[lcol],db[colB + lcol]);

  __local float4 blocka[4][1];
  __local float4 blockb[4][1];
  size_t nblock = colA / TILE_DIM;

  // matmul
  float sum = (0.0f);
  for(size_t n=0; n<nblock; ++n) {
  //for(size_t n=1; n<=1; ++n) {
      const int idxa  = (gprow + lrow) * colA + TILE_DIM * n + lcol + 0;
      const int idxa1 = (gprow + lrow) * colA + TILE_DIM * n + lcol + 1;
      const int idxa2 = (gprow + lrow) * colA + TILE_DIM * n + lcol + 2;
      const int idxa3 = (gprow + lrow) * colA + TILE_DIM * n + lcol + 3;
    if(lcol == 0) {
      blocka[lrow][lcol] = (float4) (da[idxa], da[idxa1], da[idxa2], da[idxa3]);
    }

      const int idxb  = (TILE_DIM * n + lrow + 0) * colB + gpcol + lcol;
      const int idxb1 = (TILE_DIM * n + lrow + 1) * colB + gpcol + lcol;
      const int idxb2 = (TILE_DIM * n + lrow + 2) * colB + gpcol + lcol;
      const int idxb3 = (TILE_DIM * n + lrow + 3) * colB + gpcol + lcol;
    if(lrow == 0) {
      blockb[lcol][lrow] = (float4) (db[idxb], db[idxb1], db[idxb2], db[idxb3]);
    }
    for(size_t i=0; i<4; ++i) {
      printf("[Thread %u, blocka]\t(%f,%f,%f,%f)\n", thread_id, blocka[i][0].s0,blocka[i][0].s1,blocka[i][0].s2,blocka[i][0].s3);
    }
    for(size_t i=0; i<4; ++i) {
      printf("[Thread %u, blockb]\t(%f,%f,%f,%f)\n", thread_id, blockb[i][0].s0,blockb[i][0].s1,blockb[i][0].s2,blockb[i][0].s3);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for(size_t i=0; i<4; ++i) {
      printf("[Thread %u, blocka]\t(%f,%f,%f,%f)\n", thread_id, blocka[i][0].s0,blocka[i][0].s1,blocka[i][0].s2,blocka[i][0].s3);
    }
    for(size_t i=0; i<4; ++i) {
      printf("[Thread %u, blockb]\t(%f,%f,%f,%f)\n", thread_id, blockb[i][0].s0,blockb[i][0].s1,blockb[i][0].s2,blockb[i][0].s3);
    }

    printf("[Thread_%u] blocka(%d,%d)=[(%d,%d,%d,%d):(%f,%f,%f,%f)]\tblockb(%d,%d)=[(%d,%d,%d,%d):(%f,%f,%f,%f)]\n", thread_id, 
      lrow,0, idxa,idxa1,idxa2,idxa3, blocka[lrow][0].s0, blocka[lrow][0].s1, blocka[lrow][0].s2, blocka[lrow][0].s3,
      lcol,0, idxb,idxb1,idxb2,idxb3, blockb[lcol][0].s0, blockb[lcol][0].s1, blockb[lcol][0].s2, blockb[lcol][0].s3
    );

    sum += dot(blocka[lrow][0] * blockb[lcol][0], (float4) (1,1,1,1));
    printf("[Thread %u]\tsum(%f)\tblocka(%d,%d):[%f,%f,%f,%f]\tblockb(%d,%d):[%f,%f,%f,%f]\n", thread_id, sum, 
      lrow,0, blocka[lrow][0].s0,blocka[lrow][0].s1,blocka[lrow][0].s2,blocka[lrow][0].s3,
      lcol,0, blockb[lcol][0].s0,blockb[lcol][0].s1,blockb[lcol][0].s2,blockb[lcol][0].s3);

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  dc[(gprow + lrow)*colB + gpcol + lcol] = sum;
  return;
}

