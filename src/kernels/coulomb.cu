#include <math.h>
#include <float.h>


__global__ void coulombDirect_cuda(int number_of_targets_in_batch, int number_of_source_points_in_cluster,
        int starting_index_of_target, int starting_index_of_source,
        double *target_x, double *target_y, double *target_z,
        double *source_x, double *source_y, double *source_z, double *source_charge, double *source_weight,
        double *potential, int gpu_async_stream_id)
{

    // compute block and thread indices.  Note: need to have launched exactly number_of_targets_in_batch blocks.
    int bid = blockIdx.x*blockDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    
    int targetID = starting_index_of_target+bid;
    int sourceID = starting_index_of_source+tid;
     
    extern __shared__ double temporary_potential[];

    double3 t;
    t.x = target_x[targetID];
    t.y = target_y[targetID];
    t.z = target_z[targetID];


    if(threadIdx.x<number_of_source_points_in_cluster){

        double3 d;
        d.x = t.x - source_x[sourceID];
        d.y = t.y - source_y[sourceID];
        d.z = t.z - source_z[sourceID];
        double r  = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);

        if (r > DBL_MIN) {
            temporary_potential[threadIdx.x] = source_charge[sourceID] * source_weight[sourceID] / r;
        }
    } // end if tid<num
    __syncthreads();

    // PERFORM REDUCTION OVER temporary_potential
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            temporary_potential[threadIdx.x] += temporary_potential[threadIdx.x + s];
        }
        __syncthreads();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) potential[targetID] += temporary_potential[0];

    return;
}
