#include <math.h>
#include <float.h>
#include <stdio.h>

#include "../../run_params/struct_run_params.h"
#include "coulomb_direct.h"

void K_Coulomb_Direct(int number_of_targets_in_batch, int number_of_source_points_in_cluster,
        int starting_index_of_target, int starting_index_of_source,
        double *target_x, double *target_y, double *target_z,
        double *source_x, double *source_y, double *source_z, double *source_charge, double *source_weight,
        struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{

#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(target_x, target_y, target_z, \
                        source_x, source_y, source_z, source_charge, source_weight, potential)
    {
    #pragma acc loop independent
#endif
    for (int i = 0; i < number_of_targets_in_batch; i++) {

        int ii = starting_index_of_target + i;
        double temporary_potential = 0.0;

        double tx = target_x[ii];
        double ty = target_y[ii];
        double tz = target_z[ii];

#ifdef OPENACC_ENABLED
        #pragma acc loop independent reduction(+:temporary_potential)
#endif
        for (int j = 0; j < number_of_source_points_in_cluster; j++) {
#ifdef OPENACC_ENABLED
            #pragma acc cache(source_x[starting_index_of_source : starting_index_of_source+number_of_source_points_in_cluster], \
                              source_y[starting_index_of_source : starting_index_of_source+number_of_source_points_in_cluster], \
                              source_z[starting_index_of_source : starting_index_of_source+number_of_source_points_in_cluster], \
                              source_charge[starting_index_of_source : starting_index_of_source+number_of_source_points_in_cluster], \
                              source_weight[starting_index_of_source : starting_index_of_source+number_of_source_points_in_cluster])
#endif


            int jj = starting_index_of_source + j;
            double dx = tx - source_x[jj];
            double dy = ty - source_y[jj];
            double dz = tz - source_z[jj];
            double r2  = dx*dx + dy*dy + dz*dz;

            if (r2 > DBL_MIN) {
                temporary_potential += source_charge[jj] * source_weight[jj] / sqrt(r2);
            }
        } // end loop over interpolation points
#ifdef OPENACC_ENABLED
        #pragma acc atomic
#endif
        potential[ii] += temporary_potential;
    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}
