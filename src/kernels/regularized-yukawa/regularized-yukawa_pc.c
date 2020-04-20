#include <math.h>
#include <float.h>
#include <stdio.h>

#include "../../run_params/struct_run_params.h"
#include "regularized-yukawa_pc.h"


void K_RegularizedYukawa_PC_Lagrange(int number_of_targets_in_batch, int number_of_interpolation_points_in_cluster,
        int starting_index_of_target, int starting_index_of_cluster,
        double *target_x, double *target_y, double *target_z,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_charge,
        struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{

    double kappa    = run_params->kernel_params[0];
    double epsilon2 = run_params->kernel_params[1] * run_params->kernel_params[1];

#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(target_x, target_y, target_z, \
                        cluster_x, cluster_y, cluster_z, cluster_charge, potential)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i = 0; i < number_of_targets_in_batch; i++) {

        double temporary_potential = 0.0;

        double tx = target_x[starting_index_of_target + i];
        double ty = target_y[starting_index_of_target + i];
        double tz = target_z[starting_index_of_target + i];

#ifdef OPENACC_ENABLED
        #pragma acc loop independent reduction(+:temporary_potential)
#endif
        for (int j = 0; j < number_of_interpolation_points_in_cluster; j++) {

            int jj = starting_index_of_cluster + j;
            double dx = tx - cluster_x[jj];
            double dy = ty - cluster_y[jj];
            double dz = tz - cluster_z[jj];
            double r2 = dx*dx + dy*dy + dz*dz;
            double r  = sqrt(r2);

            temporary_potential += cluster_charge[jj] * exp(-kappa * r) / sqrt(r2 + epsilon2);
        } // end loop over interpolation points
#ifdef OPENACC_ENABLED
        #pragma acc atomic
#endif
        potential[starting_index_of_target + i] += temporary_potential;
    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}



/*
void K_RegularizedYukawa_PC_Hermite(int number_of_targets_in_batch, int number_of_interpolation_points_in_cluster,
        int starting_index_of_target, int starting_index_of_cluster, int total_number_interpolation_points,
        double *target_x, double *target_y, double *target_z,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_charge,
        struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{

    double *cluster_charge_          = &cluster_charge[8*starting_index_of_cluster + 0*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_x   = &cluster_charge[8*starting_index_of_cluster + 1*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_y   = &cluster_charge[8*starting_index_of_cluster + 2*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_z   = &cluster_charge[8*starting_index_of_cluster + 3*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_xy  = &cluster_charge[8*starting_index_of_cluster + 4*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_yz  = &cluster_charge[8*starting_index_of_cluster + 5*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_xz  = &cluster_charge[8*starting_index_of_cluster + 6*number_of_interpolation_points_in_cluster];
    double *cluster_charge_delta_xyz = &cluster_charge[8*starting_index_of_cluster + 7*number_of_interpolation_points_in_cluster];

    double kappa    = run_params->kernel_params[0];
    double epsilon2 = run_params->kernel_params[1] * run_params->kernel_params[1];
    double kappa2   = kappa * kappa;
    double kappa3   = kappa * kappa2;

#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(target_x, target_y, target_z, \
                        cluster_x, cluster_y, cluster_z, cluster_charge, potential, \
                        cluster_charge_, cluster_charge_delta_x, cluster_charge_delta_y, cluster_charge_delta_z, \
                        cluster_charge_delta_xy, cluster_charge_delta_yz, cluster_charge_delta_xz, \
                        cluster_charge_delta_xyz)
    {
#endif
#ifdef OPENACC_ENABLED
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
        for (int j = 0; j < number_of_interpolation_points_in_cluster; j++) {

            int jj = starting_index_of_cluster + j;
            double dx = tx - cluster_x[jj];
            double dy = ty - cluster_y[jj];
            double dz = tz - cluster_z[jj];

            double r  = sqrt(dx*dx + dy*dy + dz*dz);
            double q  = sqrt(dx*dx + dy*dy * dz*dz + epsilon2);

            double rinv  = 1 / r;
            double r2inv = rinv * rinv;
            double r3inv = rinv * r2inv;
            double r4inv = rinv * r3inv;
            double r5inv = rinv * r4inv;

            double q2    = q  * q;
            double q4    = q2 * q2;
            double qinv  = 1 / q;
            double q2inv = qinv  * qinv;
            double q3inv = qinv  * q2inv;
            double q5inv = q2inv * q3inv;


            
            if (r > DBL_MIN) {
                temporary_potential += exp(-kappa * r)

                         * (qinv * (cluster_charge_[j])

                         + rinv * q3inv * (kappa * q2 + r)
                                * (cluster_charge_delta_x[j]*dx + cluster_charge_delta_y[j]*dy
                                 + cluster_charge_delta_z[j]*dz)

                         + q5inv * (3  +  kappa2 * q4 * r2inv  +  kappa * q4 * r3inv  +  2 * kappa * q2 * rinv)
                                 * (cluster_charge_delta_xy[j]*dx*dy + cluster_charge_delta_yz[j]*dy*dz
                                  + cluster_charge_delta_xz[j]*dx*dz)

                         + (kappa3 * r3inv * qinv  +   3 * kappa2 * r2inv * q3inv   +   3 * kappa2 * r4inv * qinv
                                                   -   6 * kappa  * rinv  * q5inv   
                                                   +   3 * kappa  * r3inv * q3inv   +   3 * kappa  * r5inv * qinv
                                                   +  15 * kappa  * rinv  * qinv    +  15 * q3inv)
                                 * cluster_charge_delta_xyz[j]*dx*dy*dz);


            } else {
            
                temporary_potential += exp(-kappa * r) * qinv * (cluster_charge_[j]);

            }


        } // end loop over interpolation points
#ifdef OPENACC_ENABLED
        #pragma acc atomic
#endif
        potential[starting_index_of_target + i] += temporary_potential;
    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}
*/
