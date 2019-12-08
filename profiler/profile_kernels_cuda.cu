/*
 *Procedures for Particle-Cluster Treecode
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>

#include "../src/kernels/coulomb.cu"

#include "profile_kernels_cuda.h"


void Interaction_PC_Kernels_cuda(double *target_x, double *target_y, double *target_z, double *target_charge,
                            double *cluster_x, double *cluster_y, double *cluster_z,
                            double *cluster_charge, double *cluster_weight,
                            double *pointwisePotential, int interpolationOrder,
                            int numSources, int numTargets, int numberOfInterpolationPoints,
                            char *kernelName, double kernel_parameter, char *singularityHandling,
                            char *approximationName, int numLaunches)
{

    int numberOfClusterCharges = numberOfInterpolationPoints;
    int numberOfClusterWeights = numberOfInterpolationPoints;

    if (strcmp(approximationName, "hermite") == 0)
        numberOfClusterCharges = 8 * numberOfInterpolationPoints;

    if ((strcmp(approximationName, "hermite") == 0) && (strcmp(singularityHandling, "subtraction") == 0))
        numberOfClusterWeights = 8 * numberOfInterpolationPoints;

    for (int i = 0; i < numLaunches; ++i) {
        int streamID = i%3;

/**********************************************************/
/************** POTENTIAL FROM APPROX *********************/
/**********************************************************/


    /***********************************************/
    /***************** Coulomb *********************/
    /***********************************************/
        if (strcmp(kernelName, "coulomb") == 0) {

            if (strcmp(approximationName, "lagrange") == 0) {

                if (strcmp(singularityHandling, "skipping") == 0) {
        
                } else if (strcmp(singularityHandling, "subtraction") == 0) {

                } else {
                    printf("Invalid choice of singularityHandling. Exiting. \n");
                    exit(1);
                }

            } else if (strcmp(approximationName, "hermite") == 0) {

                if (strcmp(singularityHandling, "skipping") == 0) {

                } else if (strcmp(singularityHandling, "subtraction") == 0) {

                } else {
                    printf("Invalid choice of singularityHandling. Exiting. \n");
                    exit(1);
                }


            }else{
                printf("Invalid approximationName.  Was set to %s\n", approximationName);
                exit(1);
            }

/***********************************************/
/***************** Yukawa **********************/
/***********************************************/

        } else if (strcmp(kernelName, "yukawa") == 0) {

            if (strcmp(approximationName, "lagrange") == 0) {

                if (strcmp(singularityHandling, "skipping") == 0) {

                } else if (strcmp(singularityHandling, "subtraction") == 0) {
                    
                } else {
                    printf("Invalid choice of singularityHandling. Exiting. \n");
                    exit(1);
                }

            } else if (strcmp(approximationName, "hermite") == 0) {

                if (strcmp(singularityHandling, "skipping") == 0) {

                } else if (strcmp(singularityHandling, "subtraction") == 0) {

                } else {
                    printf("Invalid choice of singularityHandling. Exiting. \n");
                    exit(1);
                }

            } else {
                printf("Invalid approximationName.\n");
                exit(1);
            }

        } else {
            printf("Invalid kernelName. Exiting.\n");
            exit(1);
        }

    }

    return;

} /* END of function pc_treecode */




void Interaction_Direct_Kernels_cuda(double *source_x, double *source_y, double *source_z,
                                double *source_q, double *source_w,
                                double *target_x, double *target_y, double *target_z, double *target_q,
                                double *pointwisePotential, int numSources, int numTargets,
                                char *kernelName, double kernel_parameter, char *singularityHandling,
                                char *approximationName, int numLaunches)
{
    cudaError_t err;

    double *dev_source_x, *dev_source_y, *dev_source_z, *dev_source_q, *dev_source_w;
    double *dev_target_x, *dev_target_y, *dev_target_z, *dev_target_q;
    double *dev_pointwisePotential;

    err = cudaMalloc(&dev_source_x, numSources*sizeof(double));
    err = cudaMalloc(&dev_source_y, numSources*sizeof(double));
    err = cudaMalloc(&dev_source_z, numSources*sizeof(double));
    err = cudaMalloc(&dev_source_q, numSources*sizeof(double));
    err = cudaMalloc(&dev_source_w, numSources*sizeof(double));
    err = cudaMalloc(&dev_target_x, numTargets*sizeof(double));
    err = cudaMalloc(&dev_target_y, numTargets*sizeof(double));
    err = cudaMalloc(&dev_target_z, numTargets*sizeof(double));
    err = cudaMalloc(&dev_target_q, numTargets*sizeof(double));
    err = cudaMalloc(&dev_pointwisePotential, numTargets*sizeof(double));

    err = cudaMemcpy(dev_source_x, source_x, numSources*sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_source_y, source_y, numSources*sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_source_z, source_z, numSources*sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_source_q, source_q, numSources*sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_source_w, source_w, numSources*sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_target_x, target_x, numTargets*sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_target_y, target_y, numTargets*sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_target_z, target_z, numTargets*sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_target_q, target_q, numTargets*sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemset(dev_pointwisePotential, 0, numTargets*sizeof(double));


    for (int i = 0; i < numLaunches; ++i) {
        int streamID = i%3;

/**********************************************************/
/************** POTENTIAL FROM DIRECT *********************/
/**********************************************************/


        /***************************************/
        /********* Coulomb *********************/
        /***************************************/
    
        if (strcmp(kernelName, "coulomb") == 0) {
    
            if (strcmp(singularityHandling, "skipping") == 0) {
    
                coulombDirect_cuda<<<numTargets, numSources, numSources*sizeof(double)>>>
                       (numTargets, numSources, 0, 0,
                        dev_target_x, dev_target_y, dev_target_z,
                        dev_source_x, dev_source_y, dev_source_z, dev_source_q, dev_source_w,
                        dev_pointwisePotential);

                err = cudaGetLastError();
                if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
    
            } else if (strcmp(singularityHandling, "subtraction") == 0) {
    
            }else {
                printf("Invalid choice of singularityHandling. Exiting. \n");
                exit(1);
            }
    
        /***************************************/
        /********* Yukawa **********************/
        /***************************************/
    
        } else if (strcmp(kernelName, "yukawa") == 0) {
    
            if (strcmp(singularityHandling, "skipping") == 0) {
    
            } else if (strcmp(singularityHandling, "subtraction") == 0) {
    
            } else {
                printf("Invalid choice of singularityHandling. Exiting. \n");
                exit(1);
            }
    
    
        } else {
            printf("Invalid kernelName. Exiting.\n");
            exit(1);
        }

    }

    err = cudaDeviceSynchronize();

    err = cudaMemcpy(pointwisePotential, dev_pointwisePotential, numTargets*sizeof(double), cudaMemcpyDeviceToHost);
    err = cudaFree(dev_source_x); 
    err = cudaFree(dev_source_y); 
    err = cudaFree(dev_source_z); 
    err = cudaFree(dev_source_q); 
    err = cudaFree(dev_source_w); 
    err = cudaFree(dev_target_x); 
    err = cudaFree(dev_target_y); 
    err = cudaFree(dev_target_z); 
    err = cudaFree(dev_target_q); 
    err = cudaFree(dev_pointwisePotential); 


    return;
}
