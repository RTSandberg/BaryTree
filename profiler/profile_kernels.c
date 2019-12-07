/*
 *Procedures for Particle-Cluster Treecode
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>

#include "../src/kernels/coulomb.h"
#include "../src/kernels/yukawa.h"
#include "../src/kernels/coulomb_singularity_subtraction.h"
#include "../src/kernels/yukawa_singularity_subtraction.h"

#include "profile_kernels.h"


void Interaction_PC_Kernels(double *target_x, double *target_y, double *target_z, double *target_charge,
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

#ifdef OPENACC_ENABLED
    #pragma acc data copyin(target_x[0:numTargets], target_y[0:numTargets], target_z[0:numTargets], \
                        target_charge[0:numTargets], \
                        cluster_x[0:numberOfInterpolationPoints], cluster_y[0:numberOfInterpolationPoints], \
                        cluster_z[0:numberOfInterpolationPoints], \
                        cluster_charge[0:numberOfClusterCharges], cluster_weight[0:numberOfClusterWeights]), \
                        copy(pointwisePotential[0:numTargets])
#endif
    {

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
        
                    coulombApproximationLagrange(numTargets,
                            numberOfInterpolationPoints, 0, 0,
                            target_x, target_y, target_z,
                            cluster_x, cluster_y, cluster_z, cluster_charge,
                            pointwisePotential, streamID);

                } else if (strcmp(singularityHandling, "subtraction") == 0) {

                    coulombSingularitySubtractionApproximationLagrange(numTargets,
                            numberOfInterpolationPoints, 0, 0,
                            target_x, target_y, target_z, target_charge,
                            cluster_x, cluster_y, cluster_z, cluster_charge, cluster_weight,
                            kernel_parameter, pointwisePotential, streamID);

                } else {
                    printf("Invalid choice of singularityHandling. Exiting. \n");
                    exit(1);
                }

            } else if (strcmp(approximationName, "hermite") == 0) {

                if (strcmp(singularityHandling, "skipping") == 0) {

                    coulombApproximationHermite(numTargets,
                            numberOfInterpolationPoints, 0,
                            0, numberOfInterpolationPoints,
                            target_x, target_y, target_z,
                            cluster_x, cluster_y, cluster_z, cluster_charge,
                            pointwisePotential, streamID);

                } else if (strcmp(singularityHandling, "subtraction") == 0) {

                    coulombSingularitySubtractionApproximationHermite(numTargets,
                            numberOfInterpolationPoints, 0,
                            0, numberOfInterpolationPoints,
                            target_x, target_y, target_z, target_charge,
                            cluster_x, cluster_y, cluster_z, cluster_charge, cluster_weight,
                            kernel_parameter, pointwisePotential, streamID);

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

                    yukawaApproximationLagrange(numTargets,
                            numberOfInterpolationPoints, 0, 0,
                            target_x, target_y, target_z,
                            cluster_x, cluster_y, cluster_z, cluster_charge,
                            kernel_parameter, pointwisePotential, streamID);

                } else if (strcmp(singularityHandling, "subtraction") == 0) {
                    
                    yukawaSingularitySubtractionApproximationLagrange(numTargets,
                            numberOfInterpolationPoints, 0, 0,
                            target_x, target_y, target_z, target_charge,
                            cluster_x, cluster_y, cluster_z, cluster_charge, cluster_weight,
                            kernel_parameter, pointwisePotential, streamID);

                } else {
                    printf("Invalid choice of singularityHandling. Exiting. \n");
                    exit(1);
                }

            } else if (strcmp(approximationName, "hermite") == 0) {

                if (strcmp(singularityHandling, "skipping") == 0) {

                    yukawaApproximationHermite(numTargets,
                            numberOfInterpolationPoints, 0,
                            0, numberOfInterpolationPoints,
                            target_x, target_y, target_z,
                            cluster_x, cluster_y, cluster_z, cluster_charge,
                            kernel_parameter, pointwisePotential, streamID);

                } else if (strcmp(singularityHandling, "subtraction") == 0) {

                    yukawaSingularitySubtractionApproximationHermite(numTargets,
                            numberOfInterpolationPoints, 0,
                            0, numberOfInterpolationPoints,
                            target_x, target_y, target_z, target_charge,
                            cluster_x, cluster_y, cluster_z, cluster_charge, cluster_weight,
                            kernel_parameter, pointwisePotential, streamID);

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

#ifdef OPENACC_ENABLED
        #pragma acc wait
#endif
    } // end acc data region

    return;

} /* END of function pc_treecode */




void Interaction_Direct_Kernels(double *source_x, double *source_y, double *source_z,
                                double *source_q, double *source_w,
                                double *target_x, double *target_y, double *target_z, double *target_q,
                                double *pointwisePotential, int numSources, int numTargets,
                                char *kernelName, double kernel_parameter, char *singularityHandling,
                                char *approximationName, int numLaunches)
{


#ifdef OPENACC_ENABLED
    #pragma acc data copyin(source_x[0:numSources], source_y[0:numSources], source_z[0:numSources], \
                            source_q[0:numSources], source_w[0:numSources], \
                            target_x[0:numTargets], target_y[0:numTargets], target_z[0:numTargets], \
                            target_q[0:numTargets]), copy(pointwisePotential[0:numTargets])
#endif
    {

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
    
                coulombDirect(numTargets, numSources, 0, 0,
                        target_x, target_y, target_z,
                        source_x, source_y, source_z, source_q, source_w,
                        pointwisePotential, streamID);
    
            } else if (strcmp(singularityHandling, "subtraction") == 0) {
    
                coulombSingularitySubtractionDirect(numTargets, numSources, 0, 0,
                        target_x, target_y, target_z, target_q,
                        source_x, source_y, source_z, source_q, source_w,
                        kernel_parameter, pointwisePotential, streamID);
    
            }else {
                printf("Invalid choice of singularityHandling. Exiting. \n");
                exit(1);
            }
    
        /***************************************/
        /********* Yukawa **********************/
        /***************************************/
    
        } else if (strcmp(kernelName, "yukawa") == 0) {
    
            if (strcmp(singularityHandling, "skipping") == 0) {
    
                yukawaDirect(numTargets, numSources, 0, 0,
                        target_x, target_y, target_z,
                        source_x, source_y, source_z, source_q, source_w,
                        kernel_parameter, pointwisePotential, streamID);
    
            } else if (strcmp(singularityHandling, "subtraction") == 0) {
    
                yukawaSingularitySubtractionDirect(numTargets, numSources, 0, 0,
                        target_x, target_y, target_z, target_q,
                        source_x, source_y, source_z, source_q, source_w,
                        kernel_parameter, pointwisePotential, streamID);
    
            } else {
                printf("Invalid choice of singularityHandling. Exiting. \n");
                exit(1);
            }
    
    
        } else {
            printf("Invalid kernelName. Exiting.\n");
            exit(1);
        }

    }

#ifdef OPENACC_ENABLED
        #pragma acc wait
#endif
    } // end acc data region

    return;
}
