#ifndef H_INTERACTIONCOMPUTECUDA_H
#define H_INTERACTIONCOMPUTECUDA_H


#ifdef __cplusplus
extern "C" {
#endif

void Interaction_PC_Kernels_cuda(double *xT, double *yT, double *zT, double *qT,
                            double *xC, double *yC, double *zC, double *qC, double *wC,
                            double *pointwisePotential, int interpolationOrder,
                            int numSources, int numTargets, int numClusters,
                            char *kernelName, double kernel_parameter, char *singularityHandling,
                            char *approximationName, int numLaunches);


void Interaction_Direct_Kernels_cuda(double *source_x, double *source_y, double *source_z,
                            double *source_q, double *source_w,
                            double *target_x, double *target_y, double *target_z, double *target_q,
                            double *totalPotential, int numSources, int numTargets,
                            char *kernelName, double kernel_parameter, char *singularityHandling,
                            char *approximationName, int numLaunches);

#ifdef __cplusplus
}
#endif


#endif /* H_INTERACTIONCOMPUTE_H */
