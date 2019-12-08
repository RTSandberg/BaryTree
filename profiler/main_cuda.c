#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <float.h>

#include "../src/tools.h"
#include "profile_kernels_cuda.h"


const unsigned m = 1664525u;
const unsigned c = 1013904223u;

int main(int argc, char **argv)
{

    //run parameters
    int N, order, numLaunches;
    double kappa; 
    char *kernelName = NULL;
    char *singularityHandling = NULL;
    char *approximationName = NULL;
    char *kernelType = NULL;

    N = atoi(argv[1]);
    order = atoi(argv[2]);
    kernelName = argv[3];
    kappa = atof(argv[4]);
    singularityHandling = argv[5];
    approximationName = argv[6];
    kernelType = argv[7];
    numLaunches = atoi(argv[8]);

    int numClusterPts = (order+1)*(order+1)*(order+1);


    int rc, rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);


    /* variables for date-time calculation */
    double time_run[4], time1, time2;
    time1 = MPI_Wtime();


    double *source_x = malloc(N*sizeof(double));
    double *source_y = malloc(N*sizeof(double));
    double *source_z = malloc(N*sizeof(double));
    double *source_q = malloc(N*sizeof(double));
    double *source_w = malloc(N*sizeof(double));

    double *target_x = malloc(N*sizeof(double));
    double *target_y = malloc(N*sizeof(double));
    double *target_z = malloc(N*sizeof(double));
    double *target_q = malloc(N*sizeof(double));

    double *cluster_x = malloc(numClusterPts*sizeof(double));
    double *cluster_y = malloc(numClusterPts*sizeof(double));
    double *cluster_z = malloc(numClusterPts*sizeof(double));
    double *cluster_q = malloc(numClusterPts*sizeof(double));
    double *cluster_w = malloc(numClusterPts*sizeof(double));

    double *potential = malloc(N*sizeof(double));

    time_t t = time(NULL);
    unsigned t_hashed = (unsigned) t;
    t_hashed = m*t_hashed + c;
    //srand(t_hashed ^ rank);

    for (int i = 0; i < N; ++i) {
        source_x[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        source_y[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        source_z[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        source_q[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        source_w[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        target_x[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        target_y[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        target_z[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        target_q[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
    }

    for (int i = 0; i < numClusterPts; ++i) {
        cluster_x[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        cluster_y[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        cluster_z[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        cluster_q[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        cluster_w[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
    }

    memset(potential, 0, N * sizeof(double));


    time_run[0] = MPI_Wtime() - time1;

    /* Calling main treecode subroutine to calculate approximate energy */

    MPI_Barrier(MPI_COMM_WORLD);

    if (strcmp(kernelType,"direct") == 0) {
        if (rank == 0) fprintf(stderr,"Running direct kernels...\n");

        time1 = MPI_Wtime();
        Interaction_Direct_Kernels_cuda(source_x, source_y, source_z, source_q, source_w,
                                   target_x, target_y, target_z, target_q,
                                   potential, N, N,
                                   kernelName, kappa, singularityHandling, approximationName,
                                   numLaunches);
        time2 = MPI_Wtime() - time1;
        double potsum = sum(potential, N);
        printf("CUDA direct sum: %f, time: %f\n", potsum, time2);

        if (rank == 0) fprintf(stderr,"Done.\n");
    
    } else {

        if (rank == 0) fprintf(stderr,"Running PC kernels...\n");
        if (rank == 0) fprintf(stderr,"Done.\n");

    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    

    free(source_x);
    free(source_y);
    free(source_z);
    free(source_q);
    free(source_w);

    free(target_x);
    free(target_y);
    free(target_z);
    free(target_q);

    free(cluster_x);
    free(cluster_y);
    free(cluster_z);
    free(cluster_q);
    free(cluster_w);

    free(potential);

    MPI_Finalize();

    return 0;
}
