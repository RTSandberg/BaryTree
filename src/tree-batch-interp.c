/*
 *Procedures for Cluster-Particle Treecode
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#include "array.h"
#include "globvars.h"
#include "tnode.h"
#include "particles.h"
#include "tools.h"

#include "partition.h"
#include "tree.h"




void cc_fill_cluster_batch_interp(struct particles *target_clusters, struct particles *targets, struct batch *batches, int order, int numDevices, int numThreads)
{
    int pointsPerCluster = (order+1)*(order+1)*(order+1);
    int numInterpPoints = batches->num * pointsPerCluster;
    make_vector(target_clusters->x, numInterpPoints);
    make_vector(target_clusters->y, numInterpPoints);
    make_vector(target_clusters->z, numInterpPoints);
    make_vector(target_clusters->q, numInterpPoints);
    make_vector(target_clusters->w, numInterpPoints);
    target_clusters->num=numInterpPoints;

    for (int i = 0; i < numInterpPoints; i++) {
        target_clusters->q[i]=0.0;
    }

#pragma omp parallel num_threads(numThreads)
    {

#ifdef OPENACC_ENABLED
        if (omp_get_thread_num() < numDevices) {
            acc_set_device_num(omp_get_thread_num(), acc_get_device_type());
        }
#endif

        int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
        if (this_thread==0){printf("numDevices: %i\n", numDevices);}
        if (this_thread==0){printf("num_threads: %i\n", num_threads);}
        printf("this_thread: %i\n", this_thread);

        double *tempX, *tempY, *tempZ;
        make_vector(tempX,target_clusters->num);
        make_vector(tempY,target_clusters->num);
        make_vector(tempZ,target_clusters->num);

        for (int i = 0; i < target_clusters->num; i++) {
            tempX[i] = 0.0;
            tempY[i] = 0.0;
            tempZ[i] = 0.0;
        }

        double *xT = targets->x;
        double *yT = targets->y;
        double *zT = targets->z;
        double *qT = targets->q;

        double *xC = target_clusters->x;
        double *yC = target_clusters->y;
        double *zC = target_clusters->z;

        int clusterNum = target_clusters->num;
        int targetNum = targets->num;
        int pointsPerCluster = torderlim * torderlim * torderlim;

#pragma acc data copyin(tt[0:torderlim], \
        xT[0:targetNum], yT[0:targetNum], zT[0:targetNum], qT[0:targetNum]) \
        copy(tempX[0:clusterNum], tempY[0:clusterNum], tempZ[0:clusterNum])
        {
            #pragma omp for schedule(guided)
            for (int i = 0; i < batches->num; i++) {  // start from i=1, don't need to compute root moments
                int startingIndexInClusters = pointsPerCluster * i;
                cc_comp_batch_interp(batches, i, xT, yT, zT, qT,
                                     tempX, tempY, tempZ);
            }
            #pragma acc wait
        } // end ACC DATA REGION

        for (int j = 0; j < clusterNum; j++)
        {
            if(tempX[j] != 0.0) target_clusters->x[j] = tempX[j];
            if(tempY[j] != 0.0) target_clusters->y[j] = tempY[j];
            if(tempZ[j] != 0.0) target_clusters->z[j] = tempZ[j];
        }

        free_vector(tempX);
        free_vector(tempY);
        free_vector(tempZ);

    } // end OMP PARALLEL REGION

    return;
}




void cc_comp_batch_interp(struct batch *batches, int idx,
        double *xT, double *yT, double *zT, double *qT,
        double *clusterX, double *clusterY, double *clusterZ)
{
    int pointsPerCluster = torderlim*torderlim*torderlim;
    int startingIndexInClusters = idx * pointsPerCluster;

    double x0, x1, y0, y1, z0, z1;  // bounding box

    double nodeX[torderlim], nodeY[torderlim], nodeZ[torderlim];
    

    x0 = batches->x_min[idx];  // 1e-15 fails for large meshes, mysteriously.
    x1 = batches->x_max[idx];
    y0 = batches->y_min[idx];
    y1 = batches->y_max[idx];
    z0 = batches->z_min[idx];
    z1 = batches->z_max[idx];


    int streamID = rand() % 4;
#pragma acc kernels async(streamID) present(xT, yT, zT, qT, clusterX, clusterY, clusterZ, tt) \
    create(nodeX[0:torderlim],nodeY[0:torderlim],nodeZ[0:torderlim])
    {

    //  Fill in arrays of unique x, y, and z coordinates for the interpolation points.
    #pragma acc loop independent
    for (int i = 0; i < torderlim; i++) {
        nodeX[i] = x0 + (tt[i] + 1.0)/2.0 * (x1 - x0);
        nodeY[i] = y0 + (tt[i] + 1.0)/2.0 * (y1 - y0);
        nodeZ[i] = z0 + (tt[i] + 1.0)/2.0 * (z1 - z0);
    }


    #pragma acc loop independent
    for (int j = 0; j < pointsPerCluster; j++) { // loop over interpolation points, set (cx,cy,cz) for this point
        // compute k1, k2, k3 from j
        int k1 = j%torderlim;
        int kk = (j-k1)/torderlim;
        int k2 = kk%torderlim;
        kk = kk - k2;
        int k3 = kk / torderlim;

        // Fill cluster X, Y, and Z arrays
        clusterX[startingIndexInClusters + j] = nodeX[k1];
        clusterY[startingIndexInClusters + j] = nodeY[k2];
        clusterZ[startingIndexInClusters + j] = nodeZ[k3];
    }

    }

    return;
}












void cc_interaction_list_treecode(struct tnode_array *tree_array, struct particles *source_clusters, struct particles *target_clusters, struct batch *batches,
                                  int *tree_inter_list, int *direct_inter_list,
                                  struct particles *sources, struct particles *targets,
                                  double *tpeng, double *EnP, int numDevices, int numThreads)
{
    for (int i = 0; i < targets->num; i++) EnP[i] = 0.0;

    printf("Performing cluster-batch treecode...\n");

    #pragma omp parallel num_threads(numThreads)
    {

#ifdef OPENACC_ENABLED
        if (omp_get_thread_num()<numDevices)
            acc_set_device_num(omp_get_thread_num(),acc_get_device_type());
#endif

        int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
        if (this_thread==0){printf("numDevices: %i\n", numDevices);}
        if (this_thread==0){printf("num_threads: %i\n", num_threads);}
        printf("this_thread: %i\n", this_thread);


        double *xS = sources->x;
        double *yS = sources->y;
        double *zS = sources->z;
        double *qS = sources->q;
        double *wS = sources->w;

        double *xT = targets->x;
        double *yT = targets->y;
        double *zT = targets->z;
        double *qT = targets->q;

        double *xCS = source_clusters->x;
        double *yCS = source_clusters->y;
        double *zCS = source_clusters->z;
        double *qCS = source_clusters->q;
        
        double *xCT = target_clusters->x;
        double *yCT = target_clusters->y;
        double *zCT = target_clusters->z;
        double *qCT = target_clusters->q;

        int *ibegs = tree_array->ibeg;
        int *iends = tree_array->iend;

    #pragma acc data copyin(xS[0:sources->num], yS[0:sources->num], zS[0:sources->num], \
                            qS[0:sources->num], wS[0:sources->num], \
                            xT[0:targets->num], yT[0:targets->num], zT[0:targets->num], qT[0:targets->num], \
                            xCS[0:source_clusters->num], yCS[0:source_clusters->num], zCS[0:source_clusters->num], \
                            qCS[0:source_clusters->num], xCT[0:target_clusters->num], yCT[0:target_clusters->num], \
                            zCT[0:target_clusters->num]) \
                            copy(qCT[0:target_clusters->num], EnP[0:targets->num])
        {

        int numberOfInterpolationPoints = torderlim*torderlim*torderlim;

        #pragma omp for 
        for (int i = 0; i < batches->num; i++) {
            int batch_ibeg = batches->index[i][0];
            int batch_iend = batches->index[i][1];
            
            int numberOfClusterApproximations = batches->index[i][2];
            int numberOfDirectSums = batches->index[i][3];

            int numberOfTargets = batch_iend - batch_ibeg + 1;
            int batchStart =  batch_ibeg - 1;
            int targetClusterStart  = numberOfInterpolationPoints * i;
            

            for (int j = 0; j < numberOfClusterApproximations; j++) {
                int node_index = tree_inter_list[i * numnodes + j];
                int sourceClusterStart = numberOfInterpolationPoints*node_index;

                int streamID = j%3;
                #pragma acc kernels async(streamID)
                {
                #pragma acc loop independent
                for (int jj = 0; jj < numberOfInterpolationPoints; jj++) {
                    double tempPotential = 0.0;
                    double xi = xCT[targetClusterStart + jj];
                    double yi = yCT[targetClusterStart + jj];
                    double zi = zCT[targetClusterStart + jj];

                    #pragma acc loop reduction(+:tempPotential)
                    for (int ii = 0; ii < numberOfInterpolationPoints; ii++) {
                        // Compute x, y, and z distances between target i and interpolation point j
                        double dxt = xCS[sourceClusterStart + ii] - xi;
                        double dyt = yCS[sourceClusterStart + ii] - yi;
                        double dzt = zCS[sourceClusterStart + ii] - zi;
                        tempPotential += qCS[sourceClusterStart + ii] / sqrt(dxt*dxt + dyt*dyt + dzt*dzt);

                    }
                    #pragma acc atomic
                    qCT[targetClusterStart + jj] += tempPotential;
                }
                } // end kernel
            } // end for loop over cluster approximations

            for (int j = 0; j < numberOfDirectSums; j++) {

                int node_index = direct_inter_list[i * numleaves + j];

                int source_start=ibegs[node_index]-1;
                int source_end=iends[node_index];

                int streamID = j%3;
                # pragma acc kernels async(streamID)
                {
                #pragma acc loop independent
                for (int ii = batchStart; ii < batchStart+numberOfTargets; ii++) {
                    double d_peng = 0.0;
                    double xtgt = xT[ii];
                    double ytgt = yT[ii];
                    double ztgt = zT[ii];

                    #pragma acc loop reduction(+:d_peng)
                    for (int jj = source_start; jj < source_end; jj++) {
                        double tx = xS[jj] - xtgt;
                        double ty = yS[jj] - ytgt;
                        double tz = zS[jj] - ztgt;
                        double r = sqrt(tx*tx + ty*ty + tz*tz);

                        if (r > DBL_MIN) {
                            d_peng += qS[jj] * wS[jj] / r;
                        }
                    }

                    #pragma acc atomic
                    EnP[ii] += d_peng;
                }
                } // end kernel
            } // end loop over number of leaves
        } // end loop over target batches

        #pragma acc wait
        #pragma omp barrier
        } // end acc data region
        } // end omp parallel region

        return;

    } /* END of function pc_treecode */






void cc_compute_batch_interp_interactions(struct batch *batches, struct particles *target_clusters,
    struct particles *targets, double *tpeng, double *EnP,
    int numDevices, int numThreads)
{

    printf("Performing cluster-batch interpolation...\n");

    #pragma omp parallel num_threads(numThreads)
    {

#ifdef OPENACC_ENABLED
    if (omp_get_thread_num()<numDevices)
        acc_set_device_num(omp_get_thread_num(),acc_get_device_type());
#endif

    int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
    if (this_thread==0){printf("numDevices: %i\n", numDevices);}
    if (this_thread==0){printf("num_threads: %i\n", num_threads);}
    printf("this_thread: %i\n", this_thread);


    double *xT = targets->x;
    double *yT = targets->y;
    double *zT = targets->z;
    double *qT = targets->q;

    double *xCT = target_clusters->x;
    double *yCT = target_clusters->y;
    double *zCT = target_clusters->z;
    double *qCT = target_clusters->q;

    int **idxs = batches->index;
    
    int clusterNum = target_clusters->num;
    int targetNum = targets->num;

    #pragma acc data copyin(tt[0:torderlim], xT[0:targetNum], yT[0:targetNum], zT[0:targetNum], qCT[0:clusterNum]) \
                            copy(EnP[0:targetNum])
    {

    #pragma omp for schedule(guided)
    for (int idx = 0; idx < batches->num; idx++) {
    
        int pointsPerCluster = torderlim*torderlim*torderlim;
        int pointsInNode = idxs[idx][1] - idxs[idx][0] + 1;
        int startingIndexInClusters = idx * pointsPerCluster;
        int startingIndexInTargets = idxs[idx][0]-1;
        
        double nodeX[torderlim], nodeY[torderlim], nodeZ[torderlim];
        double weights[torderlim], dj[torderlim];
        
        
        double x0 = batches->x_min[idx];
        double x1 = batches->x_max[idx];
        double y0 = batches->y_min[idx];
        double y1 = batches->y_max[idx];
        double z0 = batches->z_min[idx];
        double z1 = batches->z_max[idx];
        
        int streamID = rand() % 4;
        #pragma acc kernels async(streamID) present(xT, yT, zT, qCT, tt) \
            create(nodeX[0:torderlim], nodeY[0:torderlim], nodeZ[0:torderlim], weights[0:torderlim], dj[0:torderlim])
        {
        
        //  Fill in arrays of unique x, y, and z coordinates for the interpolation points.
        #pragma acc loop independent
        for (int i = 0; i < torderlim; i++) {
            nodeX[i] = x0 + (tt[i] + 1.0)/2.0 * (x1 - x0);
            nodeY[i] = y0 + (tt[i] + 1.0)/2.0 * (y1 - y0);
            nodeZ[i] = z0 + (tt[i] + 1.0)/2.0 * (z1 - z0);
        }
        
        // Compute weights
        #pragma acc loop independent
        for (int j = 0; j < torder+1; j++){
            dj[j] = 1.0;
            if (j==0) dj[j] = 0.5;
            if (j==torder) dj[j]=0.5;
        }

        #pragma acc loop independent
        for (int j = 0; j < torderlim; j++) {
            weights[j] = ((j % 2 == 0)? 1 : -1) * dj[j];
        }
        
        #pragma acc loop gang independent
        for (int i = 0; i < pointsInNode; i++) { // loop through the target points

            double sumX=0.0;
            double sumY=0.0;
            double sumZ=0.0;

            double tx = xT[startingIndexInTargets+i];
            double ty = yT[startingIndexInTargets+i];
            double tz = zT[startingIndexInTargets+i];

            int eix=-1;
            int eiy=-1;
            int eiz=-1;

            #pragma acc loop vector reduction(+:sumX) reduction(+:sumY) reduction(+:sumZ)
            for (int j = 0; j < torderlim; j++) {  // loop through the degree

                double cx = tx-nodeX[j];
                double cy = ty-nodeY[j];
                double cz = tz-nodeZ[j];

                if (fabs(cx)<DBL_MIN) eix=j;
                if (fabs(cy)<DBL_MIN) eiy=j;
                if (fabs(cz)<DBL_MIN) eiz=j;

                // Increment the sums
                double w = weights[j];
                sumX += w / cx;
                sumY += w / cy;
                sumZ += w / cz;

            }

            double denominator = 1.0;
            if (eix==-1) denominator /= sumX;
            if (eiy==-1) denominator /= sumY;
            if (eiz==-1) denominator /= sumZ;
            
            double temp = 0.0;
            
            #pragma acc loop vector reduction(+:temp)
            for (int j = 0; j < pointsPerCluster; j++) { // loop over interpolation points, set (cx,cy,cz) for this point

                int k1 = j%torderlim;
                int kk = (j-k1)/torderlim;
                int k2 = kk%torderlim;
                kk = kk - k2;
                int k3 = kk / torderlim;

                double w3 = weights[k3];
                double w2 = weights[k2];
                double w1 = weights[k1];
                
                double cx = nodeX[k1];
                double cy = nodeY[k2];
                double cz = nodeZ[k3];
                double cq = qCT[startingIndexInClusters + j];
            
                double numerator = 1.0;

                // If exactInd[i] == -1, then no issues.
                // If exactInd[i] != -1, then we want to zero out terms EXCEPT when exactInd=k1.
                if (eix==-1) {
                    numerator *=  w1 / (tx - cx);
                } else {
                    if (eix!=k1) numerator *= 0;
                }

                if (eiy==-1) {
                    numerator *=  w2 / (ty - cy);
                } else {
                    if (eiy!=k2) numerator *= 0;
                }

                if (eiz==-1) {
                    numerator *=  w3 / (tz - cz);
                } else {
                    if (eiz!=k3) numerator *= 0;
                }

                temp += numerator * denominator * cq;
            }
            
#ifdef OPENACC_ENABLED
            #pragma acc atomic
#else
            #pragma omp atomic
#endif
            EnP[i+startingIndexInTargets] += temp;
        }        
        } //end ACC kernels
        
    } //end loop over nodes
    #pragma acc wait
    } //end ACC data region
    } //end OMP parallel region
    
    *tpeng = sum(EnP, targets->num);

    return;
}
