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
                                     
                for (int j = 0; j < pointsPerCluster; j++)
                {
                    target_clusters->x[startingIndexInClusters + j] = tempX[startingIndexInClusters + j];
                    target_clusters->y[startingIndexInClusters + j] = tempY[startingIndexInClusters + j];
                    target_clusters->z[startingIndexInClusters + j] = tempZ[startingIndexInClusters + j];
                }
            }
            #pragma acc wait
        } // end ACC DATA REGION

        int counter=0;
//        for (int j = 0; j < clusters->num; j++)
//        {
//            clusters->x[j] = tempX[j];
//            clusters->y[j] = tempY[j];
//            clusters->z[j] = tempZ[j];
//        }

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
    int i,j,k;
    int pointsPerCluster = torderlim*torderlim*torderlim;
    int startingIndexInClusters = idx * pointsPerCluster;

    double x0, x1, y0, y1, z0, z1;  // bounding box
    int k1, k2, k3, kk;

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
        k1 = j%torderlim;
        kk = (j-k1)/torderlim;
        k2 = kk%torderlim;
        kk = kk - k2;
        k3 = kk / torderlim;

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
    int i, j;

    for (i = 0; i < targets->num; i++)
        EnP[i] = 0.0;

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
                            xCS[0:source_clusters->num], yCS[0:source_clusters->num], zCS[0:source_clusters->num], qCS[0:source_clusters->num], xCT[0:target_clusters->num], yCT[0:target_clusters->num], zCT[0:target_clusters->num]) \
                            copy(qCT[0:target_clusters->num], EnP[0:targets->num])
        {

        int batch_ibeg, batch_iend, node_index;
        double dist;
        double tx, ty, tz;
        int i, j, k, ii, jj;
        double dxt,dyt,dzt,tempPotential;
        double temp_i[torderlim], temp_j[torderlim], temp_k[torderlim];

        int target_start, target_end;

        double d_peng, r, xi, yi, zi;

        int numberOfSources, numberOfTargets;
        int numberOfInterpolationPoints = torderlim*torderlim*torderlim;
        int sourceClusterStart, targetClusterStart, batchStart;
        
        int source_start, source_end;

        int numberOfClusterApproximations, numberOfDirectSums;
        int streamID;

        #pragma omp for private(i,j,ii,jj,batch_ibeg,batch_iend,source_start,source_end,numberOfClusterApproximations,\
                                numberOfDirectSums,numberOfTargets,batchStart,node_index,sourceClusterStart,targetClusterStart,streamID)
        for (int i = 0; i < batches->num; i++) {
            batch_ibeg = batches->index[i][0];
            batch_iend = batches->index[i][1];
            
            numberOfClusterApproximations = batches->index[i][2];
            numberOfDirectSums = batches->index[i][3];

            numberOfTargets = batch_iend - batch_ibeg + 1;
            batchStart =  batch_ibeg - 1;
            targetClusterStart  = numberOfInterpolationPoints * i;
            

            for (int j = 0; j < numberOfClusterApproximations; j++) {
                node_index = tree_inter_list[i * numnodes + j];
                sourceClusterStart = numberOfInterpolationPoints*node_index;

                streamID = j%3;
                #pragma acc kernels async(streamID)
                {
                #pragma acc loop independent
                for (int jj = 0; jj < numberOfInterpolationPoints; jj++) {
                    tempPotential = 0.0;
                    xi = xCT[targetClusterStart + jj];
                    yi = yCT[targetClusterStart + jj];
                    zi = zCT[targetClusterStart + jj];

                    for (int ii = 0; ii < numberOfInterpolationPoints; ii++) {
                        // Compute x, y, and z distances between target i and interpolation point j
                        dxt = xCS[sourceClusterStart + ii] - xi;
                        dyt = yCS[sourceClusterStart + ii] - yi;
                        dzt = zCS[sourceClusterStart + ii] - zi;
                        tempPotential += qCS[sourceClusterStart + ii] / sqrt(dxt*dxt + dyt*dyt + dzt*dzt);

                    }
                    #pragma acc atomic
                    qCT[targetClusterStart + jj] += tempPotential;
                }
                } // end kernel
            } // end for loop over cluster approximations

            for (j = 0; j < numberOfDirectSums; j++) {

                node_index = direct_inter_list[i * numleaves + j];

                source_start=ibegs[node_index]-1;
                source_end=iends[node_index];

                streamID = j%3;
                # pragma acc kernels async(streamID)
                {
                #pragma acc loop independent
                for (ii = batchStart; ii < batchStart+numberOfTargets; ii++) {
                    d_peng = 0.0;

                    for (jj = source_start; jj < source_end; jj++) {
                        tx = xS[jj] - xT[ii];
                        ty = yS[jj] - yT[ii];
                        tz = zS[jj] - zT[ii];
                        r = sqrt(tx*tx + ty*ty + tz*tz);

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

    #pragma acc data copyin(tt[0:torderlim], xCT[0:targetNum], yCT[0:targetNum], zCT[0:targetNum], qCT[0:clusterNum]) \
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
        int *exactIndX, *exactIndY, *exactIndZ;
        
        double sumX, sumY, sumZ;
        double tx, ty, tz, cx, cy, cz, cq;
        double denominator, numerator, xn, yn, zn, temp;
        int k1, k2, k3, kk;
        double w1, w2, w3, w;
        
        double x0 = batches->x_min[idx];
        double x1 = batches->x_max[idx];
        double y0 = batches->y_min[idx];
        double y1 = batches->y_max[idx];
        double z0 = batches->z_min[idx];
        double z1 = batches->z_max[idx];
        
        make_vector(exactIndX, pointsInNode);
        make_vector(exactIndY, pointsInNode);
        make_vector(exactIndZ, pointsInNode);

        int streamID = rand() % 4;
        #pragma acc kernels async(streamID) present(xT, yT, zT, qCT, tt) \
            create(exactIndX[0:pointsInNode], exactIndY[0:pointsInNode], exactIndZ[0:pointsInNode], \
            nodeX[0:torderlim], nodeY[0:torderlim], nodeZ[0:torderlim], weights[0:torderlim], dj[0:torderlim])
        {
        
        #pragma acc loop independent
        for (int j = 0; j < pointsInNode; j++) {
            exactIndX[j] = -1;
            exactIndY[j] = -1;
            exactIndZ[j] = -1;
        }

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
        
        #pragma acc loop independent
        for (int i = 0; i < pointsInNode; i++) { // loop through the target points

            sumX=0.0;
            sumY=0.0;
            sumZ=0.0;

            tx = xT[startingIndexInTargets+i];
            ty = yT[startingIndexInTargets+i];
            tz = zT[startingIndexInTargets+i];

            #pragma acc loop independent
            for (int j = 0; j < torderlim; j++) {  // loop through the degree

                cx = tx-nodeX[j];
                cy = ty-nodeY[j];
                cz = tz-nodeZ[j];

                if (fabs(cx)<DBL_MIN) exactIndX[i]=j;
                if (fabs(cy)<DBL_MIN) exactIndY[i]=j;
                if (fabs(cz)<DBL_MIN) exactIndZ[i]=j;

                // Increment the sums
                w = weights[j];
                sumX += w / (cx);
                sumY += w / (cy);
                sumZ += w / (cz);

            }

            denominator = 1.0;
            if (exactIndX[i]==-1) denominator /= sumX;
            if (exactIndY[i]==-1) denominator /= sumY;
            if (exactIndZ[i]==-1) denominator /= sumZ;
            
            temp = 0.0;
            
            #pragma acc loop independent
            for (int j = 0; j < pointsPerCluster; j++) { // loop over interpolation points, set (cx,cy,cz) for this point

                k1 = j%torderlim;
                kk = (j-k1)/torderlim;
                k2 = kk%torderlim;
                kk = kk - k2;
                k3 = kk / torderlim;

                w3 = weights[k3];
                w2 = weights[k2];
                w1 = weights[k1];
                
                cx = nodeX[k1];
                cy = nodeY[k2];
                cz = nodeZ[k3];
                cq = qCT[startingIndexInClusters + j];
            
                numerator = 1.0;

                // If exactInd[i] == -1, then no issues.
                // If exactInd[i] != -1, then we want to zero out terms EXCEPT when exactInd=k1.
                if (exactIndX[i]==-1) {
                    numerator *=  w1 / (tx - cx);
                } else {
                    if (exactIndX[i]!=k1) numerator *= 0;
                }

                if (exactIndY[i]==-1) {
                    numerator *=  w2 / (ty - cy);
                } else {
                    if (exactIndY[i]!=k2) numerator *= 0;
                }

                if (exactIndZ[i]==-1) {
                    numerator *=  w3 / (tz - cz);
                } else {
                    if (exactIndZ[i]!=k3) numerator *= 0;
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
        
        free_vector(exactIndX);
        free_vector(exactIndY);
        free_vector(exactIndZ);
    } //end loop over nodes
    } //end ACC data region
    } //end OMP parallel region
    
    *tpeng = sum(EnP, targets->num);

    return;
}
