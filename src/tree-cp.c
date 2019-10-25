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


/* Definition of variables declared extern in globvars.h */
double *cf = NULL;
double *cf1 = NULL;
double *cf2 = NULL;
double ***b1 = NULL;

int *orderarr = NULL;

int torder, torderlim, torderflat;
int minlevel, maxlevel;
int maxpars, minpars;
int numleaves;

int numnodes;

double tarpos[3];
double thetasq, tarposq;

/* variables used by Yukawa treecode */
double *cf3 = NULL;
double ***a1 = NULL;

/* variable used by kernel independent moment computation */
double *tt, *ww;

double *unscaledQuadratureWeights;




void setup(struct particles *particles, int order, double theta,
           double *xyzminmax)
{
    /* local variables */
    int i;
    double t1, xx;

    /* changing values of our extern variables */
    torder = order;
    torderlim = torder + 1;
    thetasq = theta * theta;
    torderflat = torderlim * (torderlim + 1) * (torderlim + 2) / 6;

    /* allocating global Taylor expansion variables */
    make_vector(cf, torder+1);
    make_vector(cf1, torderlim);
    make_vector(cf2, torderlim);

    make_3array(b1, torderlim, torderlim, torderlim);
    
    make_vector(tt, torderlim);
    make_vector(ww, torderlim);
    

    /* initializing array for Chev points */
    for (i = 0; i < torderlim; i++)
        tt[i] = cos(i * M_PI / torder);

    ww[0] = 0.25 * (torder*torder/3.0 + 1.0/6.0);
    ww[torder] = -ww[0];

    for (i = 1; i < torder; i++) {
        xx = i * M_PI / torder;
        ww[i] = -cos(xx) / (2 * sin(xx) * sin(xx));
    }

    /* initializing arrays for Taylor sums and coefficients */
    for (i = 0; i < torder + 1; i++)
        cf[i] = -i + 1.0;

    for (i = 0; i < torderlim; i++) {
        t1 = 1.0 / (i + 1.0);
        cf1[i] = 1.0 - (0.5 * t1);
        cf2[i] = 1.0 - t1;
    }

    /* find bounds of Cartesian box enclosing the particles */
    xyzminmax[0] = minval(particles->x, particles->num);
    xyzminmax[1] = maxval(particles->x, particles->num);
    xyzminmax[2] = minval(particles->y, particles->num);
    xyzminmax[3] = maxval(particles->y, particles->num);
    xyzminmax[4] = minval(particles->z, particles->num);
    xyzminmax[5] = maxval(particles->z, particles->num);

    make_vector(orderarr, particles->num);
    for (i = 0; i < particles->num; i++)
        orderarr[i] = i+1;

    return;
    
} /* END of function setup */




void cp_create_tree_n0(struct tnode **p, struct particles *targets,
                       int ibeg, int iend, int maxparnode,
                       double *xyzmm, int level)
{
    /*local variables*/
    double x_mid, y_mid, z_mid, xl, yl, zl, lmax, t1, t2, t3;
    int i, j, loclev, numposchild, idx;
    
    int ind[8][2];
    double xyzmms[6][8];
    double lxyzmm[6];

    for (i = 0; i < 8; i++) {
        for (j = 0; j < 2; j++) {
            ind[i][j] = 0.0;
        }
    }

    for (i = 0; i < 6; i++) {
        for (j = 0; j < 8; j++) {
            xyzmms[i][j] = 0.0;
        }
    }

    for (i = 0; i < 6; i++) {
        lxyzmm[i] = 0.0;
    }

    (*p) = malloc(sizeof(struct tnode));
    
    
    /* increment number of nodes */
    #pragma omp critical
    numnodes++;

    /* set node fields: number of particles, exist_ms, and xyz bounds */
    (*p)->numpar = iend - ibeg + 1;
    (*p)->exist_ms = 0;

    (*p)->x_min = minval(targets->x + ibeg - 1, (*p)->numpar);
    (*p)->x_max = maxval(targets->x + ibeg - 1, (*p)->numpar);
    (*p)->y_min = minval(targets->y + ibeg - 1, (*p)->numpar);
    (*p)->y_max = maxval(targets->y + ibeg - 1, (*p)->numpar);
    (*p)->z_min = minval(targets->z + ibeg - 1, (*p)->numpar);
    (*p)->z_max = maxval(targets->z + ibeg - 1, (*p)->numpar);
    
    

    /*compute aspect ratio*/
    xl = (*p)->x_max - (*p)->x_min;
    yl = (*p)->y_max - (*p)->y_min;
    zl = (*p)->z_max - (*p)->z_min;
        
    lmax = max3(xl, yl, zl);
    t1 = lmax;
    t2 = min3(xl, yl, zl);


    if (t2 != 0.0)
        (*p)->aspect = t1/t2;
    else
        (*p)->aspect = 0.0;

    
    /*midpoint coordinates, RADIUS and SQRADIUS*/
    (*p)->x_mid = ((*p)->x_max + (*p)->x_min) / 2.0;
    (*p)->y_mid = ((*p)->y_max + (*p)->y_min) / 2.0;
    (*p)->z_mid = ((*p)->z_max + (*p)->z_min) / 2.0;

    t1 = (*p)->x_max - (*p)->x_mid;
    t2 = (*p)->y_max - (*p)->y_mid;
    t3 = (*p)->z_max - (*p)->z_mid;

    (*p)->sqradius = t1*t1 + t2*t2 + t3*t3;
    (*p)->radius = sqrt((*p)->sqradius);

    
    /*set particle limits, tree level of node, and nullify child pointers*/
    (*p)->ibeg = ibeg;
    (*p)->iend = iend;
    (*p)->level = level;

    #pragma omp critical
    {
        if (maxlevel < level) maxlevel = level;
    }

    (*p)->num_children = 0;
    for (i = 0; i < 8; i++)
        (*p)->child[i] = NULL;
    

    if ((*p)->numpar > maxparnode) {
    /*
     * set IND array to 0, and then call PARTITION_8 routine.
     * IND array holds indices of the eight new subregions.
     * Also, setup XYZMMS array in the case that SHRINK = 1.
     */
        xyzmms[0][0] = (*p)->x_min;
        xyzmms[1][0] = (*p)->x_max;
        xyzmms[2][0] = (*p)->y_min;
        xyzmms[3][0] = (*p)->y_max;
        xyzmms[4][0] = (*p)->z_min;
        xyzmms[5][0] = (*p)->z_max;

        ind[0][0] = ibeg;
        ind[0][1] = iend;

        x_mid = (*p)->x_mid;
        y_mid = (*p)->y_mid;
        z_mid = (*p)->z_mid;

        cp_partition_8(targets->x, targets->y, targets->z, targets->q,
                       xyzmms, xl, yl, zl, lmax, &numposchild,
                       x_mid, y_mid, z_mid, ind);

        loclev = level + 1;

        for (i = 0; i < numposchild; i++) {
            if (ind[i][0] <= ind[i][1]) {
            
                (*p)->num_children = (*p)->num_children + 1;
                idx = (*p)->num_children - 1;

                for (j = 0; j < 6; j++)
                    lxyzmm[j] = xyzmms[j][i];
                    
                struct tnode **paddress = &((*p)->child[idx]);

                cp_create_tree_n0(paddress,
                                  targets, ind[i][0], ind[i][1],
                                  maxparnode, lxyzmm, loclev);
            }
        }
        
        #pragma omp taskwait

    } else {
    
        #pragma omp critical
        {
            if (level < minlevel) minlevel = level;
            if (minpars > (*p)->numpar) minpars = (*p)->numpar;
            if (maxpars < (*p)->numpar) maxpars = (*p)->numpar;
        
            numleaves++;
        }
    }
    
    return;

} /* end of function create_tree_n0 */


int cp_set_tree_index(struct tnode *p, int index)
{
        int current_index = index;
        p->node_index = current_index;

        for (int i = 0; i < p->num_children; i++)
            current_index = cp_set_tree_index(p->child[i], current_index + 1);

        return current_index;
}


void cp_partition_8(double *x, double *y, double *z, double *q, double xyzmms[6][8],
                    double xl, double yl, double zl, double lmax, int *numposchild,
                    double x_mid, double y_mid, double z_mid, int ind[8][2])
{

    /* local variables */
    int temp_ind, i, j;
    double critlen;

    *numposchild = 1;
    critlen = lmax / sqrt(2.0);

    if (xl >= critlen) {
        cp_partition(x, y, z, q, orderarr, ind[0][0], ind[0][1],
                     x_mid, &temp_ind);

        ind[1][0] = temp_ind + 1;
        ind[1][1] = ind[0][1];
        ind[0][1] = temp_ind;

        for (i = 0; i < 6; i++)
            xyzmms[i][1] = xyzmms[i][0];
        
        xyzmms[1][0] = x_mid;
        xyzmms[0][1] = x_mid;
        *numposchild = 2 * *numposchild;
    }

    if (yl >= critlen) {
        for (i = 0; i < *numposchild; i++) {
            cp_partition(y, x, z, q, orderarr, ind[i][0], ind[i][1],
                         y_mid, &temp_ind);
                        
            ind[*numposchild + i][0] = temp_ind + 1;
            ind[*numposchild + i][1] = ind[i][1];
            ind[i][1] = temp_ind;

            for (j = 0; j < 6; j++)
                xyzmms[j][*numposchild + i] = xyzmms[j][i];

            xyzmms[3][i] = y_mid;
            xyzmms[2][*numposchild + i] = y_mid;
        }
        
        *numposchild = 2 * *numposchild;
    }

    if (zl >= critlen) {
        for (i = 0; i < *numposchild; i++) {
            cp_partition(z, x, y, q, orderarr, ind[i][0], ind[i][1],
                         z_mid, &temp_ind);
                        
            ind[*numposchild + i][0] = temp_ind + 1;
            ind[*numposchild + i][1] = ind[i][1];
            ind[i][1] = temp_ind;

            for (j = 0; j < 6; j++)
                xyzmms[j][*numposchild + i] = xyzmms[j][i];

            xyzmms[5][i] = z_mid;
            xyzmms[4][*numposchild + i] = z_mid;
        }
        
        *numposchild = 2 * *numposchild;

    }

    return;

} /* END of function cp_partition_8 */



void cp_create_tree_array(struct tnode *p, struct tnode_array *tree_array)
{
    //    printf("Entering pc_create_tree_array.\n");
    int i;

    /*midpoint coordinates, RADIUS and SQRADIUS*/
    tree_array->x_mid[p->node_index] = p->x_mid;
    tree_array->y_mid[p->node_index] = p->y_mid;
    tree_array->z_mid[p->node_index] = p->z_mid;

    tree_array->x_min[p->node_index] = p->x_min;
    tree_array->y_min[p->node_index] = p->y_min;
    tree_array->z_min[p->node_index] = p->z_min;

    tree_array->x_max[p->node_index] = p->x_max;
    tree_array->y_max[p->node_index] = p->y_max;
    tree_array->z_max[p->node_index] = p->z_max;

    tree_array->ibeg[p->node_index] = p->ibeg;
    tree_array->iend[p->node_index] = p->iend;
    tree_array->numpar[p->node_index] = p->numpar;
    tree_array->level[p->node_index] = p->level;
    tree_array->radius[p->node_index] = p->radius;

    for (i = 0; i < p->num_children; i++) {
        cp_create_tree_array(p->child[i], tree_array);
    }

    return;

} /* END of function create_tree_n0 */






void cp_make_interaction_list(const struct tnode_array *tree_array, struct batch *batches,
                              int *tree_inter_list, int *direct_inter_list)
{
    /* local variables */
    int i, j;

    int **batches_ind;
    double **batches_center;
    double *batches_radius;
    
    int tree_numnodes;
    const int *tree_numpar, *tree_level;
    const double *tree_x_mid, *tree_y_mid, *tree_z_mid, *tree_radius;

    batches_ind = batches->index;
    batches_center = batches->center;
    batches_radius = batches->radius;

    tree_numnodes = tree_array->numnodes;
    tree_numpar = tree_array->numpar;
    tree_level = tree_array->level;
    tree_radius = tree_array->radius;
    tree_x_mid = tree_array->x_mid;
    tree_y_mid = tree_array->y_mid;
    tree_z_mid = tree_array->z_mid;

    for (i = 0; i < batches->num * numnodes; i++)
        tree_inter_list[i] = -1;

    for (i = 0; i < batches->num * numleaves; i++)
        direct_inter_list[i] = -1;
    
    for (i = 0; i < batches->num; i++)
        cp_compute_interaction_list(tree_numnodes, tree_level, tree_numpar,
                tree_radius, tree_x_mid, tree_y_mid, tree_z_mid,
                batches_ind[i], batches_center[i], batches_radius[i],
                &(tree_inter_list[i*numnodes]), &(direct_inter_list[i*numleaves]));

    return;

} /* END of function pc_treecode */



void cp_compute_interaction_list(int tree_numnodes, const int *tree_level,
                const int *tree_numpar, const double *tree_radius,
                const double *tree_x_mid, const double *tree_y_mid, const double *tree_z_mid,
                int *batch_ind, double *batch_mid, double batch_rad,
                int *batch_tree_list, int *batch_direct_list)
{
    /* local variables */
    double tx, ty, tz, dist;
    int j, current_level;
    int tree_index_counter, direct_index_counter;
    
    tree_index_counter = 0;
    direct_index_counter = 0;
    current_level = 0;
    
    for (j = 0; j < tree_numnodes; j++) {
        if (tree_level[j] <= current_level) {

            /* determine DIST for MAC test */
            tx = batch_mid[0] - tree_x_mid[j];
            ty = batch_mid[1] - tree_y_mid[j];
            tz = batch_mid[2] - tree_z_mid[j];
            dist = sqrt(tx*tx + ty*ty + tz*tz);

            if (((tree_radius[j] + batch_rad) < dist * sqrt(thetasq))
              && (tree_radius[j] > 0.00)
              && (torder*torder*torder < tree_numpar[j])) {

              current_level = tree_level[j];
            /*
             * If MAC is accepted and there is more than 1 particle
             * in the box, use the expansion for the approximation.
             */
        
                batch_tree_list[tree_index_counter] = j;
                tree_index_counter++;
        
            } else {
            /*
             * If MAC fails check to see if there are children. If not, perform direct
             * calculation. If there are children, call routine recursively for each.
             */

                if ( (j==tree_numnodes-1) || (tree_level[j+1] <= tree_level[j]) ) {
                    
                    batch_direct_list[direct_index_counter] = j;
                    direct_index_counter++;
            
                } else {
                    
                    current_level = tree_level[j+1];
                    
                }
            }
        }
    }

    // Setting tree and direct index counter for batch
    batch_ind[2] = tree_index_counter;
    batch_ind[3] = direct_index_counter;
    
    return;
}












void cp_fill_cluster_interp(struct particles *clusters, struct particles *targets, struct tnode *troot, int order, int numDevices, int numThreads, struct tnode_array *tree_array)
{
    int pointsPerCluster = (order+1)*(order+1)*(order+1);
    int numInterpPoints = numnodes * pointsPerCluster;
    make_vector(clusters->x, numInterpPoints);
    make_vector(clusters->y, numInterpPoints);
    make_vector(clusters->z, numInterpPoints);
    make_vector(clusters->q, numInterpPoints);
    make_vector(clusters->w, numInterpPoints);
    clusters->num=numInterpPoints;

    for (int i = 0; i < numInterpPoints; i++) {
        clusters->q[i]=0.0;
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
        make_vector(tempX,clusters->num);
        make_vector(tempY,clusters->num);
        make_vector(tempZ,clusters->num);

        for (int i = 0; i < clusters->num; i++) {
            tempX[i] = 0.0;
            tempY[i] = 0.0;
            tempZ[i] = 0.0;
        }

        double *xT = targets->x;
        double *yT = targets->y;
        double *zT = targets->z;
        double *qT = targets->q;

        double *xC = clusters->x;
        double *yC = clusters->y;
        double *zC = clusters->z;

        int clusterNum = clusters->num;
        int targetNum = targets->num;
        int pointsPerCluster = torderlim * torderlim * torderlim;

#pragma acc data copyin(tt[0:torderlim], \
        xT[0:targetNum], yT[0:targetNum], zT[0:targetNum], qT[0:targetNum]) \
        copy(tempX[0:clusterNum], tempY[0:clusterNum], tempZ[0:clusterNum])
        {
            #pragma omp for schedule(guided)
            for (int i = 1; i < numnodes; i++) {  // start from i=1, don't need to compute root moments
                int startingIndexInClusters = pointsPerCluster * i;
                cp_comp_interp(tree_array, i, xT, yT, zT, qT,
                                     tempX, tempY, tempZ);
            }
            #pragma acc wait
        } // end ACC DATA REGION

        for (int j = 0; j < clusters->num; j++)
        {
            if(tempX[j] != 0.0) clusters->x[j] = tempX[j];
            if(tempY[j] != 0.0) clusters->y[j] = tempY[j];
            if(tempZ[j] != 0.0) clusters->z[j] = tempZ[j];
        }

        free_vector(tempX);
        free_vector(tempY);
        free_vector(tempZ);

    } // end OMP PARALLEL REGION

    return;
}




void cp_comp_interp(struct tnode_array *tree_array, int idx,
        double *xT, double *yT, double *zT, double *qT,
        double *clusterX, double *clusterY, double *clusterZ)
{
    int pointsPerCluster = torderlim*torderlim*torderlim;
    int startingIndexInClusters = idx * pointsPerCluster;

    double x0, x1, y0, y1, z0, z1;  // bounding box

    double nodeX[torderlim], nodeY[torderlim], nodeZ[torderlim];
    

    x0 = tree_array->x_min[idx];  // 1e-15 fails for large meshes, mysteriously.
    x1 = tree_array->x_max[idx];
    y0 = tree_array->y_min[idx];
    y1 = tree_array->y_max[idx];
    z0 = tree_array->z_min[idx];
    z1 = tree_array->z_max[idx];


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














void cp_interaction_list_treecode(struct tnode_array *tree_array, struct particles *clusters, struct batch *batches,
                                  int *tree_inter_list, int *direct_inter_list,
                                  struct particles *sources, struct particles *targets,
                                  double *tpeng, double *EnP, int numDevices, int numThreads)
{
    for (int i = 0; i < targets->num; i++)
        EnP[i] = 0.0;

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

        double *xC = clusters->x;
        double *yC = clusters->y;
        double *zC = clusters->z;
        double *qC = clusters->q;

        int *ibegs = tree_array->ibeg;
        int *iends = tree_array->iend;

    #pragma acc data copyin(xS[0:sources->num], yS[0:sources->num], zS[0:sources->num], \
                            qS[0:sources->num], wS[0:sources->num], \
                            xT[0:targets->num], yT[0:targets->num], zT[0:targets->num], qT[0:targets->num], \
                            xC[0:clusters->num], yC[0:clusters->num], zC[0:clusters->num]) \
                            copy(qC[0:clusters->num], EnP[0:targets->num])
        {

        double temp_i[torderlim], temp_j[torderlim], temp_k[torderlim];
        int numberOfInterpolationPoints = torderlim*torderlim*torderlim;

        #pragma omp for
        for (int i = 0; i < batches->num; i++) {
            int batch_ibeg = batches->index[i][0];
            int batch_iend = batches->index[i][1];
            int numberOfClusterApproximations = batches->index[i][2];
            int numberOfDirectSums = batches->index[i][3];

            int numberOfSources = batch_iend - batch_ibeg + 1;
            int batchStart =  batch_ibeg - 1;

            for (int j = 0; j < numberOfClusterApproximations; j++) {
                int node_index = tree_inter_list[i * numnodes + j];
                int clusterStart = numberOfInterpolationPoints*node_index;

                int streamID = j%3;
                #pragma acc kernels async(streamID)
                {
                #pragma acc loop independent
                for (int jj = 0; jj < numberOfInterpolationPoints; jj++) {
                    double tempPotential = 0.0;
                    double xi = xC[clusterStart + jj];
                    double yi = yC[clusterStart + jj];
                    double zi = zC[clusterStart + jj];

                    #pragma acc loop reduction(+:tempPotential)
                    for (int ii = 0; ii < numberOfSources; ii++) {
                        // Compute x, y, and z distances between target i and interpolation point j
                        double dxt = xS[batchStart + ii] - xi;
                        double dyt = yS[batchStart + ii] - yi;
                        double dzt = zS[batchStart + ii] - zi;
                        tempPotential += qS[batchStart + ii] / sqrt(dxt*dxt + dyt*dyt + dzt*dzt);

                    }
                    #pragma acc atomic
                    qC[clusterStart + jj] += tempPotential;
                }
                } // end kernel
            } // end for loop over cluster approximations

            for (int j = 0; j < numberOfDirectSums; j++) {

                int node_index = direct_inter_list[i * numleaves + j];

                int target_start=ibegs[node_index]-1;
                int target_end=iends[node_index];

                int streamID = j%3;
                # pragma acc kernels async(streamID)
                {
                #pragma acc loop independent
                for (int ii = batchStart; ii < batchStart+numberOfSources; ii++) {
                    double d_peng = 0.0;
                    double xsrc = xS[ii];
                    double ysrc = yS[ii];
                    double zsrc = zS[ii];
                    double qwsrc = qS[ii]*wS[ii];

                    #pragma acc loop reduction(+:d_peng)
                    for (int jj = target_start; jj < target_end; jj++) {
                        double tx = xsrc - xT[jj];
                        double ty = ysrc - yT[jj];
                        double tz = zsrc - zT[jj];
                        double r = sqrt(tx*tx + ty*ty + tz*tz);

                        if (r > DBL_MIN) {
                            d_peng += qwsrc / r;
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





void cp_compute_tree_interactions(struct tnode_array *tree_array, struct particles *clusters,
    struct particles *targets, double *tpeng, double *EnP,
    int numDevices, int numThreads)
{

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

    double *xC = clusters->x;
    double *yC = clusters->y;
    double *zC = clusters->z;
    double *qC = clusters->q;

    int *ibegs = tree_array->ibeg;
    int *iends = tree_array->iend;
    
    int clusterNum = clusters->num;
    int targetNum = targets->num;

    #pragma acc data copyin(tt[0:torderlim], xT[0:targetNum], yT[0:targetNum], zT[0:targetNum], qC[0:clusterNum]) \
                            copy(EnP[0:targetNum])
    {

    #pragma omp for schedule(guided)
    for (int idx = 1; idx < numnodes; idx++) {
    
        int pointsPerCluster = torderlim*torderlim*torderlim;
        int pointsInNode = iends[idx] - ibegs[idx] + 1;
        int startingIndexInClusters = idx * pointsPerCluster;
        int startingIndexInTargets = ibegs[idx]-1;
        
        double nodeX[torderlim], nodeY[torderlim], nodeZ[torderlim];
        double weights[torderlim], dj[torderlim];
        
        double sumX, sumY, sumZ;
        double tx, ty, tz, cx, cy, cz, cq;
        double denominator, numerator, xn, yn, zn;
        int k1, k2, k3, kk;
        double w1, w2, w3, w;
        
        double x0 = tree_array->x_min[idx];
        double x1 = tree_array->x_max[idx];
        double y0 = tree_array->y_min[idx];
        double y1 = tree_array->y_max[idx];
        double z0 = tree_array->z_min[idx];
        double z1 = tree_array->z_max[idx];

        int streamID = rand() % 4;
        #pragma acc kernels async(streamID) present(xT, yT, zT, qC, tt) \
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
        for (int j = 0; j < torderlim; j++){
            dj[j] = 1.0;
            if (j==0) dj[j] = 0.5;
            if (j==torder) dj[j] = 0.5;
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

            int eix = -1;
            int eiy = -1;
            int eiz = -1;

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
                sumX += w / (cx);
                sumY += w / (cy);
                sumZ += w / (cz);

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
                double cq = qC[startingIndexInClusters + j];
            
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




/*
 * cleanup deallocates allocated global variables and then calls
 * recursive function remove_node to delete the tree
 */
void cleanup(struct tnode *p)
{
    free_vector(cf);
    free_vector(cf1);
    free_vector(cf2);
    free_vector(cf3);
    
    free_3array(b1);
    free_3array(a1);

    free_vector(orderarr);

    remove_node(p);
    free(p);

    return;

} /* END function cleanup */




void remove_node(struct tnode *p)
{
    /* local variables */
    int i;

//    if (p->exist_ms == 1)
//        free(p->ms);
//    	free(p->ms2);

    if (p->num_children > 0) {
        for (i = 0; i < p->num_children; i++) {
            remove_node(p->child[i]);
            free(p->child[i]);
        }
    }

    return;

} /* END function remove_node */








void compute_cp2(struct tnode *ap, double *x, double *y, double *z,
                 double *EnP)
{
    /* local variables */
    double tx, ty, peng;
    double xm, ym, zm, dx, dy, dz;
    int i, nn, j, k1, k2, k3, kk, porder, porder1;

    porder = torder;
    porder1 = porder - 1;

    if (ap->exist_ms == 1) {
        xm = ap->x_mid;
        ym = ap->y_mid;
        zm = ap->z_mid;

        for (i = ap->ibeg-1; i < ap->iend; i++) {
            nn = orderarr[i];
            dx = x[i] - xm;
            dy = y[i] - ym;
            dz = z[i] - zm;

            kk=0;
            peng = ap->ms[kk];

            for (k3 = porder1; k3 > -1; k3--) {
                ty = ap->ms[++kk];

                for (k2 = porder1 - k3; k2 > -1; k2--) {
                    tx = ap->ms[++kk];

                    for (k1 = porder1 - k3 - k2; k1 > -1; k1--) {
                        tx = dx*tx + ap->ms[++kk];
                    }

                    ty = dy * ty + tx;
                }

                peng = dz * peng + ty;
            }

            EnP[nn-1] = EnP[nn-1] + peng;
        }
    }

    for (j = 0; j < ap->num_children; j++)
        compute_cp2(ap->child[j], x, y, z, EnP);

    return;

} /* END function compute_cp2 */


void cp_comp_ms(struct tnode *p)
{
    int k1, k2, k3, kk=-1;

    for (k3 = torder; k3 > -1; k3--) {
        for (k2 = torder - k3; k2 > -1; k2--) {
            for (k1 = torder - k3 - k2; k1 > -1; k1--) {
                kk++;
                p->ms[kk] = p->ms[kk]
                          + tarposq * b1[k1][k2][k3];
            }
        }
    }

    return;

} /* END function comp_cms */
