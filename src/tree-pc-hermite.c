/*
 *Procedures for Particle-Cluster Treecode
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "array.h"
#include "globvars.h"
#include "tnode.h"
#include "batch.h"
#include "particles.h"
#include "tools.h"

#include "partition.h"
#include "tree.h"

#include "mkl.h"
#include <omp.h>


void fill_in_cluster_data_hermite(struct particles *clusters, struct particles *sources, struct tnode *troot, int order){

	int pointsPerCluster = (order+1)*(order+1)*(order+1);
	int numInterpPoints = numnodes * pointsPerCluster;
	make_vector(clusters->x, numInterpPoints);
	make_vector(clusters->y, numInterpPoints);
	make_vector(clusters->z, numInterpPoints);
	make_vector(clusters->q, numInterpPoints*8);
	make_vector(clusters->w, numInterpPoints);  // will be used in singularity subtraction
//	memset(clusters->q,0,numInterpPoints*sizeof(double));
	clusters->num=numInterpPoints;

	for (int i=0; i< numInterpPoints; i++){
		clusters->w[i]=0.0;
	}
	for (int i=0; i< 8*numInterpPoints; i++){
		clusters->q[i]=0.0;
	}

#pragma acc data copyin(tt[0:torderlim],ww[0:torderlim], \
		sources->x[0:sources->num], sources->y[0:sources->num], sources->z[0:sources->num], sources->q[0:sources->num], sources->w[0:sources->num]) \
		copy(clusters->x[0:clusters->num], clusters->y[0:clusters->num], clusters->z[0:clusters->num], clusters->q[0:8*numInterpPoints], clusters->w[0:clusters->num])

	addNodeToArray_hermite(troot, sources, clusters, order, numInterpPoints, pointsPerCluster);

	return;
}



void pc_treecode_hermite(struct tnode *p, struct batch *batches,
                 struct particles *sources, struct particles *targets, struct particles *clusters,
                 double *tpeng, double *EnP)
{
	printf("Entered pc_treecoode_hermite.\n");
    /* local variables */
    int i, j;

    for (i = 0; i < targets->num; i++)
        EnP[i] = 0.0;
    
#pragma acc data copyin(sources->x[0:sources->num], sources->y[0:sources->num], sources->z[0:sources->num], sources->q[0:sources->num], sources->w[0:sources->num], \
		targets->x[0:targets->num], targets->y[0:targets->num], targets->z[0:targets->num], targets->q[0:targets->num], \
		clusters->x[0:clusters->num], clusters->y[0:clusters->num], clusters->z[0:clusters->num], clusters->q[0:8*(clusters->num)]) \
		copy(EnP[0:targets->num])
    {
//#pragma acc data copyin(targets->x[0:targets->num], targets->y[0:targets->num], targets->z[0:targets->num], targets->q[0:targets->num], EnP[0:targets->num]) \
//		copyout(EnP[0:targets->num]) \
//		present(sources->x[0:sources->num], sources->y[0:sources->num], sources->z[0:sources->num], sources->q[0:sources->num], sources->w[0:sources->num], \
//				clusters->x[0:clusters->num], clusters->y[0:clusters->num], clusters->z[0:clusters->num], clusters->q[0:clusters->num])
//    {

    for (i = 0; i < batches->num; i++) {
        for (j = 0; j < p->num_children; j++) {
            compute_pc_hermite(p->child[j],
                batches->index[i], batches->center[i], batches->radius[i],
                sources->x, sources->y, sources->z, sources->q, sources->w,
                targets->x, targets->y, targets->z, targets->q, EnP,
				clusters->x, clusters->y, clusters->z, clusters->q);
        }
    }
}

    printf("Exited the main comp_pc call.\n");
    *tpeng = sum(EnP, targets->num);

    return;

} /* END of function pc_treecode */




void compute_pc_hermite(struct tnode *p,
                int *batch_ind, double *batch_mid, double batch_rad,
                double *xS, double *yS, double *zS, double *qS, double *wS,
                double *xT, double *yT, double *zT, double *qT, double *EnP,
				double * clusterX, double * clusterY, double * clusterZ, double * clusterM )
{
//	printf("Entering compute_cp_hermite.  Batch start: %d.  Batch end: %d.\n", batch_ind[0]-1, batch_ind[1]);
    /* local variables */
    double dist;
    double tx, ty, tz;
    int i, j, k, ii, kk;
    double dxt,dyt,dzt,tempPotential;
    double temp_i[torderlim], temp_j[torderlim], temp_k[torderlim];

    /* determine DIST for MAC test */
    tx = batch_mid[0] - p->x_mid;
    ty = batch_mid[1] - p->y_mid;
    tz = batch_mid[2] - p->z_mid;
    dist = sqrt(tx*tx + ty*ty + tz*tz);



    if (((p->radius + batch_rad) < dist * sqrt(thetasq)) && (p->sqradius != 0.00) && (torderlim*torderlim*torderlim < p->numpar) ) {


	int numberOfTargets = batch_ind[1] - batch_ind[0] + 1;
	int numberOfInterpolationPoints = torderlim*torderlim*torderlim;
	int clusterStart = numberOfInterpolationPoints*p->node_index;

//	double clusterX[numberOfInterpolationPoints], clusterY[numberOfInterpolationPoints], clusterZ[numberOfInterpolationPoints], localMoments[numberOfInterpolationPoints];
//	double clusterXYZM[4*numberOfInterpolationPoints];


//	// Fill some local cluster arrays from the cluster itself
//	int k1,k2,k3;
//	kk = -1;
//	for (k3 = 0; k3 < torderlim; k3++) {
//		for (k2 = 0; k2 < torderlim; k2++) {
//			for (k1 = 0; k1 < torderlim; k1++) {
//				kk++;
//				localMoments[kk] = p->ms[kk];
//				clusterX[kk] = p->tx[k1];
//				clusterY[kk] = p->ty[k2];
//				clusterZ[kk] = p->tz[k3];
//			}
//		}
//	}





	double xi,yi,zi,rinv,r3inv,r5inv,r7inv;
	int batchStart = batch_ind[0] - 1, sourceIdx;
//	double pointvals[4];
# pragma acc kernels present(xT,yT,zT,qT,EnP, clusterX, clusterY, clusterZ, clusterM)
    {
	#pragma acc loop independent
	for (i = 0; i < numberOfTargets; i++){
		tempPotential = 0.0;
		xi = xT[ batchStart + i];
		yi = yT[ batchStart + i];
		zi = zT[ batchStart + i];
		for (j = 0; j < numberOfInterpolationPoints; j++){
			sourceIdx = clusterStart + j;
			// Compute x, y, and z distances between target i and interpolation point j
			dxt = xi - clusterX[sourceIdx];
			dyt = yi - clusterY[sourceIdx];
			dzt = zi - clusterZ[sourceIdx];


//			tempPotential += localMoments[j] / sqrt(dxt*dxt + dyt*dyt + dzt*dzt);
			rinv = 1 / sqrt(dxt*dxt + dyt*dyt + dzt*dzt);
			r3inv = rinv*rinv*rinv;
			r5inv = r3inv*rinv*rinv;
			r7inv = r5inv*rinv*rinv;
//			printf("%1.2e\n", rinv);
//			if (isnan(rinv)){
//				printf("xi = %1.2e\n", xi);
//				printf("yi = %1.2e\n", yi);
//				printf("zi = %1.2e\n", zi);
//				printf("xC = %1.2e\n", clusterX[sourceIdx]);
//				printf("yC = %1.2e\n", clusterY[sourceIdx]);
//				printf("zC = %1.2e\n\n", clusterZ[sourceIdx]);
//			}

//			printf("%1.2e \n", clusterM[8*sourceIdx+0]);
//			printf("%1.2e \n\n", clusterM[8*sourceIdx+7]);
			tempPotential += 	( clusterM[8*sourceIdx+0] * rinv
								+ r3inv * ( clusterM[8*sourceIdx+1]*dxt +  clusterM[8*sourceIdx+2]*dyt +  clusterM[8*sourceIdx+3]*dzt )
								+ 3*r5inv * ( clusterM[8*sourceIdx+4]*dxt*dyt +  clusterM[8*sourceIdx+5]*dyt*dzt +  clusterM[8*sourceIdx+6]*dxt*dzt )
								+ 15*r7inv * clusterM[8*sourceIdx+7]*dxt*dyt*dzt
								);

						}
//		printf("%1.2e \n", tempPotential);
		EnP[batchStart + i] += tempPotential;
	}
    }



    } else {
    /*
     * If MAC fails check to see if there are children. If not, perform direct
     * calculation. If there are children, call routine recursively for each.
     */


        if (p->num_children == 0) {
//        	printf("MAC rejected, and node has no children.  Calling pc_comp_dierct()...\n");
            pc_comp_direct(p->ibeg, p->iend, batch_ind[0], batch_ind[1],
                           xS, yS, zS, qS, wS, xT, yT, zT, qT, EnP);
        } else {
//        	printf("MAC rejected, recursing over children...\n");
            for (i = 0; i < p->num_children; i++) {
                compute_pc_hermite(p->child[i], batch_ind, batch_mid, batch_rad,
                			xS, yS, zS, qS, wS, xT, yT, zT, qT, EnP,
							clusterX, clusterY, clusterZ, clusterM);
            }
        }
    }

    return;

} /* END of function compute_pc */









void pc_comp_ms_modifiedF_hermite(struct tnode *p, double *xS, double *yS, double *zS, double *qS, double *wS,
		double *clusterX, double *clusterY, double *clusterZ, double *clusterQ){

	int i,j,k;
	int pointsPerCluster = torderlim*torderlim*torderlim;
	int pointsInNode = p->numpar;
	int startingIndexInClusters = p->node_index * pointsPerCluster;
	int startingIndexInSources = p->ibeg-1;

	double x0, x1, y0, y1, z0, z1;  // bounding box


	double weights[torderlim];
	double dj[torderlim],wx[torderlim],wy[torderlim],wz[torderlim];
	double *modifiedF;
	make_vector(modifiedF,pointsInNode);

	double nodeX[torderlim], nodeY[torderlim], nodeZ[torderlim];

	int *exactIndX, *exactIndY, *exactIndZ;
	make_vector(exactIndX, pointsInNode);
	make_vector(exactIndY, pointsInNode);
	make_vector(exactIndZ, pointsInNode);

	// Set the bounding box.

	x0 = p->x_min;
	x1 = p->x_max;
	y0 = p->y_min;
	y1 = p->y_max;
	z0 = p->z_min;
	z1 = p->z_max;

	// Make and zero-out arrays to store denominator sums
	double sumX, sumY, sumZ;


#pragma acc kernels present(xS, yS, zS, qS, wS, clusterX, clusterY, clusterZ, clusterQ,tt,ww) \
	create(modifiedF[0:pointsInNode],exactIndX[0:pointsInNode],exactIndY[0:pointsInNode],exactIndZ[0:pointsInNode], \
			nodeX[0:torderlim],nodeY[0:torderlim],nodeZ[0:torderlim],weights[0:torderlim],dj[0:torderlim], \
			wx[0:torderlim],wy[0:torderlim],wz[0:torderlim])
	{

	#pragma acc loop independent
	for (j=0;j<pointsInNode;j++){
		modifiedF[j] = qS[startingIndexInSources+j]*wS[startingIndexInSources+j];
		exactIndX[j] = -1;
		exactIndY[j] = -1;
		exactIndZ[j] = -1;
	}

	//  Fill in arrays of unique x, y, and z coordinates for the interpolation points.
	#pragma acc loop independent
	for (i = 0; i < torderlim; i++) {
		nodeX[i] = x0 + (tt[i] + 1.0)/2.0 * (x1 - x0);
		nodeY[i] = y0 + (tt[i] + 1.0)/2.0 * (y1 - y0);
		nodeZ[i] = z0 + (tt[i] + 1.0)/2.0 * (z1 - z0);

	}

	// Compute weights

	#pragma acc loop independent
	for (j = 0; j < torderlim; j++){
		dj[j] = 1.0;
		wx[j] = -4.0 * ww[j] / (x1 - x0);
		wy[j] = -4.0 * ww[j] / (y1 - y0);
		wz[j] = -4.0 * ww[j] / (z1 - z0);
	}
	dj[0] = 0.25;
	dj[torder] = 0.25;
//	printf("wx: %1.2e\n", wx[0]);
//	#pragma acc loop independent
//	for (j = 0; j < torderlim; j++) {
//		weights[j] = ((j % 2 == 0)? 1 : -1) * dj[j];
//	}


	// Compute modified f values
	double sx,sy,sz,dx,dy,dz,denominator,w;



	#pragma acc loop independent
	for (i=0; i<pointsInNode;i++){ // loop through the source points

		sumX=0.0;
		sumY=0.0;
		sumZ=0.0;

		sx = xS[startingIndexInSources+i];
		sy = yS[startingIndexInSources+i];
		sz = zS[startingIndexInSources+i];


		#pragma acc loop independent
		for (j=0;j<torderlim;j++){  // loop through the degree

			dx = sx-nodeX[j];
			dy = sy-nodeY[j];
			dz = sz-nodeZ[j];

			if (fabs(dx)<DBL_MIN) exactIndX[i]=j;
			if (fabs(dy)<DBL_MIN) exactIndY[i]=j;
			if (fabs(dz)<DBL_MIN) exactIndZ[i]=j;

			// Increment the sums
//			w = weights[j];
			sumX += dj[j] / (dx*dx) + wx[j]/dx;
			sumY += dj[j] / (dy*dy) + wy[j]/dy;
			sumZ += dj[j] / (dz*dz) + wz[j]/dz;

		}

//		denominator = sumX*sumY*sumZ;

		denominator = 1.0;
		if (exactIndX[i]==-1) denominator *= sumX;
		if (exactIndY[i]==-1) denominator *= sumY;
		if (exactIndZ[i]==-1) denominator *= sumZ;

		modifiedF[i] /= denominator;
//		printf("%1.2e \n", modifiedF[i]);
	}


	// Compute moments for each interpolation point
	double numerator0,numerator1,numerator2,numerator3,numerator4,numerator5,numerator6,numerator7,  xn, yn, zn;
	double Ax,Ay,Az,Bx,By,Bz;
	double temp0,temp1,temp2,temp3,temp4,temp5,temp6,temp7;
	int k1, k2, k3, kk;
	double w1,w2,w3;
	double cx,cy,cz;

	#pragma acc loop independent
	for (j=0;j<pointsPerCluster;j++){ // loop over interpolation points, set (dx,dy,dz) for this point
		// compute k1, k2, k3 from j
		k1 = j%torderlim;
		kk = (j-k1)/torderlim;
		k2 = kk%torderlim;
		kk = kk - k2;
		k3 = kk / torderlim;

		cz = nodeZ[k3];
		w3 = weights[k3];

		cy = nodeY[k2];
		w2 = weights[k2];

		cx = nodeX[k1];
		w1 = weights[k1];


		// Fill cluster X, Y, and Z arrays
		clusterX[startingIndexInClusters + j] = cx;
		clusterY[startingIndexInClusters + j] = cy;
		clusterZ[startingIndexInClusters + j] = cz;


		// Increment cluster Q array
		temp0 = 0.0; temp1=0.0; temp2 = 0.0; temp3=0.0;
		temp4 = 0.0; temp5=0.0; temp6 = 0.0; temp7=0.0;
		#pragma acc loop independent
		for (i=0;i<pointsInNode; i++){  // loop over source points
			dx = xS[startingIndexInSources+i]-cx;
			dy = yS[startingIndexInSources+i]-cy;
			dz = zS[startingIndexInSources+i]-cz;

			numerator0=1.0;numerator1=1.0;numerator2=1.0;numerator3=1.0;
			numerator4=1.0;numerator5=1.0;numerator6=1.0;numerator7=1.0;

			Ax = ( dj[k1]/(dx*dx) + wx[k1]/dx );
			Ay = ( dj[k2]/(dy*dy) + wy[k2]/dy );
			Az = ( dj[k3]/(dz*dz) + wz[k3]/dz );
			Bx = ( dj[k1]/(dx) );
			By = ( dj[k2]/(dy) );
			Bz = ( dj[k3]/(dz) );
			if (exactIndX[i]==-1){
				numerator0 *=  Ax; 					// Aaa

				numerator1 *=  Bx; 					// Baa
				numerator2 *=  Ax; 					// Aba
				numerator3 *=  Ax; 					// Aab

				numerator4 *=  Bx; 					// Bba
				numerator5 *=  Ax; 					// Abb
				numerator6 *=  Bx; 					// Bab

				numerator7 *=  Bx; 					// Bbb

			}else{
				if (exactIndX[i]!=k1){ numerator0 *= 0; numerator1 *= 0; numerator2 *= 0; numerator3 *= 0; numerator4 *= 0; numerator5 *= 0; numerator6 *= 0; numerator7 *= 0;}
			}

			if (exactIndY[i]==-1){
				numerator0 *=  Ay;					// aAa

				numerator1 *=  Ay; 					// bAa
				numerator2 *=  By; 					// aBa
				numerator3 *=  Ay; 					// aAb

				numerator4 *=  By; 					// bBa
				numerator5 *=  By; 					// aBb
				numerator6 *=  Ay; 					// bAb

				numerator7 *=  By; 					// bBb

			}else{
				if (exactIndY[i]!=k2) { numerator0 *= 0; numerator1 *= 0; numerator2 *= 0; numerator3 *= 0; numerator4 *= 0; numerator5 *= 0; numerator6 *= 0; numerator7 *= 0;}
			}

			if (exactIndZ[i]==-1){
				numerator0 *=  Az;					// aaA

				numerator1 *=  Az;					// baA
				numerator2 *=  Az;					// abA
				numerator3 *=  Bz;					// aaB

				numerator4 *=  Az; 					// bbA
				numerator5 *=  Bz; 					// abB
				numerator6 *=  Bz; 					// baB

				numerator7 *=  Bz; 					// bbB

			}else{
				if (exactIndZ[i]!=k3) { numerator0 *= 0; numerator1 *= 0; numerator2 *= 0; numerator3 *= 0; numerator4 *= 0; numerator5 *= 0; numerator6 *= 0; numerator7 *= 0;}
			}





			temp0 += numerator0*modifiedF[i];
			temp1 += numerator1*modifiedF[i];
			temp2 += numerator2*modifiedF[i];
			temp3 += numerator3*modifiedF[i];
			temp4 += numerator4*modifiedF[i];
			temp5 += numerator5*modifiedF[i];
			temp6 += numerator6*modifiedF[i];
			temp7 += numerator7*modifiedF[i];


		}

		clusterQ[8*(startingIndexInClusters + j) + 0] += temp0;
		clusterQ[8*(startingIndexInClusters + j) + 1] += temp1;
		clusterQ[8*(startingIndexInClusters + j) + 2] += temp2;
		clusterQ[8*(startingIndexInClusters + j) + 3] += temp3;
		clusterQ[8*(startingIndexInClusters + j) + 4] += temp4;
		clusterQ[8*(startingIndexInClusters + j) + 5] += temp5;
		clusterQ[8*(startingIndexInClusters + j) + 6] += temp6;
		clusterQ[8*(startingIndexInClusters + j) + 7] += temp7;

//		printf("\n\n %1.2e\n", clusterQ[8*(startingIndexInClusters + j) + 0]);
//		printf("%1.2e\n", clusterQ[8*(startingIndexInClusters + j) + 1]);
//		printf("%1.2e\n", clusterQ[8*(startingIndexInClusters + j) + 2]);
//		printf("%1.2e\n", clusterQ[8*(startingIndexInClusters + j) + 3]);
//		printf("%1.2e\n", clusterQ[8*(startingIndexInClusters + j) + 5]);
//		printf("%1.2e\n\n", clusterQ[8*(startingIndexInClusters + j) + 7]);

	}


	}

	free_vector(modifiedF);
	free_vector(exactIndX);
	free_vector(exactIndY);
	free_vector(exactIndZ);


	return;
}


void addNodeToArray_hermite(struct tnode *p, struct particles *sources, struct particles *clusters, int order, int numInterpPoints, int pointsPerCluster)
{
	int torderlim = order+1;
	int startingIndex = p->node_index * pointsPerCluster;
	int i;


	if (torderlim*torderlim*torderlim < p->numpar){ // don't compute moments for clusters that won't get used

		pc_comp_ms_modifiedF_hermite(p, sources->x, sources->y, sources->z, sources->q, sources->w, \
				clusters->x,clusters->y,clusters->z,clusters->q);

		p->exist_ms = 1;


	}

	for (i = 0; i < p->num_children; i++) {
		addNodeToArray_hermite(p->child[i],sources,clusters,order,numInterpPoints,pointsPerCluster);
	}

	return;
}

