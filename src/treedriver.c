#include <stdio.h>
#include <mpi.h>
#include <limits.h>
#include <omp.h>

#include "array.h"
#include "globvars.h"
#include "tnode.h"
#include "batch.h"
#include "particles.h"
#include "tools.h"
#include "tree.h"
#include "structAllocations.h"
#include "interaction-masks.h"
#include "kernels/kernels.h"

#include "treedriver.h"


/* definition of primary treecode driver */

void treedriver(struct particles *sources, struct particles *targets,
                int interpolationOrder, double theta, int maxparnode, int batch_size,
                double kappa,
                double (*directKernel)(double,  double,  double,  double,  double,  double,  double,  double,  double, double),
                double (*approxKernel)(double,  double,  double,  double,  double,  double,  double,  double,  double, double),
                int tree_type, double *tEn, double *tpeng, double *time_tree)
{

    double time_beg = MPI_Wtime();

    int verbosity = 0;

    int rank=0, numProcs=1, ierr;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (verbosity > 0) printf("Set rank %i and numProcs %i.\n", rank, numProcs);


    /* local variables */
    struct tnode *troot = NULL;
    double xyzminmax[6];
    
    /* batch variables */
    struct batch *batches = NULL;
    double batch_lim[6];
    
    /* date and time */
    double time1;
    
    int *tree_inter_list, *local_tree_inter_list;
    int *direct_inter_list, *local_direct_inter_list;

    struct tnode_array *tree_array = NULL;
    int numnodes = 0, numleaves = 0;

    struct particles *clusters = NULL;
    clusters = malloc(sizeof(struct particles));

    int max_batch_approx = 0;
    int max_batch_direct = 0;


    /* call setup to allocate arrays for Taylor expansions and setup global vars */
    if (tree_type == 0) {

        fprintf(stderr, "ERROR: Cluster-particle treecode currently disabled.\n");
        exit(1);

        //setup(targets, order, theta, xyzminmax);
        //cp_create_tree_n0(&troot, targets, 1, targets->num,
        //                  maxparnode, xyzminmax, level);
        //setup_batch(&batches, batch_lim, targets, batch_size);
        //create_source_batch(batches, sources, 1, sources->num,
        //                    batch_size, batch_lim);
        
    } else if (tree_type == 1) {

        time1 = MPI_Wtime();
        setup(sources, interpolationOrder, theta, xyzminmax);
        pc_create_tree_n0(&troot, sources, 1, sources->num,
                          maxparnode, xyzminmax, 0, &numnodes, &numleaves);
        int final_index = pc_set_tree_index(troot, 0);
        time_tree[1] = MPI_Wtime() - time1; //time_treebuild
        
        
        time1 = MPI_Wtime();

        tree_array = malloc(sizeof(struct tnode_array));
        allocate_tree_array(tree_array, numnodes);
        pc_create_tree_array(troot, tree_array);

        time_tree[2] = MPI_Wtime() - time1; //time_maketreearray
        

        time1 = MPI_Wtime();

        setup_batch(&batches, batch_lim, targets, batch_size);
        create_target_batch(batches, targets, 1, targets->num, batch_size, batch_lim);

        time_tree[3] = MPI_Wtime() - time1; //time_createbatch
        

        time1 = MPI_Wtime();

        if        ((directKernel == &coulombKernel) || (directKernel == &yukawaKernel)) {
//            if (rank == 0) printf("Calling fill_in_cluster_data.\n");
            fill_in_cluster_data(clusters, sources, troot, interpolationOrder, tree_array);

        } else if ((directKernel == &coulombKernel_SS_direct) || (directKernel == &yukawaKernel_SS_direct)) {
//            if (rank == 0) printf("Calling fill_in_cluster_data_SS.\n");
            fill_in_cluster_data_SS(clusters, sources, troot, interpolationOrder, tree_array);
//            printf("First element of clusterQ and clusterW: %f, %f\n",clusters->q[0], clusters->w[0] );

        } else {
            if (rank == 0) printf("Not sure how to fill cluster data... aborting.\n");
            exit(1);
        }

        ////// For now, just fill in cluster data for Coulomb and Yukawa Lagrange.  Need to handle Hermite and SS separately.

//        if         ((pot_type == 0) || (pot_type==1)) {
//            fill_in_cluster_data(   clusters, sources, troot, interpolationOrder,
//                                  tree_array);
//
//        } else if  ((pot_type == 2) || (pot_type==3)) {
////            fill_in_cluster_data_SS(clusters, sources, troot, interpolationOrder);
//            fill_in_cluster_data_SS(   clusters, sources, troot, interpolationOrder,
//                                              tree_array);
//
//        } else if  ((pot_type == 4) || (pot_type==5)) {
//            fill_in_cluster_data_hermite(clusters, sources, troot, interpolationOrder);
//
//        } else if  ((pot_type == 6) || (pot_type==7)) {
//            fill_in_cluster_data_hermite(clusters, sources, troot, interpolationOrder);
//        }
        time_tree[4] = MPI_Wtime() - time1; //time_fillclusters
    }
    

    if (verbosity > 0) {
        printf("Tree information: \n\n");

        printf("                      numpar: %d\n", troot->numpar);
        printf("                       x_mid: %e\n", troot->x_mid);
        printf("                       y_mid: %e\n", troot->y_mid);
        printf("                       z_mid: %e\n\n", troot->z_mid);
        printf("                      radius: %f\n\n", troot->radius);
        printf("                       x_len: %e\n", troot->x_max - troot->x_min);
        printf("                       y_len: %e\n", troot->y_max - troot->y_min);
        printf("                       z_len: %e\n\n", troot->z_max - troot->z_min);
        printf("                      torder: %d\n", interpolationOrder);
        printf("                       theta: %f\n", theta);
        printf("                  maxparnode: %d\n", maxparnode);
        printf("            number of leaves: %d\n", numleaves);
        printf("             number of nodes: %d\n", numnodes);
        printf("           target batch size: %d\n", batch_size);
        printf("           number of batches: %d\n\n", batches->num);
    }


    if (tree_type == 0) {
        fprintf(stderr, "ERROR: Cluster-particle treecode is currently not disabled.\n");
        exit(1);
//        if (pot_type == 0) {
//            cp_treecode(troot, batches, sources, targets,
//                        tpeng, tEn, &timetree[1]);
//        }
//        } else if (pot_type == 1) {
//            cp_treecode_yuk(troot, batches, sources, targets, kappa,
//                            tpeng, tEn, &timetree[1]);
//        }
    } else if (tree_type == 1) {

        MPI_Barrier(MPI_COMM_WORLD);
  
        time1 = MPI_Wtime();

        int numNodesOnProc[numProcs];
        MPI_Allgather(&numnodes, 1, MPI_INT, numNodesOnProc, 1, MPI_INT, MPI_COMM_WORLD);

        int pointsPerCluster = (interpolationOrder+1)*(interpolationOrder+1)*(interpolationOrder+1);

        struct tnode_array *let_tree_array = NULL;
        let_tree_array = malloc(sizeof(struct tnode_array));
        int let_tree_array_length = 0;

        struct particles *let_clusters = NULL;
        let_clusters = malloc(sizeof(struct particles)); // let_clusters will hold all cluster data for LET
        int let_clusters_length = 0; // previously let_clusters included the local.  Now it should not

        struct particles *let_sources = NULL;
        let_sources = malloc(sizeof(struct particles));  // let_sources will hold all source nodes needed for direct interactions
        int let_sources_length = 0;  // previously let_sources included local.  Now it should not


        
        MPI_Win win_x_mid, win_y_mid, win_z_mid, win_radius, win_numpar, win_ibeg, win_iend, win_level;
        MPI_Win win_clusters_x, win_clusters_y, win_clusters_z, win_clusters_q, win_clusters_w;
        MPI_Win win_sources_x, win_sources_y, win_sources_z, win_sources_q, win_sources_w;
        MPI_Win win_children, win_num_children;
        
        MPI_Win_create(tree_array->x_mid,  numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_x_mid);
        MPI_Win_create(tree_array->y_mid,  numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_y_mid);
        MPI_Win_create(tree_array->z_mid,  numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_z_mid);
        MPI_Win_create(tree_array->radius, numnodes*sizeof(double), sizeof(double),  MPI_INFO_NULL, MPI_COMM_WORLD, &win_radius);
        MPI_Win_create(tree_array->numpar, numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_numpar);
        MPI_Win_create(tree_array->ibeg,   numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_ibeg);
        MPI_Win_create(tree_array->iend,   numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_iend);
        MPI_Win_create(tree_array->level,  numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_level);
        MPI_Win_create(tree_array->num_children,  numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_num_children);
        MPI_Win_create(tree_array->children,    8*numnodes*sizeof(int),    sizeof(int),     MPI_INFO_NULL, MPI_COMM_WORLD, &win_children);

        MPI_Win_create(clusters->x, numnodes*pointsPerCluster*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_x);
        MPI_Win_create(clusters->y, numnodes*pointsPerCluster*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_y);
        MPI_Win_create(clusters->z, numnodes*pointsPerCluster*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_z);
        MPI_Win_create(clusters->q, numnodes*pointsPerCluster*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_q);
        MPI_Win_create(clusters->w, numnodes*pointsPerCluster*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_clusters_w);

        MPI_Win_create(sources->x, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_x);
        MPI_Win_create(sources->y, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_y);
        MPI_Win_create(sources->z, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_z);
        MPI_Win_create(sources->q, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_q);
        MPI_Win_create(sources->w, troot->numpar*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_sources_w);

        // Perform MPI round robin, filling LET with remote data
        int num_remote_approx_array[numProcs], new_sources_length_array[numProcs];
        int previous_let_clusters_length_array[numProcs], previous_let_sources_length_array[numProcs];
        MPI_Datatype approx_type[numProcs], direct_type[numProcs];
        int let_clusters_num = 0;

        int *sizeof_batch_approx, *sizeof_batch_direct;
        int *offset_batch_approx, *offset_batch_direct;

        make_vector(sizeof_batch_approx, batches->num);
        make_vector(sizeof_batch_direct, batches->num);
        make_vector(offset_batch_approx, batches->num);
        make_vector(offset_batch_direct, batches->num);

        int loopsize = batches->num;
        for (int i = 0; i < loopsize; i++) sizeof_batch_approx[i] = 0;
        for (int i = 0; i < loopsize; i++) sizeof_batch_direct[i] = 0;
        for (int i = 0; i < loopsize; i++) offset_batch_approx[i] = 0;
        for (int i = 0; i < loopsize; i++) offset_batch_approx[i] = 0;

        int total_batch_direct = 0;
        int total_batch_approx = 0;
         


        for (int procID = 1; procID < numProcs; ++procID) {

            int getFrom = (numProcs+rank-procID) % numProcs;

            // Allocate remote_tree_array
            struct tnode_array *remote_tree_array = NULL;
            remote_tree_array = malloc(sizeof(struct tnode_array));
            allocate_tree_array(remote_tree_array, numNodesOnProc[getFrom]);

            // Get remote_tree_array
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_x_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_y_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_z_mid);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_radius);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_numpar);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_ibeg);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_iend);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_level);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_children);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_num_children);

            
            MPI_Get(remote_tree_array->x_mid, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_x_mid);
            MPI_Get(remote_tree_array->y_mid, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_y_mid);
            MPI_Get(remote_tree_array->z_mid, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_z_mid);
            MPI_Get(remote_tree_array->radius, numNodesOnProc[getFrom], MPI_DOUBLE,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_DOUBLE, win_radius);
            MPI_Get(remote_tree_array->numpar, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_numpar);
            MPI_Get(remote_tree_array->ibeg, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_ibeg);
            MPI_Get(remote_tree_array->iend, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_iend);
            MPI_Get(remote_tree_array->level, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_level);

            MPI_Get(remote_tree_array->children, 8*numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, 8*numNodesOnProc[getFrom], MPI_INT, win_children);
            MPI_Get(remote_tree_array->num_children, numNodesOnProc[getFrom], MPI_INT,
                    getFrom, 0, numNodesOnProc[getFrom], MPI_INT, win_num_children);
            
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Win_unlock(getFrom, win_x_mid);
            MPI_Win_unlock(getFrom, win_y_mid);
            MPI_Win_unlock(getFrom, win_z_mid);
            MPI_Win_unlock(getFrom, win_radius);
            MPI_Win_unlock(getFrom, win_numpar);
            MPI_Win_unlock(getFrom, win_ibeg);
            MPI_Win_unlock(getFrom, win_iend);
            MPI_Win_unlock(getFrom, win_level);
            MPI_Win_unlock(getFrom, win_children);
            MPI_Win_unlock(getFrom, win_num_children);
            
            MPI_Barrier(MPI_COMM_WORLD);

            // Construct masks
            int *approx_list_packed; int *approx_list_unpacked; int *direct_list; int *direct_ibeg_list; int *direct_length_list;
            make_vector(approx_list_packed, numNodesOnProc[getFrom]);
            make_vector(approx_list_unpacked, numNodesOnProc[getFrom]);
            make_vector(direct_list, numNodesOnProc[getFrom]);
            make_vector(direct_ibeg_list, numNodesOnProc[getFrom]);
            make_vector(direct_length_list, numNodesOnProc[getFrom]);

            remote_interaction_lists(remote_tree_array, batches, approx_list_unpacked, approx_list_packed,
                                     direct_list, sizeof_batch_approx, sizeof_batch_direct);

            //MPI_Barrier(MPI_COMM_WORLD);

            // Count number of unique clusters adding to LET
            int numberOfUniqueClusters =  0;
            int previousTreeArrayLength = let_tree_array_length;
            for (int i = 0; i < numNodesOnProc[getFrom]; ++i) {
                numberOfUniqueClusters++;
                let_tree_array_length++;
            }
            //MPI_Barrier(MPI_COMM_WORLD);
            
            if (procID==1){
                allocate_tree_array(let_tree_array, let_tree_array_length);
            }else{
                reallocate_tree_array(let_tree_array, let_tree_array_length);
            }
            //MPI_Barrier(MPI_COMM_WORLD);

            int numberOfRemoteApprox = 0;
            int previous_let_clusters_length = let_clusters_length;

            int numberOfRemoteDirect = 0;
            int previous_let_sources_length = let_sources_length;

            // Fill in LET tree array from Remote tree array.
            int appendCounter = 0;
            for (int i = 0; i < numNodesOnProc[getFrom]; ++i) {

                let_tree_array->x_mid[previousTreeArrayLength + appendCounter] = remote_tree_array->x_mid[i];
                let_tree_array->y_mid[previousTreeArrayLength + appendCounter] = remote_tree_array->y_mid[i];
                let_tree_array->z_mid[previousTreeArrayLength + appendCounter] = remote_tree_array->z_mid[i];
                let_tree_array->radius[previousTreeArrayLength + appendCounter] = remote_tree_array->radius[i];
                let_tree_array->numpar[previousTreeArrayLength + appendCounter] = remote_tree_array->numpar[i];
                let_tree_array->level[previousTreeArrayLength + appendCounter] = remote_tree_array->level[i];
                    
                if (approx_list_unpacked[i] != -1) {
                    let_tree_array->cluster_ind[previousTreeArrayLength + appendCounter] = let_clusters_num;
                    let_clusters_length += pointsPerCluster;
                    let_clusters_num++;
                    numberOfRemoteApprox++;
                }
                    
                if (direct_list[i] != -1) {
                        
                    // Set the beginning and ending particle indices for the associated nodes in the local sources list
                    let_tree_array->ibeg[previousTreeArrayLength + appendCounter] = let_sources_length + 1;  // These are one-index based!!!
                    let_tree_array->iend[previousTreeArrayLength + appendCounter] = let_sources_length + remote_tree_array->numpar[i];
                    let_sources_length += remote_tree_array->numpar[i];
                        
                    // Determine displacements and lengths for getting prticles from remote sources list
                    direct_ibeg_list[numberOfRemoteDirect] = remote_tree_array->ibeg[i] - 1; // These are zero-index based!!!
                    direct_length_list[numberOfRemoteDirect] = remote_tree_array->numpar[i];
                    numberOfRemoteDirect++;
                }
                
                appendCounter+=1;
            }
            
            num_remote_approx_array[getFrom] = numberOfRemoteApprox;
            new_sources_length_array[getFrom] = let_sources_length - previous_let_sources_length;
            previous_let_clusters_length_array[getFrom] = previous_let_clusters_length; 
            previous_let_sources_length_array[getFrom] = previous_let_sources_length; 
            
            //MPI_Barrier(MPI_COMM_WORLD);
            
            // Use masks to get remote data
            for (int ii = 0; ii < numberOfRemoteApprox; ++ii)
                approx_list_packed[ii] *= pointsPerCluster;
            
            MPI_Type_create_indexed_block(numberOfRemoteApprox, pointsPerCluster, approx_list_packed, MPI_DOUBLE, &approx_type[getFrom]);
            MPI_Type_commit(&approx_type[getFrom]);
            MPI_Type_indexed(numberOfRemoteDirect, direct_length_list, direct_ibeg_list, MPI_DOUBLE, &direct_type[getFrom]);
            MPI_Type_commit(&direct_type[getFrom]);

            free_vector(approx_list_packed);
            free_vector(approx_list_unpacked);
            free_vector(direct_list);
            free_vector(direct_ibeg_list);
            free_vector(direct_length_list);
            free_tree_array(remote_tree_array);
        } //end loop over numProcs

        if (let_sources_length > 0) allocate_sources(let_sources, let_sources_length);
        if (let_clusters_length > 0) allocate_cluster(let_clusters, let_clusters_length);
    

        for (int procID = 1; procID < numProcs; ++procID) {

            int getFrom = (numProcs+rank-procID) % numProcs;

            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_x);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_y);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_z);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_w);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_clusters_q);

            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_x);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_y);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_z);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_q);
            MPI_Win_lock(MPI_LOCK_SHARED, getFrom, 0, win_sources_w);


            MPI_Get(&(let_clusters->x[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_x);
            MPI_Get(&(let_clusters->y[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_y);
            MPI_Get(&(let_clusters->z[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_z);
            MPI_Get(&(let_clusters->q[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_q);
            MPI_Get(&(let_clusters->w[previous_let_clusters_length_array[getFrom]]),
                    num_remote_approx_array[getFrom] * pointsPerCluster, MPI_DOUBLE,
                    getFrom, 0, 1, approx_type[getFrom], win_clusters_w);


            MPI_Get(&(let_sources->x[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_x);
            MPI_Get(&(let_sources->y[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_y);
            MPI_Get(&(let_sources->z[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_z);
            MPI_Get(&(let_sources->q[previous_let_sources_length_array[getFrom]]), 
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_q);
            MPI_Get(&(let_sources->w[previous_let_sources_length_array[getFrom]]),
                    new_sources_length_array[getFrom], MPI_DOUBLE,
                    getFrom, 0, 1, direct_type[getFrom], win_sources_w);

            MPI_Barrier(MPI_COMM_WORLD);
            
            MPI_Win_unlock(getFrom, win_clusters_x);
            MPI_Win_unlock(getFrom, win_clusters_y);
            MPI_Win_unlock(getFrom, win_clusters_z);
            MPI_Win_unlock(getFrom, win_clusters_q);
            MPI_Win_unlock(getFrom, win_clusters_w);
            
            MPI_Win_unlock(getFrom, win_sources_x);
            MPI_Win_unlock(getFrom, win_sources_y);
            MPI_Win_unlock(getFrom, win_sources_z);
            MPI_Win_unlock(getFrom, win_sources_q);
            MPI_Win_unlock(getFrom, win_sources_w);

            MPI_Barrier(MPI_COMM_WORLD);

        } // end loop over numProcs


        // Local particles
//MAKE THIS BETTER!
        make_vector(local_tree_inter_list, batches->num * tree_array->numnodes);
        make_vector(local_direct_inter_list, batches->num * tree_array->numnodes);
        pc_make_interaction_list(tree_array, batches, local_tree_inter_list, local_direct_inter_list,
        tree_array->numnodes, tree_array->numnodes);

        pc_interaction_list_treecode(tree_array, batches,
                        local_tree_inter_list, local_direct_inter_list,
                        sources->x, sources->y, sources->z, sources->q, sources->w,
                        targets->x, targets->y, targets->z, targets->q,
                        clusters->x, clusters->y, clusters->z, clusters->q, clusters->w,
                        tpeng, tEn, interpolationOrder,
                        sources->num, targets->num, clusters->num,
                        tree_array->numnodes, tree_array->numnodes,
                        (*directKernel),(*approxKernel),kappa);

        time_tree[5] = MPI_Wtime() - time1; //time_constructlet


        // Compute interaction lists based on LET

        if (numProcs > 1) {
        time1 = MPI_Wtime();

        max_batch_approx = maxval_int(sizeof_batch_approx, batches->num);
        max_batch_direct = maxval_int(sizeof_batch_direct, batches->num);
        
        if (max_batch_approx > 0) make_vector(tree_inter_list, batches->num * max_batch_approx);
        if (max_batch_direct > 0) make_vector(direct_inter_list, batches->num * max_batch_direct);
        pc_make_interaction_list(let_tree_array, batches, tree_inter_list, direct_inter_list,
                                 max_batch_approx, max_batch_direct);

        time_tree[6] = MPI_Wtime() - time1; //time_makeglobintlist
        time_tree[0] = MPI_Wtime() - time_beg; //time_setup
        
        //MPI_Barrier(MPI_COMM_WORLD);


        // After filling LET, call interaction_list_treecode
        time1 = MPI_Wtime(); // start timer for tree evaluation
//        if (pot_type == 0) {
            if (verbosity>0) printf("Entering particle-cluster, pot_type=0 (Coulomb).\n");
//            pc_interaction_list_treecode(let_tree_array, let_clusters, batches,
//                                         tree_inter_list, direct_inter_list, let_sources, targets,
//                                         tpeng, tEn, interpolationOrder);
            pc_interaction_list_treecode(let_tree_array, batches,
                                    tree_inter_list, direct_inter_list,
                                    let_sources->x, let_sources->y, let_sources->z, let_sources->q, let_sources->w,
                                    targets->x, targets->y, targets->z, targets->q,
                                    let_clusters->x, let_clusters->y, let_clusters->z, let_clusters->q, let_clusters->w,
                                    tpeng, tEn, interpolationOrder,
                                    let_sources->num, targets->num, let_clusters->num,
                                    max_batch_approx, max_batch_direct,
                                    (*directKernel),(*approxKernel), kappa);


            if (verbosity>0) printf("Exiting particle-cluster, pot_type=0 (Coulomb).\n");

//        } else if (pot_type == 1) {
//            if (verbosity>0) printf("Entering particle-cluster, pot_type=1 (Yukawa).  Need to modify to call on let_tree_array, let_clusters, etc. ????\n");
//            pc_interaction_list_treecode_yuk(let_tree_array, let_clusters, batches,
//                                             tree_inter_list, direct_inter_list, let_sources, targets,
//                                             tpeng, kappa, tEn);
//
//        } else if (pot_type == 2) {
//            if (verbosity>0) printf("Entering particle-cluster, pot_type=2 (Coulomb w/ singularity subtraction).\n");
////            pc_treecode_coulomb_SS(troot, batches, sources, targets,clusters,
////                                   kappa, tpeng, tEn);
//          pc_interaction_list_treecode_Coulomb_SS(let_tree_array, let_clusters, batches,
//                                                  tree_inter_list, direct_inter_list, let_sources, targets,
//                                                  tpeng, kappa, tEn);
//
//        } else if (pot_type == 3) {
//            if (verbosity>0) printf("Entering particle-cluster, pot_type=3 (Yukawa w/ singularity subtraction).\n");
////            pc_treecode_yuk_SS(troot, batches, sources, targets,clusters,
////                               kappa, tpeng, tEn);
//            pc_interaction_list_treecode_yuk_SS(let_tree_array, let_clusters, batches,
//                                             tree_inter_list, direct_inter_list, let_sources, targets,
//                                             tpeng, kappa, tEn);
//
//        } else if (pot_type == 4) {
//            if (verbosity>0) printf("Entering particle-cluster, pot_type=4 (Coulomb Hermite).\n");
//            pc_interaction_list_treecode_hermite_coulomb(let_tree_array, clusters, batches,
//                                                    tree_inter_list, direct_inter_list, sources, targets,
//                                                    tpeng, tEn);
//
//        }else if (pot_type == 5) {
//            if (verbosity>0) printf("Entering particle-cluster, pot_type=4 (Yukawa Hermite).\n");
//            pc_interaction_list_treecode_hermite_yukawa(let_tree_array, clusters, batches,
//                                                    tree_inter_list, direct_inter_list, sources, targets,
//                                                    tpeng, kappa, tEn);
//
//        }else if (pot_type == 6) {
//            if (verbosity>0) printf("Entering particle-cluster, pot_type=6 (Coulomb Hermite w/ singularity subtraction).\n");
//            pc_treecode_hermite_coulomb_SS(troot, batches, sources, targets, clusters,
//                                           kappa, tpeng, tEn);
//        }
        }
        
        reorder_energies(batches->reorder, targets->num, tEn);
    }
    time_tree[7] = MPI_Wtime()-time1; // end time for tree evaluation


    time1 = MPI_Wtime();
    cleanup(troot);

    // free interaction lists
    free_vector(local_tree_inter_list);
    free_vector(local_direct_inter_list);

    if (numProcs > 1) {
        if (max_batch_approx > 0) free_vector(tree_inter_list);
        if (max_batch_direct > 0) free_vector(direct_inter_list);
    }

    // free clusters
    free_vector(clusters->x);
    free_vector(clusters->y);
    free_vector(clusters->z);
    free_vector(clusters->q);
    free_vector(clusters->w);
    free(clusters);

    // free tree_array
    free_tree_array(tree_array);

    // free target batches
    free_vector(batches->reorder);
    free_matrix(batches->index);
    free_matrix(batches->center);
    free_vector(batches->radius);
    free(batches);
    
    time_tree[8] = MPI_Wtime() - time1; //time_cleanup

    MPI_Barrier(MPI_COMM_WORLD);

    return;
} /* END function treecode */

