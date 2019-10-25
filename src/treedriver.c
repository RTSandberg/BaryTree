#include <stdio.h>
#include <limits.h>
#include <omp.h>

#include "array.h"
#include "globvars.h"
#include "tnode.h"
#include "batch.h"
#include "particles.h"
#include "tools.h"
#include "tree.h"

#include "treedriver.h"


/* definition of primary treecode driver */

void treedriver(struct particles *sources, struct particles *targets,
                int order, double theta, int maxparnode, int batch_size,
                int pot_type, double kappa, int tree_type,
                double *tEn, double *tpeng, double *timetree, int numThreads)
{

    int verbosity = 0;
    /* local variables */
    struct tnode *troot = NULL;
    int level;
    double xyzminmax[6];
    
    /* batch variables */
    struct batch *batches = NULL;
    double batch_lim[6];
    
    /* date and time */
    double time1, time2, timeFillClusters1, timeFillClusters2;

    time1 = omp_get_wtime();
    
    level = 0;
    numleaves = 0;
    minlevel = INT_MAX;
    minpars = INT_MAX;
    maxlevel = 0;
    maxpars = 0;
    
    int *tree_inter_list, *tree_inter_list2;
    int *direct_inter_list, *direct_inter_list2;
    struct tnode_array *tree_array = NULL;
    numnodes = 0;
    struct particles *clusters = NULL;
    struct particles *target_clusters = NULL;
    
    clusters = malloc(sizeof(struct particles));

    /* call setup to allocate arrays for Taylor expansions and setup global vars */
    if (tree_type == 0) {
//        fprintf(stderr, "ERROR: Cluster-particle treecode currently disabled.\n");
//        exit(1);
        
        setup(targets, order, theta, xyzminmax);
        
        #pragma omp parallel
        {
            #pragma omp single
            {
                cp_create_tree_n0(&troot, targets, 1, targets->num,
                                  maxparnode, xyzminmax, level);
            }
        }
        
        int final_index = cp_set_tree_index(troot, 0);
        
        tree_array = malloc(sizeof(struct tnode_array));
        tree_array->numnodes = numnodes;
        make_vector(tree_array->ibeg, numnodes);
        make_vector(tree_array->iend, numnodes);
        make_vector(tree_array->numpar, numnodes);
        make_vector(tree_array->x_mid, numnodes);
        make_vector(tree_array->y_mid, numnodes);
        make_vector(tree_array->z_mid, numnodes);
        make_vector(tree_array->x_min, numnodes);

        make_vector(tree_array->y_min, numnodes);
        make_vector(tree_array->z_min, numnodes);
        make_vector(tree_array->x_max, numnodes);
        make_vector(tree_array->y_max, numnodes);
        make_vector(tree_array->z_max, numnodes);
        make_vector(tree_array->level, numnodes);
        make_vector(tree_array->cluster_ind, numnodes);
        make_vector(tree_array->radius, numnodes);

        cp_create_tree_array(troot, tree_array);
        setup_batch(&batches, batch_lim, sources, batch_size);
        create_source_batch(batches, sources, 1, sources->num, batch_size, batch_lim);
        
        if ((pot_type == 0) || (pot_type == 1)) {
            cp_fill_cluster_interp(clusters, targets, troot, order,
                                 numThreads, numThreads, tree_array);
        }
        
    } else if (tree_type == 2) {
        
        target_clusters = malloc(sizeof(struct particles));
        
        setup(sources, order, theta, xyzminmax);
        
        #pragma omp parallel
        {
            #pragma omp single
            {
                pc_create_tree_n0(&troot, sources, 1, sources->num,
                                  maxparnode, xyzminmax, level);
            }
        }

        int final_index = pc_set_tree_index(troot, 0);
        
        tree_array = malloc(sizeof(struct tnode_array));
        tree_array->numnodes = numnodes;
        make_vector(tree_array->ibeg, numnodes);
        make_vector(tree_array->iend, numnodes);
        make_vector(tree_array->numpar, numnodes);
        make_vector(tree_array->x_mid, numnodes);
        make_vector(tree_array->y_mid, numnodes);
        make_vector(tree_array->z_mid, numnodes);
        make_vector(tree_array->x_min, numnodes);

        make_vector(tree_array->y_min, numnodes);
        make_vector(tree_array->z_min, numnodes);
        make_vector(tree_array->x_max, numnodes);
        make_vector(tree_array->y_max, numnodes);
        make_vector(tree_array->z_max, numnodes);
        make_vector(tree_array->level, numnodes);
        make_vector(tree_array->cluster_ind, numnodes);
        make_vector(tree_array->radius, numnodes);

        pc_create_tree_array(troot, tree_array);
        setup_batch(&batches, batch_lim, targets, batch_size);
        create_target_batch(batches, targets, 1, targets->num,batch_size, batch_lim);


        if  (pot_type == 0) {
            fill_in_cluster_data(clusters, sources, troot, order,
                                 numThreads, numThreads, tree_array);

            cc_fill_cluster_batch_interp(target_clusters, targets, batches, order,
                                 numThreads, numThreads);
        }
    
        
    } else if (tree_type == 1) {
        setup(sources, order, theta, xyzminmax);
        
        #pragma omp parallel
        {
            #pragma omp single
            { 
                pc_create_tree_n0(&troot, sources, 1, sources->num,
                                  maxparnode, xyzminmax, level);
            }
        }

        int final_index = pc_set_tree_index(troot, 0);
        
        tree_array = malloc(sizeof(struct tnode_array));
        tree_array->numnodes = numnodes;
        make_vector(tree_array->ibeg, numnodes);
        make_vector(tree_array->iend, numnodes);
        make_vector(tree_array->numpar, numnodes);
        make_vector(tree_array->x_mid, numnodes);
        make_vector(tree_array->y_mid, numnodes);
        make_vector(tree_array->z_mid, numnodes);
        make_vector(tree_array->x_min, numnodes);

        make_vector(tree_array->y_min, numnodes);
        make_vector(tree_array->z_min, numnodes);
        make_vector(tree_array->x_max, numnodes);
        make_vector(tree_array->y_max, numnodes);
        make_vector(tree_array->z_max, numnodes);
        make_vector(tree_array->level, numnodes);
        make_vector(tree_array->cluster_ind, numnodes);
        make_vector(tree_array->radius, numnodes);

        pc_create_tree_array(troot, tree_array);
        setup_batch(&batches, batch_lim, targets, batch_size);
        create_target_batch(batches, targets, 1, targets->num,batch_size, batch_lim);

        timeFillClusters1 = omp_get_wtime();
        if        ((pot_type == 0) || (pot_type == 1)) {
            fill_in_cluster_data(clusters, sources, troot, order,
                                 numThreads, numThreads, tree_array);

        } else if ((pot_type == 2) || (pot_type == 3)) {
            fill_in_cluster_data_SS(clusters, sources, troot, order);

        } else if ((pot_type == 4) || (pot_type == 5)) {
            fill_in_cluster_data_hermite(clusters, sources, troot, order);

        } else if ((pot_type == 6) || (pot_type == 7)) {
        	printf("Not set up to do singularity subtraction for Hermite yet.\n");
            fill_in_cluster_data_hermite(clusters, sources, troot, order);
        }
        timeFillClusters2 = omp_get_wtime();
        timeFillClusters1 = timeFillClusters2-timeFillClusters1;
    }

    time2 = omp_get_wtime();
    timetree[0] = time2-time1;

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
        printf("                      torder: %d\n", torder);
        printf("                       theta: %f\n", theta);
        printf("                  maxparnode: %d\n", maxparnode);
        printf("               tree maxlevel: %d\n", maxlevel);
        printf("               tree minlevel: %d\n", minlevel);
        printf("                tree maxpars: %d\n", maxpars);
        printf("                tree minpars: %d\n", minpars);
        printf("            number of leaves: %d\n", numleaves);
        printf("             number of nodes: %d\n", numnodes);
#ifdef OPENACC_ENABLED
        printf("           number of devices: %d\n", numThreads);
#else
        printf("           number of threads: %d\n", numThreads);
#endif
        printf("           target batch size: %d\n", batch_size);
        printf("           number of batches: %d\n\n", batches->num);
    }

    time1 = omp_get_wtime();
    
    if (tree_type == 0) {
//        fprintf(stderr, "ERROR: Cluster-particle treecode is currently not enabled.\n");
//        exit(1);

        make_vector(tree_inter_list, batches->num * numnodes);
        make_vector(direct_inter_list, batches->num * numleaves);

        cp_make_interaction_list(tree_array, batches, tree_inter_list, direct_inter_list);
        
        if (pot_type == 0) {
        
            double timet1 = omp_get_wtime();
            
            cp_interaction_list_treecode(tree_array, clusters, batches,
                                         tree_inter_list, direct_inter_list, sources, targets,
                                         tpeng, tEn, numThreads, numThreads);
             
            double timet2 = omp_get_wtime();
                                         
            cp_compute_tree_interactions(tree_array, clusters, targets,
                                         tpeng, tEn, numThreads, numThreads);
                                         
            double timet3 = omp_get_wtime();
            timetree[1] = timet2 - timet1;
            timetree[2] = timet3 - timet2;
                                         
            //reorder_energies(orderarr, targets->num, tEn);
        }
        
        
    } else if (tree_type == 2) {
        make_vector(tree_inter_list, batches->num * numnodes);
        make_vector(direct_inter_list, batches->num * numleaves);

        pc_make_interaction_list(tree_array, batches, tree_inter_list, direct_inter_list);

        if (pot_type == 0) {
        
            double timet1 = omp_get_wtime();
            
            cc_interaction_list_treecode(tree_array, clusters, target_clusters, batches,
                                         tree_inter_list, direct_inter_list, sources, targets,
                                         tpeng, tEn, numThreads, numThreads);
                                         
            double timet2 = omp_get_wtime();
                                         
            cc_compute_batch_interp_interactions(batches, target_clusters, targets,
                                         tpeng, tEn, numThreads, numThreads);
                                         
            double timet3 = omp_get_wtime();
            timetree[1] = timet2 - timet1;
            timetree[2] = timet3 - timet2;
            
        }
        
        reorder_energies(batches->reorder, targets->num, tEn);
        
        
                                         

    } else if (tree_type == 1) {
        make_vector(tree_inter_list, batches->num * numnodes);
        make_vector(direct_inter_list, batches->num * numleaves);

        pc_make_interaction_list(tree_array, batches, tree_inter_list, direct_inter_list);

        if (pot_type == 0) {
            if (verbosity > 0) printf("Entering particle-cluster, pot_type=0 (Coulomb).\n");
            pc_interaction_list_treecode(tree_array, clusters, batches,
                                         tree_inter_list, direct_inter_list, sources, targets,
                                         tpeng, tEn, numThreads, numThreads);

        } else if (pot_type == 1) {
            if (verbosity > 0) printf("Entering particle-cluster, pot_type=1 (Yukawa).\n");
            pc_interaction_list_treecode_yuk(tree_array, clusters, batches,
                                             tree_inter_list, direct_inter_list, sources, targets,
                                             tpeng, kappa, tEn, numThreads, numThreads);

        } else if (pot_type == 2) {
            if (verbosity > 0) printf("Entering particle-cluster, pot_type=2 (Coulomb w/ singularity subtraction).\n");
            pc_treecode_coulomb_SS(troot, batches, sources, targets,clusters,
                                   kappa, tpeng, tEn, numThreads, numThreads);
//          pc_interaction_list_treecode_Coulomb_SS(tree_array, clusters, batches,
//                                                  tree_inter_list, direct_inter_list, sources, targets,
//                                                  tpeng, kappa, tEn, numThreads, numThreads);

        } else if (pot_type == 3) {
            if (verbosity > 0) printf("Entering particle-cluster, pot_type=3 (Yukawa w/ singularity subtraction).\n");
            pc_treecode_yuk_SS(troot, batches, sources, targets,clusters,
                               kappa, tpeng, tEn, numThreads, numThreads);

        } else if (pot_type == 4) {
            if (verbosity > 0) printf("Entering particle-cluster, pot_type=4 (Coulomb Hermite).\n");
            pc_interaction_list_treecode_hermite_coulomb(tree_array, clusters, batches,
                                                    tree_inter_list, direct_inter_list, sources, targets,
                                                    tpeng, tEn, numThreads, numThreads);

        } else if (pot_type == 5) {
            if (verbosity > 0) printf("Entering particle-cluster, pot_type=4 (Yukawa Hermite).\n");
            pc_interaction_list_treecode_hermite_yukawa(tree_array, clusters, batches,
                                                    tree_inter_list, direct_inter_list, sources, targets,
                                                    tpeng, kappa, tEn, numThreads, numThreads);

        } else if (pot_type == 6) {
            if (verbosity > 0) printf("Entering particle-cluster, pot_type=6 (Coulomb Hermite w/ singularity subtraction).\n");
            pc_treecode_hermite_coulomb_SS(troot, batches, sources, targets,clusters,
                                           kappa, tpeng, tEn, numThreads, numThreads);
        }
        
        reorder_energies(batches->reorder, targets->num, tEn);
    }

    time2 = omp_get_wtime();  // end time for tree evaluation
    timetree[3] = time2-time1;

    time1 = omp_get_wtime();

    cleanup(troot);

    // free interaction lists
    free_vector(tree_inter_list);
    free_vector(direct_inter_list);

    // free clusters
    free_vector(clusters->x);
    free_vector(clusters->y);
    free_vector(clusters->z);
    free_vector(clusters->q);
    free_vector(clusters->w);
    free(clusters);

    // free tree_array
    free_vector(tree_array->ibeg);
    free_vector(tree_array->iend);
    free_vector(tree_array->x_mid);
    free_vector(tree_array->y_mid);
    free_vector(tree_array->z_mid);
    free_vector(tree_array->x_min);
    free_vector(tree_array->y_min);
    free_vector(tree_array->z_min);
    free_vector(tree_array->x_max);
    free_vector(tree_array->y_max);
    free_vector(tree_array->z_max);
    free(tree_array);

    // free target batches
    free_vector(batches->reorder);
    free_matrix(batches->index);
    free_matrix(batches->center);
    free_vector(batches->radius);
    free_vector(batches->x_min);
    free_vector(batches->y_min);
    free_vector(batches->z_min);
    free_vector(batches->x_max);
    free_vector(batches->y_max);
    free_vector(batches->z_max);
    free(batches);
    
    
    // free clusters
    if (tree_type == 2) {
        free_vector(target_clusters->x);
        free_vector(target_clusters->y);
        free_vector(target_clusters->z);
        free_vector(target_clusters->q);
        free_vector(target_clusters->w);
        free(target_clusters);
    }
    

    time2 = omp_get_wtime();
    timetree[4] = time2-time1;

    return;

} /* END function treecode */
