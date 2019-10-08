#ifndef H_TREEFUNCTIONS_H
#define H_TREEFUNCTIONS_H

#include "tnode.h"
#include "batch.h"
#include "particles.h"


/* declaration of treecode support functions */

/* used by cluster-particle and particle-cluster */
void remove_node(struct tnode *p);

void cleanup(struct tnode *p);


/* used by cluster-particle and particle-cluster Coulomb */
void setup(struct particles *particles, int order, double theta,
           double *xyzminmax);

void fill_in_cluster_data(struct particles *clusters, struct particles *sources, struct tnode *troot, int order, int numDevices, int numThreads, struct tnode_array * tree_array);
void fill_in_cluster_data_SS(struct particles *clusters, struct particles *sources, struct tnode *troot, int order);
void fill_in_cluster_data_hermite(struct particles *clusters, struct particles *sources, struct tnode *troot, int order);

void addNodeToArray_hermite(struct tnode *p, struct particles *sources, struct particles *clusters, int order, int numInterpPoints, int pointsPerCluster);
void addNodeToArray_hermite_SS(struct tnode *p, struct particles *sources, struct particles *clusters, int order, int numInterpPoints, int pointsPerCluster);
void addNodeToArray_SS(struct tnode *p, struct particles *sources, struct particles *clusters, int order, int numInterpPoints, int pointsPerCluster);


/* used by cluster-particle and particle-cluster Yukawa */
void setup_yuk(struct particles *particles, int order, double theta,
               double *xyzminmax);

void comp_tcoeff_yuk(double dx, double dy, double dz, double kappa);


/* used by cluster-particle */
void cp_create_tree_n0(struct tnode **p, struct particles *targets,
                       int ibeg, int iend, int maxparnode, double *xyzmm,
                       int level);

void cp_partition_8(double *x, double *y, double *z, double *q, double xyzmms[6][8],
                    double xl, double yl, double zl,
                    double lmax, int *numposchild,
                    double x_mid, double y_mid, double z_mid,
                    int ind[8][2]);

void cp_comp_ms(struct tnode *p);

void compute_cp2(struct tnode *ap, double *x, double *y, double *z,
                 double *EnP);


/* used by cluster-particle Coulomb */
void cp_interaction_list_treecode(struct tnode_array *tree_array, struct particles *clusters, struct batch *batches,
int *tree_inter_list, int *direct_inter_list,
struct particles *sources, struct particles *targets,
double *tpeng, double *EnP, int numDevices, int numThreads);

int cp_set_tree_index(struct tnode *p, int index);

void cp_create_tree_array(struct tnode *p, struct tnode_array *tree_array);

void cp_make_interaction_list(const struct tnode_array *tree_array, struct batch *batches,
int *tree_inter_list, int *direct_inter_list);

void cp_compute_interaction_list(int tree_numnodes, const int *tree_level,
const int *tree_numpar, const double *tree_radius,
const double *tree_x_mid, const double *tree_y_mid, const double *tree_z_mid,
int *batch_ind, double *batch_mid, double batch_rad,
int *batch_tree_list, int *batch_direct_list);

void cp_fill_cluster_interp(struct particles *clusters, struct particles *targets, struct tnode *troot, int order, int numDevices, int numThreads, struct tnode_array *tree_array);

void cp_comp_interp(struct tnode_array *tree_array, int idx,
double *xT, double *yT, double *zT, double *qT,
double *clusterX, double *clusterY, double *clusterZ);

void cp_compute_tree_interactions(struct tnode_array *tree_array, struct particles *clusters,
struct particles *targets, double *tpeng, double *EnP,
int numDevices, int numThreads);

/* used by cluster-particle Yukawa */
void cp_treecode_yuk(struct tnode *p, struct batch *batches,
                     struct particles *sources, struct particles *targets,
                     double kappa, double *tpeng, double *EnP,
                     double *timetree);


void compute_cp1_yuk(struct tnode *p, double *EnP,
                     double *x, double *y, double *z,
                     double kappa);

void cp_comp_direct_yuk(double *EnP, int ibeg, int iend,
                        double *x, double *y, double *z,
                        double kappa);


/* used by particle-cluster */
void pc_create_tree_n0(struct tnode **p, struct particles *sources,
                       int ibeg, int iend, int maxparnode, double *xyzmm,
                       int level);

int pc_set_tree_index(struct tnode *p, int index);

void pc_create_tree_array(struct tnode *p, struct tnode_array *tree_array);

void pc_partition_8(double *x, double *y, double *z, double *q, double *w,
                    double xyzmms[6][8], double xl, double yl, double zl,
                    double lmax, int *numposchild,
                    double x_mid, double y_mid, double z_mid,
                    int ind[8][2]);

void pc_comp_ms_modifiedF(struct tnode_array * tree_array, int idx, double *xS, double *yS, double *zS, double *qS, double *wS,
		double *clusterX, double *clusterY, double *clusterZ, double *clusterQ);

void pc_comp_ms_modifiedF_hermite(struct tnode *p, double *xS, double *yS, double *zS, double *qS, double *wS,
		double *clusterX, double *clusterY, double *clusterZ, double *clusterQ,
		double * clusterMx,double * clusterMy,double * clusterMz,double * clusterMxy,double * clusterMyz,double * clusterMzx,double * clusterMxyz);

void pc_comp_ms_modifiedF_SS(struct tnode *p, double *xS, double *yS, double *zS, double *qS, double *wS,
		double *clusterX, double *clusterY, double *clusterZ, double *clusterQ , double *clusterW);

void pc_comp_ms_modifiedF_hermite_SS(struct tnode *p, double *xS, double *yS, double *zS, double *qS, double *wS,
		double *clusterX, double *clusterY, double *clusterZ, double *clusterQ,
		double * clusterMx,double * clusterMy,double * clusterMz,double * clusterMxy,double * clusterMyz,double * clusterMzx,double * clusterMxyz,
		double * clusterW, double * clusterWx ,double * clusterWy,double * clusterWz,double * clusterWxy,double * clusterWyz,double * clusterWzx,double * clusterWxyz);


void pc_make_interaction_list(const struct tnode_array *tarray, struct batch *batches,
                              int *tree_inter_list, int *direct_inter_list);

void pc_compute_interaction_list(int tree_numnodes, const int *tree_level, 
                const int *tree_numpar, const double *tree_radius,
                const double *tree_x_mid, const double *tree_y_mid, const double *tree_z_mid,
                int *batch_ind, double *batch_mid, double batch_rad,
                int *batch_tree_list, int *batch_direct_list);


void pc_interaction_list_treecode(struct tnode_array *tree_array, struct particles *clusters, struct batch *batches,
                                  int *tree_inter_list, int *direct_inter_list,
                                  struct particles *sources, struct particles *targets,
                                  double *tpeng, double *EnP, int numDevices, int numThreads);

void pc_interaction_list_treecode_yuk(struct tnode_array *tree_array, struct particles *clusters, struct batch *batches,
                                  int *tree_inter_list, int *direct_inter_list,
                                  struct particles *sources, struct particles *targets,
                                  double *tpeng, double kappa, double *EnP, int numDevices, int numThreads);

void pc_interaction_list_treecode_dcf(struct tnode_array *tree_array, struct particles *clusters, struct batch *batches,
                                  int *tree_inter_list, int *direct_inter_list,
                                  struct particles *sources, struct particles *targets,
                                  double *tpeng, double eta, double *EnP, int numDevices, int numThreads);

void pc_interaction_list_treecode_tcf(struct tnode_array *tree_array, struct particles *clusters, struct batch *batches,
                                  int *tree_inter_list, int *direct_inter_list,
                                  struct particles *sources, struct particles *targets,
                                  double *tpeng, double kappa, double eta, double *EnP, int numDevices, int numThreads);

void pc_interaction_list_treecode_hermite_coulomb(struct tnode_array *tree_array, struct particles *clusters, struct batch *batches,
                                  int *tree_inter_list, int *direct_inter_list,
                                  struct particles *sources, struct particles *targets,
                                  double *tpeng, double *EnP, int numDevices, int numThreads);

void pc_interaction_list_treecode_hermite_yukawa(struct tnode_array *tree_array, struct particles *clusters, struct batch *batches,
                                  int *tree_inter_list, int *direct_inter_list,
                                  struct particles *sources, struct particles *targets,
                                  double *tpeng, double kappa, double *EnP, int numDevices, int numThreads);

void pc_interaction_list_treecode_hermite_dcf(struct tnode_array *tree_array, struct particles *clusters, struct batch *batches,
                                  int *tree_inter_list, int *direct_inter_list,
                                  struct particles *sources, struct particles *targets,
                                  double *tpeng, double eta, double *EnP, int numDevices, int numThreads);


/* used by particle-cluster Coulomb */
void pc_comp_direct(int ibeg, int iend, int batch_ibeg, int batch_iend,
                    double *xS, double *yS, double *zS, double *qS, double *wS,
                    double *xT, double *yT, double *zT, double *qT, double *EnP);

void pc_treecode_hermite_coulomb_SS(struct tnode *p, struct batch *batches,
                 struct particles *sources, struct particles *targets, struct particles *clusters,
				 double kappa, double *tpeng, double *EnP, int numDevices, int numThreads);

void compute_pc_hermite_SS(struct tnode *p,
                int *batch_ind, double *batch_mid, double batch_rad,
                double *xS, double *yS, double *zS, double *qS, double *wS,
                double *xT, double *yT, double *zT, double *qT, double kappaSq, double *EnP,
				double * clusterX, double * clusterY, double * clusterZ, double * clusterQ,
				double * clusterQx,double * clusterQy,double * clusterQz,double * clusterQxy,double * clusterQyz,double * clusterQxz,double * clusterQxyz,
				double * clusterW, double * clusterWx,double * clusterWy,double * clusterWz,double * clusterWxy,double * clusterWyz,double * clusterWxz,double * clusterWxyz);

/* used by particle-cluster Yukawa w/ singularity subtraction */

void pc_treecode_yuk_SS(struct tnode *p, struct batch *batches,
                     struct particles *sources, struct particles *targets, struct particles *clusters,
                     double kappa, double *tpeng, double *EnP, int numDevices, int numThreads);

void compute_pc_yuk_SS(struct tnode *p,
                int *batch_ind, double *batch_mid, double batch_rad,
                double *xS, double *yS, double *zS, double *qS, double *wS,
                double *xT, double *yT, double *zT, double *qT, double kappa, double *EnP,
				double * clusterX, double * clusterY, double * clusterZ, double * clusterM , double * clusterM2);

void pc_comp_direct_yuk_SS(int ibeg, int iend, int batch_ibeg, int batch_iend,
                    double *xS, double *yS, double *zS, double *qS, double *wS,
                    double *xT, double *yT, double *zT, double *qT, double kappa, double *EnP);


/* used by particle-cluster Coulomb kernel w/ singularity subtraction */

void pc_treecode_coulomb_SS(struct tnode *p, struct batch *batches,
                     struct particles *sources, struct particles *targets, struct particles *clusters,
                     double kappaSq, double *tpeng, double *EnP, int numDevices, int numThreads);

void compute_pc_coulomb_SS(struct tnode *p,
                int *batch_ind, double *batch_mid, double batch_rad,
                double *xS, double *yS, double *zS, double *qS, double *wS,
                double *xT, double *yT, double *zT, double *qT, double kappaSq, double *EnP,
				double * clusterX, double * clusterY, double * clusterZ, double * clusterM, double * clusterM2 );

void pc_comp_direct_coulomb_SS(int ibeg, int iend, int batch_ibeg, int batch_iend,
                    double *xS, double *yS, double *zS, double *qS, double *wS,
                    double *xT, double *yT, double *zT, double *qT, double kappaSq, double *EnP);


/* batch functions */
void setup_batch(struct batch **batches, double *batch_lim,
                 struct particles *particles, int batch_size);

void create_target_batch(struct batch *batches, struct particles *particles,
                     int ibeg, int iend, int maxparnode, double *xyzmm);

void cp_partition_batch(double *x, double *y, double *z, double *q, double xyzmms[6][8],
                    double xl, double yl, double zl, double lmax, int *numposchild,
                    double x_mid, double y_mid, double z_mid, int ind[8][2],
                    int *batch_reorder);

void create_source_batch(struct batch *batches, struct particles *particles,
                     int ibeg, int iend, int maxparnode, double *xyzmm);

void pc_partition_batch(double *x, double *y, double *z, double *q, double *w, double xyzmms[6][8],
                    double xl, double yl, double zl, double lmax, int *numposchild,
                    double x_mid, double y_mid, double z_mid, int ind[8][2],
                    int *batch_reorder);

void reorder_energies(int *batch_reorder, int numpars, double *tEn);
#endif /* H_TREEFUNCTIONS_H */
