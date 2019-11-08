#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <zoltan.h>
#include <time.h>

#include “zoltan.h”

const unsigned m = 1664525u;
const unsigned c = 1013904223u;

typedef struct{
  int numGlobalPoints;
  int numMyPoints;
  ZOLTAN_ID_PTR myGlobalIDs;
  double *x;
  double *y;
  double *z;
  double *q;
  double *w;
} MESH_DATA;

typedef struct{
  ZOLTAN_ID_TYPE myGlobalID;
  double x;
  double y;
  double z;
  double q;
  double w;
} SINGLE_MESH_DATA;

/* Application defined query functions */

static int get_number_of_objects(void *data, int *ierr);
static void get_object_list(void *data, int sizeGID, int sizeLID,
            ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                  int wgt_dim, float *obj_wgts, int *ierr);
static int get_num_geometry(void *data, int *ierr);
static void get_geometry_list(void *data, int sizeGID, int sizeLID,
             int num_obj, ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
             int num_dim, double *geom_vec, int *ierr);
static void ztn_pack(void *data, int num_gid_entries, int num_lid_entries,
              ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id,
              int dest, int size, char *buf, int *ierr);
static void ztn_unpack(void *data, int num_gid_entries,
                ZOLTAN_ID_PTR global_id,
                int size, char *buf, int *ierr);
static int ztn_obj_size(void *data, int num_gid_entries, int num_lid_entries,
    ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr);




void load_balance(double *x, double *y, double *z, double *q, double *w, int numpars_local, int numpars_global)
{
    int rc, rank, numProcs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (rank == 0) fprintf(stderr,"Entered load_balance with %d ranks.\n", numProcs);


    float ver;
    struct Zoltan_Struct *zz;
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    int *parts;
    MESH_DATA mySources;


//    struct particles *sources = NULL;
//    struct particles *targets = NULL;
    int *particleOrder = NULL;
//    double *potential = NULL, *potential_direct = NULL;
//    double potential_engy = 0, potential_engy_glob = 0;
//    double potential_engy_direct = 0, potential_engy_direct_glob = 0;

    /* variables for date-time calculation */
//    double time_run[3], time_tree[9], time_direct;
//    double time_run_glob[3][3], time_tree_glob[3][9], time_direct_glob[3];
//    double time1, time2;


    printf("Initializing Zoltan.\n");
    if (Zoltan_Initialize(argc, argv, &ver) != ZOLTAN_OK) {
        printf("Zoltan failed to initialize. Exiting.\n");
        MPI_Finalize();
        exit(0);
    }
    printf("Initialized Zoltan.\n");


    mySources.numGlobalPoints = numpars_global;
    mySources.numMyPoints = numpars_local;
    // Don't need to allocate new arrays.  Just point to the x,y,z,q,w that are inputs to the load balancing function
    mySources.x = x;
    mySources.y = y;
    mySources.z = z;
    mySources.q = q;
    mySources.w = w;
//    mySources.x = malloc(N*sizeof(double));
//    mySources.y = malloc(N*sizeof(double));
//    mySources.z = malloc(N*sizeof(double));
//    mySources.q = malloc(N*sizeof(double));
//    mySources.w = malloc(N*sizeof(double));
    mySources.myGlobalIDs = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * numpars_local);


    for (int i = 0; i < numpars_local; ++i) {
        mySources.myGlobalIDs[i] = (ZOLTAN_ID_TYPE)(startingIndex + i);
    }

    zz = Zoltan_Create(MPI_COMM_WORLD);

    /* General parameters */

    Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
    Zoltan_Set_Param(zz, "LB_METHOD", "RCB");
    Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");
    Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1");
    Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "0");
    Zoltan_Set_Param(zz, "RETURN_LISTS", "ALL");
    Zoltan_Set_Param(zz, "AUTO_MIGRATE", "TRUE");

    /* RCB parameters */

    Zoltan_Set_Param(zz, "RCB_OUTPUT_LEVEL", "0");
    Zoltan_Set_Param(zz, "RCB_RECTILINEAR_BLOCKS", "1");

    /* Query functions, to provide geometry to Zoltan */

    Zoltan_Set_Num_Obj_Fn(zz, get_number_of_objects, &mySources);
    Zoltan_Set_Obj_List_Fn(zz, get_object_list, &mySources);
    Zoltan_Set_Num_Geom_Fn(zz, get_num_geometry, &mySources);
    Zoltan_Set_Geom_Multi_Fn(zz, get_geometry_list, &mySources);
    Zoltan_Set_Obj_Size_Fn(zz, ztn_obj_size, &mySources);
    Zoltan_Set_Pack_Obj_Fn(zz, ztn_pack, &mySources);
    Zoltan_Set_Unpack_Obj_Fn(zz, ztn_unpack, &mySources);

    double x_min = minval(mySources.x, mySources.numMyPoints);
    double x_max = maxval(mySources.x, mySources.numMyPoints);
    double y_min = minval(mySources.y, mySources.numMyPoints);
    double y_max = maxval(mySources.y, mySources.numMyPoints);
    double z_min = minval(mySources.z, mySources.numMyPoints);
    double z_max = maxval(mySources.z, mySources.numMyPoints);

    rc = Zoltan_LB_Partition(zz, /* input (all remaining fields are output) */
                &changes,        /* 1 if partitioning was changed, 0 otherwise */
                &numGidEntries,  /* Number of integers used for a global ID */
                &numLidEntries,  /* Number of integers used for a local ID */
                &numImport,      /* Number of vertices to be sent to me */
                &importGlobalGids,  /* Global IDs of vertices to be sent to me */
                &importLocalGids,   /* Local IDs of vertices to be sent to me */
                &importProcs,    /* Process rank for source of each incoming vertex */
                &importToPart,   /* New partition for each incoming vertex */
                &numExport,      /* Number of vertices I must send to other processes*/
                &exportGlobalGids,  /* Global IDs of the vertices I must send */
                &exportLocalGids,   /* Local IDs of the vertices I must send */
                &exportProcs,    /* Process to which I send each of the vertices */
                &exportToPart);  /* Partition to which each vertex will belong */

    int i = 0;
    while (i < mySources.numMyPoints) {
        if ((int)mySources.myGlobalIDs[i] < 0) {
            mySources.x[i] = mySources.x[mySources.numMyPoints-1];
            mySources.y[i] = mySources.y[mySources.numMyPoints-1];
            mySources.z[i] = mySources.z[mySources.numMyPoints-1];
            mySources.q[i] = mySources.q[mySources.numMyPoints-1];
            mySources.w[i] = mySources.w[mySources.numMyPoints-1];
            mySources.myGlobalIDs[i] = mySources.myGlobalIDs[mySources.numMyPoints-1];
            mySources.numMyPoints--;
        } else {
          i++;
        }
    }

    x_min = minval(mySources.x, mySources.numMyPoints);
    x_max = maxval(mySources.x, mySources.numMyPoints);
    y_min = minval(mySources.y, mySources.numMyPoints);
    y_max = maxval(mySources.y, mySources.numMyPoints);
    z_min = minval(mySources.z, mySources.numMyPoints);
    z_max = maxval(mySources.z, mySources.numMyPoints);

    if (rc != ZOLTAN_OK) {
        printf("Error! Zoltan has failed. Exiting. \n");
        MPI_Finalize();
        Zoltan_Destroy(&zz);
        exit(0);
    }

    if (rank == 0) fprintf(stderr,"Zoltan load balancing has finished.\n");


    return;

} /* END of function load_balance */


