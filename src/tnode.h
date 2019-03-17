#ifndef H_TNODE_H
#define H_TNODE_H

/* declaration of struct with tag tnode */
struct tnode
{
        int numpar, ibeg, iend;
        double x_min, y_min, z_min;
        double x_max, y_max, z_max;
        double x_mid, y_mid, z_mid;
        double radius, sqradius, aspect;
        int level, num_children, exist_ms;
        double *ms;
        struct tnode *child[8];      //Child is ptr to array of ptrs to tnode children

        int xdim, ydim, zdim;
        int xlind, ylind, zlind;
        int xhind, yhind, zhind;
    
        int node_index;
    
        double *tx, *ty, *tz;
        double *wx, *wy, *wz;
        double **mms;
};


struct tnode_array
{
        int numnodes;
        int *ibeg;
        int *iend;
        double *x_mid;
        double *y_mid;
        double *z_mid;
};


#endif /* H_TNODE_H */
