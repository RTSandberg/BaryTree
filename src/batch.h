#ifndef H_BATCH_H
#define H_BATCH_H

/* declaration of struct with tag batch */
struct batch
{
        int num;
        int *reorder;
        int **index;
        double **center;
        double *radius;
        
        int *ibeg;
        int *iend;
        int *numpar;
        
        double *x_mid;
        double *y_mid;
        double *z_mid;

        double *x_min;
        double *y_min;
        double *z_min;

        double *x_max;
        double *y_max;
        double *z_max;
        
        
};

#endif /* H_BATCH_H */
